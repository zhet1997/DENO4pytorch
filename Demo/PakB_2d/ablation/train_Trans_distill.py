import os
import sys
import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch.nn as nn

# 添加项目根目录到Python路径
sys.path.append('/data/wqn/DENO4pytorch')
sys.path.append('/data/wqn/DENO4pytorch/Models')
sys.path.append('/data/wqn/DENO4pytorch/Utilizes')
from Utilizes.visual_data import MatplotlibVision, TextLogger
import matplotlib.pyplot as plt
import time
import yaml
from Demo.PakB_2d.utilizes_pakB import get_origin, PakBAntiNormLoss, get_loader_pakB, clear_value_in_hole, DistillationLoss
from Tools.train_model.model_whole_life import WorkPrj
from fno.FNOs import FNO2d
from transformer.Transformers import FourierTransformer
from Tools.model_define.define_FNO import train, valid, inference, train_random, train_mask
from Demo.PakB_2d.ablation.ablation_PakB import get_loaders, get_setting, calculate_per
from Demo.PakB_2d.trains_PakB import train_supercondition, valid_supercondition, valid_detail, supredictor_list_windows
from Tools.pre_process.data_reform import channel_to_instance, fill_channels
from Tools.model_define.define_FNO import feature_transform
import torch.nn.functional as F
# import wandb
# os.chdir('/data/wqn/DENO4pytorch/Demo/PakB_2d')  # 注释掉工作目录切换


class ThreeStageScheduler:
    """
    三阶段交替训练调度器：渐进式学习策略
    阶段1：C学习基础映射 (train_c_only) - 只训练C网络学习基本的几何→场映射
    阶段2：C+S学习叠加 (train_cs_joint) - 联合训练学习叠加规律
    阶段3：S→C知识蒸馏 (distill_s_to_c) - 将S∘C的叠加知识蒸馏回C网络
    """
    def __init__(self, c_only_steps=3, cs_joint_steps=5, distill_steps=4):
        self.c_only_steps = c_only_steps      # 只训练C的批次数
        self.cs_joint_steps = cs_joint_steps  # 联合训练C+S的批次数  
        self.distill_steps = distill_steps    # 蒸馏训练的批次数
        self.cycle_length = c_only_steps + cs_joint_steps + distill_steps
        
    def get_training_mode(self, epoch, batch_idx, batches_per_epoch):
        """
        根据当前训练进度返回训练模式
        """
        global_step = epoch * batches_per_epoch + batch_idx
        cycle_position = global_step % self.cycle_length
        
        if cycle_position < self.c_only_steps:
            return 'train_c_only'
        elif cycle_position < self.c_only_steps + self.cs_joint_steps:
            return 'train_cs_joint'
        else:
            return 'distill_s_to_c'
            
    def get_cycle_info(self, epoch, batch_idx, batches_per_epoch):
        """返回当前周期信息，用于日志记录"""
        global_step = epoch * batches_per_epoch + batch_idx
        cycle_num = global_step // self.cycle_length
        cycle_position = global_step % self.cycle_length
        return cycle_num, cycle_position


class DataDistributionGenerator:
    """
    数据分布生成器：生成更复杂的孔洞组合案例
    核心机制：
    1. 从基础数据中分解单个孔洞的SDF配置
    2. 通过随机组合和空间变换生成新的多孔配置
    3. 确保生成的案例超出训练集的复杂度范围，用于蒸馏训练
    """
    def __init__(self, channel_num=16):
        self.channel_num = channel_num
        
    def generate_complex_cases(self, base_batch, complexity_factor=2):
        """
        生成复杂案例的核心逻辑
        Args:
            base_batch: (xx, yy) 基础训练批次
            complexity_factor: 复杂度倍增因子
        Returns:
            complex_xx: 生成的复杂孔洞配置
        """
        xx, yy = base_batch
        batch_size = xx.shape[0]
        
        # 1. 分解现有的孔洞配置为单孔组件
        hole_configs = self._decompose_holes(xx)
        
        # 2. 生成新的组合配置（增加复杂度）
        complex_configs = self._combine_holes(hole_configs, complexity_factor)
        
        # 3. 应用空间变换增加多样性
        augmented_configs = self._spatial_augmentation(complex_configs)
        
        return augmented_configs
        
    def _decompose_holes(self, xx):
        """
        分解多孔配置为单孔组件
        使用channel_to_instance逻辑将多孔SDF分解为单孔列表
        """
        # 将输入按channel_num分组，每组代表一个孔洞的SDF
        hole_list = channel_to_instance(xx, channel_num=self.channel_num, list=True)
        return hole_list
        
    def _combine_holes(self, hole_configs, complexity_factor):
        """
        组合生成更复杂的配置
        从现有孔洞中随机选择和组合，生成更复杂的多孔案例
        """
        batch_size = hole_configs[0].shape[0]
        num_holes = len(hole_configs)
        
        # 计算目标孔洞数（增加复杂度）
        target_holes = min(num_holes * complexity_factor, 8)  # 最多8个孔
        
        complex_list = []
        for _ in range(target_holes):
            # 随机选择孔洞配置
            selected_idx = np.random.randint(0, num_holes)
            selected_hole = hole_configs[selected_idx]
            
            # 随机选择批次中的样本
            batch_indices = np.random.randint(0, batch_size, size=batch_size)
            combined_hole = selected_hole[batch_indices]
            
            complex_list.append(combined_hole)
            
        # 拼接成新的复杂配置
        complex_xx = torch.cat(complex_list, dim=-1)
        return complex_xx
        
    def _spatial_augmentation(self, configs):
        """
        空间变换增加多样性
        应用旋转、缩放等变换增加数据的多样性
        """
        # 简单的随机噪声增强（可以扩展为更复杂的空间变换）
        noise_scale = 0.01
        noise = torch.randn_like(configs) * noise_scale
        augmented = configs + noise
        
        # 确保SDF值的合理范围
        augmented = torch.clamp(augmented, min=-2.0, max=2.0)
        
        return augmented


if __name__ == "__main__":
    ################################################################
    # configs
    ################################################################
    wandb_run = False  # 禁用wandb
    patch_num = 1
    dataset_train_list = [
        [1],
        [1, 2],
        # [1, 2, 3],
        # [1, 2, 3, 5],
        # [1, 2, 3, 5, 10],
    ]
    basic_dict, train_dict, pred_model_dict, _, distill_dict = get_setting()
    for save_number, dataset_train in enumerate(dataset_train_list):
        dataset_valid = [1, 2, 3]  # 只使用有数据文件的孔数

        name = 'Trans_Distill'  # 修改名称以区分蒸馏版本
        work_path = os.path.join('work_ablation', name + '_patch_' + str(patch_num) + '_super_' + str(save_number))
        work = WorkPrj(work_path)
        Logger = TextLogger(os.path.join(work_path, 'train.log'))
        Device = work.device

        basic_dict, train_dict, pred_model_dict, super_model_dict, distill_dict = get_setting()
        locals().update(basic_dict)
        locals().update(train_dict)

        basic_dict['work_path'] = work_path
        channel_num = in_dim

        if wandb_run:
            wandb.init(
                project="pak_B_film_cooling_ablation_500",  # 写自己的
                entity="turbo-1997",
                notes="const=350",
                name='Super_patch_'+str(patch_num)+'_' + str(dataset_train),
                config={
                    **basic_dict,
                    **train_dict,
                    **pred_model_dict,
                    **super_model_dict,
                }
            )

        print(epochs, learning_rate, scheduler_step, scheduler_gamma)

        # ################################################################
        # # load data
        # ################################################################
        train_loader, valid_loader_list, x_normalizer, y_normalizer = get_loaders(dataset_train,
                                                                                  dataset_valid,
                                                                                  train_num=ntrain,
                                                                                  valid_num=nvalid,
                                                                                  channel_num=channel_num,
                                                                                  batch_size=batch_size,
                                                                                  )
        # ################################################################
        # #  Neural Networks
        # ################################################################
        #
        # # 建立网络
        perd_model = FourierTransformer(**pred_model_dict).to(Device)
        super_model = FNO2d(in_dim=2, out_dim=1, **super_model_dict).to(Device)
        Net_model = supredictor_list_windows(perd_model, super_model, channel_num=in_dim, win_split=patch_num).to(Device)
        # # 损失函数
        Loss_func_train = PakBAntiNormLoss(weighted_cof=0, shreshold_cof=-100, x_norm=x_normalizer, y_norm=y_normalizer)
        Loss_func_valid = PakBAntiNormLoss(weighted_cof=0, shreshold_cof=0, x_norm=x_normalizer, y_norm=y_normalizer)
        # 蒸馏损失函数：用于S→C知识蒸馏阶段
        Distill_loss_func = DistillationLoss(alpha=distill_dict['alpha'], 
                                           base_loss_weight=distill_dict['base_loss_weight'], 
                                           x_norm=x_normalizer, y_norm=y_normalizer, shreshold_cof=-100)
        # # 优化算法
        Optimizer_0 = torch.optim.Adam(Net_model.pred_net.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-7)
        Optimizer_1 = torch.optim.Adam([
            {'params': Net_model.pred_net.parameters(), 'lr': learning_rate * 0.1},  # 学习率为默认的
            {'params': Net_model.super_net.parameters()}], lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-7)
        # # 下降策略
        Scheduler_0 = torch.optim.lr_scheduler.StepLR(Optimizer_0, step_size=scheduler_step, gamma=scheduler_gamma)
        Scheduler_1 = torch.optim.lr_scheduler.StepLR(Optimizer_1, step_size=scheduler_step, gamma=scheduler_gamma)
        # # 可视化
        Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))
        
        # 三阶段训练调度器：控制训练模式切换
        scheduler = ThreeStageScheduler(c_only_steps=distill_dict['c_only_steps'], 
                                      cs_joint_steps=distill_dict['cs_joint_steps'], 
                                      distill_steps=distill_dict['distill_steps'])
        
        # 数据分布生成器：为蒸馏阶段生成复杂案例
        data_generator = DataDistributionGenerator(channel_num=in_dim)

        star_time = time.time()
        
        # 基础训练日志
        log_loss = {
            'train_loss': [],
        }
        for hole_num in dataset_valid[:-2]:
            log_loss.update({'valid_loss_hole_' + str(hole_num): []})

        # 叠加训练日志
        log_super_loss = {
            'train_super_loss': [],
        }
        
        # 三阶段蒸馏训练日志
        log_distill_loss = {
            'train_c_only_loss': [],      # 阶段1：只训练C的损失
            'train_cs_joint_loss': [],    # 阶段2：联合训练损失
            'distill_s_to_c_loss': [],    # 阶段3：蒸馏损失
            'soft_target_loss': [],       # 软目标损失分量
            'hard_target_loss': [],       # 硬目标损失分量
            'training_mode': [],          # 当前训练模式记录
        }
        for hole_num in dataset_valid:
            if hole_num <= in_dim:
                split_range = range(hole_num + 1)
            else:
                split_range = range(hole_num - in_dim, in_dim + 1)
            log_super_loss.update({'valid_super_loss_hole_' + str(hole_num): []})
            for split_num in split_range:
                log_super_loss.update({'valid_super_loss_hole_' + str(hole_num) + '_split_' + str(split_num) : []})
        ################################################################
        # 三阶段交替训练过程
        ################################################################
        for epoch in range(epochs):
            epoch_c_only_losses = []
            epoch_cs_joint_losses = []
            epoch_distill_losses = []
            epoch_soft_losses = []
            epoch_hard_losses = []
            epoch_modes = []
            
            # 遍历每个批次，根据调度器决定训练模式
            for batch_idx, (xx, yy) in enumerate(train_loader):
                xx, yy = xx.to(Device), yy.to(Device)
                
                # 获取当前训练模式
                training_mode = scheduler.get_training_mode(epoch, batch_idx, len(train_loader))
                epoch_modes.append(training_mode)
                
                if training_mode == 'train_c_only':
                    # 阶段1：只训练C网络，学习基础的几何→场映射
                    # 固定S网络，只训练C网络，不使用叠加机制
                    Net_model.pred_net.train()
                    Net_model.super_net.eval()
                    
                    # 准备数据
                    xx = fill_channels(xx, x_norm=x_normalizer, channel_num=in_dim, shuffle=True)
                    gd = feature_transform(xx).to(Device)
                    
                    # 只用C网络直接预测（绕过叠加机制）
                    pred = Net_model.pred_net(xx, gd)
                    loss = Loss_func_train(pred, yy, xx)
                    
                    # 反向传播和优化
                    Optimizer_0.zero_grad()
                    loss.backward()
                    Optimizer_0.step()
                    
                    epoch_c_only_losses.append(loss.item())
                    
                elif training_mode == 'train_cs_joint':
                    # 阶段2：联合训练C+S，学习叠加规律
                    # 两个网络都参与训练，使用完整的叠加机制
                    Net_model.train()
                    
                    # 准备数据
                    xx = fill_channels(xx, x_norm=x_normalizer, channel_num=in_dim * 2, shuffle=True)  # 增加复杂度
                    gd = feature_transform(xx).to(Device)
                    
                    # 使用完整的叠加网络
                    pred = Net_model(xx, gd)
                    loss = Loss_func_train(pred, yy, xx)
                    
                    # 反向传播和优化
                    Optimizer_1.zero_grad()
                    loss.backward()
                    Optimizer_1.step()
                    
                    epoch_cs_joint_losses.append(loss.item())
                    
                elif training_mode == 'distill_s_to_c':
                    # 阶段3：S→C知识蒸馏，将叠加能力转移到C网络
                    # 固定S网络，训练C网络学习S∘C的输出
                    Net_model.pred_net.train()
                    Net_model.super_net.eval()
                    
                    # 生成复杂案例用于蒸馏
                    complex_xx = data_generator.generate_complex_cases((xx, yy), complexity_factor=distill_dict['complexity_factor'])
                    complex_xx = complex_xx.to(Device)
                    gd = feature_transform(complex_xx).to(Device)
                    
                    # 生成软目标：使用S∘C的完整输出作为教师信号
                    with torch.no_grad():
                        soft_targets = Net_model(complex_xx, gd)  # S∘C组合预测
                    
                    # 学生网络：C网络直接预测
                    student_pred = Net_model.pred_net(complex_xx, gd)
                    
                    # 蒸馏损失：让C学习S∘C的输出
                    distill_loss = Distill_loss_func(student_pred, soft_targets, yy, complex_xx)
                    
                    # 反向传播和优化
                    Optimizer_0.zero_grad()
                    distill_loss.backward()
                    Optimizer_0.step()
                    
                    epoch_distill_losses.append(distill_loss.item())
                    
                    # 记录软目标和硬目标损失分量（用于分析）
                    with torch.no_grad():
                        soft_loss = torch.nn.MSELoss()(student_pred, soft_targets).item()
                        hard_loss = torch.nn.MSELoss()(student_pred, yy).item()
                        epoch_soft_losses.append(soft_loss)
                        epoch_hard_losses.append(hard_loss)
            
            # 更新学习率调度器
            Scheduler_0.step()
            Scheduler_1.step()
            
            # 记录每个epoch的平均损失
            if epoch_c_only_losses:
                log_distill_loss['train_c_only_loss'].append(np.mean(epoch_c_only_losses))
            if epoch_cs_joint_losses:
                log_distill_loss['train_cs_joint_loss'].append(np.mean(epoch_cs_joint_losses))
            if epoch_distill_losses:
                log_distill_loss['distill_s_to_c_loss'].append(np.mean(epoch_distill_losses))
            if epoch_soft_losses:
                log_distill_loss['soft_target_loss'].append(np.mean(epoch_soft_losses))
            if epoch_hard_losses:
                log_distill_loss['hard_target_loss'].append(np.mean(epoch_hard_losses))
            
            # 记录训练模式分布
            mode_counts = {mode: epoch_modes.count(mode) for mode in ['train_c_only', 'train_cs_joint', 'distill_s_to_c']}
            log_distill_loss['training_mode'].append(mode_counts)

            # 验证阶段：使用完整的S∘C网络进行验证
            Net_model.eval()
            for hole_num, valid_loader_single in zip(dataset_valid[:-2], valid_loader_list[:-2]):
                log_loss['valid_loss_hole_' + str(hole_num)].append(
                    valid_supercondition(valid_loader_single, Net_model, Device, Loss_func_valid
                                         , x_norm=x_normalizer, super_num=0, channel_num=in_dim)
                )
            for hole_num, valid_loader_single in zip(dataset_valid, valid_loader_list):
                if hole_num <= in_dim:
                    split_range = range(hole_num + 1)
                else:
                    split_range = range(hole_num - in_dim, in_dim + 1)
                per_list = calculate_per(hole_num)
                total_sum = 0
                for split_num in split_range:
                    super_loss = valid_detail(valid_loader_single, Net_model, Device, Loss_func_valid,
                                     x_norm=x_normalizer,
                                     channel_num=in_dim,
                                     super_num=1, hole_num=hole_num, split_num=split_num,
                                     )
                    log_super_loss['valid_super_loss_hole_' + str(hole_num) + '_split_' + str(split_num)].append(super_loss)
                    total_sum = total_sum + super_loss * per_list[int(split_num)]
                log_super_loss['valid_super_loss_hole_' + str(hole_num)].append(total_sum)

            if epoch % 10 == 0:
                # 可视化三阶段训练损失
                fig, axs = plt.subplots(2, 2, figsize=(15, 12), num=1)
                
                # 绘制各阶段损失
                if log_distill_loss['train_c_only_loss']:
                    axs[0,0].plot(log_distill_loss['train_c_only_loss'], label='C Only Loss')
                if log_distill_loss['train_cs_joint_loss']:
                    axs[0,1].plot(log_distill_loss['train_cs_joint_loss'], label='CS Joint Loss')
                if log_distill_loss['distill_s_to_c_loss']:
                    axs[1,0].plot(log_distill_loss['distill_s_to_c_loss'], label='Distill Loss')
                if log_distill_loss['soft_target_loss'] and log_distill_loss['hard_target_loss']:
                    axs[1,1].plot(log_distill_loss['soft_target_loss'], label='Soft Target Loss')
                    axs[1,1].plot(log_distill_loss['hard_target_loss'], label='Hard Target Loss')
                
                for ax in axs.flat:
                    ax.legend()
                    ax.grid(True)
                
                fig.suptitle('Three-Stage Distillation Training Loss')
                fig.savefig(work.svg)
                plt.close(fig)
                
                # 保存模型和日志（包含蒸馏日志）
                torch.save({
                    'log_loss': log_loss, 
                    'log_super_loss': log_super_loss,
                    'log_distill_loss': log_distill_loss,  # 新增蒸馏日志
                    'net_model': Net_model.state_dict(), 
                    'optimizer_0': Optimizer_0.state_dict(),
                    'optimizer_1': Optimizer_1.state_dict()
                }, work.pth)
                torch.save(Net_model, work.fpth)
                
            # 打印三阶段训练信息
            current_mode_counts = log_distill_loss['training_mode'][-1] if log_distill_loss['training_mode'] else {}
            c_only_loss = log_distill_loss['train_c_only_loss'][-1] if log_distill_loss['train_c_only_loss'] else 0.0
            cs_joint_loss = log_distill_loss['train_cs_joint_loss'][-1] if log_distill_loss['train_cs_joint_loss'] else 0.0
            distill_loss = log_distill_loss['distill_s_to_c_loss'][-1] if log_distill_loss['distill_s_to_c_loss'] else 0.0
            
            print('epoch: {:6d}, '
                  'lr: {:.3e}, '
                  'C_only: {:.3e}({:d}), '
                  'CS_joint: {:.3e}({:d}), '
                  'Distill: {:.3e}({:d}), '
                  'cost: {:.2f}'.
                  format(epoch,
                         Optimizer_0.state_dict()['param_groups'][0]['lr'],
                         c_only_loss, current_mode_counts.get('train_c_only', 0),
                         cs_joint_loss, current_mode_counts.get('train_cs_joint', 0),
                         distill_loss, current_mode_counts.get('distill_s_to_c', 0),
                         time.time() - star_time)
                  )
            star_time = time.time()
            if wandb_run:
                # 记录所有日志到wandb，包括三阶段蒸馏日志
                log_dict = {}
                log_dict.update({key: value[-1] for key, value in log_loss.items() if value})
                log_dict.update({key: value[-1] for key, value in log_super_loss.items() if value})
                log_dict.update({key: value[-1] for key, value in log_distill_loss.items() if value and key != 'training_mode'})
                log_dict.update(current_mode_counts)  # 添加训练模式计数
                wandb.log(log_dict)
        if wandb_run: wandb.finish()







