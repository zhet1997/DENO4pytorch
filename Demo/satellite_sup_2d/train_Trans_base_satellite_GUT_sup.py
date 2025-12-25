"""
卫星数据叠加训练脚本 - DualHeadFourierTransformer + 超分辨率

训练流程:
1. 使用 DualHeadFourierTransformer 作为预测器（G+U -> T）
2. 使用 FNO2d 作为超分辨率模型
3. 两阶段训练:
   - 阶段1: 只训练 predictor (Optimizer_0)
   - 阶段2: 同时训练 predictor + super model (Optimizer_1)

数据格式:
- G: [B, H, W, 4] - 条件场 (x, y, scale_x, scale_y)
- U: [B, H, W, 1] - 源项场
- T: [B, H, W, 1] - 目标温度场

使用方法:
    python train_Trans_base_satellite_GUT_sup.py --epochs 200 --batch_size 16 --lr 1e-3
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import argparse

sys.path.append('/data/wqn/DENO4pytorch')
sys.path.append('/data/wqn/DENO4pytorch/Models')
sys.path.append('/data/wqn/DENO4pytorch/Utilizes')

from Utilizes.visual_data import MatplotlibVision, TextLogger
import matplotlib.pyplot as plt
from Tools.train_model.model_whole_life import WorkPrj
from fno.FNOs import FNO2d
from transformer.DualHeadTransformer import DualHeadFourierTransformer
from Demo.satellite_sup_2d.ablation_satellite import (
    get_setting_satellite,
    get_loaders_satellite_multi_GUT,
    transform_U_channels
)
from Demo.satellite_sup_2d.trains_satellite import supredictor_list_windows, train_supercondition, valid_supercondition

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntrain', type=int, default=None, help='训练样本数，默认80%')
    parser.add_argument('--nvalid', type=int, default=None, help='验证样本数，默认剩余')
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--work_name', type=str, default='DualHead_satellite_super')
    parser.add_argument('--win_split', type=int, default=1, help='窗口分割数量')
    parser.add_argument('--shuffle', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    basic_dict, train_dict, pred_model_dict, super_model_dict = get_setting_satellite()
    if args.batch_size: train_dict['batch_size'] = args.batch_size
    if args.epochs: train_dict['epochs'] = args.epochs
    if args.lr: train_dict['learning_rate'] = args.lr
    if args.ntrain: basic_dict['ntrain'] = args.ntrain
    if args.nvalid: basic_dict['nvalid'] = args.nvalid

    # 工作路径
    work_path = os.path.join('work_satellite', args.work_name)
    work = WorkPrj(work_path)
    Logger = TextLogger(os.path.join(work_path, 'train.log'))
    Device = work.device

    locals().update(basic_dict)
    locals().update(train_dict)
    target_U_channels = 16

    # 数据加载（固定V2/17通道）
    train_loader, valid_loader, normalizers, meta = get_loaders_satellite_multi_GUT(
        component_nums=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        target_U_channels=target_U_channels,
        empty_channel_value=1.0,
        train_num=ntrain,
        valid_num=nvalid,
        batch_size=batch_size,
        shuffled=True
    )

    # 网络
    pred_model = DualHeadFourierTransformer(**pred_model_dict).to(Device)
    super_model = FNO2d(in_dim=2, out_dim=1, **super_model_dict).to(Device)
    Net_model = supredictor_list_windows(pred_model, super_model, channel_num=in_dim).to(Device)    

    # 损失与优化器
    Loss_func_train = nn.MSELoss()
    Loss_func_valid = nn.MSELoss()
    
    Optimizer_0 = torch.optim.Adam(Net_model.pred_net.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-7)
    Optimizer_1 = torch.optim.Adam([
        {'params': Net_model.pred_net.parameters(), 'lr': learning_rate * 0.1},  # 学习率为默认的
        {'params': Net_model.super_net.parameters()}], lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-7)
    # # 下降策略
    Scheduler_0 = torch.optim.lr_scheduler.StepLR(Optimizer_0, step_size=scheduler_step, gamma=scheduler_gamma)
    Scheduler_1 = torch.optim.lr_scheduler.StepLR(Optimizer_1, step_size=scheduler_step, gamma=scheduler_gamma)
    
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))

    # 训练循环
    start_time = time.time()
    log_loss = {
        'train_step_loss': [],       # predictor 训练损失
        'valid_step_loss': [],       # predictor 验证损失
    }
    log_super_loss = {
        'train_super_loss': [],      # super 训练损失
        'valid_super_loss': [],      # super 验证损失
    }

    Logger.write("\n" + "=" * 60 + "\n")
    Logger.write("Starting training...\n")
    Logger.write("=" * 60 + "\n")

    for epoch in range(epochs):
        Net_model.train()
        
        # 阶段1: 训练 predictor (Optimizer_0)
        train_loss_0 = train_supercondition(
            train_loader, Net_model, Device, Loss_func_train, Optimizer_0, Scheduler_0,
            x_norm=normalizers['G'],  # 假设 G 用作归一化参考
            super_num=0,              # super_num=0 表示只训练 predictor
            channel_num=in_dim
        )
        log_loss['train_step_loss'].append(train_loss_0)
        
        # 阶段2: 训练 super model (Optimizer_1)
        train_loss_1 = train_supercondition(
            train_loader, Net_model, Device, Loss_func_train, Optimizer_1, Scheduler_1,
            x_norm=normalizers['G'],
            super_num=1,              # super_num=1 表示训练 super model
            channel_num=in_dim
        )
        log_super_loss['train_super_loss'].append(train_loss_1)

        # 验证
        Net_model.eval()
        if valid_loader is not None:
            valid_loss_0 = valid_supercondition(
                valid_loader, Net_model, Device, Loss_func_valid,
                x_norm=normalizers['G'],
                super_num=0,
                channel_num=in_dim
            )
            log_loss['valid_step_loss'].append(valid_loss_0)
            
            valid_loss_1 = valid_supercondition(
                valid_loader, Net_model, Device, Loss_func_valid,
                x_norm=normalizers['G'],
                super_num=1,
                channel_num=in_dim
            )
            log_super_loss['valid_super_loss'].append(valid_loss_1)

        # 定期保存和可视化
        if epoch % 10 == 0:
            # 绘制损失曲线
            fig, axs = plt.subplots(1, 2, figsize=(20, 8), num=1)
            
            # 左图: predictor loss
            Visual.plot_loss(fig, axs[0], np.arange(len(log_loss['train_step_loss'])), 
                           np.array(log_loss['train_step_loss']), 'train_step')
            if len(log_loss['valid_step_loss']) > 0:
                Visual.plot_loss(fig, axs[0], np.arange(len(log_loss['valid_step_loss'])), 
                               np.array(log_loss['valid_step_loss']), 'valid_step')
            axs[0].set_title('Predictor Loss')
            axs[0].legend()
            
            # 右图: super loss
            Visual.plot_loss(fig, axs[1], np.arange(len(log_super_loss['train_super_loss'])), 
                           np.array(log_super_loss['train_super_loss']), 'train_super')
            if len(log_super_loss['valid_super_loss']) > 0:
                Visual.plot_loss(fig, axs[1], np.arange(len(log_super_loss['valid_super_loss'])), 
                               np.array(log_super_loss['valid_super_loss']), 'valid_super')
            axs[1].set_title('Super Model Loss')
            axs[1].legend()
            
            fig.suptitle('Training Loss')
            fig.savefig(work.svg)
            plt.close(fig)
            
            # 保存模型
            torch.save({
                'log_loss': log_loss,
                'log_super_loss': log_super_loss,
                'net_model': Net_model.state_dict(),
                'optimizer_0': Optimizer_0.state_dict(),
                'optimizer_1': Optimizer_1.state_dict(),
            }, work.pth)
            torch.save(Net_model, work.fpth)
            
            Logger.write(f"[Epoch {epoch:04d}] Model saved\n")

        # 打印训练信息
        if valid_loader is not None:
            print('epoch: {:6d}, lr: {:.3e}, train_step: {:.3e}, train_super: {:.3e}, '
                  'valid_step: {:.3e}, valid_super: {:.3e}, cost: {:.2f}'.format(
                epoch,
                Optimizer_0.state_dict()['param_groups'][0]['lr'],
                log_loss['train_step_loss'][-1],
                log_super_loss['train_super_loss'][-1],
                log_loss['valid_step_loss'][-1],
                log_super_loss['valid_super_loss'][-1],
                time.time() - start_time,
            ))
            
            Logger.write('epoch: {:6d}, lr: {:.3e}, train_step: {:.3e}, train_super: {:.3e}, '
                        'valid_step: {:.3e}, valid_super: {:.3e}, cost: {:.2f}\n'.format(
                epoch,
                Optimizer_0.state_dict()['param_groups'][0]['lr'],
                log_loss['train_step_loss'][-1],
                log_super_loss['train_super_loss'][-1],
                log_loss['valid_step_loss'][-1],
                log_super_loss['valid_super_loss'][-1],
                time.time() - start_time,
            ))
        else:
            print('epoch: {:6d}, lr: {:.3e}, train_step: {:.3e}, train_super: {:.3e}, cost: {:.2f}'.format(
                epoch,
                Optimizer_0.state_dict()['param_groups'][0]['lr'],
                log_loss['train_step_loss'][-1],
                log_super_loss['train_super_loss'][-1],
                time.time() - start_time,
            ))
        
        start_time = time.time()
    
    Logger.write("\n" + "=" * 60 + "\n")
    Logger.write("Training completed!\n")
    Logger.write("=" * 60 + "\n")


