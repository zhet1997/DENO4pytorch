"""
叠加训练 + 一致性训练脚本

训练流程（每个epoch）：
1. 监督训练阶段：使用真值标签训练C+S网络
2. 一致性训练阶段：冻结C，使用多路径一致性训练S网络

使用方法:
    python train_consistency.py --epochs 200 --batch_size 16 --lr 1e-3 --consistency_epochs 10
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import argparse

sys.path.append('/data/wqn/Code/DENO4pytorch')
sys.path.append('/data/wqn/Code/DENO4pytorch/Models')
sys.path.append('/data/wqn/Code/DENO4pytorch/Utilizes')

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
from Demo.satellite_sup_2d.trains_satellite import (
    supredictor_list_windows, 
    train_supercondition, 
    valid_supercondition
)
from Demo.satellite_sup_2d.consistency_modules import PathSampler, ConsistencyTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='叠加训练 + 一致性训练')
    parser.add_argument('--ntrain', type=int, default=None, help='训练样本数，默认80%')
    parser.add_argument('--nvalid', type=int, default=None, help='验证样本数，默认剩余')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200, help='总训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--work_name', type=str, default='DualHead_consistency')
    parser.add_argument('--consistency_epochs', type=int, default=5, 
                        help='每多少个epoch执行一次一致性训练')
    parser.add_argument('--consistency_weight', type=float, default=1.0,
                        help='一致性损失权重')
    parser.add_argument('--K', type=int, default=8, help='多路径数量')
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

    # 数据加载
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
    
    # 监督训练优化器（C+S）
    Optimizer_supervised = torch.optim.Adam(
        Net_model.parameters(), 
        lr=learning_rate, 
        betas=(0.7, 0.9), 
        weight_decay=1e-7
    )
    Scheduler_supervised = torch.optim.lr_scheduler.StepLR(
        Optimizer_supervised, 
        step_size=scheduler_step, 
        gamma=scheduler_gamma
    )
    
    # 一致性训练优化器（只包含S的参数）
    Optimizer_consistency = torch.optim.Adam(
        Net_model.super_net.parameters(), 
        lr=learning_rate * 0.1,  # 一致性训练使用较小学习率
        betas=(0.7, 0.9), 
        weight_decay=1e-7
    )
    
    # 路径采样器和一致性训练器
    path_sampler = PathSampler(K=args.K, channel_num=in_dim)
    consistency_trainer = ConsistencyTrainer(
        model=Net_model,
        path_sampler=path_sampler,
        device=Device,
        K=args.K,
        consistency_weight=args.consistency_weight
    )
    
    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))

    # 训练循环
    start_time = time.time()
    log_loss = {
        'train_supervised_loss': [],
        'valid_supervised_loss': [],
        'train_consistency_loss': [],
        'train_consistency_std': [],
        'valid_consistency_loss': [],
        'valid_consistency_std': [],
    }

    Logger.write("\n" + "=" * 80 + "\n")
    Logger.write("开始训练: 叠加训练 + 一致性训练\n")
    Logger.write("=" * 80 + "\n")
    Logger.write(f"配置:\n")
    Logger.write(f"  - 总epochs: {epochs}\n")
    Logger.write(f"  - 一致性训练间隔: 每{args.consistency_epochs}个epoch\n")
    Logger.write(f"  - 多路径数量K: {args.K}\n")
    Logger.write(f"  - 一致性权重: {args.consistency_weight}\n")
    Logger.write(f"  - 批次大小: {batch_size}\n")
    Logger.write(f"  - 学习率: {learning_rate}\n")
    Logger.write("=" * 80 + "\n\n")

    for epoch in range(epochs):
        # ==================== 阶段1: 监督训练（C+S） ====================
        Net_model.train()
        
        # 确保所有参数可训练
        for param in Net_model.parameters():
            param.requires_grad = True
        
        train_supervised_loss = train_supercondition(
            train_loader, Net_model, Device, Loss_func_train, 
            Optimizer_supervised, Scheduler_supervised,
            x_norm=normalizers['G'],
            super_num=0,
            channel_num=in_dim
        )
        log_loss['train_supervised_loss'].append(train_supervised_loss)
        
        # 监督验证
        Net_model.eval()
        if valid_loader is not None:
            valid_supervised_loss = valid_supercondition(
                valid_loader, Net_model, Device, Loss_func_valid,
                x_norm=normalizers['G'],
                super_num=0,
                channel_num=in_dim
            )
            log_loss['valid_supervised_loss'].append(valid_supervised_loss)
        
        # ==================== 阶段2: 一致性训练（冻结C，更新S） ====================
        perform_consistency = (epoch + 1) % args.consistency_epochs == 0
        
        if perform_consistency:
            Logger.write(f"\n[Epoch {epoch}] 执行一致性训练...\n")
            
            # 一致性训练
            train_consistency_loss, train_consistency_std = consistency_trainer.train_one_epoch(
                train_loader,
                Optimizer_consistency,
                log_interval=20
            )
            log_loss['train_consistency_loss'].append(train_consistency_loss)
            log_loss['train_consistency_std'].append(train_consistency_std)
            
            # 一致性验证
            if valid_loader is not None:
                valid_consistency_loss, valid_consistency_std = consistency_trainer.validate_one_epoch(
                    valid_loader
                )
                log_loss['valid_consistency_loss'].append(valid_consistency_loss)
                log_loss['valid_consistency_std'].append(valid_consistency_std)
                
                Logger.write(f"  一致性训练 - Loss: {train_consistency_loss:.6f}, Std: {train_consistency_std:.6f}\n")
                Logger.write(f"  一致性验证 - Loss: {valid_consistency_loss:.6f}, Std: {valid_consistency_std:.6f}\n")
        
        # ==================== 定期保存和可视化 ====================
        if epoch % 10 == 0:
            # 绘制损失曲线（3图）
            fig, axs = plt.subplots(1, 3, figsize=(24, 6), num=1)
            
            # 左图: 监督loss
            Visual.plot_loss(fig, axs[0], np.arange(len(log_loss['train_supervised_loss'])), 
                           np.array(log_loss['train_supervised_loss']), 'train_supervised')
            if len(log_loss['valid_supervised_loss']) > 0:
                Visual.plot_loss(fig, axs[0], np.arange(len(log_loss['valid_supervised_loss'])), 
                               np.array(log_loss['valid_supervised_loss']), 'valid_supervised')
            axs[0].set_title('Supervised Loss (C+S)')
            axs[0].legend()
            axs[0].set_yscale('log')
            
            # 中图: 一致性loss
            if len(log_loss['train_consistency_loss']) > 0:
                consistency_epochs_idx = np.array([i * args.consistency_epochs - 1 
                                                   for i in range(1, len(log_loss['train_consistency_loss']) + 1)])
                Visual.plot_loss(fig, axs[1], consistency_epochs_idx, 
                               np.array(log_loss['train_consistency_loss']), 'train_consistency')
                if len(log_loss['valid_consistency_loss']) > 0:
                    Visual.plot_loss(fig, axs[1], consistency_epochs_idx, 
                                   np.array(log_loss['valid_consistency_loss']), 'valid_consistency')
            axs[1].set_title('Consistency Loss (S-only)')
            axs[1].legend()
            axs[1].set_yscale('log')
            
            # 右图: 路径标准差（稳定性指标）
            if len(log_loss['train_consistency_std']) > 0:
                consistency_epochs_idx = np.array([i * args.consistency_epochs - 1 
                                                   for i in range(1, len(log_loss['train_consistency_std']) + 1)])
                axs[2].plot(consistency_epochs_idx, log_loss['train_consistency_std'], 
                           'o-', label='train_std', linewidth=2)
                if len(log_loss['valid_consistency_std']) > 0:
                    axs[2].plot(consistency_epochs_idx, log_loss['valid_consistency_std'], 
                               's-', label='valid_std', linewidth=2)
            axs[2].set_title('Path Output Std (Stability)')
            axs[2].set_xlabel('Epoch')
            axs[2].set_ylabel('Std')
            axs[2].legend()
            axs[2].grid(True, alpha=0.3)
            
            fig.suptitle(f'Training Progress (Epoch {epoch})', fontsize=14)
            fig.savefig(work.svg, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'log_loss': log_loss,
                'net_model': Net_model.state_dict(),
                'optimizer_supervised': Optimizer_supervised.state_dict(),
                'optimizer_consistency': Optimizer_consistency.state_dict(),
                'args': vars(args),
            }, work.pth)
            torch.save(Net_model, work.fpth)
            
            Logger.write(f"[Epoch {epoch:04d}] 模型已保存\n")

        # ==================== 打印训练信息 ====================
        if valid_loader is not None:
            msg = 'epoch: {:6d}, lr: {:.3e}, train_sup: {:.3e}, valid_sup: {:.3e}'.format(
                epoch,
                Optimizer_supervised.state_dict()['param_groups'][0]['lr'],
                log_loss['train_supervised_loss'][-1],
                log_loss['valid_supervised_loss'][-1],
            )
            if perform_consistency and len(log_loss['train_consistency_loss']) > 0:
                msg += ', consistency_loss: {:.3e}, consistency_std: {:.6f}'.format(
                    log_loss['train_consistency_loss'][-1],
                    log_loss['train_consistency_std'][-1]
                )
            msg += ', cost: {:.2f}s'.format(time.time() - start_time)
            Logger.write(msg)
        else:
            msg = 'epoch: {:6d}, lr: {:.3e}, train_sup: {:.3e}'.format(
                epoch,
                Optimizer_supervised.state_dict()['param_groups'][0]['lr'],
                log_loss['train_supervised_loss'][-1],
            )
            if perform_consistency and len(log_loss['train_consistency_loss']) > 0:
                msg += ', consistency_loss: {:.3e}, consistency_std: {:.6f}'.format(
                    log_loss['train_consistency_loss'][-1],
                    log_loss['train_consistency_std'][-1]
                )
            msg += ', cost: {:.2f}s'.format(time.time() - start_time)
            Logger.write(msg)
        
        start_time = time.time()
    
    Logger.write("\n" + "=" * 80 + "\n")
    Logger.write("训练完成！\n")
    Logger.write("=" * 80 + "\n")
