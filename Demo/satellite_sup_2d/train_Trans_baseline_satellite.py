#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星数据baseline训练脚本
使用单个FourierTransformer网络直接预测卫星温度场（无叠加机制）
"""
import os
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
from transformer.Transformers import FourierTransformer
from Tools.model_define.define_FNO import train, valid, inference

from Demo.satellite_sup_2d.ablation_satellite import get_setting_satellite, get_loaders_satellite


class MSELoss3Args(nn.Module):
    """MSELoss包装器，适配三参数损失函数接口"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, xx):
        # xx参数被忽略，保持接口兼容
        return self.mse(pred, target)


def parse_args():
    parser = argparse.ArgumentParser(description='卫星数据baseline训练')
    parser.add_argument('--h5', type=str, default=None, help='H5数据路径，默认使用utilizes_satellite内置路径')
    parser.add_argument('--ntrain', type=int, default=None, help='训练样本数，默认使用配置')
    parser.add_argument('--nvalid', type=int, default=None, help='验证样本数，默认使用配置')
    parser.add_argument('--batch_size', type=int, default=None, help='batch大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--work_name', type=str, default='Trans_baseline_v2', help='工作目录名称')
    parser.add_argument('--shuffle', action='store_true', help='是否打乱数据')
    parser.add_argument('--cpu', action='store_true', help='使用CPU训练（避免GPU内存不足）')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 获取配置
    basic_dict, train_dict, pred_model_dict, _ = get_setting_satellite()
    
    # 应用命令行参数覆盖
    if args.batch_size: train_dict['batch_size'] = args.batch_size
    if args.epochs: train_dict['epochs'] = args.epochs
    if args.lr: train_dict['learning_rate'] = args.lr
    if args.ntrain: basic_dict['ntrain'] = args.ntrain
    if args.nvalid: basic_dict['nvalid'] = args.nvalid

    # 工作路径
    work_path = os.path.join('work_satellite', args.work_name)
    work = WorkPrj(work_path)
    Logger = TextLogger(os.path.join(work_path, 'train.log'))
    
    # 设备选择
    if args.cpu:
        Device = torch.device('cpu')
        print("使用CPU训练")
    else:
        Device = work.device
        print(f"使用GPU训练: {Device}")

    locals().update(basic_dict)
    locals().update(train_dict)

    print(f"训练配置:")
    print(f"  - 训练样本数: {ntrain}")
    print(f"  - 验证样本数: {nvalid}")
    print(f"  - Batch大小: {batch_size}")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 工作路径: {work_path}")

    # 数据加载（V2格式17通道）
    train_loader, valid_loader, x_normalizer, y_normalizer, meta = get_loaders_satellite(
        train_num=ntrain,
        valid_num=nvalid,
        batch_size=batch_size,
        h5_path=args.h5,
        shuffled=args.shuffle,
    )

    print(f"\n数据加载完成:")
    print(f"  - 数据格式: {meta['format']}")
    print(f"  - 输入通道数: {meta['in_dim']}")
    print(f"  - 元件SDF通道数: {meta['num_components']}")

    # 网络结构
    Net_model = FourierTransformer(**pred_model_dict).to(Device)
    print(f"\n网络结构:")
    print(f"  - 模型: FourierTransformer")
    print(f"  - 输入通道: {meta['in_dim']}")
    print(f"  - 输出通道: 1")
    print(f"  - 参数量: {sum(p.numel() for p in Net_model.parameters()):,}")

    # 损失函数与优化器
    Loss_func = MSELoss3Args()
    Optimizer = torch.optim.Adam(Net_model.parameters(), lr=learning_rate, betas=(0.7, 0.9), weight_decay=1e-7)
    Scheduler = torch.optim.lr_scheduler.StepLR(Optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # 可视化
    Visual = MatplotlibVision(work_path, input_name=('x', 'y'), field_name=('T',))

    # 训练循环
    start_time = time.time()
    log_loss = {
        'train_loss': [],
        'valid_loss': [],
    }

    print(f"\n开始训练...")
    print(f"{'Epoch':<6} {'LR':<10} {'Train Loss':<12} {'Valid Loss':<12} {'Time(s)':<8}")
    print("-" * 60)

    for epoch in range(epochs):
        # 训练阶段
        Net_model.train()
        train_loss = train(train_loader, Net_model, Device, Loss_func, Optimizer, Scheduler)
        log_loss['train_loss'].append(train_loss)

        # 验证阶段
        Net_model.eval()
        if valid_loader is not None:
            valid_loss = valid(valid_loader, Net_model, Device, Loss_func)
            log_loss['valid_loss'].append(valid_loss)
        else:
            valid_loss = 0.0

        # 每10个epoch保存和可视化
        if epoch % 10 == 0:
            # 绘制loss曲线
            fig, axs = plt.subplots(1, 1, figsize=(15, 8), num=1)
            Visual.plot_loss(fig, axs, np.arange(len(log_loss['train_loss'])), np.array(log_loss['train_loss']), 'train_loss')
            if len(log_loss['valid_loss']) > 0:
                Visual.plot_loss(fig, axs, np.arange(len(log_loss['valid_loss'])), np.array(log_loss['valid_loss']), 'valid_loss')
            fig.suptitle('Training Loss - Satellite Baseline')
            fig.savefig(work.svg)
            plt.close(fig)

            # 保存模型
            torch.save({
                'log_loss': log_loss, 
                'net_model': Net_model.state_dict(), 
                'optimizer': Optimizer.state_dict(),
                'x_normalizer': x_normalizer,
                'y_normalizer': y_normalizer,
                'meta': meta
            }, work.pth)
            torch.save(Net_model, work.fpth)

        # 打印进度
        current_time = time.time()
        elapsed = current_time - start_time
        print(f"{epoch:<6} {Optimizer.state_dict()['param_groups'][0]['lr']:<10.2e} {train_loss:<12.6f} {valid_loss:<12.6f} {elapsed:<8.2f}")
        start_time = current_time

    print("\n训练完成！")
    print(f"最终训练损失: {log_loss['train_loss'][-1]:.6f}")
    if len(log_loss['valid_loss']) > 0:
        print(f"最终验证损失: {log_loss['valid_loss'][-1]:.6f}")
    print(f"模型已保存至: {work_path}")

    # 可选：进行推理测试
    if valid_loader is not None:
        print("\n进行推理测试...")
        Net_model.eval()
        with torch.no_grad():
            # 获取一个batch进行测试
            batch_x, batch_y = next(iter(valid_loader))
            batch_x = batch_x.to(Device)
            batch_y = batch_y.to(Device)
            
            # 推理（需要feature_transform）
            from Tools.model_define.define_FNO import feature_transform
            gd = feature_transform(batch_x).to(Device)
            pred = Net_model(batch_x, gd)
            
            # 计算物理空间的误差
            pred_physical = y_normalizer.back(pred.cpu().numpy())
            true_physical = y_normalizer.back(batch_y.cpu().numpy())
            
            mse_physical = np.mean((pred_physical - true_physical)**2)
            mae_physical = np.mean(np.abs(pred_physical - true_physical))
            
            print(f"推理测试结果:")
            print(f"  - 物理空间MSE: {mse_physical:.6f}")
            print(f"  - 物理空间MAE: {mae_physical:.6f}")
            print(f"  - 预测温度范围: [{pred_physical.min():.2f}, {pred_physical.max():.2f}] K")
            print(f"  - 真实温度范围: [{true_physical.min():.2f}, {true_physical.max():.2f}] K")
