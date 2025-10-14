#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星导热数据集使用示例
展示如何在叠加神经网络训练中使用satellite数据加载器
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from Demo.satellite_sup_2d.utilizes_satellite import get_origin_satellite, get_loader_satellite
import torch
import numpy as np


def example_basic_usage():
    """示例1: 基本数据加载"""
    print("=" * 60)
    print("示例1: 基本数据加载与DataLoader创建")
    print("=" * 60)
    
    # 加载数据（兼容多格式返回值）
    train_x, train_y, valid_x, valid_y, fmt, nc = get_origin_satellite(
        train_num=4000,
        valid_num=500,
        shuffled=False
    )
    print(f"数据格式: {fmt}, 元件通道数: {nc}")
    
    # 创建DataLoader
    train_loader, valid_loader, x_norm, y_norm = get_loader_satellite(
        train_x, train_y,
        valid_x, valid_y,
        batch_size=32
    )
    
    print("\n数据加载完成！")
    return train_loader, valid_loader, x_norm, y_norm


def example_channel_analysis(train_loader, x_norm):
    """示例2: 分析各通道的归一化统计信息"""
    print("\n" + "=" * 60)
    print("示例2: 各通道归一化统计信息")
    print("=" * 60)
    
    channel_names = ['component_sdf', 'component_power', 'cooling_sdf', 
                     'cooling_temp', 'coord_x', 'coord_y']
    
    print("\n各通道统计信息:")
    for i, name in enumerate(channel_names):
        print(f"\n{i}. {name}:")
        print(f"   Mean: {x_norm.mean[i]:.6f}")
        print(f"   Std:  {x_norm.std[i]:.6f}")
    
    # 获取一个batch查看归一化后的值范围
    batch_x, _ = next(iter(train_loader))
    print("\n归一化后各通道值范围:")
    for i, name in enumerate(channel_names):
        channel_data = batch_x[:, :, :, i]
        print(f"{i}. {name}: [{channel_data.min():.4f}, {channel_data.max():.4f}]")


def example_denormalization(train_loader, y_norm):
    """示例3: 反归一化预测结果"""
    print("\n" + "=" * 60)
    print("示例3: 反归一化预测结果")
    print("=" * 60)
    
    batch_x, batch_y = next(iter(train_loader))
    
    # 模拟模型预测（这里用真实值代替）
    predictions_norm = batch_y
    
    # 反归一化
    predictions_physical = y_norm.back(predictions_norm.numpy())
    ground_truth_physical = y_norm.back(batch_y.numpy())
    
    print(f"\n归一化空间:")
    print(f"  预测值范围: [{predictions_norm.min():.4f}, {predictions_norm.max():.4f}]")
    
    print(f"\n物理空间 (温度K):")
    print(f"  预测值范围: [{predictions_physical.min():.2f}, {predictions_physical.max():.2f}]")
    print(f"  真实值范围: [{ground_truth_physical.min():.2f}, {ground_truth_physical.max():.2f}]")


def example_comparison_with_pakb():
    """示例4: 与PakB数据集的对比"""
    print("\n" + "=" * 60)
    print("示例4: Satellite vs PakB 数据集对比")
    print("=" * 60)
    
    comparison = """
    ┌─────────────────┬──────────────────────────┬──────────────────────────┐
    │     特性        │     PakB数据集           │    Satellite数据集       │
    ├─────────────────┼──────────────────────────┼──────────────────────────┤
    │ 数据格式        │ .mat文件                 │ .h5文件                  │
    │ 数据源          │ 多个mat文件拼接          │ 单个h5文件               │
    │ 加载方式        │ get_struct_quanlity      │ h5py直接读取             │
    │ 划分策略        │ 全部从前面取             │ train前取,valid后取      │
    │ 归一化轴        │ axis=(0,1,2,3)           │ axis=(0,1,2)每通道独立   │
    │ 输入通道        │ 变化                     │ 固定6通道                │
    │ 坐标信息        │ 需要单独处理             │ 包含在输入通道中         │
    │ 网格类型        │ struct/unstruct          │ 规则网格256x256          │
    └─────────────────┴──────────────────────────┴──────────────────────────┘
    """
    print(comparison)
    
    print("\n主要改进:")
    print("  ✓ 每通道独立归一化，更适合不同物理量混合输入")
    print("  ✓ 从后面取验证集，避免数据泄露")
    print("  ✓ h5格式加载更快，内存占用更小")
    print("  ✓ 包含完整的坐标和几何信息")


def example_training_loop_skeleton(train_loader, valid_loader, y_norm):
    """示例5: 训练循环框架"""
    print("\n" + "=" * 60)
    print("示例5: 训练循环框架（伪代码）")
    print("=" * 60)
    
    code_example = '''
# 训练循环示例
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)  # (B, 256, 256, 6)
        batch_y = batch_y.to(device)  # (B, 256, 256, 1)
        
        # 前向传播
        pred = model(batch_x)  # (B, 256, 256, 1)
        
        # 计算损失（在归一化空间）
        loss = criterion(pred, batch_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in valid_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            pred = model(batch_x)
            
            # 反归一化到物理空间评估
            pred_physical = y_norm.back(pred.cpu().numpy())
            true_physical = y_norm.back(batch_y.cpu().numpy())
            
            # 计算物理空间的误差
            mse_physical = np.mean((pred_physical - true_physical)**2)
            print(f"Physical MSE: {mse_physical:.6f}")
    '''
    print(code_example)


def example_superposition_network_data_prep(train_loader, x_norm):
    """示例6: 叠加神经网络的数据准备"""
    print("\n" + "=" * 60)
    print("示例6: 叠加神经网络数据准备")
    print("=" * 60)
    
    print("\n叠加神经网络架构:")
    print("  Complex Network (C): 处理复杂几何")
    print("  Simple Network (S): 处理简单特征")
    print("  S∘C: 组合网络")
    
    print("\n可能的通道划分方案:")
    print("\n方案1 - 按物理意义划分:")
    print("  Complex输入: [component_sdf, component_power, cooling_sdf, coord_x, coord_y]")
    print("  Simple输入:  [cooling_temp] + S∘C输出")
    
    print("\n方案2 - 按复杂度划分:")
    print("  Complex输入: [component_sdf, cooling_sdf, coord_x, coord_y]  # 几何信息")
    print("  Simple输入:  [component_power, cooling_temp] + S∘C输出      # 物理量")
    
    # 演示如何提取特定通道
    batch_x, batch_y = next(iter(train_loader))
    
    print("\n通道索引:")
    channel_names = ['component_sdf', 'component_power', 'cooling_sdf', 
                     'cooling_temp', 'coord_x', 'coord_y']
    for i, name in enumerate(channel_names):
        print(f"  {i}: {name}")
    
    print("\n示例代码:")
    code = '''
# 提取complex网络输入（几何+坐标）
complex_input = batch_x[:, :, :, [0, 2, 4, 5]]  # shape: (B, 256, 256, 4)

# Complex网络预测
complex_output = complex_model(complex_input)  # shape: (B, 256, 256, 1)

# 拼接simple网络输入（物理量+complex输出）
simple_input = torch.cat([
    batch_x[:, :, :, [1, 3]],  # component_power, cooling_temp
    complex_output
], dim=-1)  # shape: (B, 256, 256, 3)

# Simple网络预测
simple_output = simple_model(simple_input)  # shape: (B, 256, 256, 1)
    '''
    print(code)


def main():
    """主函数：运行所有示例"""
    
    # 示例1: 基本数据加载
    train_loader, valid_loader, x_norm, y_norm = example_basic_usage()
    
    # 示例2: 通道分析
    example_channel_analysis(train_loader, x_norm)
    
    # 示例3: 反归一化
    example_denormalization(train_loader, y_norm)
    
    # 示例4: 与PakB对比
    example_comparison_with_pakb()
    
    # 示例5: 训练循环框架
    example_training_loop_skeleton(train_loader, valid_loader, y_norm)
    
    # 示例6: 叠加网络数据准备
    example_superposition_network_data_prep(train_loader, x_norm)
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

