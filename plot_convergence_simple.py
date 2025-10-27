#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的收敛图绘制脚本
使用项目现有的visual_data.py模块

用法:
    python plot_convergence_simple.py --exp1 path1 --exp2 path2
    python plot_convergence_simple.py --exp1 path1  # 单个实验
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from Utilizes.visual_data import MatplotlibVision

def load_loss_history(file_path):
    """加载loss_history.npy文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    data = data.item()        # 0维 object 数组 → dict
    train = np.asarray(data['train'], dtype=float)
    valid = np.asarray(data['valid'], dtype=float)
    train_self = np.asarray(data['train_self'], dtype=float)
    return {'train': train, 'valid': valid, 'train_self': train_self}

def plot_convergence_comparison(exp1_path, exp2_path=None, output_path='convergence_comparison.png'):
    """
    绘制收敛对比图
    
    Args:
        exp1_path: 第一个实验的loss_history.npy文件路径
        exp2_path: 第二个实验的loss_history.npy文件路径（可选）
        output_path: 输出图片路径
    """
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 初始化可视化工具
    visual = MatplotlibVision('.', input_name=('x',), field_name=('loss',))
    
    # 加载第一个实验数据
    print(f"加载实验1: {exp1_path}")
    loss_data1 = load_loss_history(exp1_path)
    
    if loss_data1:
        # 定义颜色和线型方案
        colors = {'train': '#1f77b4', 'valid': '#ff7f0e', 'train_self': '#2ca02c'}
        linestyles = {'train': '-', 'valid': '--', 'train_self': ':'}
        
        # 绘制第一个实验的损失曲线
        for loss_type, values in loss_data1.items():
            # 检查values是否为可迭代对象且非空
            try:
                if hasattr(values, '__len__') and len(values) > 0:
                    epochs = np.arange(len(values))
                    # 使用项目现有的绘图方法
                    visual.plot_loss(fig, ax, epochs, values, 
                                   label=f"实验1_{loss_type}", 
                                   color=colors.get(loss_type, '#1f77b4'), 
                                   linestyle=linestyles.get(loss_type, '-'))
                    print(f"  绘制 {loss_type}: {len(values)} 个数据点")
                else:
                    print(f"跳过 {loss_type}: 数据为空或不可迭代")
            except Exception as e:
                print(f"处理 {loss_type} 时出错: {e}")
                continue
    
    # 加载第二个实验数据（如果提供）
    if exp2_path:
        print(f"加载实验2: {exp2_path}")
        loss_data2 = load_loss_history(exp2_path)
        
        if loss_data2:
            # 定义第二个实验的颜色和线型方案（使用不同颜色）
            colors2 = {'train': '#d62728', 'valid': '#9467bd', 'train_self': '#8c564b'}
            linestyles2 = {'train': '-', 'valid': '--', 'train_self': ':'}
            
            # 绘制第二个实验的损失曲线
            for loss_type, values in loss_data2.items():
                # 检查values是否为可迭代对象且非空
                try:
                    if hasattr(values, '__len__') and len(values) > 0:
                        epochs = np.arange(len(values))
                        # 使用项目现有的绘图方法
                        visual.plot_loss(fig, ax, epochs, values, 
                                       label=f"实验2_{loss_type}", 
                                       color=colors2.get(loss_type, '#d62728'), 
                                       linestyle=linestyles2.get(loss_type, '-'))
                        print(f"  绘制 {loss_type}: {len(values)} 个数据点")
                    else:
                        print(f"跳过 {loss_type}: 数据为空或不可迭代")
                except Exception as e:
                    print(f"处理 {loss_type} 时出错: {e}")
                    continue
    
    # 设置图形属性
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Convergence Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局并保存
    plt.tight_layout()
    plt.show()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"收敛图已保存到: {output_path}")
    
    # 显示图片
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='绘制训练收敛图')
    parser.add_argument('--exp1', 
                        default='./work_satellite_dssl_fno_20251021/FNO_BASE_n100_20251021_214701/loss_history.npy',
                        help='第一个实验的loss_history.npy文件路径')
    parser.add_argument('--exp2', help='第二个实验的loss_history.npy文件路径（可选）')
    parser.add_argument('--output', default='convergence_comparison.png', help='输出图片路径')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.exp1):
        print(f"错误: 文件不存在 {args.exp1}")
        return
    
    if args.exp2 and not os.path.exists(args.exp2):
        print(f"错误: 文件不存在 {args.exp2}")
        return
    
    # 绘制对比图
    plot_convergence_comparison(
        exp1_path=args.exp1,
        exp2_path=args.exp2,
        output_path=args.output
    )

if __name__ == '__main__':
    main()
