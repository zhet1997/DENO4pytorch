#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
温度场回归误差图绘制脚本

功能：
- 从评估输出目录加载 train_pred.npy, train_true.npy, valid_pred.npy, valid_true.npy
- 将温度场聚合为标量（平均值或最大值）
- 绘制对角线回归误差图（train和valid叠加显示）
- 保存图片到指定目录

用法：
    python plot_regression_from_npy.py --eval_dir work_satellite_eval/eval_mlp_20251024_184758
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# 添加项目路径以导入 MatplotlibVision
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Utilizes.visual_data import MatplotlibVision


def load_npy_files(eval_dir):
    """
    加载评估输出的npy文件
    
    Args:
        eval_dir: 评估输出目录路径
        
    Returns:
        train_true, train_pred, valid_true, valid_pred: 四个numpy数组
    """
    print("加载数据...")
    
    # 检查目录是否存在
    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"目录不存在: {eval_dir}")
    
    # 定义文件路径
    files = {
        'train_true': os.path.join(eval_dir, 'train_true.npy'),
        'train_pred': os.path.join(eval_dir, 'train_pred.npy'),
        'valid_true': os.path.join(eval_dir, 'valid_true.npy'),
        'valid_pred': os.path.join(eval_dir, 'valid_pred.npy'),
    }
    
    # 检查文件是否存在
    for name, path in files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
    
    # 加载数据
    train_true = np.load(files['train_true'])
    train_pred = np.load(files['train_pred'])
    valid_true = np.load(files['valid_true'])
    valid_pred = np.load(files['valid_pred'])
    
    # 打印形状信息
    print(f"  train_true.npy: {train_true.shape}")
    print(f"  train_pred.npy: {train_pred.shape}")
    print(f"  valid_true.npy: {valid_true.shape}")
    print(f"  valid_pred.npy: {valid_pred.shape}")
    
    return train_true, train_pred, valid_true, valid_pred


def aggregate_temperature_field(data, method='mean'):
    """
    将温度场聚合为标量
    
    Args:
        data: 温度场数据，形状 (N, H, W, 1)
        method: 聚合方法，'mean' 或 'max'
        
    Returns:
        aggregated: 聚合后的标量数组，形状 (N,)
    """
    # 温度场聚合方式（可修改）
    if method == 'mean':
        # 选项1：平均温度（默认）
        aggregated = data.mean(axis=(1, 2, 3))
    elif method == 'max':
        # 选项2：最高温度（备选）
        aggregated = data.max(axis=(1, 2, 3))
    else:
        raise ValueError(f"未知的聚合方法: {method}. 支持 'mean' 或 'max'")
    
    return aggregated


def plot_regression(train_true, train_pred, valid_true, valid_pred, output_path):
    """
    绘制回归误差图
    
    Args:
        train_true: 训练集真实值 (N_train,)
        train_pred: 训练集预测值 (N_train,)
        valid_true: 验证集真实值 (N_valid,)
        valid_pred: 验证集预测值 (N_valid,)
        output_path: 输出图片路径
    """
    print("\n绘制回归误差图...")
    
    # 计算R²
    r2_train = r2_score(train_true, train_pred)
    r2_valid = r2_score(valid_true, valid_pred)
    
    print(f"  Train R²: {r2_train:.4f}")
    print(f"  Valid R²: {r2_valid:.4f}")
    
    # 初始化MatplotlibVision（用于字体配置）
    visual = MatplotlibVision(log_dir=os.path.dirname(output_path), field_name=('Temperature',))
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 计算整体数据范围（用于设置坐标轴）
    all_true = np.concatenate([train_true, valid_true])
    all_pred = np.concatenate([train_pred, valid_pred])
    min_value = min(all_true.min(), all_pred.min())
    max_value = max(all_true.max(), all_pred.max())
    
    # 绘制训练集散点（红色）
    ax.scatter(train_true[::4], train_pred[::4], marker='o', color='firebrick', s=120, 
               linewidth=0.5, facecolor='firebrick', edgecolor='k', alpha=0.7, 
               label=f'Train (R²={r2_train:.4f})')
    
    # 绘制验证集散点（蓝色）
    ax.scatter(valid_true[::4], valid_pred[::4], marker='s', color='steelblue', s=120, 
               linewidth=0.5, facecolor='steelblue', edgecolor='k', alpha=0.2, 
               label=f'Valid (R²={r2_valid:.4f})')
    
    # 绘制对角线 y=x
    ax.plot([min_value, max_value], [min_value, max_value], 'k--', 
            linewidth=2.0, label='y=x', zorder=1)
    
    # 添加±1%误差带（半透明填充）
    ax.fill_between([0.995 * min_value, 1.005 * max_value], 
                     [0.995**2 * min_value, 0.995*1.005 * max_value],
                     [1.005*0.995 * min_value, 1.005**2 * max_value],
                     alpha=0.2, color='darkcyan', label='±1% error band')
    
    # 设置坐标轴范围
    ax.set_xlim((0.995 * min_value, 1.005 * max_value))
    ax.set_ylim((0.995 * min_value, 1.005 * max_value))
    
    # 设置网格和标签
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", prop=visual.font, framealpha=0.9)
    ax.set_xlabel('True Temperature (K)', fontdict=visual.font)
    ax.set_ylabel('Predicted Temperature (K)', fontdict=visual.font)
    ax.tick_params('both', labelsize=visual.font["size"])
    ax.set_aspect('equal', adjustable='box')
    
    # 添加标题
    fig.suptitle('Temperature Prediction Regression', fontsize=visual.font["size"]+4, y=0.98)
    
    # 保存图片
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n保存图片: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制温度场回归误差图')
    parser.add_argument('--eval_dir', type=str, required=True,
                        help='评估输出目录路径（包含npy文件）')
    parser.add_argument('--method', type=str, default='mean', choices=['mean', 'max'],
                        help='温度场聚合方法：mean(平均值) 或 max(最大值)，默认mean')
    parser.add_argument('--output_name', type=str, default='regression_plot.png',
                        help='输出图片文件名，默认regression_plot.png')
    args = parser.parse_args()
    
    try:
        # 1. 加载npy文件
        train_true, train_pred, valid_true, valid_pred = load_npy_files(args.eval_dir)
        
        # 2. 聚合温度场为标量
        print(f"\n聚合温度场为标量（使用{args.method}）...")
        train_true_scalar = aggregate_temperature_field(train_true, method=args.method)
        train_pred_scalar = aggregate_temperature_field(train_pred, method=args.method)
        valid_true_scalar = aggregate_temperature_field(valid_true, method=args.method)
        valid_pred_scalar = aggregate_temperature_field(valid_pred, method=args.method)
        
        print(f"  Train: {len(train_true_scalar)} samples")
        print(f"  Valid: {len(valid_true_scalar)} samples")
        
        # 3. 绘制回归误差图
        output_path = os.path.join(args.eval_dir, args.output_name)
        plot_regression(train_true_scalar, train_pred_scalar, 
                       valid_true_scalar, valid_pred_scalar, 
                       output_path)
        
        print("\n完成！")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

