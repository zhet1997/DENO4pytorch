#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
温度场误差分布图绘制脚本

功能：
- 从评估输出目录加载 train_pred.npy, train_true.npy, valid_pred.npy, valid_true.npy
- 计算每个样本的归一化误差（MSE 或 MAE）
- 绘制误差分布图（直方图 + KDE 拟合曲线）
- 保存图片到指定目录

用法：
    # 单目录模式
    python plot_error_hist_from_npy.py --eval_dir work_satellite_eval/eval_mlp_20251024_184758 --error_type mse --plot_mode valid
    
    # 逐像素统计模式
    python plot_error_hist_from_npy.py --eval_dir work_satellite_eval/eval_mlp_xxx --pixel_wise --plot_mode valid
    
    # 多目录模式（叠加绘图）
    python plot_error_hist_from_npy.py --eval_dirs dir1 dir2 dir3 --error_type mse --output_name comparison.png
    python plot_error_hist_from_npy.py --eval_dirs dir1 dir2 --labels "Model A" "Model B" --error_type mae
    python plot_error_hist_from_npy.py --eval_dirs dir1 dir2 --pixel_wise --output_name pixel_comparison.png
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 添加项目路径以导入 MatplotlibVision
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Utilizes.visual_data import MatplotlibVision


def load_npy_files(eval_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载评估输出的npy文件
    
    Args:
        eval_dir: 评估输出目录路径
        
    Returns:
        train_true, train_pred, valid_true, valid_pred: 四个numpy数组
    """
    print("加载数据...")
    
    if not os.path.exists(eval_dir):
        raise FileNotFoundError(f"目录不存在: {eval_dir}")
    
    files = {
        'train_true': os.path.join(eval_dir, 'train_true.npy'),
        'train_pred': os.path.join(eval_dir, 'train_pred.npy'),
        'valid_true': os.path.join(eval_dir, 'valid_true.npy'),
        'valid_pred': os.path.join(eval_dir, 'valid_pred.npy'),
    }
    
    for path in files.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件不存在: {path}")
    
    train_true = np.load(files['train_true'])
    train_pred = np.load(files['train_pred'])
    valid_true = np.load(files['valid_true'])
    valid_pred = np.load(files['valid_pred'])
    
    print(f"  train_true.npy: {train_true.shape}")
    print(f"  train_pred.npy: {train_pred.shape}")
    print(f"  valid_true.npy: {valid_true.shape}")
    print(f"  valid_pred.npy: {valid_pred.shape}")
    
    return train_true, train_pred, valid_true, valid_pred


def compute_errors(true: np.ndarray, pred: np.ndarray, error_type: str, pixel_wise: bool = False) -> np.ndarray:
    """
    计算误差
    
    Args:
        true: 真实温度场，形状 (N, H, W, 1)
        pred: 预测温度场，形状 (N, H, W, 1)
        error_type: 误差类型，'mse'/'mae'(样本级) 或 'diff'(逐像素差值)
        pixel_wise: 是否逐像素统计（True: 展平所有像素；False: 每个样本一个误差值）
        
    Returns:
        errors: 误差数组，形状 (N,) 或 (N*H*W,)
    """
    std_true = true.std()
    
    if pixel_wise:
        # 逐像素模式：直接计算每个像素的差值（pred - true），不取绝对值
        diff = (pred - true).flatten()  # 展平为一维
        errors = diff / std_true  # 归一化到标准差单位
    else:
        # 样本模式：每个样本计算一个归一化误差
        if error_type == 'mse':
            # MSE: 对每个样本计算均方误差，归一化到方差单位
            errors = ((pred - true) ** 2).mean(axis=(1, 2, 3)) / (std_true ** 2)
        elif error_type == 'mae':
            # MAE: 对每个样本计算平均绝对误差，归一化到标准差单位
            errors = np.abs(pred - true).mean(axis=(1, 2, 3)) / std_true
        else:
            raise ValueError(f"未知的误差类型: {error_type}. 支持 'mse' 或 'mae'")
    
    return errors


def extract_label_from_path(eval_dir: str) -> str:
    """
    从目录路径提取标签
    
    Args:
        eval_dir: 评估目录路径
        
    Returns:
        label: 提取的标签字符串
        
    示例：
        /path/to/FNO_BASE_n500_20251024_195555 -> FNO_BASE_n500
    """
    dirname = os.path.basename(eval_dir.rstrip('/'))
    parts = dirname.split('_')
    
    # 去除最后两部分（通常是日期和时间戳）
    if len(parts) > 2:
        label = '_'.join(parts[:-2])
    else:
        label = dirname
    
    return label


def plot_error_distribution(
    train_errors: np.ndarray,
    valid_errors: np.ndarray,
    error_type: str,
    output_path: str,
    plot_mode: str = 'valid',
    pixel_wise: bool = False
) -> None:
    """
    绘制误差分布图（直方图 + KDE 曲线）
    
    Args:
        train_errors: 训练集误差
        valid_errors: 验证集误差
        error_type: 误差类型，'mse' 或 'mae'
        output_path: 输出图片路径
        plot_mode: 绘制模式，'train', 'valid' 或 'both'
        pixel_wise: 是否为逐像素模式
    """
    print(f"\n绘制误差分布图（模式: {plot_mode}）...")
    
    # 初始化 MatplotlibVision（用于字体配置）
    visual = MatplotlibVision(log_dir=os.path.dirname(output_path), field_name=('Temperature',))
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 计算误差范围（根据绘制模式）
    if plot_mode == 'train':
        all_errors = train_errors
    elif plot_mode == 'valid':
        all_errors = valid_errors
    else:  # both
        all_errors = np.concatenate([train_errors, valid_errors])
    
    min_error = all_errors.min()
    max_error = all_errors.max()
    x_grid = np.linspace(min_error, max_error, 500)
    
    # 绘制训练集
    if plot_mode in ['train', 'both']:
        ax.hist(train_errors, bins=50, alpha=0.5, color='firebrick', 
                density=True, label='Train', edgecolor='darkred', linewidth=0.5)
        kde_train = gaussian_kde(train_errors)
        ax.plot(x_grid, kde_train(x_grid), 'r-', linewidth=2.5, label='Train KDE', alpha=0.8)
    
    # 绘制验证集
    if plot_mode in ['valid', 'both']:
        ax.hist(valid_errors, bins=50, alpha=0.5, color='steelblue', 
                density=True, label='Valid', edgecolor='darkblue', linewidth=0.5)
        kde_valid = gaussian_kde(valid_errors)
        ax.plot(x_grid, kde_valid(x_grid), 'b-', linewidth=2.5, label='Valid KDE', alpha=0.8)
    
    # 设置坐标轴标签
    if pixel_wise:
        error_label = 'Normalized Difference (pred - true) / std'
        title = 'Pixel-wise Error Distribution'
    else:
        error_label = 'Normalized MSE' if error_type == 'mse' else 'Normalized MAE'
        title = f'Error Distribution ({error_type.upper()})'
    
    ax.set_xlabel(error_label, fontdict=visual.font)
    ax.set_ylabel('Probability Density', fontdict=visual.font)
    
    # 设置网格和图例
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", prop=visual.font, framealpha=0.9)
    ax.tick_params('both', labelsize=visual.font["size"])
    
    # 添加标题
    fig.suptitle(title, fontsize=visual.font["size"] + 4, y=0.98)
    
    # 保存图片
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"保存图片: {output_path}")


def plot_error_distribution_multi(
    errors_list: list[np.ndarray],
    labels: list[str],
    error_type: str,
    output_path: str,
    pixel_wise: bool = False
) -> None:
    """
    绘制多个误差分布在同一张图上（仅KDE曲线，无直方图）
    
    Args:
        errors_list: 多个误差数组的列表
        labels: 对应的标签列表
        error_type: 误差类型，'mse' 或 'mae'
        output_path: 输出图片路径
        pixel_wise: 是否为逐像素模式
    """
    print(f"\n绘制多曲线误差分布图（{len(labels)} 条曲线）...")
    
    # 初始化 MatplotlibVision（用于字体配置）
    visual = MatplotlibVision(log_dir=os.path.dirname(output_path), field_name=('Temperature',))
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 计算全局误差范围
    all_errors = np.concatenate(errors_list)
    min_error = all_errors.min()
    max_error = all_errors.max()
    x_grid = np.linspace(min_error, max_error, 500)
    
    # 使用颜色映射
    cmap = plt.get_cmap('tab10' if len(labels) <= 10 else 'tab20')
    colors = cmap(np.linspace(0, 1, len(labels)))
    
    # 绘制每条KDE曲线
    for i, (errors, label) in enumerate(zip(errors_list, labels)):
        kde = gaussian_kde(errors)
        ax.plot(x_grid, kde(x_grid), color=colors[i], 
                linewidth=2.5, label=label, alpha=0.85)
    
    # 设置坐标轴标签
    if pixel_wise:
        error_label = 'Normalized Difference (pred - true) / std'
        title = 'Pixel-wise Error Distribution Comparison'
    else:
        error_label = 'Normalized MSE' if error_type == 'mse' else 'Normalized MAE'
        title = f'Error Distribution Comparison ({error_type.upper()})'
    
    ax.set_xlabel(error_label, fontdict=visual.font)
    ax.set_ylabel('Probability Density', fontdict=visual.font)
    
    # 设置网格和图例
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", prop=visual.font, framealpha=0.9)
    ax.tick_params('both', labelsize=visual.font["size"])
    
    # 添加标题
    fig.suptitle(title, fontsize=visual.font["size"] + 4, y=0.98)
    
    # 保存图片
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"保存图片: {output_path}")


def print_statistics(train_errors: np.ndarray, valid_errors: np.ndarray, error_type: str) -> None:
    """
    打印误差统计信息
    
    Args:
        train_errors: 训练集误差
        valid_errors: 验证集误差
        error_type: 误差类型
    """
    print(f"\n误差统计信息 ({error_type.upper()}):")
    print(f"  Train - 均值: {train_errors.mean():.6f}, 中位数: {np.median(train_errors):.6f}, "
          f"标准差: {train_errors.std():.6f}")
    print(f"  Valid - 均值: {valid_errors.mean():.6f}, 中位数: {np.median(valid_errors):.6f}, "
          f"标准差: {valid_errors.std():.6f}")


def main():
    parser = argparse.ArgumentParser(description='绘制温度场误差分布图')
    parser.add_argument('--eval_dir', type=str, default=None,
                        help='单个评估输出目录路径（包含npy文件）')
    parser.add_argument('--eval_dirs', type=str, nargs='+', default=None,
                        help='多个评估目录路径（用于叠加绘图）')
    parser.add_argument('--labels', type=str, nargs='+', default=None,
                        help='每个目录的标签（可选，默认从目录名提取）')
    parser.add_argument('--error_type', type=str, default='mse', choices=['mse', 'mae'],
                        help='误差类型：mse(均方误差) 或 mae(平均绝对误差)，默认mse')
    parser.add_argument('--plot_mode', type=str, default='valid', choices=['train', 'valid', 'both'],
                        help='绘制模式：train(仅训练集), valid(仅验证集), both(两者都绘制)，默认valid（仅用于单目录模式）')
    parser.add_argument('--pixel_wise', action='store_true',
                        help='逐像素统计模式：直接统计每个像素的差值(pred-true)，不取绝对值')
    parser.add_argument('--output_name', type=str, default='error_distribution.png',
                        help='输出图片文件名，默认error_distribution.png')
    args = parser.parse_args()
    
    try:
        # 验证参数互斥性
        if args.eval_dir and args.eval_dirs:
            raise ValueError("--eval_dir 和 --eval_dirs 不能同时使用")
        if not args.eval_dir and not args.eval_dirs:
            raise ValueError("必须提供 --eval_dir 或 --eval_dirs")
        
        # 单目录模式
        if args.eval_dir:
            # 1. 加载npy文件
            train_true, train_pred, valid_true, valid_pred = load_npy_files(args.eval_dir)
            
            # 2. 计算误差
            mode_str = "逐像素差值" if args.pixel_wise else f"{args.error_type.upper()}误差"
            print(f"\n计算{mode_str}...")
            train_errors = compute_errors(train_true, train_pred, args.error_type, args.pixel_wise)
            valid_errors = compute_errors(valid_true, valid_pred, args.error_type, args.pixel_wise)
            
            if args.pixel_wise:
                print(f"  Train: {len(train_errors)} pixels")
                print(f"  Valid: {len(valid_errors)} pixels")
            else:
                print(f"  Train: {len(train_errors)} samples")
                print(f"  Valid: {len(valid_errors)} samples")
            
            # 3. 打印统计信息
            print_statistics(train_errors, valid_errors, args.error_type)
            
            # 4. 绘制误差分布图
            output_path = os.path.join(args.eval_dir, args.output_name)
            plot_error_distribution(train_errors, valid_errors, args.error_type, output_path, args.plot_mode, args.pixel_wise)
        
        # 多目录模式
        else:
            eval_dirs = args.eval_dirs
            
            # 处理标签
            if args.labels:
                if len(args.labels) != len(eval_dirs):
                    raise ValueError(f"标签数量({len(args.labels)})必须与目录数量({len(eval_dirs)})一致")
                labels = args.labels
            else:
                labels = [extract_label_from_path(d) for d in eval_dirs]
                print(f"\n自动提取的标签: {labels}")
            
            # 加载所有目录数据并计算误差（仅valid）
            mode_str = "逐像素差值" if args.pixel_wise else f"{args.error_type.upper()}误差"
            print(f"\n加载 {len(eval_dirs)} 个目录的数据（{mode_str}）...")
            errors_list = []
            for i, eval_dir in enumerate(eval_dirs):
                print(f"\n[{i+1}/{len(eval_dirs)}] 处理: {eval_dir}")
                _, _, valid_true, valid_pred = load_npy_files(eval_dir)
                valid_errors = compute_errors(valid_true, valid_pred, args.error_type, args.pixel_wise)
                errors_list.append(valid_errors)
                unit = "pixels" if args.pixel_wise else "samples"
                print(f"  Valid: {len(valid_errors)} {unit}, "
                      f"均值: {valid_errors.mean():.6f}, 标准差: {valid_errors.std():.6f}")
            
            # 绘制叠加图
            output_path = args.output_name if os.path.isabs(args.output_name) \
                          else os.path.join(os.getcwd(), args.output_name)
            plot_error_distribution_multi(errors_list, labels, args.error_type, output_path, args.pixel_wise)
        
        print("\n完成！")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

