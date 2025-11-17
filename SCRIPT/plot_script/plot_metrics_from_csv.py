#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标对比图绘制脚本

功能：
- 从CSV文件读取评估指标
- 绘制指标随样本数变化的曲线图（多模型、多方法对比）
- 支持选择不同指标（MSE/MAE/R²）

用法：
    # 绘制MSE对比图
    python plot_metrics_from_csv.py --csv metrics_summary.csv --metric mse
    
    # 绘制R²对比图
    python plot_metrics_from_csv.py --csv metrics_summary.csv --metric r2
    
    # 指定输出文件名
    python plot_metrics_from_csv.py --csv metrics_summary.csv --metric mae --output mae_comparison.png
    
    # 仅绘制特定模型
    python plot_metrics_from_csv.py --csv metrics_summary.csv --filter_model FNO
    
    # 仅绘制特定方法
    python plot_metrics_from_csv.py --csv metrics_summary.csv --filter_method DSSL
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# 配置：字体样式
FONT_CONFIG = {
    'family': 'DejaVu Sans',
    'weight': 'normal',
    'size': 14
}

# 配置：指标名称映射
METRIC_CONFIGS = {
    'mse': {
        'col_name': 'avg_mse',
        'ylabel': 'Mean Squared Error (MSE)',
        'scale': 'log',  # 'log' 或 'linear'
        'title': 'MSE Comparison'
    },
    'mae': {
        'col_name': 'avg_mae',
        'ylabel': 'Mean Absolute Error (MAE)',
        'scale': 'log',
        'title': 'MAE Comparison'
    },
    'r2': {
        'col_name': 'avg_r2',
        'ylabel': 'R² Score',
        'scale': 'linear',
        'title': 'R² Score Comparison'
    }
}

# 配置：颜色和线型
COLOR_SCHEME = {
    'FNO_BASE': '#1f77b4',    # 蓝色
    'FNO_DSSL': '#ff7f0e',    # 橙色
    'MLP_BASE': '#2ca02c',    # 绿色
    'MLP_DSSL': '#d62728',    # 红色
}

LINESTYLE_SCHEME = {
    'BASE': '--',  # 虚线
    'DSSL': '-',   # 实线
}

MARKER_SCHEME = {
    'FNO': 'o',    # 圆形
    'MLP': 's',    # 方形
}


def load_csv(csv_path: str) -> pd.DataFrame:
    """加载CSV文件"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"加载CSV: {csv_path}")
    print(f"  总记录数: {len(df)}")
    return df


def filter_data(df: pd.DataFrame, 
                filter_model: Optional[str] = None,
                filter_method: Optional[str] = None,
                filter_split: Optional[str] = None) -> pd.DataFrame:
    """过滤数据"""
    df_filtered = df.copy()
    
    if filter_model:
        df_filtered = df_filtered[df_filtered['model_type'] == filter_model.upper()]
        print(f"  过滤模型: {filter_model}")
    
    if filter_method:
        df_filtered = df_filtered[df_filtered['method'] == filter_method.upper()]
        print(f"  过滤方法: {filter_method}")
    
    if filter_split:
        df_filtered = df_filtered[df_filtered['split'] == filter_split]
        print(f"  过滤划分: {filter_split}")
    
    print(f"  过滤后记录数: {len(df_filtered)}")
    return df_filtered


def prepare_plot_data(df: pd.DataFrame, metric_col: str):
    """
    准备绘图数据：按 (model_type, method) 分组
    
    Returns:
        plot_groups: {
            'FNO_BASE': {'ntrain': [...], 'metric': [...]},
            'FNO_DSSL': {'ntrain': [...], 'metric': [...]},
            ...
        }
    """
    plot_groups = {}
    
    # 按 model_type 和 method 分组
    for (model, method), group_df in df.groupby(['model_type', 'method']):
        key = f"{model}_{method}"
        
        # 按 ntrain 排序
        group_df = group_df.sort_values('ntrain')
        
        # 提取数据（去除NaN）
        valid_mask = group_df[metric_col].notna()
        ntrain_vals = group_df.loc[valid_mask, 'ntrain'].values
        metric_vals = group_df.loc[valid_mask, metric_col].values
        
        if len(ntrain_vals) > 0:
            plot_groups[key] = {
                'ntrain': ntrain_vals,
                'metric': metric_vals,
                'model': model,
                'method': method
            }
    
    return plot_groups


def plot_metrics_comparison(plot_groups: dict, 
                            metric_name: str,
                            output_path: str,
                            figsize: tuple = (12, 8)):
    """
    绘制指标对比图
    
    Args:
        plot_groups: 准备好的绘图数据
        metric_name: 指标名称 ('mse', 'mae', 'r2')
        output_path: 输出图片路径
        figsize: 图像尺寸
    """
    config = METRIC_CONFIGS[metric_name]
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    print(f"\n绘制 {metric_name.upper()} 对比图...")
    
    # 绘制每条曲线
    for key, data in plot_groups.items():
        model = data['model']
        method = data['method']
        ntrain = data['ntrain']
        metric = data['metric']
        
        # 确定颜色、线型、标记
        color = COLOR_SCHEME.get(key, '#333333')
        linestyle = LINESTYLE_SCHEME.get(method, '-')
        marker = MARKER_SCHEME.get(model, 'o')
        
        label = f"{model}-{method}"
        
        # 绘制曲线
        ax.plot(ntrain, metric, 
                color=color, 
                linestyle=linestyle, 
                marker=marker,
                markersize=8,
                linewidth=2.5,
                label=label,
                alpha=0.85)
        
        print(f"  [✓] {label}: {len(ntrain)} 个数据点")
    
    # 设置坐标轴
    ax.set_xlabel('Training Sample Size (ntrain)', fontdict=FONT_CONFIG)
    ax.set_ylabel(config['ylabel'], fontdict=FONT_CONFIG)
    
    # 设置Y轴尺度
    if config['scale'] == 'log':
        ax.set_yscale('log')
    
    # 设置网格和图例
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='best', prop=FONT_CONFIG, framealpha=0.9)
    ax.tick_params('both', labelsize=FONT_CONFIG["size"])
    
    # 添加标题
    fig.suptitle(config['title'], fontsize=FONT_CONFIG["size"] + 4, y=0.98)
    
    # 保存图片
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n[成功] 保存图片: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='从CSV绘制指标对比图')
    parser.add_argument('--csv', type=str, required=True,
                        help='输入CSV文件路径')
    parser.add_argument('--metric', type=str, default='mse', 
                        choices=['mse', 'mae', 'r2'],
                        help='指标类型：mse, mae, r2，默认mse')
    parser.add_argument('--output', type=str, default=None,
                        help='输出图片路径（默认：metrics_{metric}_comparison.png）')
    parser.add_argument('--filter_model', type=str, default=None,
                        help='仅绘制指定模型（如：FNO 或 MLP）')
    parser.add_argument('--filter_method', type=str, default=None,
                        help='仅绘制指定方法（如：BASE 或 DSSL）')
    parser.add_argument('--filter_split', type=str, default=None,
                        help='仅绘制指定数据划分（如：train, valid, all）')
    parser.add_argument('--figsize', type=float, nargs=2, default=[12, 8],
                        help='图像尺寸（宽 高），默认 12 8')
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output is None:
        args.output = f'metrics_{args.metric}_comparison.png'
    
    print("=" * 60)
    print("指标对比图绘制脚本")
    print(f"输入CSV: {args.csv}")
    print(f"指标类型: {args.metric.upper()}")
    print(f"输出图片: {args.output}")
    print("=" * 60)
    
    try:
        # 1. 加载CSV
        df = load_csv(args.csv)
        
        # 2. 过滤数据
        df_filtered = filter_data(df, 
                                  filter_model=args.filter_model,
                                  filter_method=args.filter_method,
                                  filter_split=args.filter_split)
        
        if len(df_filtered) == 0:
            print("\n[错误] 过滤后没有数据，请检查过滤条件")
            sys.exit(1)
        
        # 3. 准备绘图数据
        metric_col = METRIC_CONFIGS[args.metric]['col_name']
        plot_groups = prepare_plot_data(df_filtered, metric_col)
        
        if len(plot_groups) == 0:
            print(f"\n[错误] 没有有效的 {metric_col} 数据")
            sys.exit(1)
        
        # 4. 绘制对比图
        plot_metrics_comparison(plot_groups, args.metric, args.output, 
                               figsize=tuple(args.figsize))
        
        print("=" * 60)
        print("完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[错误] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


