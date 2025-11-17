#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文收敛图绘制脚本
用于绘制BASE和DSSL两种方法的训练收敛对比图

用法：直接运行，修改配置区域的参数即可
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 路径注入
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ========== 配置区域（集中管理所有参数） ==========

# 输入参数
MODEL_TYPE = 'fno'  # 模型类型：'fno' 或 'mlp'
SAMPLE_NUM = 2000    # 样本数量：100, 500, 1000, 2000, 4000
SHOW_LEGEND = False           # 是否显示图例

# 画布参数
FIGURE_WIDTH = 10   # 图像宽度（英寸）
FIGURE_HEIGHT = 6   # 图像高度（英寸）
DPI = 400           # 分辨率

# 坐标轴参数
XLABEL = 'Epoch'    # X轴标签
YLABEL = 'Loss'     # Y轴标签
XLIM_MIN = 0        # X轴下界（可选，None为自动）
XLIM_MAX = 950     # X轴上界（可选，None为自动）
YLIM_MIN = 1e-4     # Y轴下界（对数坐标）
YLIM_MAX = 3e0      # Y轴上界（对数坐标）
TITLE = f'{MODEL_TYPE.upper()} (n={SAMPLE_NUM})'  # 图标题

# 线条样式
LINEWIDTH_BASE = 2.5    # BASE方法线宽（保留但不再直接使用）
LINEWIDTH_DSSL = 2.5    # DSSL方法线宽（保留但不再直接使用）
LINESTYLE_BASE = '-'    # 保留但不再直接使用
LINESTYLE_DSSL = '-'    # 保留但不再直接使用
ALPHA_BASE = 0.5        # 保留但不再直接使用
ALPHA_DSSL = 1.0        # 保留但不再直接使用

# 新的线宽/透明度策略（强调valid）
LINEWIDTH_VALID = 3.2
LINEWIDTH_OTHERS = 2.0
ALPHA_VALID = 1.0
ALPHA_OTHERS = 0.5

# 配色方案（仅按方法区分颜色）
METHOD_COLORS = {
    'base': '#6BAED6',  # BASE 浅蓝
    'dssl': '#D95319',  # DSSL 深橙
}

# 字体参数
FONT_FAMILY = 'DejaVu Serif'  # 字体家族
FONT_SIZE_LABEL = 20          # 坐标轴标签字号
FONT_SIZE_TITLE = 22          # 标题字号
FONT_SIZE_TICK = 18           # 刻度字号

# 图例参数

LEGEND_LOC = 'lower left'     # 图例位置（图内左下角）
LEGEND_FONTSIZE = 14          # 图例字号

# 图例名称（可自定义）
LEGEND_LABELS = {
    'base_train': 'BASE-train',           # BASE方法train的图例名称
    'base_valid': 'BASE-valid',           # BASE方法valid的图例名称
    'base_train_self': 'BASE-consist', # BASE方法train_self的图例名称
    'dssl_train': 'DSSL-train',           # DSSL方法train的图例名称
    'dssl_valid': 'DSSL-valid',           # DSSL方法valid的图例名称
    'dssl_train_self': 'DSSL-consist'  # DSSL方法train_self的图例名称
}

# Marker参数（增加区分度）
USE_MARKERS = True            # 是否使用markers
MARKER_INTERVAL = 100         # Marker间隔（每N个epoch显示一个marker）
# 按损失类型选择marker（仅用于辅助区分线型）
LOSS_MARKERS = {
    'train': 'o',
    'valid': 's',
    'train_self': '^',
}
MARKER_SIZE = 13               # Marker大小
MARKER_EDGE_WIDTH = 2       # Marker边框宽度
# Marker内部填充颜色：valid=黑色，train=白色，consist=方法颜色（通过函数动态获取）

# 网格参数
GRID_ALPHA = 0.3              # 网格透明度

# 线型按损失类型区分
LOSS_LINESTYLES = {
    'train': '-',            # 实线
    'valid': '-',    # 长虚线
    'train_self': '-'# 点划线
}

# 输出参数
OUTPUT_DIR = 'work_post_statistic'                              # 输出目录
OUTPUT_FILENAME = f'convergence_{MODEL_TYPE}_n{SAMPLE_NUM}_new.png'  # 输出文件名

# ========== 配置区域结束 ==========


def get_marker_face_color(loss_type, method_color):
    """
    根据损失类型获取marker的填充颜色
    
    Args:
        loss_type: 'train', 'valid', 或 'train_self'
        method_color: 方法的颜色（用于consist类型）
    
    Returns:
        str: marker填充颜色
    """
    if loss_type == 'valid':
        return 'black'
    elif loss_type == 'train':
        return 'white'
    elif loss_type == 'train_self':  # consist
        return method_color
    else:
        return 'black'  # 默认


def find_loss_file(model, method, sample_num):
    """
    查找loss_history.npy文件
    
    Args:
        model: 模型类型 'fno' 或 'mlp'
        method: 训练方法 'base' 或 'dssl'
        sample_num: 样本数量
    
    Returns:
        str: loss_history.npy文件的完整路径
    
    Raises:
        FileNotFoundError: 如果找不到文件
    """
    model_upper = model.upper()
    
    # 构建目录模式
    if method.lower() == 'base':
        # work_sate_20251024/work_satellite_BASE_fno_20251024/FNO_BASE_n100_*/loss_history.npy
        dir_pattern = os.path.join(
            PROJECT_ROOT,
            'work_sate_20251024',
            f'work_satellite_BASE_{model}_20251024',
            f'{model_upper}_BASE_n{sample_num}_*',
            'loss_history.npy'
        )
    else:  # dssl
        # work_sate_20251024/work_satellite_dssl_fno_20251021/FNO_DSSL_n100_*/loss_history.npy
        dir_pattern = os.path.join(
            PROJECT_ROOT,
            'work_sate_20251024',
            f'work_satellite_dssl_{model}_20251021',
            f'{model_upper}_DSSL_n{sample_num}_*',
            'loss_history.npy'
        )
    
    # 使用glob查找匹配的文件
    matches = glob.glob(dir_pattern)
    
    if not matches:
        raise FileNotFoundError(
            f"找不到文件: {dir_pattern}\n"
            f"模型={model}, 方法={method}, 样本数={sample_num}"
        )
    
    if len(matches) > 1:
        print(f"警告: 找到多个匹配文件，使用第一个: {matches[0]}")
    
    return matches[0]


def load_loss_data(file_path):
    """
    加载loss_history.npy文件
    
    Args:
        file_path: .npy文件路径
    
    Returns:
        dict: 包含'train', 'valid', 'train_self'的字典
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 加载npy文件（0维object数组）
    data = np.load(file_path, allow_pickle=True)
    data = data.item()  # 转换为dict
    
    # 提取三个损失曲线
    train = np.asarray(data['train'], dtype=float)
    valid = np.asarray(data['valid'], dtype=float)
    train_self = np.asarray(data['train_self'], dtype=float)
    
    return {
        'train': train,
        'valid': valid,
        'train_self': train_self
    }


def plot_paper_convergence():
    """
    主绘图函数：绘制论文级别的收敛对比图
    """
    # 配置matplotlib
    config = {
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE_LABEL,
        "mathtext.fontset": 'stix',
    }
    rcParams.update(config)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    print(f"正在绘制 {MODEL_TYPE.upper()} 模型，样本数 {SAMPLE_NUM} 的收敛图...")
    
    # 加载BASE方法的数据
    try:
        base_file = find_loss_file(MODEL_TYPE, 'base', SAMPLE_NUM)
        print(f"  加载BASE数据: {base_file}")
        base_data = load_loss_data(base_file)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 加载DSSL方法的数据
    try:
        dssl_file = find_loss_file(MODEL_TYPE, 'dssl', SAMPLE_NUM)
        print(f"  加载DSSL数据: {dssl_file}")
        dssl_data = load_loss_data(dssl_file)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return
    
    # 绘制BASE方法的3条线（颜色=方法；线型/marker=损失；valid加粗且不透明）
    for loss_type in ['train', 'valid', 'train_self']:
        values = base_data[loss_type]
        if len(values) > 0:
            epochs = np.arange(len(values))
            color = METHOD_COLORS['base']
            linestyle = LOSS_LINESTYLES[loss_type]
            linewidth = LINEWIDTH_VALID if loss_type == 'valid' else LINEWIDTH_OTHERS
            alpha = ALPHA_VALID if loss_type == 'valid' else ALPHA_OTHERS
            
            # 绘制主曲线（不带label，图例由后面的虚拟线条生成）
            ax.semilogy(
                epochs, values,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha
            )
            
            # 添加markers（每隔MARKER_INTERVAL个点，排除头尾）
            if USE_MARKERS and len(values) > 2 * MARKER_INTERVAL:
                marker_indices = np.arange(MARKER_INTERVAL, len(values) - 1, MARKER_INTERVAL)
                marker_face_color = get_marker_face_color(loss_type, color)
                ax.semilogy(
                    epochs[marker_indices], values[marker_indices],
                    marker=LOSS_MARKERS[loss_type],
                    markersize=MARKER_SIZE,
                    markerfacecolor=marker_face_color,
                    markeredgecolor=color,
                    markeredgewidth=MARKER_EDGE_WIDTH,
                    linestyle='None',
                    alpha=1.0  # marker不透明
                )
            
            print(f"  绘制 BASE-{loss_type}: {len(values)} 个epoch")
    
    # 绘制DSSL方法的3条线（颜色=方法；线型/marker=损失；valid加粗且不透明）
    for loss_type in ['train', 'valid', 'train_self']:
        values = dssl_data[loss_type]
        if len(values) > 0:
            epochs = np.arange(len(values))
            color = METHOD_COLORS['dssl']
            linestyle = LOSS_LINESTYLES[loss_type]
            linewidth = LINEWIDTH_VALID if loss_type == 'valid' else LINEWIDTH_OTHERS
            alpha = ALPHA_VALID if loss_type == 'valid' else ALPHA_OTHERS
            
            # 绘制主曲线（不带label，图例由后面的虚拟线条生成）
            ax.semilogy(
                epochs, values,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                alpha=alpha
            )
            
            # 添加markers（每隔MARKER_INTERVAL个点，排除头尾）
            if USE_MARKERS and len(values) > 2 * MARKER_INTERVAL:
                marker_indices = np.arange(MARKER_INTERVAL, len(values) - 1, MARKER_INTERVAL)
                marker_face_color = get_marker_face_color(loss_type, color)
                ax.semilogy(
                    epochs[marker_indices], values[marker_indices],
                    marker=LOSS_MARKERS[loss_type],
                    markersize=MARKER_SIZE,
                    markerfacecolor=marker_face_color,
                    markeredgecolor=color,
                    markeredgewidth=MARKER_EDGE_WIDTH,
                    linestyle='None',
                    alpha=1.0  # marker不透明
                )
            
            print(f"  绘制 DSSL-{loss_type}: {len(values)} 个epoch")
    
    # 创建图例用的虚拟线条（在不可见区域绘制，仅用于生成图例）
    if SHOW_LEGEND:
        # 在图外位置（负坐标）绘制虚拟线条
        dummy_x = np.array([-1000, -999])
        dummy_y = np.array([1, 1])

        # 组合图例：方法决定颜色；损失决定线型与marker
        for method in ['base', 'dssl']:
            method_color = METHOD_COLORS[method]
            for loss_type in ['train', 'valid', 'train_self']:
                linestyle = LOSS_LINESTYLES[loss_type]
                linewidth = LINEWIDTH_VALID if loss_type == 'valid' else LINEWIDTH_OTHERS
                alpha = ALPHA_VALID if loss_type == 'valid' else ALPHA_OTHERS
                label_key = f"{method}_{loss_type}".lower()
                # 若用户自定义了LEGEND_LABELS键名，兼容读取；否则构造默认
                label = LEGEND_LABELS.get(label_key, f"{method.upper()}-{ 'consist' if loss_type=='train_self' else loss_type }")

                if USE_MARKERS:
                    marker_face_color = get_marker_face_color(loss_type, method_color)
                    ax.semilogy(
                        dummy_x, dummy_y,
                        label=label,
                        color=method_color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=alpha,
                        marker=LOSS_MARKERS[loss_type],
                        markersize=MARKER_SIZE,
                        markerfacecolor=marker_face_color,
                        markeredgecolor=method_color,
                        markeredgewidth=MARKER_EDGE_WIDTH,
                        markevery=1
                    )
                else:
                    ax.semilogy(
                        dummy_x, dummy_y,
                        label=label,
                        color=method_color,
                        linestyle=linestyle,
                        linewidth=linewidth,
                        alpha=alpha
                    )
    
    # 设置坐标轴
    ax.set_xlabel(XLABEL, fontsize=FONT_SIZE_LABEL)
    ax.set_ylabel(YLABEL, fontsize=FONT_SIZE_LABEL)
    
    # 设置X轴范围（如果指定）
    if XLIM_MIN is not None or XLIM_MAX is not None:
        ax.set_xlim(XLIM_MIN, XLIM_MAX)
    
    # 设置Y轴范围
    ax.set_ylim(YLIM_MIN, YLIM_MAX)
    
    # 设置标题
    ax.set_title(TITLE, fontsize=FONT_SIZE_TITLE, fontweight='bold')
    
    # 设置网格
    ax.grid(True, alpha=GRID_ALPHA)
    
    # 设置刻度字号
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICK)
    
    # 设置图例（可选）
    if SHOW_LEGEND:
        ax.legend(
            loc=LEGEND_LOC,
            fontsize=LEGEND_FONTSIZE,
            frameon=True,
            fancybox=True,
            shadow=False,
            framealpha=0.9
        )
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir_path = os.path.join(PROJECT_ROOT, OUTPUT_DIR)
    os.makedirs(output_dir_path, exist_ok=True)
    
    # 保存图片
    output_path = os.path.join(output_dir_path, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    print(f"\n收敛图已保存到: {output_path}")
    
    # 关闭图形以释放内存
    plt.close(fig)


if __name__ == '__main__':
    plot_paper_convergence()

