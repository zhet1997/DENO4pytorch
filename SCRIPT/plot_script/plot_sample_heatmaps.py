#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
样本对比云图绘制脚本（论文版）
功能：绘制 N行4列 的对比图
每行：[布局图] [真实温度] [预测温度] [误差]
所有参数集中在顶部配置区
"""

import os
import sys
import json
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle, Circle, Polygon
from matplotlib import cm

# 路径注入
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from Demo.satellite_2d_base.dataset_satellite import load_satellite_data


# ========== 全局配置区（所有参数集中在此，方便修改） ==========

# --- 数据路径配置 ---
EVAL_DIR = '/data/wqn/DENO4pytorch/work_post_eval/work_satellite_BASE_fno_20251024/FNO_BASE_n2000_20251025_034709'  # 评估结果目录
DATA_PATH = '/data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5'  # 原始数据集路径（用于计算总样本数）
SAMPLE_JSON_DIR = '/data/wqn/datasets/dataset20251017_6c/samples'  # sample_definition.json所在目录
SPLIT = 'valid'  # 使用的数据集划分: 'train' 或 'valid'
NTRAIN = 1000    # 训练集样本数（需与训练时一致）
NVALID = 500     # 验证集样本数（需与训练时一致）

# --- 样本选择配置 ---
USE_RANDOM_SAMPLES = False   # True: 随机选择 | False: 使用指定索引
NUM_SAMPLES = 5             # 随机选择时的样本数量
SPECIFIED_INDICES = [0, 10, 17, 30, 40]  # 指定索引（相对于当前split）

# --- 图像尺寸与分辨率 ---
FIG_WIDTH = 16              # 图像宽度（英寸）
FIG_HEIGHT_PER_ROW = 4      # 每行高度（英寸）
DPI = 150                   # 分辨率

# --- 温度场colormap配置 ---
TEMP_CMAP = 'jet'           # 温度场色图（推荐：jet, hot, inferno）
TEMP_VMIN = 270            # 温度场最小值（None则自动计算全局最小值）
TEMP_VMAX = 370            # 温度场最大值（None则自动计算全局最大值）
TEMP_CONTOUR_LEVELS = 15    # 等高线数量（等间距）
TEMP_CONTOUR_COLOR = 'black'  # 等高线颜色
TEMP_CONTOUR_LINEWIDTH = 1  # 等高线线宽
TEMP_CONTOUR_ALPHA = 0.6      # 等高线透明度

# --- 误差场colormap配置 ---
ERROR_CMAP = 'RdBu_r'       # 误差场色图（发散色图）
ERROR_VMAX = 10           # 误差场绝对值上限（None则自动计算）

# --- 布局图绘制参数（继承自plot_satellite_layout.py） ---
LAYOUT_COMP_LINEWIDTH = 1.0      # 元件轮廓线宽
LAYOUT_COMP_ALPHA = 0.5          # 元件透明度
LAYOUT_POWER_CMAP_NAME = 'YlOrRd'  # 元件功率色图名称
LAYOUT_POWER_MIN = 0             # 功率色图下限（W/m²）
LAYOUT_POWER_MAX = 1000         # 功率色图上限（W/m²）
LAYOUT_LABEL_SIZE = 12           # 元件内功率标注字号
LAYOUT_LABEL_COLOR = 'black'     # 标注颜色
LAYOUT_LABEL_WEIGHT = 'bold'     # 标注字重
LAYOUT_COOLING_WIDTH = 5         # 冷却线宽
LAYOUT_COOLING_COLOR = 'darkred' # 冷却线颜色
LAYOUT_COOLING_CAP_SIZE = 0.015  # 冷却线端点标记长度（相对布局尺寸）
LAYOUT_BORDER_WIDTH = 2.2        # 边界线宽
LAYOUT_CAPSULE_RES = 160         # 胶囊外轮廓插值点数

# --- Colorbar配置 ---
USE_COLORBAR = False      # 是否显示colorbar（False则完全不显示，True则每列一个）

# ===============================================================


def _split_dataset(inputs: np.ndarray,
                   outputs: np.ndarray,
                   ntrain: Optional[int],
                   nvalid: Optional[int]) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    数据划分逻辑（复用 eval_satellite_predict.py）
    返回 ((train_x, train_y), (valid_x, valid_y))
    """
    N = inputs.shape[0]
    if ntrain is None or nvalid is None:
        ntrain = int(N * 0.9)
        nvalid = N - ntrain
    else:
        assert ntrain + nvalid <= N, f"ntrain({ntrain}) + nvalid({nvalid}) 超过数据规模 {N}"

    train_x = inputs[:ntrain]
    train_y = outputs[:ntrain]
    valid_x = inputs[N - nvalid:]
    valid_y = outputs[N - nvalid:]
    
    return (train_x, train_y), (valid_x, valid_y)


def load_prediction_data(eval_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从后处理目录加载预测结果
    
    Returns:
        (pred_data, true_data): 形状均为 (N, H, W, 1)
    """
    pred_path = os.path.join(eval_dir, f'{split}_pred.npy')
    true_path = os.path.join(eval_dir, f'{split}_true.npy')
    
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"真实值文件不存在: {true_path}")
    
    pred_data = np.load(pred_path)
    true_data = np.load(true_path)
    
    print(f"加载预测数据: {pred_path}")
    print(f"  预测形状: {pred_data.shape}")
    print(f"  真实形状: {true_data.shape}")
    
    return pred_data, true_data


def map_split_index_to_global(split_idx: int, split: str, ntrain: int, nvalid: int, total_samples: int) -> str:
    """
    将split内的相对索引映射到全局样本索引
    
    Args:
        split_idx: split内的索引（0-based）
        split: 'train' 或 'valid'
        ntrain: 训练集样本数
        nvalid: 验证集样本数
        total_samples: 数据集总样本数
    
    Returns:
        sample文件夹名称，如 'sample_0042'
    """
    if split == 'train':
        global_idx = split_idx
    elif split == 'valid':
        global_idx = total_samples - nvalid + split_idx
    else:
        raise ValueError(f"不支持的 split 参数: {split}")
    
    return f'sample_{global_idx:04d}'


def load_layout_json(json_path: str) -> Tuple[float, float, list, list]:
    """
    从sample_definition.json读取布局信息
    
    Returns:
        (width, height, components, cooling_lines)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    layout = data['layout_domain']
    if isinstance(layout, dict):
        width, height = layout['width'], layout['height']
    else:
        width, height = layout[0], layout[1]
    
    return width, height, data['components'], data.get('cooling_lines', [])


def plot_capsule_on_ax(ax, cx: float, cy: float, length: float, width: float, 
                       angle: float = 0., facecolor='w', edgecolor='k', 
                       lw: float = 1.5, alpha: float = 1.0):
    """
    在给定ax上绘制胶囊形状
    采用正确几何：rect_length = length - width, radius = width/2
    """
    radius = width / 2
    rect_length = length - width
    
    # 左半圆: 从上方(π/2)到下方(3π/2)
    theta_left = np.linspace(np.pi/2, 3*np.pi/2, LAYOUT_CAPSULE_RES//2)
    x_left = (cx - rect_length/2) + radius * np.cos(theta_left)
    y_left = cy + radius * np.sin(theta_left)
    
    # 右半圆: 从下方(-π/2)到上方(π/2)
    theta_right = np.linspace(-np.pi/2, np.pi/2, LAYOUT_CAPSULE_RES//2)
    x_right = (cx + rect_length/2) + radius * np.cos(theta_right)
    y_right = cy + radius * np.sin(theta_right)
    
    # 合并轮廓
    x = np.concatenate([x_left, x_right])
    y = np.concatenate([y_left, y_right])
    
    # 旋转处理
    if angle != 0:
        A = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                      [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]])
        pts = np.stack([x-cx, y-cy], axis=0)
        rot_pts = A @ pts
        x = rot_pts[0,:] + cx
        y = rot_pts[1,:] + cy
    
    # 绘制多边形
    patch = Polygon(np.stack([x,y], axis=1), closed=True, 
                   facecolor=facecolor, edgecolor=edgecolor, 
                   lw=lw, alpha=alpha, zorder=2)
    ax.add_patch(patch)


def plot_layout_on_ax(ax, json_path: str):
    """
    在给定的ax上绘制卫星元件布局图
    """
    # 读取数据
    W, H, components, cooling_lines = load_layout_json(json_path)
    
    # 元件功率范围，用于colormap
    powers = [c['power'] for c in components if 'power' in c]
    pmin = LAYOUT_POWER_MIN if LAYOUT_POWER_MIN is not None else min(powers)
    pmax = LAYOUT_POWER_MAX if LAYOUT_POWER_MAX is not None else max(powers)
    norm = plt.Normalize(vmin=pmin, vmax=pmax)
    power_cmap = cm.get_cmap(LAYOUT_POWER_CMAP_NAME)
    
    # 绘制元件
    for comp in components:
        cx, cy = comp['center']
        shape = comp['shape']
        pow_val = comp['power']
        color = power_cmap(norm(pow_val))
        
        if shape == 'rect':
            w, h = comp['width'], comp['height']
            ax.add_patch(Rectangle((cx-w/2, cy-h/2), w, h,
                                   facecolor=color, edgecolor='black',
                                   lw=LAYOUT_COMP_LINEWIDTH, zorder=2, alpha=LAYOUT_COMP_ALPHA))
        elif shape == 'circle':
            r = comp['radius']
            ax.add_patch(Circle((cx, cy), r,
                                facecolor=color, edgecolor='black',
                                lw=LAYOUT_COMP_LINEWIDTH, zorder=2, alpha=LAYOUT_COMP_ALPHA))
        elif shape == 'capsule':
            l, w = comp['length'], comp['width']
            rot = comp.get('rotation', 0)
            plot_capsule_on_ax(ax, cx, cy, l, w, angle=rot,
                              facecolor=color, edgecolor='black',
                              lw=LAYOUT_COMP_LINEWIDTH, alpha=LAYOUT_COMP_ALPHA)
        
        # 功率标注
        total_p = comp.get('total_power', 0)
        ax.text(cx, cy, f"{total_p:.1f}W", fontsize=LAYOUT_LABEL_SIZE,
                color=LAYOUT_LABEL_COLOR, weight=LAYOUT_LABEL_WEIGHT, 
                ha='center', va='center', zorder=3)
    
    # 绘制冷却线
    for cl in cooling_lines:
        ep = cl['endpoints']
        if len(ep) != 2: 
            continue
        (x1, y1), (x2, y2) = ep
        
        # 主线段
        ax.plot([x1, x2], [y1, y2], color=LAYOUT_COOLING_COLOR, 
                lw=LAYOUT_COOLING_WIDTH, solid_capstyle='round', zorder=4)
        
        # 端点标记小线条
        cap_len = LAYOUT_COOLING_CAP_SIZE * max(W, H)
        if abs(x2 - x1) > abs(y2 - y1):  # 横向线
            ax.plot([x1, x1], [y1 - cap_len, y1 + cap_len], 
                   color=LAYOUT_COOLING_COLOR, lw=LAYOUT_COOLING_WIDTH*0.8, zorder=4)
            ax.plot([x2, x2], [y2 - cap_len, y2 + cap_len], 
                   color=LAYOUT_COOLING_COLOR, lw=LAYOUT_COOLING_WIDTH*0.8, zorder=4)
        else:  # 纵向线
            ax.plot([x1 - cap_len, x1 + cap_len], [y1, y1], 
                   color=LAYOUT_COOLING_COLOR, lw=LAYOUT_COOLING_WIDTH*0.8, zorder=4)
            ax.plot([x2 - cap_len, x2 + cap_len], [y2, y2], 
                   color=LAYOUT_COOLING_COLOR, lw=LAYOUT_COOLING_WIDTH*0.8, zorder=4)
    
    # 绘制外边界
    ax.add_patch(Rectangle((0, 0), W, H, fill=False, edgecolor='black', 
                          lw=LAYOUT_BORDER_WIDTH, zorder=1))
    
    # 设置坐标范围和样式
    ax.set_xlim(-0.05*W, W*1.05)
    ax.set_ylim(-0.05*H, H*1.05)
    ax.set_aspect('equal')
    ax.axis('off')


def select_sample_indices(total_samples: int) -> np.ndarray:
    """
    根据配置选择样本索引
    
    Returns:
        indices: 选中的样本索引数组（相对于当前split）
    """
    if USE_RANDOM_SAMPLES:
        num = min(NUM_SAMPLES, total_samples)
        indices = np.random.choice(total_samples, size=num, replace=False)
        indices = np.sort(indices)
        print(f"随机选择 {num} 个样本: {indices}")
    else:
        indices = np.array(SPECIFIED_INDICES)
        valid_indices = indices[indices < total_samples]
        if len(valid_indices) < len(indices):
            print("警告: 部分指定索引超出范围，已过滤")
        indices = valid_indices
        print(f"使用指定索引: {indices}")
    
    return indices


def save_colorbar_figures(temp_vmin: float, temp_vmax: float, error_vmax: float, 
                          output_dir: str, split: str):
    """
    保存温度和误差的colorbar为独立图片
    
    Args:
        temp_vmin, temp_vmax: 温度范围
        error_vmax: 误差绝对值上限
        output_dir: 输出目录
        split: 数据集划分名称
    """
    # 温度colorbar
    fig_temp = plt.figure(figsize=(0.5, 6))
    ax_temp = fig_temp.add_axes([0, 0, 1, 1])
    sm_temp = plt.cm.ScalarMappable(cmap=TEMP_CMAP, 
                                     norm=plt.Normalize(vmin=temp_vmin, vmax=temp_vmax))
    sm_temp.set_array([])
    cbar_temp = plt.colorbar(sm_temp, cax=ax_temp)
    cbar_temp.set_label('Temperature (K)', fontsize=14)
    plt.savefig(os.path.join(output_dir, f'temp_colorbar_{split}.png'), 
                dpi=DPI, bbox_inches='tight')
    plt.close(fig_temp)
    
    # 误差colorbar
    fig_error = plt.figure(figsize=(0.5, 6))
    ax_error = fig_error.add_axes([0, 0, 1, 1])
    sm_error = plt.cm.ScalarMappable(cmap=ERROR_CMAP, 
                                     norm=mcolors.TwoSlopeNorm(vmin=-error_vmax, vcenter=0, vmax=error_vmax))
    sm_error.set_array([])
    cbar_error = plt.colorbar(sm_error, cax=ax_error)
    cbar_error.set_label('Error (K)', fontsize=14)
    plt.savefig(os.path.join(output_dir, f'error_colorbar_{split}.png'), 
                dpi=DPI, bbox_inches='tight')
    plt.close(fig_error)
    
    print(f"已保存colorbar图片: temp_colorbar_{split}.png, error_colorbar_{split}.png")


def plot_sample_heatmaps(true_data: np.ndarray,
                         pred_data: np.ndarray,
                         sample_indices: np.ndarray,
                         split: str,
                         ntrain: int,
                         nvalid: int,
                         total_samples: int,
                         save_path: str):
    """
    绘制样本对比云图：N 行 4 列
    每行：[布局图] [真实温度] [预测温度] [误差]
    
    Returns:
        (temp_vmin, temp_vmax, error_vmax): 温度范围和误差范围
    """
    n_samples = len(sample_indices)
    
    # 计算全局温度范围
    temp_vmin = TEMP_VMIN if TEMP_VMIN is not None else min(true_data.min(), pred_data.min())
    temp_vmax = TEMP_VMAX if TEMP_VMAX is not None else max(true_data.max(), pred_data.max())
    
    # 计算全局误差范围
    all_errors = pred_data - true_data
    error_vmax = ERROR_VMAX if ERROR_VMAX is not None else np.abs(all_errors).max()
    
    print(f"温度范围: [{temp_vmin:.2f}, {temp_vmax:.2f}]")
    print(f"误差范围: [-{error_vmax:.2f}, {error_vmax:.2f}]")
    
    # 创建子图网格
    fig, axes = plt.subplots(n_samples, 4, figsize=(FIG_WIDTH, FIG_HEIGHT_PER_ROW * n_samples))
    
    # 确保 axes 是 2D 数组
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(sample_indices):
        # 获取全局样本文件夹名
        sample_folder = map_split_index_to_global(idx, split, ntrain, nvalid, total_samples)
        json_path = os.path.join(SAMPLE_JSON_DIR, sample_folder, 'sample_definition.json')
        
        # 提取温度和误差数据
        true_field = true_data[idx, :, :, 0]  # (64, 64)
        pred_field = pred_data[idx, :, :, 0]  # (64, 64)
        error_field = pred_field - true_field
        
        # 列1: 布局图
        ax = axes[i, 0]
        if os.path.exists(json_path):
            plot_layout_on_ax(ax, json_path)
        else:
            ax.text(0.5, 0.5, f'JSON not found:\n{sample_folder}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
        
        # 列2: 真实温度场
        ax = axes[i, 1]
        ax.imshow(true_field, cmap=TEMP_CMAP, origin='lower', 
                 vmin=temp_vmin, vmax=temp_vmax)
        # 添加等高线
        contour_levels = np.linspace(temp_vmin, temp_vmax, TEMP_CONTOUR_LEVELS)
        ax.contour(true_field, levels=contour_levels, colors=TEMP_CONTOUR_COLOR,
                  linewidths=TEMP_CONTOUR_LINEWIDTH, alpha=TEMP_CONTOUR_ALPHA, origin='lower')
        ax.axis('off')
        
        # 列3: 预测温度场
        ax = axes[i, 2]
        ax.imshow(pred_field, cmap=TEMP_CMAP, origin='lower', 
                 vmin=temp_vmin, vmax=temp_vmax)
        # 添加等高线
        ax.contour(pred_field, levels=contour_levels, colors=TEMP_CONTOUR_COLOR,
                  linewidths=TEMP_CONTOUR_LINEWIDTH, alpha=TEMP_CONTOUR_ALPHA, origin='lower')
        ax.axis('off')
        
        # 列4: 误差场
        ax = axes[i, 3]
        norm = mcolors.TwoSlopeNorm(vmin=-error_vmax, vcenter=0, vmax=error_vmax)
        ax.imshow(error_field, cmap=ERROR_CMAP, origin='lower', norm=norm)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    
    print(f"已保存图像: {save_path}")
    
    return temp_vmin, temp_vmax, error_vmax


def main():
    print("=" * 60)
    print("样本对比云图绘制脚本（论文版）")
    print(f"数据集划分: {SPLIT}")
    print(f"训练集: {NTRAIN}, 验证集: {NVALID}")
    print("=" * 60)
    
    # 加载预测数据
    pred_data, true_data = load_prediction_data(EVAL_DIR, SPLIT)
    
    # 计算总样本数（用于索引映射）
    inputs, _ = load_satellite_data(DATA_PATH)
    total_samples = inputs.shape[0]
    print(f"数据集总样本数: {total_samples}")
    
    # 验证数据形状一致
    assert pred_data.shape[0] == true_data.shape[0], \
        f"数据样本数不一致: pred={pred_data.shape[0]}, true={true_data.shape[0]}"
    
    split_samples = pred_data.shape[0]
    print(f"{SPLIT} 集样本数: {split_samples}")
    
    # 选择样本索引
    sample_indices = select_sample_indices(split_samples)
    
    if len(sample_indices) == 0:
        print("错误: 没有有效的样本索引")
        return
    
    # 绘制并保存
    save_path = os.path.join(EVAL_DIR, f'sample_heatmaps_{SPLIT}.png')
    temp_vmin, temp_vmax, error_vmax = plot_sample_heatmaps(true_data, pred_data, sample_indices, 
                                                              SPLIT, NTRAIN, NVALID, total_samples, save_path)
    
    # 保存colorbar图片
    save_colorbar_figures(temp_vmin, temp_vmax, error_vmax, EVAL_DIR, SPLIT)
    
    print("=" * 60)
    print("绘制完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
