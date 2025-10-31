#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
卫星元件论文布局图（极简，无依赖项目其他模块）
输入：sample_definition.json
输出：同目录 layout_new.png
所有作图参数在顶部集中设置并注释！
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib import cm

# ====== 【全局参数集中设置，调整样式直接修改此处】 ======
FIGSIZE = (7, 7)        # 图片尺寸（英寸），建议正方
DPI = 500               # 分辨率
COMP_LINEWIDTH = 1.6    # 元件外轮廓线宽
COMP_ALPHA = 0.5        # 元件透明度（所有元件形状皆适用）
CAPSULE_RES = 160       # 胶囊外轮廓插值点数，建议大于100平滑
POWER_CMAP = cm.get_cmap('YlOrRd')  # 颜色映射类型：暖色系（黄→橙→红），功率高=暖色
POWER_MIN = 0        # 若None则自取min
POWER_MAX = 1000        # None则自取max
LABEL_FONT_SIZE = 10     # 元件内功率标注字号
LABEL_FONT_COLOR = 'black'
LABEL_FONT_WEIGHT = 'bold'
COOLING_LINE_WIDTH = 7  # 冷却线宽
COOLING_LINE_COLOR = 'darkred'  # 冷却线颜色：暗红色
COOLING_CAP_SIZE = 0.015  # 冷却线端点标记小线条长度（相对布局尺寸）
BORDER_LINE_WIDTH = 2.2 # 外部包络线宽
DIM_TEXT_SIZE = 11      # 底部尺寸标注字号
DIM_TEXT_COLOR = 'black'
# ========================================================

# ---- JSON 读取 ----
def load_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    layout = data['layout_domain']
    if isinstance(layout, dict):
        width, height = layout['width'], layout['height']
    else:
        width, height = layout[0], layout[1]
    return width, height, data['components'], data.get('cooling_lines', [])

# ---- 胶囊外轮廓（基于原始visualization.py正确几何，改为单一Polygon避免内部线）----
def plot_capsule_patch(ax, cx, cy, length, width, angle=0., facecolor='w', edgecolor='k', lw=1.5, alpha=1):
    """
    绘制胶囊形状的单一多边形外轮廓
    采用原始正确几何：rect_length = length - width, radius = width/2
    """
    from matplotlib.patches import Polygon
    
    radius = width / 2
    rect_length = length - width  # 中间矩形部分长度
    
    # 构造外轮廓点（逆时针，横向orientation）
    # 左半圆圆心: (cx - rect_length/2, cy)
    # 右半圆圆心: (cx + rect_length/2, cy)
    
    # 左半圆: 从上方(π/2)经过左侧(π)到下方(3π/2) - 面向左的半圆
    theta_left = np.linspace(np.pi/2, 3*np.pi/2, CAPSULE_RES//2)
    x_left = (cx - rect_length/2) + radius * np.cos(theta_left)
    y_left = cy + radius * np.sin(theta_left)
    
    # 右半圆: 从下方(-π/2)经过右侧(0)到上方(π/2) - 面向右的半圆
    theta_right = np.linspace(-np.pi/2, np.pi/2, CAPSULE_RES//2)
    x_right = (cx + rect_length/2) + radius * np.cos(theta_right)
    y_right = cy + radius * np.sin(theta_right)
    
    # 合并: 左半圆(上→下) + 右半圆(下→上) = 封闭轮廓
    x = np.concatenate([x_left, x_right])
    y = np.concatenate([y_left, y_right])
    
    # 旋转处理（如果是纵向rotation=90）
    if angle != 0:
        A = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                      [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]])
        pts = np.stack([x-cx, y-cy], axis=0)
        rot_pts = A @ pts
        x = rot_pts[0,:] + cx
        y = rot_pts[1,:] + cy
    
    # 绘制单一多边形
    patch = Polygon(np.stack([x,y],axis=1), closed=True, 
                   facecolor=facecolor, edgecolor=edgecolor, 
                   lw=lw, alpha=alpha, zorder=2)
    ax.add_patch(patch)

# ---- 主要绘图逻辑 ----
def plot_layout(json_path):
    # 读取数据
    W, H, components, cooling_lines = load_json(json_path)
    # 元件power范围, 用于colormap
    powers = [c['power'] for c in components if 'power' in c]
    pmin, pmax = (min(powers), max(powers))
    if POWER_MIN is not None: pmin = POWER_MIN
    if POWER_MAX is not None: pmax = POWER_MAX
    norm = plt.Normalize(vmin=pmin, vmax=pmax)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # ---- 画元件 ----
    for comp in components:
        cx, cy = comp['center']
        shape = comp['shape']
        pow_val = comp['power']
        color = POWER_CMAP(norm(pow_val))
        # 统一透明度设置
        if shape == 'rect':
            w, h = comp['width'], comp['height']
            ax.add_patch(Rectangle((cx-w/2, cy-h/2), w, h,
                                   facecolor=color, edgecolor='black',
                                   lw=COMP_LINEWIDTH, zorder=2, alpha=COMP_ALPHA))
        elif shape == 'circle':
            r = comp['radius']
            ax.add_patch(Circle((cx, cy), r,
                                facecolor=color, edgecolor='black',
                                lw=COMP_LINEWIDTH, zorder=2, alpha=COMP_ALPHA))
        elif shape == 'capsule':
            l, w = comp['length'], comp['width']
            rot = comp.get('rotation', 0)
            plot_capsule_patch(ax, cx, cy, l, w, angle=rot,
                              facecolor=color, edgecolor='black',
                              lw=COMP_LINEWIDTH, alpha=COMP_ALPHA)
        # 加总功率标注
        total_p = comp.get('total_power', 0)
        ax.text(cx, cy, f"{total_p:.1f}W", fontsize=LABEL_FONT_SIZE,
                color=LABEL_FONT_COLOR, weight=LABEL_FONT_WEIGHT, ha='center', va='center', zorder=3)
    # ---- 画冷却线（暗红色粗线，端点加小线条标记）----
    for cl in cooling_lines:
        ep = cl['endpoints']
        if len(ep)!=2: continue
        (x1, y1), (x2, y2) = ep
        # 主线段
        ax.plot([x1, x2], [y1, y2], color=COOLING_LINE_COLOR, lw=COOLING_LINE_WIDTH,
                solid_capstyle='round', zorder=4)
        # 端点标记小线条
        cap_len = COOLING_CAP_SIZE * max(W, H)
        # 判断线段方向，画垂直于线段的小线条
        if abs(x2 - x1) > abs(y2 - y1):  # 横向线
            ax.plot([x1, x1], [y1 - cap_len, y1 + cap_len], color=COOLING_LINE_COLOR, lw=COOLING_LINE_WIDTH*0.8, zorder=4)
            ax.plot([x2, x2], [y2 - cap_len, y2 + cap_len], color=COOLING_LINE_COLOR, lw=COOLING_LINE_WIDTH*0.8, zorder=4)
        else:  # 纵向线
            ax.plot([x1 - cap_len, x1 + cap_len], [y1, y1], color=COOLING_LINE_COLOR, lw=COOLING_LINE_WIDTH*0.8, zorder=4)
            ax.plot([x2 - cap_len, x2 + cap_len], [y2, y2], color=COOLING_LINE_COLOR, lw=COOLING_LINE_WIDTH*0.8, zorder=4)
    # ---- 画外包络 ----
    ax.add_patch(Rectangle((0,0), W, H, fill=False, edgecolor='black', lw=BORDER_LINE_WIDTH, zorder=1))
    
    # ---- 底部尺寸标注 ----
    dim_y = -0.01*H
    ax.text(W/2, dim_y, f"{W:.3f} m × {H:.3f} m", 
            color=DIM_TEXT_COLOR, ha='center', va='top', 
            fontsize=DIM_TEXT_SIZE, weight='normal')
    
    # ---- 画布美化 ----
    ax.set_xlim(-0.05*W, W*1.05)
    ax.set_ylim(-0.09*H, H*1.06)
    ax.axis('off')          # 去除所有坐标/边框
    # ---- 保存 ----
    out_path = Path(json_path).parent / 'layout_new.png'
    fig.savefig(str(out_path), dpi=DPI, bbox_inches='tight', pad_inches=0.12)
    plt.close(fig)
    print(f"✓ 论文级布局图已保存至：{out_path}")

# ---- 主入口 ----
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('使用方法: python plot_satellite_layout.py /path/sample_definition.json')
        sys.exit(1)
    plot_layout(sys.argv[1])

