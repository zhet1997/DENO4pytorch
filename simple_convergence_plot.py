#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最简单的收敛图绘制脚本
不依赖外部库，直接使用matplotlib

用法:
    python simple_convergence_plot.py
"""

import os
import matplotlib.pyplot as plt

def find_loss_files():
    """查找所有loss_history.npy文件"""
    base_dirs = [
        'work_satellite_dssl_mlp_20251021',
        'work_satellite_dssl_fno_20251021'
    ]
    
    loss_files = []
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            for root, dirs, files in os.walk(base_dir):
                if 'loss_history.npy' in files:
                    loss_files.append(os.path.join(root, 'loss_history.npy'))
    
    return loss_files

def plot_simple_convergence():
    """绘制简单的收敛图"""
    # 查找文件
    loss_files = find_loss_files()
    
    if not loss_files:
        print("未找到任何loss_history.npy文件")
        return
    
    print(f"找到 {len(loss_files)} 个损失文件:")
    for i, file_path in enumerate(loss_files):
        print(f"  {i+1}. {file_path}")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # 绘制每个实验
    for i, file_path in enumerate(loss_files[:2]):  # 最多绘制前两个
        print(f"\n处理文件 {i+1}: {file_path}")
        
        try:
            # 尝试加载数据
            import numpy as np
            data = np.load(file_path, allow_pickle=True)
            
            if isinstance(data, dict):
                for j, (loss_type, values) in enumerate(data.items()):
                    if len(values) > 0:
                        epochs = list(range(len(values)))
                        ax.plot(epochs, values, 
                               color=colors[i], 
                               linestyle='-' if i == 0 else '--',
                               linewidth=2,
                               label=f"实验{i+1}_{loss_type}")
                        print(f"  绘制 {loss_type}: {len(values)} 个数据点")
            else:
                # 如果是数组格式
                epochs = list(range(len(data)))
                ax.plot(epochs, data, 
                       color=colors[i], 
                       linestyle='-' if i == 0 else '--',
                       linewidth=2,
                       label=f"实验{i+1}_loss")
                print(f"  绘制 loss: {len(data)} 个数据点")
                
        except Exception as e:
            print(f"  加载文件失败: {e}")
            continue
    
    # 设置图形属性
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Convergence Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 保存图片
    output_path = 'convergence_simple.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n收敛图已保存到: {output_path}")
    
    # 显示图片
    plt.show()

if __name__ == '__main__':
    plot_simple_convergence()
