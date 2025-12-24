#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
估计空通道填充值的脚本

功能：
1. 遍历所有指定的数据集
2. 加载U通道数据，记录全局最大值和统计信息
3. 输出建议的empty_value（略大于最大值，确保不影响min()操作）
"""
import sys
import os
import numpy as np

sys.path.append('/data/wqn/DENO4pytorch')

from Demo.satellite_sup_2d.utilizes_satellite import load_h5_dataset


def estimate_empty_channel_value(
    component_nums=None,
    base_path="/data/wqn/datasets/dataset_20251218",
    multiplier=1.2
):
    """
    估计空通道填充值
    
    Args:
        component_nums: 要分析的元件数量列表，None=分析所有25个
        base_path: 数据集基础路径
        multiplier: 乘数因子，建议值为全局最大值的倍数
    
    Returns:
        recommended_value: 建议的空通道填充值
        stats: 统计信息字典
    """
    if component_nums is None:
        component_nums = list(range(1, 26))  # 1-25
    
    print("=" * 60)
    print("估计空通道填充值")
    print("=" * 60)
    print(f"分析数据集: source_{min(component_nums)} 到 source_{max(component_nums)}")
    print(f"基础路径: {base_path}")
    print(f"乘数因子: {multiplier}\n")
    
    all_max_values = []
    all_min_values = []
    all_mean_values = []
    dataset_stats = []
    
    for comp_num in component_nums:
        h5_path = f"{base_path}/packaged_heat_dataset_source_{comp_num}/heat_dataset_source_{comp_num}.h5"
        
        if not os.path.exists(h5_path):
            print(f"[跳过] source_{comp_num:2d} - 文件不存在")
            continue
        
        try:
            # 加载数据集
            dataset = load_h5_dataset(h5_path)
            U = dataset['U']  # (N, H, W, k)
            
            # 计算统计量
            u_max = U.max()
            u_min = U.min()
            u_mean = U.mean()
            u_std = U.std()
            
            all_max_values.append(u_max)
            all_min_values.append(u_min)
            all_mean_values.append(u_mean)
            
            dataset_stats.append({
                'component_num': comp_num,
                'samples': U.shape[0],
                'channels': U.shape[-1],
                'max': u_max,
                'min': u_min,
                'mean': u_mean,
                'std': u_std,
            })
            
            print(f"[{comp_num:2d}元件] 样本:{U.shape[0]:4d}, 通道:{U.shape[-1]:2d}, "
                  f"U范围:[{u_min:7.4f}, {u_max:7.4f}], 均值:{u_mean:7.4f}")
        
        except Exception as e:
            print(f"[错误] source_{comp_num:2d} - {e}")
            continue
    
    if len(all_max_values) == 0:
        raise ValueError("没有成功加载任何数据集")
    
    # 计算全局统计量
    global_max = max(all_max_values)
    global_min = min(all_min_values)
    global_mean = np.mean(all_mean_values)
    
    # 建议的空通道值
    recommended_value = global_max * multiplier
    
    # 输出结果
    print("\n" + "=" * 60)
    print("统计结果:")
    print("=" * 60)
    print(f"全局最大值: {global_max:.6f}")
    print(f"全局最小值: {global_min:.6f}")
    print(f"全局平均值: {global_mean:.6f}")
    print(f"\n建议的空通道填充值: {recommended_value:.6f}")
    print(f"  (= 全局最大值 × {multiplier})")
    print("\n说明:")
    print("  - 空通道值应大于所有真实U值，确保不影响min()操作")
    print("  - 该值会被归一化，因此不会直接影响模型训练")
    print("  - 可以在 get_loaders_satellite_multi_GUT() 中使用此值")
    print("=" * 60)
    
    stats = {
        'global_max': global_max,
        'global_min': global_min,
        'global_mean': global_mean,
        'recommended_value': recommended_value,
        'multiplier': multiplier,
        'datasets_analyzed': len(all_max_values),
        'dataset_stats': dataset_stats,
    }
    
    return recommended_value, stats


if __name__ == "__main__":
    """运行估计脚本"""
    
    # 示例1: 分析特定的数据集
    print("\n【示例1】分析常用的数据集 (1-10元件)")
    print("-" * 60)
    
    recommended_value, stats = estimate_empty_channel_value(
        component_nums=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        multiplier=1.2
    )
    
    # 示例2: 分析所有数据集
    print("\n\n【示例2】分析所有数据集 (1-25元件)")
    print("-" * 60)
    
    recommended_value_all, stats_all = estimate_empty_channel_value(
        component_nums=None,  # None表示所有
        multiplier=1.2
    )
    
    print("\n\n建议使用值:")
    print("=" * 60)
    print(f"如果使用1-10元件: empty_channel_value = {recommended_value:.6f}")
    print(f"如果使用所有元件:  empty_channel_value = {recommended_value_all:.6f}")
    print("=" * 60)
