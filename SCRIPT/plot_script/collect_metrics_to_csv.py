#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标收集脚本

功能：
- 遍历指定目录，收集所有 metrics.json 文件
- 提取模型类型、样本数、MSE、MAE、R² 等关键信息
- 输出为 CSV 表格

用法：
    python collect_metrics_to_csv.py --root_dir work_post_eval --output metrics_summary.csv
    python collect_metrics_to_csv.py --root_dir work_post_eval --recursive
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def extract_info_from_path(eval_dir: str) -> Dict[str, Optional[str]]:
    """
    从目录路径提取模型信息
    
    示例路径：
        work_post_eval/work_satellite_dssl_fno_20251021/FNO_DSSL_n4000_20251025_014123
    
    返回：
        {
            'model_type': 'FNO',
            'method': 'DSSL',
            'ntrain': 4000,
            'timestamp': '20251025_014123'
        }
    """
    dirname = os.path.basename(eval_dir.rstrip('/'))
    parts = dirname.split('_')
    
    info = {
        'model_type': None,
        'method': None,
        'ntrain': None,
        'timestamp': None
    }
    
    # 提取模型类型（第一个部分，通常是FNO或MLP）
    if len(parts) > 0:
        info['model_type'] = parts[0].upper()
    
    # 提取训练方法（BASE或DSSL）
    for part in parts:
        if part.upper() in ['BASE', 'DSSL']:
            info['method'] = part.upper()
            break
    
    # 提取样本数（格式：n1000, n4000等）
    for part in parts:
        if part.startswith('n') and part[1:].isdigit():
            info['ntrain'] = int(part[1:])
            break
    
    # 提取时间戳（最后两部分，格式：20251025_014123）
    if len(parts) >= 2:
        if parts[-2].isdigit() and len(parts[-2]) == 8:  # 日期格式
            info['timestamp'] = f"{parts[-2]}_{parts[-1]}"
    
    return info


def load_metrics_json(json_path: str) -> Optional[Dict]:
    """加载 metrics.json 文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[警告] 无法读取 {json_path}: {e}")
        return None


def collect_metrics(root_dir: str, recursive: bool = True) -> List[Dict]:
    """
    收集所有评估指标
    
    Args:
        root_dir: 根目录路径
        recursive: 是否递归搜索子目录
    
    Returns:
        metrics_list: 包含所有指标的字典列表
    """
    metrics_list = []
    root_path = Path(root_dir)
    
    if not root_path.exists():
        print(f"[错误] 目录不存在: {root_dir}")
        return metrics_list
    
    # 搜索所有 metrics.json 文件
    if recursive:
        json_files = list(root_path.rglob('metrics.json'))
    else:
        json_files = list(root_path.glob('*/metrics.json'))
    
    print(f"找到 {len(json_files)} 个 metrics.json 文件")
    
    for json_path in json_files:
        eval_dir = str(json_path.parent)
        
        # 加载指标数据
        metrics = load_metrics_json(str(json_path))
        if metrics is None:
            continue
        
        # 提取路径信息
        path_info = extract_info_from_path(eval_dir)
        
        # 合并信息
        record = {
            'eval_dir': eval_dir,
            'model_type': path_info['model_type'],
            'method': path_info['method'],
            'ntrain': path_info['ntrain'],
            'timestamp': path_info['timestamp'],
            'split': metrics.get('split', 'unknown'),
            'dataset_size': metrics.get('dataset_size', 0),
            'avg_mse': metrics.get('avg_mse', None),
            'avg_mae': metrics.get('avg_mae', None),
            'avg_r2': metrics.get('avg_r2', None),
            'height': metrics.get('height', None),
            'width': metrics.get('width', None),
        }
        
        metrics_list.append(record)
        print(f"  [✓] {path_info['model_type']}_{path_info['method']}_n{path_info['ntrain']}")
    
    return metrics_list


def save_to_csv(metrics_list: List[Dict], output_path: str):
    """保存为CSV文件"""
    if len(metrics_list) == 0:
        print("[警告] 没有收集到任何指标数据")
        return
    
    df = pd.DataFrame(metrics_list)
    
    # 按模型类型、方法、样本数排序
    sort_cols = ['model_type', 'method', 'ntrain']
    existing_cols = [c for c in sort_cols if c in df.columns]
    if existing_cols:
        df = df.sort_values(by=existing_cols)
    
    # 保存CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n[成功] 已保存到: {output_path}")
    print(f"  共 {len(df)} 条记录")
    
    # 打印统计信息
    print("\n数据统计：")
    if 'model_type' in df.columns:
        print(f"  模型类型: {df['model_type'].unique().tolist()}")
    if 'method' in df.columns:
        print(f"  训练方法: {df['method'].unique().tolist()}")
    if 'ntrain' in df.columns:
        print(f"  样本数范围: {df['ntrain'].min()} ~ {df['ntrain'].max()}")


def main():
    parser = argparse.ArgumentParser(description='收集评估指标并生成CSV表格')
    parser.add_argument('--root_dir', type=str, default='work_post_eval',
                        help='根目录路径，默认 work_post_eval')
    parser.add_argument('--output', type=str, default='metrics_summary.csv',
                        help='输出CSV文件路径，默认 metrics_summary.csv')
    parser.add_argument('--recursive', action='store_true',
                        help='递归搜索所有子目录（默认：仅搜索一级子目录）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("评估指标收集脚本")
    print(f"根目录: {args.root_dir}")
    print(f"递归模式: {'是' if args.recursive else '否'}")
    print("=" * 60)
    
    # 收集指标
    metrics_list = collect_metrics(args.root_dir, args.recursive)
    
    # 保存为CSV
    if metrics_list:
        save_to_csv(metrics_list, args.output)
    else:
        print("\n[失败] 未收集到任何有效数据")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == '__main__':
    main()


