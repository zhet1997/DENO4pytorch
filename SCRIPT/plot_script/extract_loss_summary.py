#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取loss_history.npy的统计信息并保存为JSON

用法:
    python extract_loss_summary.py --train_dir path/to/train --eval_dir path/to/eval
"""

import os
import sys
import argparse
import json
import numpy as np

# 路径注入，支持绝对路径运行
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def load_loss_history(file_path):
    """加载loss_history.npy文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    data = np.load(file_path, allow_pickle=True)
    data = data.item()
    train = np.asarray(data['train'], dtype=float)
    valid = np.asarray(data['valid'], dtype=float)
    train_self = np.asarray(data['train_self'], dtype=float)
    return {'train': train, 'valid': valid, 'train_self': train_self}


def extract_loss_summary(loss_data):
    """
    提取loss的最终值和最小值
    
    Args:
        loss_data: dict，包含'train', 'valid', 'train_self'三个numpy数组
    
    Returns:
        dict，包含6个值
    """
    summary = {}
    
    for loss_type in ['train', 'valid', 'train_self']:
        values = loss_data[loss_type]
        
        if len(values) == 0:
            raise ValueError(f"{loss_type} 数组为空")
        
        summary[f"{loss_type}_final"] = float(values[-1])
        summary[f"{loss_type}_min"] = float(np.min(values))
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='提取loss统计信息并保存为JSON')
    parser.add_argument('--train_dir', required=True, help='训练文件夹路径（包含loss_history.npy）')
    parser.add_argument('--eval_dir', required=True, help='eval输出文件夹路径')
    
    args = parser.parse_args()
    
    # 检查训练文件夹和loss_history.npy
    loss_file = os.path.join(args.train_dir, 'loss_history.npy')
    if not os.path.exists(loss_file):
        print(f"错误: 文件不存在 {loss_file}")
        return 1
    
    # 检查eval目录，不存在则创建
    if not os.path.exists(args.eval_dir):
        print(f"[创建] eval目录不存在，正在创建: {args.eval_dir}")
        os.makedirs(args.eval_dir, exist_ok=True)
    
    try:
        # 加载loss数据
        print(f"加载loss数据: {loss_file}")
        loss_data = load_loss_history(loss_file)
        
        # 提取统计信息
        summary = extract_loss_summary(loss_data)
        
        # 保存为JSON
        output_file = os.path.join(args.eval_dir, 'loss_summary.json')
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[成功] loss统计信息已保存到: {output_file}")
        print(f"  train_final: {summary['train_final']:.6e}, train_min: {summary['train_min']:.6e}")
        print(f"  valid_final: {summary['valid_final']:.6e}, valid_min: {summary['valid_min']:.6e}")
        print(f"  train_self_final: {summary['train_self_final']:.6e}, train_self_min: {summary['train_self_min']:.6e}")
        
        return 0
        
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())

