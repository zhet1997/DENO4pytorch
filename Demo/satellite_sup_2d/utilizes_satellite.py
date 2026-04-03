#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卫星导热数据集加载工具
功能：从h5文件加载数据、划分训练/验证集、每通道独立归一化、创建DataLoader
支持格式：
  - V1格式 (6通道): 合并元件SDF
  - V2格式 (17通道): 分层元件SDF（12个独立通道）
详细格式说明见：DATASET_FORMATS.md
"""
import numpy as np
import h5py
import os
import sys
import torch
from pathlib import Path
from typing import Dict, Any
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Utilizes.process_data import DataNormer


def _detect_format(inputs_shape):
    """
    自动检测数据集格式
    
    Args:
        inputs_shape: 输入数据的shape，例如 (N, 256, 256, C)
    
    Returns:
        format_version: 'v1' 或 'v2'
        num_component_channels: 元件SDF通道数（v1为1，v2为12）
    """
    num_channels = inputs_shape[-1]
    
    if num_channels == 6:
        return 'v1', 1  # V1格式，1个合并的component_sdf
    elif num_channels == 17:
        return 'v2', 12  # V2格式，12个独立component_sdf
    else:
        raise ValueError(
            f"未知的数据格式: 检测到{num_channels}个通道。\n"
            f"支持的格式: V1(6通道) 或 V2(17通道)。\n"
            f"详情请参考 DATASET_FORMATS.md"
        )
        
           
def load_h5_dataset(h5_file: Path) -> Dict[str, Any]:
    """
    加载G-U-T格式的H5数据集
    
    Args:
        h5_file: H5文件路径
    
    Returns:
        Dict[str, Any]: 包含G, U, T数据及元信息的字典
            - 'G': (N, H, W, 4) - 通用全场参数
            - 'U': (N, H, W, k) - 功率调制SDF
            - 'T': (N, H, W, 1) or None - 温度场
            - 'sample_ids': (N,) - 样本ID
            - 'num_samples': int - 样本数
            - 'grid_size': str - 网格尺寸
            - 'channels': dict - 通道数信息
    """
    h5_file = Path(h5_file)
    
    if not h5_file.exists():
        raise FileNotFoundError(f"H5文件不存在: {h5_file}")
    
    dataset = {}
    
    with h5py.File(h5_file, 'r') as h5f:
        # 加载G-U-T数据
        dataset['G'] = h5f['G'][:]
        dataset['U'] = h5f['U'][:]
        dataset['T'] = h5f['T'][:] if 'T' in h5f else None
        if 'sample_ids' in h5f:
            dataset['sample_ids'] = h5f['sample_ids'][:]
        else:
            dataset['sample_ids'] = np.arange(dataset['G'].shape[0], dtype=np.int64)
            logger.warning(f"{h5_file} 缺少 sample_ids，已使用顺序索引代替")

        # 加载元数据
        num_samples = h5f.attrs.get('num_samples')
        if num_samples is None:
            num_samples = h5f.attrs.get('success_samples')
        if num_samples is None:
            num_samples = dataset['G'].shape[0]

        grid_size = h5f.attrs.get('grid_size')
        if grid_size is None:
            grid_size = f"{dataset['G'].shape[1]}x{dataset['G'].shape[2]}"

        dataset['num_samples'] = int(num_samples)
        dataset['grid_size'] = grid_size
        dataset['channels'] = {
            'G': int(h5f.attrs.get('G_channels', dataset['G'].shape[-1])),
            'U': int(h5f.attrs.get('U_channels', dataset['U'].shape[-1])),
            'T': int(h5f.attrs.get('T_channels', dataset['T'].shape[-1] if dataset['T'] is not None else 0))
        }
        dataset['attrs'] = {key: h5f.attrs[key] for key in h5f.attrs.keys()}
        dataset['G_channel_names'] = h5f.attrs.get('G_channel_names', None)

        if 'num_samples' not in h5f.attrs:
            logger.warning(f"{h5_file} 缺少 num_samples，已回退为 {dataset['num_samples']}")
        if 'T_channels' not in h5f.attrs and dataset['T'] is not None:
            logger.warning(f"{h5_file} 缺少 T_channels，已回退为 {dataset['channels']['T']}")
    
    logger.info(f"加载数据集: {h5_file}")
    logger.info(f"  样本数: {dataset['num_samples']}, 网格: {dataset['grid_size']}")
    logger.info(f"  G: {dataset['G'].shape}, U: {dataset['U'].shape}, T: {dataset['T'].shape if dataset['T'] is not None else 'None'}")
    
    return dataset

def get_origin_satellite(
    h5_path=None,
    train_num=None,
    valid_num=None,
    shuffled=False,
    num_component_channels=None,
    dataset_format='auto'
):
    """
    从h5文件加载卫星导热数据集并划分训练/验证集
    支持V1格式(6通道)和V2格式(17通道)的自动检测
    
    Args:
        h5_path: h5文件路径，默认为 /data/wqn/datasets/packaged_dataset20251011/heat_dataset.h5
        train_num: 训练集样本数量，从前面取
        valid_num: 验证集样本数量，从后面取
        shuffled: 是否在划分前打乱数据（使用固定seed=8905）
        num_component_channels: 元件SDF通道数（仅用于验证，None=不验证）
        dataset_format: 数据集格式，'auto'(自动检测)、'v1'(6通道)、'v2'(17通道)
    
    Returns:
        train_inputs: 训练集输入 (train_num, 256, 256, C)，C=6或17
        train_outputs: 训练集输出 (train_num, 256, 256, 1)
        valid_inputs: 验证集输入 (valid_num, 256, 256, C) 或 None
        valid_outputs: 验证集输出 (valid_num, 256, 256, 1) 或 None
        format_version: 检测到的格式版本 'v1' 或 'v2'
        num_components: 元件SDF通道数（v1为1, v2为12）
    """
    # 设置默认h5路径
    if h5_path is None:
        h5_path = "/data/wqn/datasets/packaged_heat_dataset_17c_20251211/heat_dataset.h5"
    
    # 检查文件是否存在
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5文件不存在: {h5_path}")
    
    # 加载数据
    with h5py.File(h5_path, 'r') as f:
        inputs = f['inputs'][:]  # (5000, 256, 256, 6)
        outputs = f['outputs'][:]  # (5000, 256, 256, 1)
        total_samples = inputs.shape[0]
        
        print(f"数据集信息:")
        print(f"  - 总样本数: {total_samples}")
        print(f"  - 输入形状: {inputs.shape}")
        print(f"  - 输出形状: {outputs.shape}")
        if 'input_channels' in f.attrs:
            print(f"  - 输入通道: {f.attrs['input_channels']}")
        if 'output_channels' in f.attrs:
            print(f"  - 输出通道: {f.attrs['output_channels']}")
    
    # 检测或验证数据格式
    detected_format, detected_num_components = _detect_format(inputs.shape)
    
    if dataset_format != 'auto':
        # 如果指定了格式，验证是否匹配
        if dataset_format != detected_format:
            raise ValueError(
                f"格式不匹配: 指定格式为'{dataset_format}'，但检测到'{detected_format}'"
            )
    
    if num_component_channels is not None:
        # 如果指定了元件通道数，验证是否匹配
        if num_component_channels != detected_num_components:
            raise ValueError(
                f"元件通道数不匹配: 指定{num_component_channels}个，但检测到{detected_num_components}个"
            )
    
    # 打印格式信息
    print(f"  - 数据格式: {detected_format.upper()}")
    print(f"  - 元件SDF通道数: {detected_num_components}")
    if detected_format == 'v1':
        print(f"  - 通道布局: component_sdf(合并), component_power, cooling_sdf, cooling_temp, coord_x, coord_y")
    else:  # v2
        print(f"  - 通道布局: cooling_sdf, component_power, cooling_temp, coord_x, coord_y, component_sdf_1~12")
    
    # 设置默认样本数
    if train_num is None:
        train_num = int(total_samples * 0.8)  # 默认80%作为训练集
    if valid_num is None:
        valid_num = total_samples - train_num  # 剩余作为验证集
    
    # 校验样本数
    if train_num + valid_num > total_samples:
        raise ValueError(
            f"样本数超出范围: train_num({train_num}) + valid_num({valid_num}) > total_samples({total_samples})"
        )
    
    # 如果需要打乱数据
    if shuffled:
        np.random.seed(8905)  # 使用固定seed保证可复现
        idx = np.random.permutation(total_samples)
        inputs = inputs[idx]
        outputs = outputs[idx]
        print(f"  - 数据已打乱（seed=8905）")
    
    # 划分数据集
    train_inputs = inputs[:train_num]
    train_outputs = outputs[:train_num]
    valid_inputs = inputs[-valid_num:] if valid_num > 0 else None
    valid_outputs = outputs[-valid_num:] if valid_num > 0 else None
    
    print(f"\n数据划分:")
    print(f"  - 训练集: {train_num} 个样本 (索引 0 到 {train_num-1})")
    if valid_num > 0:
        print(f"  - 验证集: {valid_num} 个样本 (索引 {total_samples-valid_num} 到 {total_samples-1})")
    
    return train_inputs, train_outputs, valid_inputs, valid_outputs, detected_format, detected_num_components


def get_loader_satellite(
    train_x, train_y,
    valid_x=None, valid_y=None,
    x_normalizer=None,
    y_normalizer=None,
    batch_size=32
):
    """
    对数据进行每通道独立归一化并创建DataLoader
    
    Args:
        train_x: 训练集输入 (N, 256, 256, 6)
        train_y: 训练集输出 (N, 256, 256, 1)
        valid_x: 验证集输入 (M, 256, 256, 6)，可选
        valid_y: 验证集输出 (M, 256, 256, 1)，可选
        x_normalizer: 输入归一化器，如果为None则创建
        y_normalizer: 输出归一化器，如果为None则创建
        batch_size: DataLoader的batch大小
    
    Returns:
        train_loader: 训练集DataLoader
        valid_loader: 验证集DataLoader (如果提供了验证集)
        x_normalizer: 输入归一化器
        y_normalizer: 输出归一化器
    """
    # 创建归一化器（每通道独立）
    # axis=(0,1,2) 表示在前3个维度上计算统计量，保留最后一个维度（通道维度）
    if x_normalizer is None:
        x_normalizer = DataNormer(train_x, method='mean-std', axis=(0, 1, 2))
        print(f"\n输入归一化器统计信息 (每通道独立):")
        print(f"  - Mean shape: {x_normalizer.mean.shape}")
        print(f"  - Std shape: {x_normalizer.std.shape}")
        print(f"  - Mean: {x_normalizer.mean}")
        print(f"  - Std: {x_normalizer.std}")
    
    if y_normalizer is None:
        y_normalizer = DataNormer(train_y, method='mean-std', axis=(0, 1, 2))
        print(f"\n输出归一化器统计信息:")
        print(f"  - Mean: {y_normalizer.mean}")
        print(f"  - Std: {y_normalizer.std}")
    
    # 归一化训练集
    train_x_norm = x_normalizer.norm(train_x)
    train_y_norm = y_normalizer.norm(train_y)
    
    # 转换为Tensor
    train_x_tensor = torch.as_tensor(train_x_norm, dtype=torch.float)
    train_y_tensor = torch.as_tensor(train_y_norm, dtype=torch.float)
    
    # 创建训练集DataLoader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # 处理验证集
    valid_loader = None
    if valid_x is not None and valid_y is not None:
        valid_x_norm = x_normalizer.norm(valid_x)
        valid_y_norm = y_normalizer.norm(valid_y)
        
        valid_x_tensor = torch.as_tensor(valid_x_norm, dtype=torch.float)
        valid_y_tensor = torch.as_tensor(valid_y_norm, dtype=torch.float)
        
        valid_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(valid_x_tensor, valid_y_tensor),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        print(f"\nDataLoader创建完成:")
        print(f"  - 训练集: {len(train_loader)} batches (batch_size={batch_size}, shuffle=True)")
        print(f"  - 验证集: {len(valid_loader)} batches (batch_size={batch_size}, shuffle=False)")
    else:
        print(f"\nDataLoader创建完成:")
        print(f"  - 训练集: {len(train_loader)} batches (batch_size={batch_size}, shuffle=True)")
    
    return train_loader, valid_loader, x_normalizer, y_normalizer


if __name__ == "__main__":
    """测试数据加载功能"""
    
    print("=" * 60)
    print("卫星导热数据集加载测试 - 多格式支持")
    print("=" * 60)
    
    # 测试V1格式（6通道）
    print("\n【测试1】V1格式 (6通道) - 默认数据集")
    print("-" * 60)
    train_inputs, train_outputs, valid_inputs, valid_outputs, fmt, nc = get_origin_satellite(
        train_num=4000,
        valid_num=500,
        shuffled=False
    )
    print(f"\n返回格式信息: format={fmt}, num_components={nc}")
    
    # 2. 创建DataLoader
    train_loader, valid_loader, x_norm, y_norm = get_loader_satellite(
        train_inputs, train_outputs,
        valid_inputs, valid_outputs,
        batch_size=32
    )
    
    # 3. 测试数据加载
    print("\n" + "=" * 60)
    print("测试数据加载:")
    print("=" * 60)
    
    for batch_x, batch_y in train_loader:
        print(f"\n训练集第一个batch:")
        print(f"  - 输入形状: {batch_x.shape}")
        print(f"  - 输出形状: {batch_y.shape}")
        print(f"  - 输入值范围: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
        print(f"  - 输出值范围: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
        break
    
    if valid_loader is not None:
        for batch_x, batch_y in valid_loader:
            print(f"\n验证集第一个batch:")
            print(f"  - 输入形状: {batch_x.shape}")
            print(f"  - 输出形状: {batch_y.shape}")
            print(f"  - 输入值范围: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
            print(f"  - 输出值范围: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
            break
    
    # 4. 测试反归一化
    print("\n" + "=" * 60)
    print("测试反归一化:")
    print("=" * 60)
    
    batch_x, batch_y = next(iter(train_loader))
    batch_y_denorm = y_norm.back(batch_y.numpy())
    print(f"  - 归一化后输出范围: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
    print(f"  - 反归一化后输出范围: [{batch_y_denorm.min():.4f}, {batch_y_denorm.max():.4f}]")
    
    # 测试V2格式（17通道）
    print("\n" + "=" * 60)
    print("【测试2】V2格式 (17通道) - 新数据集")
    print("=" * 60)
    
    v2_path = '/data/wqn/datasets/packaged_dataset_test/heat_dataset.h5'
    if os.path.exists(v2_path):
        train_inputs_v2, train_outputs_v2, valid_inputs_v2, valid_outputs_v2, fmt_v2, nc_v2 = get_origin_satellite(
            h5_path=v2_path,
            train_num=80,
            valid_num=20,
            shuffled=False
        )
        print(f"\n返回格式信息: format={fmt_v2}, num_components={nc_v2}")
        
        # 创建DataLoader
        train_loader_v2, valid_loader_v2, x_norm_v2, y_norm_v2 = get_loader_satellite(
            train_inputs_v2, train_outputs_v2,
            valid_inputs_v2, valid_outputs_v2,
            batch_size=16
        )
        
        # 测试数据
        batch_x_v2, batch_y_v2 = next(iter(train_loader_v2))
        print(f"\nV2格式测试batch:")
        print(f"  - 输入形状: {batch_x_v2.shape}")
        print(f"  - 输出形状: {batch_y_v2.shape}")
        print(f"  - 基础通道 (0-4): cooling_sdf, component_power, cooling_temp, coord_x, coord_y")
        print(f"  - 元件通道 (5-16): component_sdf_1 ~ component_sdf_12")
        
        # 展示元件SDF统计
        component_sdfs = batch_x_v2[:, :, :, 5:17]
        print(f"  - 元件SDF通道形状: {component_sdfs.shape}")
        print(f"  - 元件SDF值范围: [{component_sdfs.min():.4f}, {component_sdfs.max():.4f}]")
    else:
        print(f"\nV2格式数据集不存在: {v2_path}")
        print("跳过V2格式测试")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)
    print("\n格式说明:")
    print("  - V1格式 (6通道):  适用于全局优化和快速训练")
    print("  - V2格式 (17通道): 适用于元件级分析和精细控制")
    print("  - 详细格式对比请参考: DATASET_FORMATS.md")
    print("=" * 60)

