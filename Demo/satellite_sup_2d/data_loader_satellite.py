"""
数据加载器 - 卫星热传导数据集（G-U-T格式）

功能：
1. 统一数据加载接口，支持按K桶分组或混合
2. 根据super_train_mode自动加载最大U通道数
3. 支持下采样（256→128）
4. 数据划分：前ntrain_perK→train，后nvalid_perK→valid
"""

import os
import sys
import json
import numpy as np
import torch
from typing import List, Dict, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset

sys.path.append('/data/wqn/Code/DENO4pytorch')

from Utilizes.process_data import DataNormer
from Demo.satellite_sup_2d.utilizes_satellite import load_h5_dataset
from Demo.satellite_sup_2d.ablation_satellite import transform_U_channels


def resolve_satellite_h5_path(base_path: str, component_num: int) -> str:
    """
    解析指定K桶对应的H5文件路径，兼容旧版与_mc新版命名。

    优先级：
    1. heat_dataset_source_{K}_mc.h5
    2. heat_dataset_source_{K}.h5
    """
    candidates = [
        os.path.join(base_path, f"heat_dataset_source_{component_num}_mc.h5"),
        os.path.join(base_path, f"heat_dataset_source_{component_num}.h5"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"未找到K={component_num}对应的数据文件。已尝试: {candidates}"
    )


def parse_satellite_g_structure(
    G: Union[np.ndarray, torch.Tensor],
    meta: Dict,
) -> Dict[str, Union[np.ndarray, torch.Tensor, None]]:
    """统一解析 G 的语义结构，兼容 legacy 与 _mc 数据。"""
    g_channels = int(meta.get('G_channels', G.shape[-1]))
    data_variant = meta.get('data_variant', 'unknown')
    channel_names = meta.get('G_channel_names')

    if isinstance(channel_names, str):
        try:
            channel_names = json.loads(channel_names)
        except json.JSONDecodeError:
            channel_names = None

    if data_variant == 'mc' or (
        g_channels == 5 and channel_names == ['cooling_sdf_0', 'cooling_sdf_1', 'cooling_temp', 'coord_x', 'coord_y']
    ):
        if G.shape[-1] != 5:
            raise ValueError(f"_mc 数据期望 G 有 5 个通道，实际为 {G.shape[-1]}")
        if channel_names is not None and channel_names != ['cooling_sdf_0', 'cooling_sdf_1', 'cooling_temp', 'coord_x', 'coord_y']:
            raise ValueError(f"_mc 数据的 G_channel_names 不符合预期: {channel_names}")
        return {
            'global_G': G[..., 3:5],
            'bc_sdf': G[..., 0:2],
            'cooling_temp': G[..., 2:3],
            'coord': G[..., 3:5],
            'drop_cooling_temp': True,
        }

    return {
        'global_G': G,
        'bc_sdf': None,
        'cooling_temp': None,
        'coord': None,
        'drop_cooling_temp': False,
    }


def get_loaders_satellite_by_K(
    component_nums: List[int],
    train_component_nums: List[int],
    ntrain_perK: int,
    nvalid_perK: int,
    target_U_channels: int,
    empty_channel_value: float,
    batch_size: int,
    base_path: str = "/data/wqn/datasets/dataset_20251218",
    shuffled: bool = True,
    split_by_K: bool = True,
    downsample: int = 2,
) -> Tuple[DataLoader, Dict[int, DataLoader], Dict[str, DataNormer], Dict]:
    """
    统一数据加载接口，支持按K桶分组或混合模式

    Args:
        component_nums: 要加载的K桶列表（如[1,2,...,25]）
        train_component_nums: 用于训练的K桶子集
        ntrain_perK: 每个K桶的train样本数
        nvalid_perK: 每个K桶的valid样本数
        target_U_channels: U的目标通道数（根据super_train_mode确定）
        empty_channel_value: 空通道填充值
        batch_size: DataLoader的batch大小
        base_path: 数据集基础路径
        shuffled: 是否打乱训练数据
        split_by_K: 验证集是否分桶返回（True=分桶，False=混合）
        downsample: 下采样因子（1=256, 2=128, 4=64）

    Returns:
        train_loader: 混合训练集（只包含train_component_nums的train部分）
        valid_loaders: {K: DataLoader} 每桶独立（包含所有component_nums的valid部分）
        normalizers: {'G': g_norm, 'U': u_norm, 'T': t_norm}
        meta: 元信息字典
    """
    print("=" * 60)
    print("加载数据集（按K桶分组）")
    print("=" * 60)
    print(f"训练K桶: {train_component_nums}")
    print(f"验证K桶: {component_nums}")
    print(f"目标U通道数: {target_U_channels}")
    print(f"下采样因子: {downsample}")
    print(f"每K桶train样本: {ntrain_perK}, valid样本: {nvalid_perK}\n")

    # 分别收集train和valid数据
    train_G_list, train_U_list, train_T_list = [], [], []
    valid_data = {}  # {K: (G, U, T)}
    g_channels_detected = None
    data_variant = None
    resolved_files = {}
    g_channel_names = None

    for K in component_nums:
        try:
            h5_path = resolve_satellite_h5_path(base_path, K)
            resolved_files[K] = h5_path
        except FileNotFoundError as e:
            print(f"警告: {e}")
            continue

        try:
            # 1. 加载数据
            dataset = load_h5_dataset(h5_path)
            G = dataset['G']  # (N, H, W, Gc)
            U = dataset['U']  # (N, H, W, k)
            T = dataset['T']  # (N, H, W, 1)
            channels = dataset.get('channels', {})
            attrs = dataset.get('attrs', {})
            current_g_channels = int(channels.get('G', G.shape[-1]))
            current_channel_names = dataset.get('G_channel_names')

            if g_channels_detected is None:
                g_channels_detected = current_g_channels
                g_channel_names = current_channel_names
            elif g_channels_detected != current_g_channels:
                raise ValueError(
                    f"G通道数不一致: 之前为{g_channels_detected}，K={K}检测到{current_g_channels}"
                )

            current_variant = 'mc' if h5_path.endswith('_mc.h5') else 'legacy'
            if data_variant is None:
                data_variant = current_variant
            elif data_variant != current_variant:
                data_variant = 'mixed'

            n_samples = G.shape[0]
            n_channels = U.shape[-1]

            # 2. 下采样
            if downsample > 1:
                G = G[:, ::downsample, ::downsample, :]
                U = U[:, ::downsample, ::downsample, :]
                T = T[:, ::downsample, ::downsample, :]

            # 3. 通道转换
            U = transform_U_channels(U, target_U_channels, empty_channel_value)

            print(f"[K={K:2d}] 样本:{n_samples:4d}, G通道:{current_g_channels:2d}, U通道:{n_channels:2d}→{target_U_channels}, "
                  f"下采样:{G.shape[1]}×{G.shape[2]}, 文件:{os.path.basename(h5_path)}")

            # 4. 划分train/valid（固定位置）
            if K in train_component_nums:
                train_G_list.append(G[:ntrain_perK])
                train_U_list.append(U[:ntrain_perK])
                train_T_list.append(T[:ntrain_perK])

            # 验证集：取最后nvalid_perK个样本
            valid_data[K] = (
                G[-nvalid_perK:],
                U[-nvalid_perK:],
                T[-nvalid_perK:]
            )

        except Exception as e:
            print(f"错误: 无法加载 source_{K} - {e}")
            continue

    if len(train_G_list) == 0:
        raise ValueError("没有成功加载任何训练数据")

    # 5. 合并train数据
    print(f"\n合并训练数据...")
    train_G = np.concatenate(train_G_list, axis=0)
    train_U = np.concatenate(train_U_list, axis=0)
    train_T = np.concatenate(train_T_list, axis=0)

    print(f"训练集总样本数: {train_G.shape[0]}")
    print(f"  - G形状: {train_G.shape}")
    print(f"  - U形状: {train_U.shape}")
    print(f"  - T形状: {train_T.shape}")

    # 6. 打乱train数据
    if shuffled:
        print(f"\n打乱训练数据 (seed=8905)...")
        np.random.seed(8905)
        idx = np.random.permutation(train_G.shape[0])
        train_G = train_G[idx]
        train_U = train_U[idx]
        train_T = train_T[idx]

    # 7. 归一化（基于训练集统计）
    print(f"\n创建归一化器...")
    g_normalizer = DataNormer(train_G, method='mean-std', axis=(0, 1, 2))
    u_normalizer = DataNormer(train_U, method='mean-std', axis=(0, 1, 2))
    t_normalizer = DataNormer(train_T, method='mean-std', axis=(0, 1, 2))

    print(f"  - G归一化器: mean shape={g_normalizer.mean.shape}, std shape={g_normalizer.std.shape}")
    print(f"  - U归一化器: mean shape={u_normalizer.mean.shape}, std shape={u_normalizer.std.shape}")
    print(f"  - T归一化器: mean shape={t_normalizer.mean.shape}, std shape={t_normalizer.std.shape}")

    # 8. 归一化并转换为Tensor
    train_G_norm = g_normalizer.norm(train_G)
    train_U_norm = u_normalizer.norm(train_U)
    train_T_norm = t_normalizer.norm(train_T)

    train_G_tensor = torch.as_tensor(train_G_norm, dtype=torch.float)
    train_U_tensor = torch.as_tensor(train_U_norm, dtype=torch.float)
    train_T_tensor = torch.as_tensor(train_T_norm, dtype=torch.float)

    # 9. 创建训练集DataLoader
    train_dataset = TensorDataset(train_G_tensor, train_U_tensor, train_T_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # 10. 创建验证集DataLoaders（分桶）
    print(f"\n创建验证集DataLoaders...")
    valid_loaders = {}

    for K, (valid_G, valid_U, valid_T) in valid_data.items():
        valid_G_norm = g_normalizer.norm(valid_G)
        valid_U_norm = u_normalizer.norm(valid_U)
        valid_T_norm = t_normalizer.norm(valid_T)

        valid_G_tensor = torch.as_tensor(valid_G_norm, dtype=torch.float)
        valid_U_tensor = torch.as_tensor(valid_U_norm, dtype=torch.float)
        valid_T_tensor = torch.as_tensor(valid_T_norm, dtype=torch.float)

        valid_dataset = TensorDataset(valid_G_tensor, valid_U_tensor, valid_T_tensor)
        valid_loaders[K] = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

    print(f"  - 训练集: {len(train_loader)} batches (batch_size={batch_size})")
    print(f"  - 验证集: {len(valid_loaders)} 个K桶")

    # 11. 构建元信息
    normalizers = {
        'G': g_normalizer,
        'U': u_normalizer,
        'T': t_normalizer,
    }

    meta = {
        'format': 'multi_GUT_by_K',
        'component_nums': component_nums,
        'train_component_nums': train_component_nums,
        'datasets_loaded': len(valid_data),
        'train_samples': train_G.shape[0],
        'valid_samples_perK': nvalid_perK,
        'G_channels': int(g_channels_detected if g_channels_detected is not None else train_G.shape[-1]),
        'U_channels': target_U_channels,
        'T_channels': int(train_T.shape[-1]),
        'empty_channel_value': empty_channel_value,
        'downsample': downsample,
        'grid_size': (train_G.shape[1], train_G.shape[2]),
        'data_variant': data_variant or 'unknown',
        'G_channel_names': g_channel_names,
        'bc_window_count': 2 if (data_variant == 'mc' and g_channel_names is not None) else None,
        'coord_channels': 2 if data_variant == 'mc' else None,
        'drop_cooling_temp': data_variant == 'mc',
        'resolved_files': resolved_files,
    }

    print("\n" + "=" * 60)

    return train_loader, valid_loaders, normalizers, meta

