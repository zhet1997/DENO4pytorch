import os
import sys
import yaml
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any

# 确保项目根目录在路径中，便于以 `Demo.*` 导入
sys.path.append('/data/wqn/DENO4pytorch')

from Demo.satellite_sup_2d.utilizes_satellite import (
    get_origin_satellite,
    get_loader_satellite,
    load_h5_dataset,
)

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from Utilizes.process_data import DataNormer

def get_setting_satellite() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    定义卫星数据(17通道)的默认训练/模型配置。
    返回: basic_dict, train_dict, pred_model_dict, super_model_dict
    """
    basic_dict = {
        'in_dim': 8,   # 固定V2格式(17通道)
        'out_dim': 1,
        'ntrain': 400,
        'nvalid': 100,
    }

    train_dict = {
        'batch_size': 4,  # 减小batch size以适应内存限制
        'epochs': 200,
        'learning_rate': 1e-3,
        'scheduler_step': 100,
        'scheduler_gamma': 0.5,
    }

    # 读取与PakB相同的Transformer配置，设置 node_feats=in_dim
    # with open(os.path.join('data', 'configs', 'transformer_config_sate.yml')) as f:
    #     config = yaml.full_load(f)
    #     pred_model_dict = config['PakB_2d']
    #     pred_model_dict['node_feats'] = basic_dict['in_dim']
        
    
    with open(os.path.join('configs', 'dualhead_transformer_config_sate.yml')) as f:
        config = yaml.full_load(f)
        pred_model_dict = config['DualHead_GUT_2d']
        pred_model_dict['U_dim'] = basic_dict['in_dim']


    super_model_dict = {
        'modes': (16, 16),
        'width': 64,
        'depth': 2,
        'steps': 1,
        'padding': 0,
        'dropout': 0.1,
    }

    return basic_dict, train_dict, pred_model_dict, super_model_dict


def get_loaders_satellite(
    train_num: Optional[int] = None,
    valid_num: Optional[int] = None,
    batch_size: int = 16,
    h5_path: Optional[str] = None,
    shuffled: bool = False,
) -> Tuple[Any, Optional[Any], Any, Any, Dict[str, Any]]:
    """
    构建卫星数据(固定V2/17通道)的训练与验证 DataLoader。
    返回: train_loader, valid_loader, x_normalizer, y_normalizer, meta
    meta: { 'format': 'v2', 'num_components': 12, 'in_dim': 17 }
    """
    train_x, train_y, valid_x, valid_y, fmt, num_components = get_origin_satellite(
        h5_path=h5_path,
        train_num=train_num,
        valid_num=valid_num,
        shuffled=shuffled,
        dataset_format='v2',  # 固定为V2格式
    )

    # 构建 DataLoader（每通道独立归一化）
    train_loader, valid_loader, x_norm, y_norm = get_loader_satellite(
        train_x, train_y,
        valid_x, valid_y,
        batch_size=batch_size,
    )

    meta = {
        'format': fmt,
        'num_components': num_components,
        'in_dim': train_x.shape[-1],
    }
    return train_loader, valid_loader, x_norm, y_norm, meta
def get_loaders_satellite_GUT(
    train_num: Optional[int] = None,
    valid_num: Optional[int] = None,
    batch_size: int = 16,
    h5_path: Optional[str] = None,
    shuffled: bool = False,
) -> Tuple[Any, Optional[Any], Any, Any, Dict[str, Any]]:
    """
    构建卫星数据的训练与验证 DataLoader (使用G-U-T格式)。
    
    数据处理流程:
    1. 加载G-U-T格式数据
    2. 将U的所有通道用min()合并为1个通道
    3. 合并G和U_min作为输入 (5通道: G[4] + U_min[1])
    4. T作为输出 (1通道)
    
    Args:
        train_num: 训练集样本数量，从前面取，None=使用80%
        valid_num: 验证集样本数量，从后面取，None=使用剩余样本
        batch_size: DataLoader的batch大小
        h5_path: h5文件路径，None=使用默认路径
        shuffled: 是否在划分前打乱数据（使用固定seed=8905）
    
    Returns:
        train_loader: 训练集DataLoader
        valid_loader: 验证集DataLoader (如果valid_num>0)
        x_normalizer: 输入归一化器
        y_normalizer: 输出归一化器
        meta: 元信息字典 {'in_dim': 5, 'out_dim': 1, 'format': 'GUT', ...}
    """
    # 设置默认h5路径
    if h5_path is None:
        h5_path = "/data/wqn/datasets/SDNO_test/15c_data_test.h5"
    
    # 1. 加载G-U-T格式数据
    dataset = load_h5_dataset(h5_path)
    
    G = dataset['G']  # (N, H, W, 4)
    U = dataset['U']  # (N, H, W, k) k=15
    T = dataset['T']  # (N, H, W, 1)
    
    total_samples = G.shape[0]
    
    print(f"\nG-U-T数据加载:")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - G形状: {G.shape}")
    print(f"  - U形状: {U.shape}")
    print(f"  - T形状: {T.shape}")
    
    # 2. 将U的所有通道合并为1个通道 (取最小值)
    U_min = U.min(axis=-1, keepdims=True)  # (N, H, W, 1)
    print(f"\n通道处理:")
    print(f"  - U_min形状: {U_min.shape}")
    print(f"  - U_min范围: [{U_min.min():.4f}, {U_min.max():.4f}]")
    
    # 3. 合并G和U_min作为输入
    inputs = np.concatenate([G, U_min], axis=-1)  # (N, H, W, 5)
    outputs = T  # (N, H, W, 1)
    
    print(f"\n合并后数据:")
    print(f"  - 输入形状: {inputs.shape} (G[4] + U_min[1])")
    print(f"  - 输出形状: {outputs.shape}")
    
    # 4. 设置默认样本数
    if train_num is None:
        train_num = int(total_samples * 0.8)  # 默认80%作为训练集
    if valid_num is None:
        valid_num = total_samples - train_num  # 剩余作为验证集
    
    # 校验样本数
    if train_num + valid_num > total_samples:
        raise ValueError(
            f"样本数超出范围: train_num({train_num}) + valid_num({valid_num}) > total_samples({total_samples})"
        )
    
    # 5. 如果需要打乱数据
    if shuffled:
        np.random.seed(8905)  # 使用固定seed保证可复现
        idx = np.random.permutation(total_samples)
        inputs = inputs[idx]
        outputs = outputs[idx]
        print(f"\n  - 数据已打乱（seed=8905）")
    
    # 6. 划分数据集
    train_x = inputs[:train_num]
    train_y = outputs[:train_num]
    valid_x = inputs[-valid_num:] if valid_num > 0 else None
    valid_y = outputs[-valid_num:] if valid_num > 0 else None
    
    print(f"\n数据划分:")
    print(f"  - 训练集: {train_num} 个样本 (索引 0 到 {train_num-1})")
    if valid_num > 0:
        print(f"  - 验证集: {valid_num} 个样本 (索引 {total_samples-valid_num} 到 {total_samples-1})")
    
    # 7. 构建 DataLoader（每通道独立归一化）
    train_loader, valid_loader, x_norm, y_norm = get_loader_satellite(
        train_x, train_y,
        valid_x, valid_y,
        batch_size=batch_size,
    )
    
    # 8. 构建元信息
    meta = {
        'format': 'GUT',
        'in_dim': 5,  # G(4) + U_min(1)
        'out_dim': 1,
        'G_channels': 4,
        'U_channels_original': U.shape[-1],
        'U_channels_processed': 1,
        'T_channels': 1,
        'num_samples': total_samples,
        'train_num': train_num,
        'valid_num': valid_num,
    }
    
    return train_loader, valid_loader, x_norm, y_norm, meta


def transform_U_channels(
    U: np.ndarray,
    target_channels: int,
    empty_channel_value: float = 1.0
) -> np.ndarray:
    """
    将U通道从n个转换为K个通道。
    
    转换规则:
    - n == K: 直接返回，无需转换
    - n > K: 将n个通道分为K组，每组内取min()
    - n < K: 保留前n个通道，补充(K-n)个空通道（填充empty_channel_value）
    
    Args:
        U: 输入U数据，形状为 (N, H, W, n)
        target_channels: 目标通道数 K
        empty_channel_value: 空通道填充值，应略大于所有U数据的最大值
    
    Returns:
        U_transformed: 转换后的U数据，形状为 (N, H, W, K)
    """
    N, H, W, n = U.shape
    
    if n == target_channels:
        # 情况1: 通道数相同，直接返回
        return U
    
    elif n > target_channels:
        # 情况2: n > K，需要合并通道
        # 将n个通道分为K组，每组内取min
        U_new = np.zeros((N, H, W, target_channels), dtype=U.dtype)
        groups = np.array_split(range(n), target_channels)
        
        for k, group in enumerate(groups):
            if len(group) == 1:
                # 单个通道，直接复制
                U_new[:, :, :, k] = U[:, :, :, group[0]]
            else:
                # 多个通道，取min
                U_new[:, :, :, k] = U[:, :, :, group].min(axis=-1)
        
        return U_new
    
    else:
        # 情况3: n < K，需要补充空通道
        U_new = np.full((N, H, W, target_channels), empty_channel_value, dtype=U.dtype)
        U_new[:, :, :, :n] = U  # 前n个通道保留原值
        
        # 打乱通道顺序
        channel_indices = np.arange(target_channels)
        np.random.shuffle(channel_indices)
        U_new = U_new[:, :, :, channel_indices]
        
        return U_new


def get_loaders_satellite_multi_GUT(
    component_nums,
    target_U_channels=15,
    empty_channel_value=1.0,
    train_num=None,
    valid_num=None,
    batch_size=16,
    base_path="/data/wqn/datasets/dataset_20251218",
    shuffled=False,
):
    """
    构建多数据集的训练与验证 DataLoader (使用G-U-T格式)。
    
    数据处理流程:
    1. 加载多个不同元件数量的数据集
    2. 对每个数据集的U通道进行n→K转换
    3. 合并所有数据集的样本
    4. 可选地混合打乱不同数据集的样本
    5. 划分训练/验证集
    6. G、U、T分别独立归一化
    7. 返回DataLoader，每次迭代返回 (batch_G, batch_U, batch_T)
    
    Args:
        component_nums: 要加载的元件数量列表，例如 [1, 2, 3, 4, 5, 10]
        target_U_channels: 目标U通道数 K
        empty_channel_value: 空通道填充值（建议使用estimate_empty_channel_value.py估计）
        train_num: 训练集样本数量，从前面取，None=使用80%
        valid_num: 验证集样本数量，从后面取，None=使用剩余样本
        batch_size: DataLoader的batch大小
        base_path: 数据集基础路径
        shuffled: 是否混合打乱不同数据集的样本（使用固定seed=8905）
    
    Returns:
        train_loader: 训练集DataLoader，迭代返回 (batch_G, batch_U, batch_T)
        valid_loader: 验证集DataLoader (如果valid_num>0)
        normalizers: 归一化器字典 {'G': g_norm, 'U': u_norm, 'T': t_norm}
        meta: 元信息字典
    """
    print("=" * 60)
    print("加载多数据集 (G-U-T格式)")
    print("=" * 60)
    print(f"元件数量列表: {component_nums}")
    print(f"目标U通道数: {target_U_channels}")
    print(f"空通道填充值: {empty_channel_value}")
    print(f"基础路径: {base_path}\n")
    
    # 1. 加载多个数据集
    all_G, all_U, all_T = [], [], []
    dataset_info = []
    
    for comp_num in component_nums:
        h5_path = f"{base_path}/packaged_heat_dataset_source_{comp_num}/heat_dataset_source_{comp_num}.h5"
        
        if not os.path.exists(h5_path):
            print(f"警告: 文件不存在，跳过 - source_{comp_num}")
            continue
        
        try:
            # 加载数据集
            dataset = load_h5_dataset(h5_path)
            G = dataset['G']  # (N, H, W, 4)
            U = dataset['U']  # (N, H, W, k)
            T = dataset['T']  # (N, H, W, 1)
            
            n_samples = G.shape[0]
            n_channels = U.shape[-1]
            
            print(f"[{comp_num:2d}元件] 加载样本:{n_samples:4d}, U通道:{n_channels:2d} → {target_U_channels}")
            
            # 2. 转换U通道
            U_transformed = transform_U_channels(U, target_U_channels, empty_channel_value)
            
            all_G.append(G)
            all_U.append(U_transformed)
            all_T.append(T)
            
            dataset_info.append({
                'component_num': comp_num,
                'samples': n_samples,
                'original_U_channels': n_channels,
            })
        
        except Exception as e:
            print(f"错误: 无法加载 source_{comp_num} - {e}")
            continue
    
    if len(all_G) == 0:
        raise ValueError("没有成功加载任何数据集")
    
    # 3. 合并数据
    print(f"\n合并 {len(all_G)} 个数据集...")
    G_combined = np.concatenate(all_G, axis=0)  # (N_total, H, W, 4)
    U_combined = np.concatenate(all_U, axis=0)  # (N_total, H, W, K)
    T_combined = np.concatenate(all_T, axis=0)  # (N_total, H, W, 1)
    
    total_samples = G_combined.shape[0]
    
    print(f"合并后总样本数: {total_samples}")
    print(f"  - G形状: {G_combined.shape}")
    print(f"  - U形状: {U_combined.shape}")
    print(f"  - T形状: {T_combined.shape}")
    
    # 4. 可选打乱
    if shuffled:
        print(f"\n混合打乱所有样本 (seed=8905)...")
        np.random.seed(8905)
        idx = np.random.permutation(total_samples)
        G_combined = G_combined[idx]
        U_combined = U_combined[idx]
        T_combined = T_combined[idx]
    
    # 5. 设置默认样本数
    if train_num is None:
        train_num = int(total_samples * 0.8)  # 默认80%作为训练集
    if valid_num is None:
        valid_num = total_samples - train_num  # 剩余作为验证集
    
    # 校验样本数
    if train_num + valid_num > total_samples:
        raise ValueError(
            f"样本数超出范围: train_num({train_num}) + valid_num({valid_num}) > total_samples({total_samples})"
        )
    
    # 6. 划分数据集
    train_G = G_combined[:train_num]
    train_U = U_combined[:train_num]
    train_T = T_combined[:train_num]
    
    valid_G = G_combined[-valid_num:] if valid_num > 0 else None
    valid_U = U_combined[-valid_num:] if valid_num > 0 else None
    valid_T = T_combined[-valid_num:] if valid_num > 0 else None
    
    print(f"\n数据划分:")
    print(f"  - 训练集: {train_num} 个样本")
    if valid_num > 0:
        print(f"  - 验证集: {valid_num} 个样本")
    
    # 7. 分别归一化G、U、T (每通道独立)
    print(f"\n创建归一化器 (G、U、T分别独立)...")
    g_normalizer = DataNormer(train_G, method='mean-std', axis=(0, 1, 2))
    u_normalizer = DataNormer(train_U, method='mean-std', axis=(0, 1, 2))
    t_normalizer = DataNormer(train_T, method='mean-std', axis=(0, 1, 2))
    
    print(f"  - G归一化器: mean shape={g_normalizer.mean.shape}, std shape={g_normalizer.std.shape}")
    print(f"  - U归一化器: mean shape={u_normalizer.mean.shape}, std shape={u_normalizer.std.shape}")
    print(f"  - T归一化器: mean shape={t_normalizer.mean.shape}, std shape={t_normalizer.std.shape}")
    
    # 归一化训练集
    train_G_norm = g_normalizer.norm(train_G)
    train_U_norm = u_normalizer.norm(train_U)
    train_T_norm = t_normalizer.norm(train_T)
    
    # 转换为Tensor
    train_G_tensor = torch.as_tensor(train_G_norm, dtype=torch.float)
    train_U_tensor = torch.as_tensor(train_U_norm, dtype=torch.float)
    train_T_tensor = torch.as_tensor(train_T_norm, dtype=torch.float)
    
    # 8. 创建训练集DataLoader
    train_dataset = torch.utils.data.TensorDataset(
        train_G_tensor, train_U_tensor, train_T_tensor
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # 处理验证集
    valid_loader = None
    if valid_G is not None and valid_num > 0:
        valid_G_norm = g_normalizer.norm(valid_G)
        valid_U_norm = u_normalizer.norm(valid_U)
        valid_T_norm = t_normalizer.norm(valid_T)
        
        valid_G_tensor = torch.as_tensor(valid_G_norm, dtype=torch.float)
        valid_U_tensor = torch.as_tensor(valid_U_norm, dtype=torch.float)
        valid_T_tensor = torch.as_tensor(valid_T_norm, dtype=torch.float)
        
        valid_dataset = torch.utils.data.TensorDataset(
            valid_G_tensor, valid_U_tensor, valid_T_tensor
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
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
    
    # 9. 构建元信息
    normalizers = {
        'G': g_normalizer,
        'U': u_normalizer,
        'T': t_normalizer,
    }
    
    meta = {
        'format': 'multi_GUT',
        'component_nums': component_nums,
        'datasets_loaded': len(all_G),
        'dataset_info': dataset_info,
        'total_samples': total_samples,
        'train_num': train_num,
        'valid_num': valid_num,
        'G_channels': 4,
        'U_channels': target_U_channels,
        'T_channels': 1,
        'empty_channel_value': empty_channel_value,
        'shuffled': shuffled,
    }
    
    print("\n" + "=" * 60)
    
    return train_loader, valid_loader, normalizers, meta


