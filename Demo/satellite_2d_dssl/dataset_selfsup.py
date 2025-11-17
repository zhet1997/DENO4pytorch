import os
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from Demo.satellite_2d_base.utils import load_yaml_config
from Utilizes.process_data import DataNormer

def load_selfsup_data(data_dir: str,
                      sample_limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    从目录中加载自监督数据：inputs(H5) 与 alphas(NPY)。

    约定：
    - H5 仅包含 inputs，形状 (N,256,256,6)
    - NPY 形状 (N, K, 4)，K 为每样本可用的不同相似系数个数（如 8）
    - 两者按样本索引一一对应
    """
    h5_path = os.path.join(data_dir, 'heat_dataset.h5')
    npy_path = os.path.join(data_dir, 'heat_dataset.alphas.npy')

    with h5py.File(h5_path, 'r') as f:
        inputs = np.array(f['inputs'], dtype=np.float32)
    alphas = np.load(npy_path)
    if inputs.shape[0] != alphas.shape[0]:
        raise ValueError(f"样本数不一致：inputs={inputs.shape[0]} vs alphas={alphas.shape[0]}")
    return inputs, alphas


class SelfSupDataset(Dataset):
    """
    自监督数据集：返回 (inputs[i], alphas[i])。
    alphas[i] 形状为 (K, 4)，K 为该样本可用的相似系数个数。
    """
    def __init__(self, inputs: np.ndarray, alphas: np.ndarray) -> None:
        assert inputs.shape[0] == alphas.shape[0]
        self.inputs = inputs.astype(np.float32)
        self.alphas = alphas.astype(np.float32)
        self.len_samples = inputs.shape[0]

    def __len__(self) -> int:
        return self.len_samples

    def __getitem__(self, idx: int):
        x = self.inputs[idx]
        a = self.alphas[idx]
        return x, a


def create_selfsup_dataloader(inputs: np.ndarray,
                              alphas: np.ndarray,
                              batch_size: int = 32,
                              shuffle: bool = True,
                              num_workers: int = 0) -> DataLoader:
    ds = SelfSupDataset(inputs, alphas)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers)


def dimension_scaling_Tensor(samples, bc_dim_mat=None, bc_dim_coef= None):
    """
    PyTorch张量版本的量纲缩放函数
    
    功能与dimension_scaling相同，但支持GPU计算和自动微分
    
    Args:
        samples (torch.Tensor): 待缩放的物理数据张量
        bc_dim_mat (torch.Tensor): 量纲矩阵
        bc_dim_coef (torch.Tensor): 量纲系数
        
    Returns:
        torch.Tensor: 缩放后的物理数据张量
    """
    # 转换为PyTorch张量并确保设备一致性
    
    bc_dim_mat = bc_dim_mat.to(dtype=torch.float32, device=bc_dim_coef.device)

    
    # 计算量纲缩放矩阵
    scale_mat = torch.exp(torch.matmul(bc_dim_coef, bc_dim_mat))
    
    # 扩展维度以匹配样本数据
    scale_mat = torch.unsqueeze(scale_mat, dim=1)  # 添加空间维度
    scale_mat = torch.unsqueeze(scale_mat, dim=1)  # 添加另一空间维度
    
    # 应用物理相似性缩放
    rst = samples * scale_mat
    return rst


def prepare_satellite_dataloaders(
    data_path: str,
    selfsup_dir: str,
    ntrain: int,
    nvalid: int,
    batch_size: int = 32,
    self_batch_size: int = 32,
    down: int = 4,
    work_path: str = None,
    self_sample_limit: Optional[int] = None,
    noise_std: float = 0.0,
    noise_type: str = 'independent',
    num_workers: int = 4,
    pin_memory: bool = True,
    use_cache: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, DataNormer, DataNormer]:
    """
    一站式准备监督和自监督训练所需的所有 DataLoader 和归一化器
    
    参数:
        data_path: 监督数据 H5 文件路径
        selfsup_dir: 自监督数据目录路径
        ntrain: 训练样本数
        nvalid: 验证样本数
        batch_size: 监督训练批次大小
        self_batch_size: 自监督训练批次大小
        down: 下采样倍数
        work_path: 工作目录（用于保存归一化器信息），可选
        self_sample_limit: 自监督数据样本数限制，可选
        noise_std: 训练集输出噪声标准差（归一化空间），默认 0.0 表示无噪声
        num_workers: DataLoader 工作进程数（默认 4）
        pin_memory: 是否使用锁页内存加速 GPU 传输（默认 True）
        use_cache: 是否使用预处理数据缓存（默认 True）
        
    返回:
        train_loader: 监督训练 DataLoader
        valid_loader: 监督验证 DataLoader
        ss_loader: 自监督训练 DataLoader
        x_normalizer: 输入归一化器
        y_normalizer: 输出归一化器
    """
    from torch.utils.data import TensorDataset
    from Demo.satellite_2d_base.dataset_satellite import load_satellite_data
    
    # 生成缓存文件路径（保存到 data/ 目录）
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
    os.makedirs(cache_dir, exist_ok=True)
    cache_filename = f'cache_n{ntrain}_v{nvalid}_d{down}.npz'
    cache_path = os.path.join(cache_dir, cache_filename)
    
    
    # 尝试加载缓存
    if use_cache and os.path.exists(cache_path):
        print(f"  [数据加载] 从缓存加载: {cache_path}")
        cache_data = np.load(cache_path)
        train_x = torch.from_numpy(cache_data['train_x']).float()
        train_y = torch.from_numpy(cache_data['train_y']).float()
        valid_x = torch.from_numpy(cache_data['valid_x']).float()
        valid_y = torch.from_numpy(cache_data['valid_y']).float()
        
        # 重建归一化器
        x_normalizer = DataNormer(None, method='mean-std')
        x_normalizer.mean = cache_data['x_mean']
        x_normalizer.std = cache_data['x_std']
        y_normalizer = DataNormer(None, method='mean-std')
        y_normalizer.mean = cache_data['y_mean']
        y_normalizer.std = cache_data['y_std']
        
        print(f"  [数据加载] 缓存加载完成")
    else:
        # 正常加载和处理
        print(f"  [数据加载] 从原始数据加载...")
        inputs, outputs = load_satellite_data(data_path, noise_scale=noise_std, noise_type=noise_type)
        N = inputs.shape[0]
        train_indices = np.zeros(N, dtype=bool)
        assert ntrain + nvalid <= N, f'ntrain({ntrain}) + nvalid({nvalid}) 超过数据规模 {N}'
        
        
        train_indices[:ntrain] = True
        
        train_x = torch.from_numpy(inputs[train_indices, ::down, ::down, :].copy()).float()
        train_y = torch.from_numpy(outputs[train_indices, ::down, ::down, :].copy()).float()
        valid_x = torch.from_numpy(inputs[N - nvalid:, ::down, ::down, :].copy()).float()
        valid_y = torch.from_numpy(outputs[N - nvalid:, ::down, ::down, :].copy()).float()
        
        # 归一化
        x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
        y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
        train_x = x_normalizer.norm(train_x)
        valid_x = x_normalizer.norm(valid_x)
        train_y = y_normalizer.norm(train_y)
        valid_y = y_normalizer.norm(valid_y)
        
        # 保存缓存
        if use_cache:
            print(f"  [数据加载] 保存缓存到: {cache_path}")
            np.savez_compressed(
                cache_path,
                train_x=train_x.numpy(),
                train_y=train_y.numpy(),
                valid_x=valid_x.numpy(),
                valid_y=valid_y.numpy(),
                x_mean=x_normalizer.mean,
                x_std=x_normalizer.std,
                y_mean=y_normalizer.mean,
                y_std=y_normalizer.std
            )
            print(f"  [数据加载] 缓存保存完成")
    
    # 5. 保存归一化器信息（如果提供了 work_path）
    if work_path is not None:
        normalizer_info = {
            'x_mean': x_normalizer.mean.tolist(),
            'x_std': x_normalizer.std.tolist(),
            'y_mean': y_normalizer.mean.tolist(),
            'y_std': y_normalizer.std.tolist(),
            'method': 'mean-std'
        }
        with open(os.path.join(work_path, 'normalizers.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(normalizer_info, f, allow_unicode=True)
    
    # 6. 创建监督 DataLoader
    train_dataset = TensorDataset(train_x, train_y)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    valid_loader = DataLoader(
        TensorDataset(valid_x, valid_y), 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 7. 加载自监督数据
    ss_inputs_np, ss_alphas_np = load_selfsup_data(selfsup_dir, sample_limit=self_sample_limit)
    
    # 8. 自监督数据下采样
    ss_inputs_np = ss_inputs_np[:, ::down, ::down, :].astype(np.float32)
    ss_alphas_np = ss_alphas_np.astype(np.float32)
    
    # 9. 自监督数据归一化
    ss_inputs_tensor = torch.tensor(ss_inputs_np, dtype=torch.float32)
    ss_alphas_tensor = torch.tensor(ss_alphas_np, dtype=torch.float32)
    ss_inputs_normalized = x_normalizer.norm(ss_inputs_tensor)
    
    # 10. 创建自监督 DataLoader（策略1优化）
    ss_dataset = TensorDataset(ss_inputs_normalized, ss_alphas_tensor)
    ss_loader = DataLoader(
        ss_dataset, 
        batch_size=self_batch_size, 
        shuffle=True, 
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, valid_loader, ss_loader, x_normalizer, y_normalizer