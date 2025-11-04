import os
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from Utilizes.process_data import DataNormer


def load_augment_data(data_dir: str,
                      sample_limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从目录中加载数据增强所需数据：inputs(H5)、outputs(H5) 与 alphas(NPY)。

    约定：
    - H5 包含 inputs 和 outputs，形状分别为 (N,256,256,6) 和 (N,256,256,1)
    - NPY 形状 (N, K, 4)，K 为每样本可用的不同相似系数个数（如 8）
    - 三者按样本索引一一对应
    
    参数：
        data_dir: 数据目录路径
        sample_limit: 样本数限制，None 表示加载全部
        
    返回：
        inputs: 输入数据，形状 (N, 256, 256, 6)
        outputs: 输出数据，形状 (N, 256, 256, 1)
        alphas: 量纲系数，形状 (N, K, 4)
    """
    h5_path = os.path.join(data_dir, 'heat_dataset.h5')
    npy_path = os.path.join(data_dir, 'heat_dataset.alphas.npy')

    with h5py.File(h5_path, 'r') as f:
        inputs = np.array(f['inputs'], dtype=np.float32)
        outputs = np.array(f['outputs'], dtype=np.float32)
    
    alphas = np.load(npy_path)
    
    if sample_limit is not None:
        inputs = inputs[:sample_limit]
        outputs = outputs[:sample_limit]
        alphas = alphas[:sample_limit]
    
    if inputs.shape[0] != outputs.shape[0] or inputs.shape[0] != alphas.shape[0]:
        raise ValueError(
            f"样本数不一致：inputs={inputs.shape[0]}, "
            f"outputs={outputs.shape[0]}, alphas={alphas.shape[0]}"
        )
    
    return inputs, outputs, alphas


def _pick_random_alpha_single(alphas: np.ndarray) -> np.ndarray:
    """
    从单个样本的多组 alpha 中随机选择一组。
    
    参数：
        alphas: 形状 (K, 4)，K 为可用系数组数
        
    返回：
        形状 (4,) 的单组 alpha
    """
    K = 4
    idx = np.random.randint(0, K)
    return alphas[idx]


def dimension_scaling_Tensor(samples, bc_dim_mat=None, bc_dim_coef=None):
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


class AugmentedDataset(Dataset):
    """
    数据增强数据集：在 __getitem__ 中对 (input, output) 动态应用量纲增强。
    
    增强流程：
    1. 随机从 alphas[i] 中选择一组系数
    2. 反归一化到物理空间
    3. 对 input 和 output 应用量纲缩放（使用相同 alpha）
    4. 归一化后返回
    
    每次访问同一样本会返回不同的随机增强结果。
    """
    def __init__(self,
                 inputs: torch.Tensor,
                 outputs: torch.Tensor,
                 alphas: np.ndarray,
                 input_dim_mat: torch.Tensor,
                 output_dim_mat: torch.Tensor,
                 x_normalizer: DataNormer,
                 y_normalizer: DataNormer) -> None:
        """
        初始化数据增强数据集。
        
        参数：
            inputs: 归一化后的输入数据，形状 (N, H, W, C_in)
            outputs: 归一化后的输出数据，形状 (N, H, W, C_out)
            alphas: 量纲系数，形状 (N, K, 4)
            input_dim_mat: 输入量纲矩阵，形状 (4, C_in)
            output_dim_mat: 输出量纲矩阵，形状 (4, C_out)
            x_normalizer: 输入归一化器
            y_normalizer: 输出归一化器
        """
        assert inputs.shape[0] == outputs.shape[0] == alphas.shape[0], \
            f"样本数不一致: inputs={inputs.shape[0]}, outputs={outputs.shape[0]}, alphas={alphas.shape[0]}"
        
        self.inputs = inputs
        self.outputs = outputs
        self.alphas = alphas
        self.input_dim_mat = input_dim_mat
        self.output_dim_mat = output_dim_mat
        self.x_normalizer = x_normalizer
        self.y_normalizer = y_normalizer
        self.len_samples = inputs.shape[0]

    def __len__(self) -> int:
        return self.len_samples

    def __getitem__(self, idx: int):
        """
        获取增强后的样本。
        
        返回：
            x_aug: 增强后的归一化输入
            y_aug: 增强后的归一化输出
        """
        # 1. 获取归一化后的数据
        x_norm = self.inputs[idx]
        y_norm = self.outputs[idx]
        
        # 2. 随机选择一组 alpha
        alpha = _pick_random_alpha_single(self.alphas[idx])  # (4,)
        alpha_tensor = torch.from_numpy(alpha).float()
        
        # 确保设备一致性
        if x_norm.is_cuda:
            alpha_tensor = alpha_tensor.to(x_norm.device)
        
        # 3. 反归一化到物理空间
        x_phys = self.x_normalizer.back(x_norm)
        y_phys = self.y_normalizer.back(y_norm)
        
        # 4. 物理空间量纲缩放（使用相同的 alpha）
        x_phys_aug = dimension_scaling_Tensor(x_phys, self.input_dim_mat, alpha_tensor)
        y_phys_aug = dimension_scaling_Tensor(y_phys, self.output_dim_mat, alpha_tensor)
        
        # 5. 归一化后返回
        x_norm_aug = self.x_normalizer.norm(x_phys_aug)
        y_norm_aug = self.y_normalizer.norm(y_phys_aug)
        
        return x_norm_aug, y_norm_aug


class NoisyOutputDataset(Dataset):
    """
    监督训练数据集（带输出噪声）：在 __getitem__ 中对 Y 动态加高斯噪声。
    噪声在归一化后的数据空间添加，每次访问重新采样。
    """
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor, noise_std: float = 0.0):
        self.inputs = inputs
        self.outputs = outputs
        self.noise_std = noise_std
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.outputs[idx]
        if self.noise_std > 0:
            noise = torch.randn_like(y) * self.noise_std
            y = y + noise
        return x, y


def create_augment_dataloader(inputs: torch.Tensor,
                               outputs: torch.Tensor,
                               alphas: np.ndarray,
                               input_dim_mat: torch.Tensor,
                               output_dim_mat: torch.Tensor,
                               x_normalizer: DataNormer,
                               y_normalizer: DataNormer,
                               batch_size: int = 32,
                               shuffle: bool = True,
                               num_workers: int = 0,
                               pin_memory: bool = False) -> DataLoader:
    """
    创建数据增强 DataLoader。
    
    参数：
        inputs: 归一化后的输入数据
        outputs: 归一化后的输出数据
        alphas: 量纲系数
        input_dim_mat: 输入量纲矩阵
        output_dim_mat: 输出量纲矩阵
        x_normalizer: 输入归一化器
        y_normalizer: 输出归一化器
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        pin_memory: 是否使用锁页内存
        
    返回：
        DataLoader
    """
    ds = AugmentedDataset(
        inputs=inputs,
        outputs=outputs,
        alphas=alphas,
        input_dim_mat=input_dim_mat,
        output_dim_mat=output_dim_mat,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )


def prepare_augmented_dataloader(
    data_path: str,
    augment_dir: str,
    ntrain: int,
    nvalid: int,
    input_dim_mat: torch.Tensor,
    output_dim_mat: torch.Tensor,
    batch_size: int = 32,
    down: int = 4,
    work_path: str = None,
    augment_sample_limit: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, DataNormer, DataNormer]:
    """
    一站式准备监督训练和数据增强训练所需的所有 DataLoader 和归一化器。
    
    参数:
        data_path: 监督数据 H5 文件路径
        augment_dir: 数据增强数据目录路径（包含 H5 和 alphas NPY）
        ntrain: 训练样本数
        nvalid: 验证样本数
        input_dim_mat: 输入量纲矩阵，形状 (4, C_in)
        output_dim_mat: 输出量纲矩阵，形状 (4, C_out)
        batch_size: 监督训练批次大小
        down: 下采样倍数
        work_path: 工作目录（用于保存归一化器信息），可选
        augment_sample_limit: 增强数据样本数限制，可选
        num_workers: DataLoader 工作进程数（默认 4）
        pin_memory: 是否使用锁页内存加速 GPU 传输（默认 True）
        
    返回:
        train_loader: 监督训练 DataLoader（普通数据）
        valid_loader: 监督验证 DataLoader
        aug_loader: 数据增强训练 DataLoader
        x_normalizer: 输入归一化器
        y_normalizer: 输出归一化器
    """
    from torch.utils.data import TensorDataset
    from Demo.satellite_2d_base.dataset_satellite import load_satellite_data
    
    print("  [数据加载] 从原始数据加载...")
    
    # 1. 加载监督数据
    inputs, outputs = load_satellite_data(data_path)
    N = inputs.shape[0]
    assert ntrain + nvalid <= N, f'ntrain({ntrain}) + nvalid({nvalid}) 超过数据规模 {N}'

    # 2. 下采样和转换为 Tensor
    train_x = torch.from_numpy(inputs[:ntrain, ::down, ::down, :].copy()).float()
    train_y = torch.from_numpy(outputs[:ntrain, ::down, ::down, :].copy()).float()
    valid_x = torch.from_numpy(inputs[N - nvalid:, ::down, ::down, :].copy()).float()
    valid_y = torch.from_numpy(outputs[N - nvalid:, ::down, ::down, :].copy()).float()
    
    # 3. 创建归一化器
    x_normalizer = DataNormer(train_x.numpy(), method='mean-std')
    y_normalizer = DataNormer(train_y.numpy(), method='mean-std')
    
    # 4. 归一化
    train_x = x_normalizer.norm(train_x)
    valid_x = x_normalizer.norm(valid_x)
    train_y = y_normalizer.norm(train_y)
    valid_y = y_normalizer.norm(valid_y)
    
    # 5. 保存归一化器信息（如果提供了 work_path）
    if work_path is not None:
        normalizer_info = {
            'x_mean': x_normalizer.mean.tolist(),
            'x_std': x_normalizer.std.tolist(),
            'y_mean': y_normalizer.mean.tolist(),
            'y_std': y_normalizer.std.tolist(),
            'method': 'mean-std'
        }
        os.makedirs(work_path, exist_ok=True)
        with open(os.path.join(work_path, 'normalizers.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(normalizer_info, f, allow_unicode=True)
    
    # 6. 创建监督训练 DataLoader（普通数据，不增强）
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 7. 创建验证 DataLoader
    valid_loader = DataLoader(
        TensorDataset(valid_x, valid_y), 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 8. 加载数据增强数据
    print(f"  [数据增强] 从 {augment_dir} 加载增强数据...")
    aug_inputs_np, aug_outputs_np, aug_alphas_np = load_augment_data(
        augment_dir,
        sample_limit=augment_sample_limit
    )
    
    # 9. 数据增强数据下采样
    aug_inputs_np = aug_inputs_np[:, ::down, ::down, :].astype(np.float32)
    aug_outputs_np = aug_outputs_np[:, ::down, ::down, :].astype(np.float32)
    aug_alphas_np = aug_alphas_np.astype(np.float32)
    
    # 10. 转换为 Tensor 并归一化
    aug_inputs_tensor = torch.tensor(aug_inputs_np, dtype=torch.float32)
    aug_outputs_tensor = torch.tensor(aug_outputs_np, dtype=torch.float32)
    aug_inputs_normalized = x_normalizer.norm(aug_inputs_tensor)
    aug_outputs_normalized = y_normalizer.norm(aug_outputs_tensor)
    
    print(f"  [数据增强] 增强数据形状: inputs={aug_inputs_normalized.shape}, "
          f"outputs={aug_outputs_normalized.shape}, alphas={aug_alphas_np.shape}")
    
    # 11. 创建数据增强 DataLoader
    aug_loader = create_augment_dataloader(
        inputs=aug_inputs_normalized,
        outputs=aug_outputs_normalized,
        alphas=aug_alphas_np,
        input_dim_mat=input_dim_mat,
        output_dim_mat=output_dim_mat,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    print(f"  [数据加载] 完成！训练集={len(train_loader.dataset)}, "
          f"验证集={len(valid_loader.dataset)}, 增强集={len(aug_loader.dataset)}")
    
    return train_loader, valid_loader, aug_loader, x_normalizer, y_normalizer
