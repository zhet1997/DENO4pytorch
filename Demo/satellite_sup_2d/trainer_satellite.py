"""
训练器模块 - 卫星热传导数据集

功能：
1. 动态通道压缩（compress_U_channels）
2. 多super_num训练（train_one_epoch）
3. 分桶验证（validate_all）
4. 采样可视化（inference_sample）
5. 日志记录（MetricsLogger）
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from Utilizes.process_data import DataNormer
from Utilizes.visual_data import MatplotlibVision
from Demo.satellite_sup_2d.trains_satellite import split_satellite_g_inputs


def compress_U_channels(
    U: torch.Tensor,
    target_channels: int,
) -> torch.Tensor:
    """
    动态压缩U通道数（torch版本的transform_U_channels）
    
    Args:
        U: (B, H, W, current_channels)
        target_channels: 目标通道数
    
    Returns:
        U_compressed: (B, H, W, target_channels)
    
    压缩规则：
        - current == target: 直接返回
        - current > target: 分target组，每组取min
        - current < target: 报错（数据加载时应已扩展）
    """
    B, H, W, current_channels = U.shape
    
    if current_channels == target_channels:
        return U
    
    elif current_channels > target_channels:
        # 分组取min
        U_new = torch.zeros(B, H, W, target_channels, device=U.device, dtype=U.dtype)
        group_size = current_channels // target_channels
        remainder = current_channels % target_channels
        
        start_idx = 0
        for k in range(target_channels):
            # 动态分组：前remainder组多分配1个通道
            end_idx = start_idx + group_size + (1 if k < remainder else 0)
            U_new[..., k] = U[..., start_idx:end_idx].min(dim=-1)[0]
            start_idx = end_idx
        
        return U_new
    
    else:
        raise ValueError(
            f"U通道数({current_channels}) < 目标通道数({target_channels})，"
            f"数据加载时应已扩展到最大通道数"
        )


def train_one_epoch(
    train_loader: DataLoader,
    model: nn.Module,
    optimizers: Dict[int, Optimizer],
    schedulers: Dict[int, _LRScheduler],
    super_nums_train: List[int],
    channel_num: int,
    device: torch.device,
    loss_func: nn.Module,
) -> Dict[int, float]:
    """
    训练一个epoch，支持多super_num
    
    Args:
        train_loader: 训练数据加载器
        model: 神经网络模型
        optimizers: {super_num: optimizer}
        schedulers: {super_num: scheduler}
        super_nums_train: 要训练的super_num列表 [0] or [0,1] or [0,1,2]
        channel_num: 基础通道数（如16）
        device: 计算设备
        loss_func: 损失函数
    
    Returns:
        {super_num: loss} 每个super_num的训练损失
    """
    model.train()
    losses = {}
    
    for super_num in super_nums_train:
        target_U_channels = channel_num * (2 ** super_num)
        optimizer = optimizers[super_num]
        scheduler = schedulers[super_num]
        
        epoch_loss = 0.0
        for batch_idx, (G, U, T) in enumerate(train_loader):
            G, U, T = G.to(device), U.to(device), T.to(device)
            
            # 动态压缩U通道
            U_compressed = compress_U_channels(U, target_U_channels)
            
            # 前向传播
            pred = model(G, U_compressed)
            loss = loss_func(pred, T)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        losses[super_num] = epoch_loss / len(train_loader)
    
    return losses


def validate_all(
    valid_loaders: Dict[int, DataLoader],
    model: nn.Module,
    super_nums_eval: List[int],
    channel_num: int,
    device: torch.device,
    loss_func: nn.Module,
    train_component_nums: List[int] = None,
) -> Dict:
    """
    验证所有K桶和super_num组合

    Args:
        valid_loaders: {K: DataLoader}
        model: 神经网络模型
        super_nums_eval: 要评估的super_num列表
        channel_num: 基础通道数
        device: 计算设备
        loss_func: 损失函数
        train_component_nums: 训练K桶列表，用于区分ID/OOD

    Returns:
        {
            "per_super": {0: 0.123, 1: 0.234, 2: 0.345},
            "per_super_id": {0: 0.111, 1: 0.222, 2: 0.333},
            "per_super_ood": {0: 0.144, 1: 0.255, 2: 0.366},
            "per_K": {
                0: {1: 0.1, 2: 0.2, ..., 25: 0.9},
                1: {...},
                2: {...}
            },
            "id_Ks": [...],
            "ood_Ks": [...],
        }
    """
    model.eval()
    train_component_set = set(train_component_nums or [])
    valid_Ks = list(valid_loaders.keys())
    id_Ks = [K for K in valid_Ks if K in train_component_set]
    ood_Ks = [K for K in valid_Ks if K not in train_component_set]

    results = {
        "per_super": {},
        "per_super_id": {},
        "per_super_ood": {},
        "per_K": {s: {} for s in super_nums_eval},
        "id_Ks": id_Ks,
        "ood_Ks": ood_Ks,
    }

    with torch.no_grad():
        for super_num in super_nums_eval:
            target_U_channels = channel_num * (2 ** super_num)

            for K, loader in valid_loaders.items():
                K_loss = 0.0
                num_batches = 0
                for G, U, T in loader:
                    G, U, T = G.to(device), U.to(device), T.to(device)

                    # 动态压缩U通道
                    U_compressed = compress_U_channels(U, target_U_channels)

                    pred = model(G, U_compressed)
                    loss = loss_func(pred, T)

                    # 检查loss是否有效
                    loss_val = loss.item()
                    if not (np.isnan(loss_val) or np.isinf(loss_val)):
                        K_loss += loss_val
                        num_batches += 1

                K_loss_avg = K_loss / num_batches if num_batches > 0 else float('nan')
                results["per_K"][super_num][K] = K_loss_avg

            all_losses = [
                loss for loss in results["per_K"][super_num].values()
                if not np.isnan(loss) and not np.isinf(loss)
            ]
            id_losses = [
                results["per_K"][super_num][K] for K in id_Ks
                if K in results["per_K"][super_num]
                and not np.isnan(results["per_K"][super_num][K])
                and not np.isinf(results["per_K"][super_num][K])
            ]
            ood_losses = [
                results["per_K"][super_num][K] for K in ood_Ks
                if K in results["per_K"][super_num]
                and not np.isnan(results["per_K"][super_num][K])
                and not np.isinf(results["per_K"][super_num][K])
            ]

            results["per_super"][super_num] = np.mean(all_losses) if all_losses else float('nan')
            results["per_super_id"][super_num] = np.mean(id_losses) if id_losses else float('nan')
            results["per_super_ood"][super_num] = np.mean(ood_losses) if ood_losses else float('nan')

    return results


def inference_sample(
    loader: DataLoader,
    model: nn.Module,
    normalizers: Dict[str, DataNormer],
    super_num: int,
    channel_num: int,
    device: torch.device,
    num_samples: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    随机采样num_samples个样本用于可视化
    
    Args:
        loader: 数据加载器
        model: 神经网络模型
        normalizers: 归一化器字典
        super_num: 叠加层数
        channel_num: 基础通道数
        device: 计算设备
        num_samples: 采样数量
    
    Returns:
        coords: (N, H, W, 2) - G的前2个通道（x, y坐标）
        true: (N, H, W, 1) - 真实T场
        pred: (N, H, W, 1) - 预测T场
    """
    model.eval()
    coords_list, true_list, pred_list = [], [], []
    samples_collected = 0
    
    with torch.no_grad():
        for G, U, T in loader:
            if samples_collected >= num_samples:
                break
            
            target_U_channels = channel_num * (2 ** super_num)
            U_compressed = compress_U_channels(U, target_U_channels)
            
            G, U_compressed, T = G.to(device), U_compressed.to(device), T.to(device)
            pred = model(G, U_compressed)
            
            # 反归一化
            parsed_G = split_satellite_g_inputs(G.cpu())
            coords_tensor = parsed_G['coord'] if parsed_G['coord'] is not None else parsed_G['global_G'][..., :2]
            coords = coords_tensor.cpu().numpy()
            true = normalizers['T'].back(T.cpu().numpy())
            pred = normalizers['T'].back(pred.cpu().numpy())
            
            coords_list.append(coords)
            true_list.append(true)
            pred_list.append(pred)
            
            samples_collected += G.shape[0]
    
    return (
        np.concatenate(coords_list, axis=0)[:num_samples],
        np.concatenate(true_list, axis=0)[:num_samples],
        np.concatenate(pred_list, axis=0)[:num_samples]
    )


def plot_loss_curves(log_loss: Dict, save_path: str, super_nums: List[int]):
    """
    绘制收敛曲线（单图，不同super_num用不同颜色/线型）
    
    Args:
        log_loss: {'train': {0: [...], 1: [...], 2: [...]}, 'valid': {...}}
        save_path: 保存路径
        super_nums: 要绘制的super_num列表
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    styles = {0: '-', 1: '--', 2: '-.'}
    colors_train = {0: 'b', 1: 'g', 2: 'r'}
    colors_valid = {0: 'cyan', 1: 'lime', 2: 'orange'}
    
    for s in super_nums:
        if s in log_loss['train'] and len(log_loss['train'][s]) > 0:
            epochs = np.arange(len(log_loss['train'][s]))
            ax.plot(epochs, log_loss['train'][s], 
                    linestyle=styles.get(s, '-'), color=colors_train.get(s, 'k'), 
                    label=f'train_S{s}')
        if s in log_loss['valid'] and len(log_loss['valid'][s]) > 0:
            epochs = np.arange(len(log_loss['valid'][s]))
            ax.plot(epochs, log_loss['valid'][s], 
                    linestyle=styles.get(s, '-'), color=colors_valid.get(s, 'gray'), 
                    label=f'valid_S{s}')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_field_comparison(
    coords: np.ndarray,
    true: np.ndarray,
    pred: np.ndarray,
    save_path: str,
    visual: MatplotlibVision
):
    """
    使用MatplotlibVision绘制真实场vs预测场
    
    Args:
        coords: (H, W, 2) - 坐标
        true: (H, W, 1) - 真实场
        pred: (H, W, 1) - 预测场
        save_path: 保存路径
        visual: MatplotlibVision实例
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    visual.plot_fields_ms(fig, axs, true, pred, None)
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


class MetricsLogger:
    """日志记录器，记录训练和验证指标"""
    
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.metrics_history = []
    
    def log_epoch(self, epoch: int, metrics: Dict):
        """
        记录一个epoch的指标
        
        Args:
            epoch: epoch编号
            metrics: 指标字典
        """
        entry = {'epoch': epoch, **metrics}
        self.metrics_history.append(entry)
        
        # 确保目录存在
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存到JSON
        with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def write_csv_summary(self):
        """导出CSV摘要（包含ID/OOD与所有分桶验证loss）"""
        try:
            import pandas as pd
        except ImportError:
            print("警告: pandas未安装，跳过CSV导出")
            return

        # 展平嵌套字典
        flat_data = []
        for entry in self.metrics_history:
            row = {'epoch': entry['epoch']}

            # 训练loss
            if 'train' in entry:
                for s, loss in entry['train'].items():
                    row[f'train_s{s}'] = loss

            # 验证loss（all / id / ood）
            if 'per_super' in entry:
                for s, loss in entry['per_super'].items():
                    row[f'valid_all_s{s}'] = loss
                    row[f'valid_s{s}'] = loss
            if 'per_super_id' in entry:
                for s, loss in entry['per_super_id'].items():
                    row[f'valid_id_s{s}'] = loss
            if 'per_super_ood' in entry:
                for s, loss in entry['per_super_ood'].items():
                    row[f'valid_ood_s{s}'] = loss

            if 'id_Ks' in entry:
                row['id_Ks'] = ','.join(str(k) for k in entry['id_Ks'])
            if 'ood_Ks' in entry:
                row['ood_Ks'] = ','.join(str(k) for k in entry['ood_Ks'])

            # 分桶验证loss（per_K）
            if 'per_K' in entry:
                for s, k_dict in entry['per_K'].items():
                    for k, loss in k_dict.items():
                        row[f'valid_s{s}_k{k}'] = loss

            flat_data.append(row)

        df = pd.DataFrame(flat_data)
        df.to_csv(os.path.join(self.save_dir, 'metrics_summary.csv'), index=False)

