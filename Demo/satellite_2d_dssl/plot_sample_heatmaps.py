import os
import sys
import argparse
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 路径注入
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
sys.path.insert(0, PROJECT_ROOT)

from Demo.satellite_2d_base.dataset_satellite import load_satellite_data


# ========== 样本选择配置（可修改） ==========
USE_RANDOM_SAMPLES = True  # True: 随机选择 | False: 使用指定索引
NUM_SAMPLES = 5            # 随机选择时的样本数量
SPECIFIED_INDICES = [0, 10, 20, 30, 40]  # 指定索引时使用此列表
# =========================================


def _split_dataset(inputs: np.ndarray,
                   outputs: np.ndarray,
                   ntrain: Optional[int],
                   nvalid: Optional[int]) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    复用 eval_satellite_predict.py 的数据划分逻辑
    返回 ((train_x, train_y), (valid_x, valid_y))
    """
    N = inputs.shape[0]
    if ntrain is None or nvalid is None:
        ntrain = int(N * 0.9)
        nvalid = N - ntrain
    else:
        assert ntrain + nvalid <= N, f"ntrain({ntrain}) + nvalid({nvalid}) 超过数据规模 {N}"

    train_x = inputs[:ntrain]
    train_y = outputs[:ntrain]
    valid_x = inputs[N - nvalid:]
    valid_y = outputs[N - nvalid:]
    
    return (train_x, train_y), (valid_x, valid_y)


def load_prediction_data(eval_dir: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从后处理目录加载预测结果
    
    Returns:
        (pred_data, true_data): 形状均为 (N, H, W, 1)
    """
    pred_path = os.path.join(eval_dir, f'{split}_pred.npy')
    true_path = os.path.join(eval_dir, f'{split}_true.npy')
    
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")
    if not os.path.exists(true_path):
        raise FileNotFoundError(f"真实值文件不存在: {true_path}")
    
    pred_data = np.load(pred_path)
    true_data = np.load(true_path)
    
    print(f"加载预测数据: {pred_path}")
    print(f"  预测形状: {pred_data.shape}")
    print(f"  真实形状: {true_data.shape}")
    
    return pred_data, true_data


def load_input_layout(data_path: str, 
                      ntrain: int, 
                      nvalid: int, 
                      split: str) -> np.ndarray:
    """
    从原始数据集加载输入数据并提取热源分布通道
    
    关键：确保与 eval_satellite_predict.py 中生成的 npy 文件对应
    - train_pred.npy 对应 inputs[:ntrain]
    - valid_pred.npy 对应 inputs[N - nvalid:]
    
    Returns:
        layout_data: 形状为 (N, 64, 64)，下采样后的热源分布
    """
    # 加载完整数据集
    inputs, outputs = load_satellite_data(data_path)
    
    # 使用相同的划分逻辑
    (train_x, _), (valid_x, _) = _split_dataset(inputs, outputs, ntrain, nvalid)
    
    # 根据 split 选择对应的数据
    if split == 'train':
        input_data = train_x
    elif split == 'valid':
        input_data = valid_x
    else:
        raise ValueError(f"不支持的 split 参数: {split}")
    
    print(f"加载输入数据 ({split}):")
    print(f"  原始形状: {input_data.shape}")
    
    # 下采样到 64x64（与预测数据一致）
    down = 4
    input_down = input_data[:, ::down, ::down, :]  # (N, 64, 64, 6)
    
    # 提取通道0：元件功率密度（热源分布）
    layout_data = input_down[:, :, :, 0]  # (N, 64, 64)
    
    print(f"  下采样后形状: {input_down.shape}")
    print(f"  热源分布形状: {layout_data.shape}")
    
    return layout_data


def select_sample_indices(total_samples: int) -> np.ndarray:
    """
    根据配置选择样本索引
    
    Returns:
        indices: 选中的样本索引数组
    """
    if USE_RANDOM_SAMPLES:
        num = min(NUM_SAMPLES, total_samples)
        indices = np.random.choice(total_samples, size=num, replace=False)
        indices = np.sort(indices)
        print(f"随机选择 {num} 个样本: {indices}")
    else:
        indices = np.array(SPECIFIED_INDICES)
        valid_indices = indices[indices < total_samples]
        if len(valid_indices) < len(indices):
            print("警告: 部分指定索引超出范围，已过滤")
        indices = valid_indices
        print(f"使用指定索引: {indices}")
    
    return indices


def plot_sample_heatmaps(layout_data: np.ndarray,
                         true_data: np.ndarray,
                         pred_data: np.ndarray,
                         sample_indices: np.ndarray,
                         save_path: str):
    """
    绘制样本对比云图：N 行 4 列
    每行：[热源分布] [真实温度] [预测温度] [误差]
    """
    n_samples = len(sample_indices)
    
    # 创建子图网格
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    # 确保 axes 是 2D 数组
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(sample_indices):
        # 提取数据
        layout = layout_data[idx]  # (64, 64)
        true_field = true_data[idx, :, :, 0]  # (64, 64)
        pred_field = pred_data[idx, :, :, 0]  # (64, 64)
        error_field = pred_field - true_field  # 绝对误差
        
        # 列1: 热源分布
        ax = axes[i, 0]
        im = ax.imshow(layout, cmap='viridis', origin='lower')
        ax.set_title(f'Sample {idx}: Heat Source', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 列2: 真实温度场
        ax = axes[i, 1]
        im = ax.imshow(true_field, cmap='hot', origin='lower')
        ax.set_title('True Temperature', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 列3: 预测温度场
        ax = axes[i, 2]
        im = ax.imshow(pred_field, cmap='hot', origin='lower')
        ax.set_title('Predicted Temperature', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # 列4: 误差（以0为中心的发散色图）
        ax = axes[i, 3]
        # 计算误差的最大绝对值，用于对称色标
        vmax = np.abs(error_field).max()
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(error_field, cmap='RdBu_r', origin='lower', norm=norm)
        ax.set_title('Error (Pred - True)', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"已保存图像: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='绘制样本对比云图')
    parser.add_argument('--eval_dir', type=str, required=True, 
                        help='评估结果目录路径')
    parser.add_argument('--data_path', type=str, 
                        default='/data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5',
                        help='原始数据集路径')
    parser.add_argument('--split', type=str, default='valid', 
                        choices=['train', 'valid'],
                        help='使用哪个子集')
    parser.add_argument('--ntrain', type=int, default=1000,
                        help='训练集样本数（需与训练时一致）')
    parser.add_argument('--nvalid', type=int, default=500,
                        help='验证集样本数（需与训练时一致）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("样本对比云图绘制脚本")
    print("=" * 60)
    
    # 加载预测数据
    pred_data, true_data = load_prediction_data(args.eval_dir, args.split)
    
    # 加载输入数据的热源分布
    layout_data = load_input_layout(args.data_path, args.ntrain, args.nvalid, args.split)
    
    # 验证数据形状一致
    assert pred_data.shape[0] == true_data.shape[0] == layout_data.shape[0], \
        f"数据样本数不一致: pred={pred_data.shape[0]}, true={true_data.shape[0]}, layout={layout_data.shape[0]}"
    
    total_samples = pred_data.shape[0]
    print(f"\n数据加载完成，共 {total_samples} 个样本")
    
    # 选择样本索引
    sample_indices = select_sample_indices(total_samples)
    
    if len(sample_indices) == 0:
        print("错误: 没有有效的样本索引")
        return
    
    # 绘制并保存
    save_path = os.path.join(args.eval_dir, f'sample_heatmaps_{args.split}.png')
    plot_sample_heatmaps(layout_data, true_data, pred_data, sample_indices, save_path)
    
    print("=" * 60)
    print("绘制完成")
    print("=" * 60)


if __name__ == '__main__':
    main()

