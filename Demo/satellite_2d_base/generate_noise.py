"""
噪声生成脚本

生成两种类型的固定噪声用于数据增强：
1. 独立高斯噪声：每个像素独立采样
2. 空间相关噪声：通过高斯滤波生成空间平滑的噪声

数学原理：
- 独立噪声: n(x) ~ N(0, σ²)，各像素独立
- 平滑噪声: ñ(x) = (n * g_ℓ)(x)，其中 g_ℓ 是尺度为 ℓ 的高斯核
- 最终添加: u'(x) = u(x) + α·ñ(x)，α 由 noise_scale 控制
"""

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def generate_independent_noise(n_samples=5000, height=256, width=256, channels=1, seed=42):
    """
    生成独立高斯噪声（各像素独立）
    
    Args:
        n_samples: 样本数量
        height: 图像高度
        width: 图像宽度
        channels: 通道数
        seed: 随机种子，确保可复现
        
    Returns:
        noise: 归一化到 [-1, 1] 的噪声，形状 (n_samples, height, width, channels)
    """
    np.random.seed(seed)
    print(f"生成独立高斯噪声...")
    print(f"  - 形状: ({n_samples}, {height}, {width}, {channels})")
    
    noise = np.random.randn(n_samples, height, width, channels).astype(np.float32)
    
    # 归一化到 [-1, 1]
    max_abs = np.max(np.abs(noise))
    noise = noise / max_abs
    
    print(f"  - 统计: min={noise.min():.4f}, max={noise.max():.4f}, "
          f"mean={noise.mean():.6f}, std={noise.std():.4f}")
    
    return noise


def generate_correlated_noise(n_samples=5000, height=256, width=256, channels=1, 
                               sigma=5.0, seed=42):
    """
    生成空间相关噪声（平滑高斯噪声）
    
    方法：先生成独立噪声，再用高斯核卷积平滑
    数学：ñ(x) = (n * g_σ)(x)
    
    Args:
        n_samples: 样本数量
        height: 图像高度
        width: 图像宽度
        channels: 通道数
        sigma: 高斯滤波器的标准差，控制空间相关尺度
        seed: 随机种子，确保可复现
        
    Returns:
        noise: 归一化到 [-1, 1] 的噪声，形状 (n_samples, height, width, channels)
    """
    np.random.seed(seed)
    print(f"生成空间相关噪声...")
    print(f"  - 形状: ({n_samples}, {height}, {width}, {channels})")
    print(f"  - 高斯滤波 sigma: {sigma} (控制空间相关尺度)")
    
    # 先生成独立噪声
    noise = np.random.randn(n_samples, height, width, channels).astype(np.float32)
    
    # 对每个样本应用高斯滤波
    print(f"  - 正在应用高斯滤波...")
    for i in range(n_samples):
        if (i + 1) % 1000 == 0:
            print(f"    处理进度: {i+1}/{n_samples}")
        for c in range(channels):
            noise[i, :, :, c] = gaussian_filter(noise[i, :, :, c], sigma=sigma)
    
    # 归一化到 [-1, 1]
    max_abs = np.max(np.abs(noise))
    noise = noise / max_abs
    
    print(f"  - 统计: min={noise.min():.4f}, max={noise.max():.4f}, "
          f"mean={noise.mean():.6f}, std={noise.std():.4f}")
    
    return noise


def save_noise_to_h5(noise, filepath):
    """
    保存噪声到 H5 文件
    
    Args:
        noise: 噪声数据
        filepath: 保存路径
    """
    print(f"保存噪声到: {filepath}")
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('noise', data=noise, dtype='float32', compression='gzip')
    print(f"  - 文件大小: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")


def visualize_noise_samples(noise_independent, noise_correlated, output_dir, n_samples=5):
    """
    可视化噪声样本
    
    Args:
        noise_independent: 独立噪声
        noise_correlated: 空间相关噪声
        output_dir: 输出目录
        n_samples: 可视化样本数
    """
    print(f"\n生成噪声可视化...")
    
    # 随机选择样本
    indices = np.random.choice(noise_independent.shape[0], n_samples, replace=False)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 3*n_samples))
    
    for i, idx in enumerate(indices):
        # 独立噪声
        im1 = axes[i, 0].imshow(noise_independent[idx, :, :, 0], cmap='RdBu_r', 
                                vmin=-1, vmax=1)
        axes[i, 0].set_title(f'独立噪声 - 样本 {idx}')
        axes[i, 0].axis('off')
        plt.colorbar(im1, ax=axes[i, 0], fraction=0.046)
        
        # 空间相关噪声
        im2 = axes[i, 1].imshow(noise_correlated[idx, :, :, 0], cmap='RdBu_r', 
                                vmin=-1, vmax=1)
        axes[i, 1].set_title(f'空间相关噪声 - 样本 {idx}')
        axes[i, 1].axis('off')
        plt.colorbar(im2, ax=axes[i, 1], fraction=0.046)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'noise_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  - 可视化保存到: {save_path}")
    plt.close()
    
    # 统计分布对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 独立噪声分布
    axes[0].hist(noise_independent.flatten(), bins=100, alpha=0.7, density=True, 
                 label='独立噪声')
    axes[0].set_xlabel('噪声值')
    axes[0].set_ylabel('密度')
    axes[0].set_title('独立高斯噪声分布')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 空间相关噪声分布
    axes[1].hist(noise_correlated.flatten(), bins=100, alpha=0.7, density=True, 
                 color='orange', label='空间相关噪声')
    axes[1].set_xlabel('噪声值')
    axes[1].set_ylabel('密度')
    axes[1].set_title('空间相关噪声分布（平滑后）')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'noise_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  - 分布图保存到: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='生成固定噪声用于数据增强')
    parser.add_argument('--output_dir', type=str, 
                        default='/data/wqn/datasets/packaged_dataset20251017_6c',
                        help='噪声文件输出目录')
    parser.add_argument('--n_samples', type=int, default=5000,
                        help='生成样本数量')
    parser.add_argument('--height', type=int, default=256,
                        help='图像高度')
    parser.add_argument('--width', type=int, default=256,
                        help='图像宽度')
    parser.add_argument('--sigma', type=float, default=5.0,
                        help='空间相关噪声的高斯滤波 sigma')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--visualize', action='store_true',
                        help='是否生成可视化')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("噪声生成脚本")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    print(f"样本数量: {args.n_samples}")
    print(f"图像尺寸: {args.height}x{args.width}")
    print(f"随机种子: {args.seed}")
    print("=" * 60)
    
    # 生成独立高斯噪声
    noise_independent = generate_independent_noise(
        n_samples=args.n_samples,
        height=args.height,
        width=args.width,
        channels=1,
        seed=args.seed
    )
    
    # 保存独立噪声
    independent_path = os.path.join(args.output_dir, 'noise_independent.h5')
    save_noise_to_h5(noise_independent, independent_path)
    
    print()
    
    # 生成空间相关噪声
    noise_correlated = generate_correlated_noise(
        n_samples=args.n_samples,
        height=args.height,
        width=args.width,
        channels=1,
        sigma=args.sigma,
        seed=args.seed
    )
    
    # 保存空间相关噪声
    correlated_path = os.path.join(args.output_dir, 'noise_correlated.h5')
    save_noise_to_h5(noise_correlated, correlated_path)
    
    # 可视化
    if args.visualize:
        visualize_noise_samples(noise_independent, noise_correlated, args.output_dir)
    
    print("\n" + "=" * 60)
    print("噪声生成完成！")
    print("=" * 60)
    print(f"独立噪声: {independent_path}")
    print(f"空间相关噪声: {correlated_path}")
    if args.visualize:
        print(f"可视化图像: {args.output_dir}/noise_*.png")
    print("=" * 60)


if __name__ == '__main__':
    main()

