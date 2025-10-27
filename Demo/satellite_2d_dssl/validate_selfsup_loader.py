import os
import sys
import argparse
import numpy as np
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Demo.satellite_2d_dssl.dataset_selfsup import load_selfsup_data
from Demo.satellite_2d_base.utils import load_yaml_config


def main():
    parser = argparse.ArgumentParser(description='验证自监督数据加载与相似映射往返流程')
    parser.add_argument('--selfsup_dir', type=str, required=True)
    parser.add_argument('--sample_limit', type=int, default=8)
    args = parser.parse_args()

    inputs, alphas = load_selfsup_data(args.selfsup_dir, sample_limit=args.sample_limit)
    print(f"inputs shape: {inputs.shape}")
    print(f"alphas shape: {alphas.shape}")

    # 构建输入量纲矩阵（使用新版本YAML）
    yaml_path = os.path.join(CURRENT_DIR, 'augmentation_satellite.yml')
    cfg = load_yaml_config(yaml_path)
    channel_mapping = cfg.get('channel_mapping', {})
    dim_exponents = cfg.get('dim_exponents', {})
    
    # 构建量纲矩阵
    num_channels = 6
    input_dim_mat = np.zeros((4, num_channels), dtype=np.float32)
    for channel_name, channel_idx in channel_mapping.items():
        if channel_name in dim_exponents:
            exps = np.array(dim_exponents[channel_name], dtype=np.float32)
            if len(exps) == 4:
                input_dim_mat[:, channel_idx] = exps
                print(f"通道 {channel_name} (索引 {channel_idx}): {exps}")
    
    input_dim_mat_t = torch.tensor(input_dim_mat, dtype=torch.float32)

    # 取一小批样本，执行一次映射与反映射
    xs = torch.tensor(inputs[:4], dtype=torch.float32)
    a = torch.tensor(alphas[:4], dtype=torch.float32)  # (B,K,4)
    B, K, _ = a.shape
    idx = torch.randint(low=0, high=K, size=(B,))
    alpha = a[torch.arange(B), idx, :]  # (B,4)

    def _sim_np(x, mat, coef):
        # 仿照 dimension_scaling_Tensor 的 numpy 版本（仅用于验证）
        mat_t = torch.tensor(mat, dtype=torch.float32)
        coef_t = coef
        scale = torch.exp(torch.matmul(coef_t, mat_t))  # (B,C)
        # NHWC -> 按通道缩放
        scale = scale[:, None, None, :]
        return x * scale

    x_far = _sim_np(xs, input_dim_mat, alpha)
    x_back = _sim_np(x_far, input_dim_mat, -alpha)

    def _stat(t):
        return float(t.mean().item()), float(t.std().item())

    m0, s0 = _stat(xs)
    m1, s1 = _stat(x_far)
    m2, s2 = _stat(x_back)
    print(f"orig mean/std: {m0:.6f}/{s0:.6f}")
    print(f"far  mean/std: {m1:.6f}/{s1:.6f}")
    print(f"back mean/std: {m2:.6f}/{s2:.6f}")
    print("✅ 验证完成（往返应基本回到原分布，存在数值漂移属正常）")


if __name__ == '__main__':
    main()


