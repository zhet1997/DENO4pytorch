"""
测试噪声功能
"""

import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Demo.satellite_2d_base.dataset_satellite import load_satellite_data


def test_no_noise():
    """测试向后兼容：不加噪声"""
    print("=" * 60)
    print("测试 1: 不加噪声（默认行为，向后兼容）")
    print("=" * 60)
    
    data_path = '/data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5'
    inputs, outputs = load_satellite_data(data_path, sample_limit=100)
    
    print(f"\n✓ 加载成功")
    print(f"  - inputs.shape: {inputs.shape}")
    print(f"  - outputs.shape: {outputs.shape}")
    print()


def test_independent_noise():
    """测试独立噪声"""
    print("=" * 60)
    print("测试 2: 添加独立高斯噪声")
    print("=" * 60)
    
    data_path = '/data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5'
    inputs, outputs = load_satellite_data(
        data_path, 
        sample_limit=5100,
        noise_type='independent',
        noise_scale=0.1
    )
    
    print(f"\n✓ 加载成功")
    print(f"  - inputs.shape: {inputs.shape}")
    print(f"  - outputs.shape: {outputs.shape}")
    print()


def test_correlated_noise():
    """测试空间相关噪声"""
    print("=" * 60)
    print("测试 3: 添加空间相关噪声")
    print("=" * 60)
    
    data_path = '/data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5'
    inputs, outputs = load_satellite_data(
        data_path, 
        sample_limit=5100,
        noise_type='correlated',
        noise_scale=0.05
    )
    
    print(f"\n✓ 加载成功")
    print(f"  - inputs.shape: {inputs.shape}")
    print(f"  - outputs.shape: {outputs.shape}")
    print()


if __name__ == '__main__':
    test_no_noise()
    
    print("\n" + "=" * 60)
    print("提示: 在测试噪声功能前，请先运行 generate_noise.py 生成噪声文件")
    print("=" * 60)
    
    # 检查噪声文件是否存在
    noise_dir = '/data/wqn/datasets/packaged_dataset20251017_6c'
    independent_exists = os.path.exists(os.path.join(noise_dir, 'noise_independent.h5'))
    correlated_exists = os.path.exists(os.path.join(noise_dir, 'noise_correlated.h5'))
    
    if independent_exists and correlated_exists:
        print("\n噪声文件已存在，继续测试...")
        test_independent_noise()
        test_correlated_noise()
        print("\n" + "=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
    else:
        print(f"\n噪声文件状态:")
        print(f"  - noise_independent.h5: {'✓ 存在' if independent_exists else '✗ 不存在'}")
        print(f"  - noise_correlated.h5: {'✓ 存在' if correlated_exists else '✗ 不存在'}")
        print(f"\n请先运行以下命令生成噪声:")
        print(f"  cd {os.path.dirname(__file__)}")
        print(f"  python generate_noise.py --output_dir {noise_dir} --visualize")

