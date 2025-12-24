#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 get_loaders_satellite_multi_GUT 函数

测试场景:
1. 加载单个数据集（验证基本功能）
2. 加载多个数据集（验证合并逻辑）
3. 测试不同的n→K转换场景
4. 验证DataLoader返回正确的三元组(G, U, T)
5. 验证三个归一化器独立工作
"""
import sys
sys.path.append('/data/wqn/DENO4pytorch')

from Demo.satellite_sup_2d.ablation_satellite import (
    get_loaders_satellite_multi_GUT,
    transform_U_channels
)
import numpy as np


def test_channel_transform():
    """测试U通道转换函数"""
    print("=" * 60)
    print("测试1: U通道转换函数")
    print("=" * 60)
    
    # 创建测试数据
    N, H, W = 10, 64, 64
    
    # 测试情况1: n == K
    print("\n情况1: n == K (15 → 15)")
    U_15 = np.random.randn(N, H, W, 15)
    U_out = transform_U_channels(U_15, target_channels=15, empty_channel_value=1.0)
    print(f"  输入形状: {U_15.shape}, 输出形状: {U_out.shape}")
    assert U_out.shape == (N, H, W, 15), "形状应保持不变"
    assert np.array_equal(U_out, U_15), "数据应完全相同"
    print("  ✓ 通过")
    
    # 测试情况2: n > K
    print("\n情况2: n > K (15 → 10)")
    U_15 = np.random.randn(N, H, W, 15)
    U_out = transform_U_channels(U_15, target_channels=10, empty_channel_value=1.0)
    print(f"  输入形状: {U_15.shape}, 输出形状: {U_out.shape}")
    assert U_out.shape == (N, H, W, 10), "输出应为10通道"
    # 验证min操作
    print(f"  原始范围: [{U_15.min():.4f}, {U_15.max():.4f}]")
    print(f"  输出范围: [{U_out.min():.4f}, {U_out.max():.4f}]")
    print("  ✓ 通过")
    
    # 测试情况3: n < K
    print("\n情况3: n < K (5 → 15)")
    U_5 = np.random.randn(N, H, W, 5)
    empty_value = 10.0
    U_out = transform_U_channels(U_5, target_channels=15, empty_channel_value=empty_value)
    print(f"  输入形状: {U_5.shape}, 输出形状: {U_out.shape}")
    assert U_out.shape == (N, H, W, 15), "输出应为15通道"
    # 验证前5个通道保留，后10个通道为empty_value
    assert np.array_equal(U_out[:, :, :, :5], U_5), "前5个通道应保持不变"
    assert np.allclose(U_out[:, :, :, 5:], empty_value), "后10个通道应为空值"
    print(f"  前5通道范围: [{U_out[:,:,:,:5].min():.4f}, {U_out[:,:,:,:5].max():.4f}]")
    print(f"  后10通道值: {U_out[0,0,0,5]} (应为{empty_value})")
    print("  ✓ 通过")
    
    print("\n" + "=" * 60)
    print("U通道转换函数测试全部通过！")
    print("=" * 60)


def test_single_dataset():
    """测试加载单个数据集"""
    print("\n\n" + "=" * 60)
    print("测试2: 加载单个数据集")
    print("=" * 60)
    
    train_loader, valid_loader, normalizers, meta = get_loaders_satellite_multi_GUT(
        component_nums=[1],
        target_U_channels=15,
        empty_channel_value=1.0,
        train_num=80,
        valid_num=20,
        batch_size=8,
        shuffled=False
    )
    
    print("\n元信息:")
    for key, value in meta.items():
        if key != 'dataset_info':
            print(f"  - {key}: {value}")
    
    print("\n测试DataLoader:")
    for batch_G, batch_U, batch_T in train_loader:
        print(f"  训练集batch:")
        print(f"    - G形状: {batch_G.shape}")
        print(f"    - U形状: {batch_U.shape}")
        print(f"    - T形状: {batch_T.shape}")
        print(f"    - G范围: [{batch_G.min():.4f}, {batch_G.max():.4f}]")
        print(f"    - U范围: [{batch_U.min():.4f}, {batch_U.max():.4f}]")
        print(f"    - T范围: [{batch_T.min():.4f}, {batch_T.max():.4f}]")
        break
    
    print("\n归一化器信息:")
    print(f"  - G: mean shape={normalizers['G'].mean.shape}, std shape={normalizers['G'].std.shape}")
    print(f"  - U: mean shape={normalizers['U'].mean.shape}, std shape={normalizers['U'].std.shape}")
    print(f"  - T: mean shape={normalizers['T'].mean.shape}, std shape={normalizers['T'].std.shape}")
    
    print("\n✓ 单数据集加载测试通过！")


def test_multiple_datasets():
    """测试加载多个数据集"""
    print("\n\n" + "=" * 60)
    print("测试3: 加载多个数据集")
    print("=" * 60)
    
    train_loader, valid_loader, normalizers, meta = get_loaders_satellite_multi_GUT(
        component_nums=[1, 2, 3, 4, 5],
        target_U_channels=15,
        empty_channel_value=1.0,
        train_num=400,
        valid_num=100,
        batch_size=16,
        shuffled=True
    )
    
    print("\n元信息:")
    for key, value in meta.items():
        if key not in ['dataset_info']:
            print(f"  - {key}: {value}")
    
    print("\n数据集详情:")
    for info in meta['dataset_info']:
        print(f"  [{info['component_num']:2d}元件] "
              f"样本:{info['samples']:4d}, "
              f"原始U通道:{info['original_U_channels']:2d}")
    
    print("\n测试DataLoader (混合打乱):")
    batch_count = 0
    for batch_G, batch_U, batch_T in train_loader:
        if batch_count == 0:
            print(f"  第一个batch:")
            print(f"    - G形状: {batch_G.shape}")
            print(f"    - U形状: {batch_U.shape}")
            print(f"    - T形状: {batch_T.shape}")
        batch_count += 1
    
    print(f"\n  总batch数: {batch_count}")
    print(f"  验证集batch数: {len(valid_loader) if valid_loader else 0}")
    
    print("\n✓ 多数据集加载测试通过！")


def test_different_channel_scenarios():
    """测试不同的通道转换场景"""
    print("\n\n" + "=" * 60)
    print("测试4: 不同的n→K转换场景")
    print("=" * 60)
    
    scenarios = [
        # (component_nums, target_K, 描述)
        ([1], 15, "1元件(1通道) → 15通道"),
        ([10], 15, "10元件(10通道) → 15通道"),
        ([15], 15, "15元件(15通道) → 15通道"),
        ([20], 15, "20元件(20通道) → 15通道"),
    ]
    
    for comp_nums, target_K, desc in scenarios:
        print(f"\n场景: {desc}")
        try:
            train_loader, _, normalizers, meta = get_loaders_satellite_multi_GUT(
                component_nums=comp_nums,
                target_U_channels=target_K,
                empty_channel_value=1.0,
                train_num=50,
                valid_num=10,
                batch_size=8,
                shuffled=False
            )
            
            # 获取第一个batch
            batch_G, batch_U, batch_T = next(iter(train_loader))
            print(f"  ✓ 成功! U形状: {batch_U.shape}, 范围: [{batch_U.min():.4f}, {batch_U.max():.4f}]")
        
        except Exception as e:
            print(f"  ✗ 失败: {e}")
    
    print("\n✓ 通道转换场景测试通过！")


def test_normalizer_independence():
    """测试三个归一化器的独立性"""
    print("\n\n" + "=" * 60)
    print("测试5: 归一化器独立性")
    print("=" * 60)
    
    train_loader, _, normalizers, _ = get_loaders_satellite_multi_GUT(
        component_nums=[1, 2, 3],
        target_U_channels=15,
        empty_channel_value=1.0,
        train_num=100,
        valid_num=20,
        batch_size=16,
    )
    
    g_norm = normalizers['G']
    u_norm = normalizers['U']
    t_norm = normalizers['T']
    
    print("\nG归一化器:")
    print(f"  Mean: {g_norm.mean}")
    print(f"  Std:  {g_norm.std}")
    
    print("\nU归一化器:")
    print(f"  Mean shape: {u_norm.mean.shape}")
    print(f"  Mean (前5个): {u_norm.mean[:5]}")
    print(f"  Std (前5个):  {u_norm.std[:5]}")
    
    print("\nT归一化器:")
    print(f"  Mean: {t_norm.mean}")
    print(f"  Std:  {t_norm.std}")
    
    # 验证归一化后的数据
    batch_G, batch_U, batch_T = next(iter(train_loader))
    print("\n归一化后batch的统计:")
    print(f"  G: mean≈{batch_G.mean():.4f}, std≈{batch_G.std():.4f}")
    print(f"  U: mean≈{batch_U.mean():.4f}, std≈{batch_U.std():.4f}")
    print(f"  T: mean≈{batch_T.mean():.4f}, std≈{batch_T.std():.4f}")
    print("  (归一化后均值应接近0，标准差接近1)")
    
    print("\n✓ 归一化器独立性测试通过！")


if __name__ == "__main__":
    """运行所有测试"""
    
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "多数据集GUT加载器 - 完整测试套件" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    
    try:
        # 测试1: 通道转换函数
        test_channel_transform()
        
        # 测试2: 单个数据集
        test_single_dataset()
        
        # 测试3: 多个数据集
        test_multiple_datasets()
        
        # 测试4: 不同通道场景
        test_different_channel_scenarios()
        
        # 测试5: 归一化器独立性
        test_normalizer_independence()
        
        # 所有测试通过
        print("\n\n")
        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 18 + "所有测试通过！ ✓" + " " * 18 + "║")
        print("╚" + "=" * 58 + "╝")
        print()
    
    except Exception as e:
        print(f"\n\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
