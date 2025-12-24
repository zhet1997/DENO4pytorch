#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 get_loaders_satellite_GUT 函数
"""
import sys
import os

# 添加项目路径
sys.path.append('/data/wqn/DENO4pytorch')

from Demo.satellite_sup_2d.ablation_satellite import get_loaders_satellite_GUT

print("=" * 60)
print("测试 get_loaders_satellite_GUT 函数")
print("=" * 60)

# 调用函数
train_loader, valid_loader, x_norm, y_norm, meta = get_loaders_satellite_GUT(
    train_num=800,
    valid_num=200,
    batch_size=16,
    h5_path=None,  # 使用默认路径
    shuffled=False
)

print("\n" + "=" * 60)
print("元信息:")
print("=" * 60)
for key, value in meta.items():
    print(f"  - {key}: {value}")

print("\n" + "=" * 60)
print("DataLoader信息:")
print("=" * 60)
print(f"  - 训练集batches: {len(train_loader)}")
if valid_loader:
    print(f"  - 验证集batches: {len(valid_loader)}")

print("\n" + "=" * 60)
print("测试第一个batch:")
print("=" * 60)

for batch_x, batch_y in train_loader:
    print(f"\n训练集第一个batch:")
    print(f"  - 输入形状: {batch_x.shape}")
    print(f"  - 输出形状: {batch_y.shape}")
    print(f"  - 输入通道数: {batch_x.shape[-1]} (应该是5)")
    print(f"  - 输出通道数: {batch_y.shape[-1]} (应该是1)")
    print(f"  - 输入值范围: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
    print(f"  - 输出值范围: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
    
    # 验证通道维度
    assert batch_x.shape[-1] == 5, f"输入通道数错误: 期望5, 实际{batch_x.shape[-1]}"
    assert batch_y.shape[-1] == 1, f"输出通道数错误: 期望1, 实际{batch_y.shape[-1]}"
    print("\n✅ 通道数验证通过!")
    break

if valid_loader:
    for batch_x, batch_y in valid_loader:
        print(f"\n验证集第一个batch:")
        print(f"  - 输入形状: {batch_x.shape}")
        print(f"  - 输出形状: {batch_y.shape}")
        print(f"  - 输入值范围: [{batch_x.min():.4f}, {batch_x.max():.4f}]")
        print(f"  - 输出值范围: [{batch_y.min():.4f}, {batch_y.max():.4f}]")
        break

print("\n" + "=" * 60)
print("归一化器信息:")
print("=" * 60)
print(f"  - x_normalizer.mean shape: {x_norm.mean.shape}")
print(f"  - x_normalizer.std shape: {x_norm.std.shape}")
print(f"  - x_normalizer.mean: {x_norm.mean}")
print(f"  - x_normalizer.std: {x_norm.std}")
print(f"\n  - y_normalizer.mean: {y_norm.mean}")
print(f"  - y_normalizer.std: {y_norm.std}")

print("\n" + "=" * 60)
print("✅ 所有测试通过!")
print("=" * 60)

