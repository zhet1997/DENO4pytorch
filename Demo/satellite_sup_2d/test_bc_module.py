"""
BC组件化模块测试脚本
测试各组件是否能正常工作
"""

import sys
import torch
import torch.nn as nn

sys.path.append('/data/wqn/Code/DENO4pytorch')

from Demo.satellite_sup_2d.models_bc import (
    SingleBCEncoder,
    MultiBCComponentEncoder,
    supredictor_bc_component,
    partition_components_with_bc,
)
from Demo.satellite_sup_2d.models_bc.bc_component_encoder import create_bc_encoder
from Demo.satellite_sup_2d.models_bc.component_partition import get_group_feature_list


def test_single_bc_encoder():
    """测试单BC编码器"""
    print("\n=== 测试 SingleBCEncoder ===")
    
    encoder = SingleBCEncoder(
        G_channels=4,
        bc_dim=16,
        encoder_type='linear'
    )
    
    G = torch.randn(2, 32, 32, 4)
    bc_features = encoder(G)
    
    print(f"输入 G: {G.shape}")
    print(f"输出 bc_features: 长度 {len(bc_features)}, 形状 {bc_features[0].shape}")
    print(f"num_bc_components: {encoder.num_bc_components}")
    print(f"bc_dim: {encoder.bc_dim}")
    
    assert len(bc_features) == 1
    assert bc_features[0].shape == (2, 32, 32, 16)
    print("测试通过!")


def test_multi_bc_encoder():
    """测试多BC编码器"""
    print("\n=== 测试 MultiBCComponentEncoder ===")

    M = 2
    encoder = MultiBCComponentEncoder(
        G_channels=1,
        bc_dim=16,
        num_bc_components=M,
        encoder_type='linear'
    )

    G = torch.randn(2, 32, 32, M)
    bc_features = encoder(G)

    print(f"输入 G: {G.shape}")
    print(f"输出 bc_features: 长度 {len(bc_features)}")
    for i, bc in enumerate(bc_features):
        print(f"  bc[{i}]: {bc.shape}")
    print(f"num_bc_components: {encoder.num_bc_components}")

    assert len(bc_features) == M
    for bc in bc_features:
        assert bc.shape == (2, 32, 32, 16)
    print("测试通过!")


def test_partition():
    """测试分组函数"""
    print("\n=== 测试 partition_components_with_bc ===")
    
    # 测试1: K=4, M=1
    groups = partition_components_with_bc(
        num_components=4,
        num_bc_components=1,
        num_groups=4,
        strategy='sequential',
    )
    print(f"K=4, M=1: {groups}")
    
    # 验证BC组件只出现一次
    bc_indices = []
    for g in groups:
        for idx in g:
            if idx >= 4:
                bc_indices.append(idx)
    print(f"BC组件索引: {bc_indices}, 去重后: {set(bc_indices)}")
    assert len(bc_indices) == 1  # 只有1个BC组件
    print("测试通过!")
    
    # 测试2: K=4, M=2
    groups = partition_components_with_bc(
        num_components=4,
        num_bc_components=2,
        num_groups=4,
        strategy='sequential',
    )
    print(f"K=4, M=2: {groups}")
    print("测试通过!")


def test_group_aggregation():
    """测试 group 聚合后保持固定通道数"""
    print("\n=== 测试 group 聚合 ===")

    class SimplePredNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.U_dim = 16
            self.fc = nn.Linear(18, 1)

        def forward(self, G, component):
            x = torch.cat([G, component], dim=-1)
            return self.fc(x)

    class SimpleSuperNet(nn.Module):
        def forward(self, x):
            return x

    model = supredictor_bc_component(
        pred_net=SimplePredNet(),
        super_net=SimpleSuperNet(),
        channel_num=16,
        G_channels=1,
        bc_dim=16,
        num_bc_components=2,
        encoder_type='linear',
        g_meta={
            'G_channels': 5,
            'data_variant': 'mc',
            'G_channel_names': ['cooling_sdf_0', 'cooling_sdf_1', 'cooling_temp', 'coord_x', 'coord_y'],
        },
    )

    group_features = [
        torch.randn(2, 32, 32, 16),
        torch.randn(2, 32, 32, 16),
        torch.randn(2, 32, 32, 16),
    ]
    aggregated = model._aggregate_group_features(group_features)
    print(f"聚合后形状: {aggregated.shape}")
    assert aggregated.shape == (2, 32, 32, 16)
    print("测试通过!")


def test_model_instantiation():
    """测试模型实例化"""
    print("\n=== 测试 supredictor_bc_component 实例化 ===")

    class SimplePredNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.U_dim = 16
            self.fc = nn.Linear(18, 1)

        def forward(self, G, component):
            x = torch.cat([G, component], dim=-1)
            return self.fc(x)

    class SimpleSuperNet(nn.Module):
        def forward(self, x):
            return x

    model = supredictor_bc_component(
        pred_net=SimplePredNet(),
        super_net=SimpleSuperNet(),
        channel_num=16,
        G_channels=1,
        bc_dim=16,
        num_bc_components=2,
        encoder_type='linear',
        g_meta={
            'G_channels': 5,
            'data_variant': 'mc',
            'G_channel_names': ['cooling_sdf_0', 'cooling_sdf_1', 'cooling_temp', 'coord_x', 'coord_y'],
        },
    )

    print(f"模型创建成功")
    print(f"  - channel_num: {model.channel_num}")
    print(f"  - num_bc_components: {model.num_bc_components}")
    print(f"  - bc_dim: {model.bc_dim}")

    G = torch.randn(2, 32, 32, 5)
    U = torch.randn(2, 32, 32, 32)

    print(f"\n前向传播测试:")
    print(f"  - G: {G.shape}")
    print(f"  - U: {U.shape}")

    output = model(G, U)
    print(f"  - output: {output.shape}")
    assert output.shape == (2, 32, 32, 1)
    print("前向传播成功!")
    print("实例化测试通过!")




def test_factory_function():
    """测试工厂函数"""
    print("\n=== 测试 create_bc_encoder 工厂函数 ===")
    
    # 单BC
    encoder1 = create_bc_encoder(
        G_channels=4, bc_dim=16, num_bc_components=1, encoder_type='linear'
    )
    print(f"单BC编码器: {type(encoder1).__name__}")
    assert isinstance(encoder1, SingleBCEncoder)
    
    # 多BC
    encoder2 = create_bc_encoder(
        G_channels=1, bc_dim=16, num_bc_components=2, encoder_type='linear'
    )
    print(f"多BC编码器: {type(encoder2).__name__}")
    assert isinstance(encoder2, MultiBCComponentEncoder)

    print("工厂函数测试通过!")


if __name__ == "__main__":
    print("=" * 60)
    print("BC组件化模块测试")
    print("=" * 60)

    test_single_bc_encoder()
    test_multi_bc_encoder()
    test_partition()
    test_group_aggregation()
    test_model_instantiation()
    test_factory_function()

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
