"""
组件分组模块

提供将普通组件和BC组件统一分组的函数。

关键约束：
- 每个BC组件只能出现在一个group中
- BC组件不能复制到多个group
"""

import torch
from typing import List, Tuple, Optional


def partition_components_with_bc(
    num_components: int,
    num_bc_components: int,
    num_groups: int,
    strategy: str = 'sequential',
) -> List[List[int]]:
    """
    将普通组件和BC组件统一分组
    
    Args:
        num_components: 普通组件数量K
        num_bc_components: BC组件数量M
        num_groups: 分组数量
        strategy: 分组策略
            - 'sequential': 顺序分组，BC组件放在最后
            - 'distribute': 将BC组件均匀分布到各group
    
    Returns:
        groups: list of component indices
                索引 0~K-1 为普通组件
                索引 K~K+M-1 为BC组件
    
    约束：
        - 每个BC组件只能出现在一个group中
        - BC组件不能复制到多个group
    """
    K = num_components
    M = num_bc_components
    total = K + M
    
    if strategy == 'sequential':
        # 顺序分组：BC组件放在最后，分配到最后几个group
        groups = _sequential_partition(K, M, num_groups)
    elif strategy == 'distribute':
        # 均匀分布：BC组件尽量均匀分布到各group
        groups = _distribute_partition(K, M, num_groups)
    else:
        raise ValueError(f"Unknown partition strategy: {strategy}")
    
    return groups


def _sequential_partition(
    K: int, M: int, num_groups: int
) -> List[List[int]]:
    """
    顺序分组策略
    
    先分配普通组件，再将BC组件分配到最后一个group中
    每个BC组件只能出现在一个group中
    """
    groups = []
    bc_start_idx = K  # BC组件的起始索引
    
    # 先分配普通组件
    components_per_group = K // num_groups
    remainder = K % num_groups
    
    idx = 0
    for g in range(num_groups):
        group_size = components_per_group + (1 if g < remainder else 0)
        group = list(range(idx, idx + group_size))
        groups.append(group)
        idx += group_size
    
    # 将所有BC组件分配到最后一个group
    # 每个BC组件只能出现一次
    if M > 0:
        groups[-1].extend(range(bc_start_idx, bc_start_idx + M))
    
    return groups


def _distribute_partition(
    K: int, M: int, num_groups: int
) -> List[List[int]]:
    """
    均匀分布策略
    
    将BC组件尽量均匀分布到各group中
    """
    groups = [[] for _ in range(num_groups)]
    bc_start_idx = K
    
    # 先分配普通组件（循环分配）
    for i in range(K):
        groups[i % num_groups].append(i)
    
    # 再分配BC组件（循环分配，确保每个BC只去一个group）
    for i in range(M):
        target_group = i % num_groups
        groups[target_group].append(bc_start_idx + i)
    
    return groups


def get_group_feature_list(
    component_features: List[torch.Tensor],
    bc_features: List[torch.Tensor],
    group_indices: List[int],
) -> List[torch.Tensor]:
    """
    根据分组索引获取对应的组件 feature 列表。

    Args:
        component_features: 普通组件列表，长度K
        bc_features: BC组件列表，长度M
        group_indices: 组件索引列表

    Returns:
        group_features: list of [B, H, W, C]
    """
    K = len(component_features)

    tensors = []
    for idx in group_indices:
        if idx < K:
            tensors.append(component_features[idx])
        else:
            bc_idx = idx - K
            tensors.append(bc_features[bc_idx])

    return tensors


def get_group_tensors(
    component_features: List[torch.Tensor],
    bc_features: List[torch.Tensor],
    group_indices: List[int],
) -> torch.Tensor:
    """
    根据分组索引获取对应的组件tensor
    
    Args:
        component_features: 普通组件列表，长度K
        bc_features: BC组件列表，长度M
        group_indices: 组件索引列表
    
    Returns:
        group_tensor: [B, H, W, sum of channels in group]
    """
    tensors = get_group_feature_list(component_features, bc_features, group_indices)
    return torch.cat(tensors, dim=-1)


def split_U_to_components(
    U: torch.Tensor,
    channel_num: int,
) -> List[torch.Tensor]:
    """
    将U拆分为组件特征列表
    
    Args:
        U: [B, H, W, K*channel_num]
        channel_num: 每个组件的通道数
    
    Returns:
        components: list of [B, H, W, channel_num]，长度K
    """
    K = U.shape[-1] // channel_num
    components = []
    for i in range(K):
        components.append(U[..., i * channel_num:(i + 1) * channel_num])
    return components


def verify_bc_partition(
    groups: List[List[int]],
    num_bc_components: int,
    bc_start_idx: int,
) -> bool:
    """
    验证BC组件分组是否满足约束
    
    约束：每个BC组件只能出现在一个group中
    
    Returns:
        True if valid, False otherwise
    """
    bc_indices = set()
    for group in groups:
        for idx in group:
            if idx >= bc_start_idx:
                if idx in bc_indices:
                    # BC组件重复出现
                    return False
                bc_indices.add(idx)
    
    # 检查是否所有BC组件都被分配
    expected_bc_indices = set(range(bc_start_idx, bc_start_idx + num_bc_components))
    return bc_indices == expected_bc_indices


class ComponentGrouper:
    """
    组件分组管理器
    
    管理普通组件和BC组件的分组逻辑
    """
    
    def __init__(
        self,
        num_components: int,
        num_bc_components: int,
        num_groups: int,
        strategy: str = 'sequential',
    ):
        """
        Args:
            num_components: 普通组件数量K
            num_bc_components: BC组件数量M
            num_groups: 分组数量
            strategy: 分组策略
        """
        self.K = num_components
        self.M = num_bc_components
        self.num_groups = num_groups
        self.strategy = strategy
        
        # 计算分组
        self.groups = partition_components_with_bc(
            self.K, self.M, self.num_groups, strategy
        )
        
        # 验证约束
        if not verify_bc_partition(self.groups, self.M, self.K):
            raise ValueError("BC partition constraint violated!")
    
    def get_group(self, group_id: int) -> List[int]:
        """获取指定group的组件索引"""
        return self.groups[group_id]
    
    def get_group_tensor(
        self,
        component_features: List[torch.Tensor],
        bc_features: List[torch.Tensor],
        group_id: int,
    ) -> torch.Tensor:
        """获取指定group的tensor"""
        group_indices = self.groups[group_id]
        return get_group_tensors(component_features, bc_features, group_indices)
    
    def __repr__(self) -> str:
        return (
            f"ComponentGrouper(K={self.K}, M={self.M}, "
            f"groups={self.num_groups}, strategy={self.strategy})\n"
            f"  groups: {self.groups}"
        )
