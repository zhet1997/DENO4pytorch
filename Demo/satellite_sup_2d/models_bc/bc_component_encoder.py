"""
BC组件编码器模块

提供将边界条件G编码为BC组件的编码器:
- BCComponentEncoderBase: 抽象基类
- SingleBCEncoder: 单BC编码器（将整个G编码为1个bc_component）
- MultiBCComponentEncoder: 多BC组件编码器（逐个 cooling window 编码，共享权重）
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List


class BCComponentEncoderBase(ABC, nn.Module):
    """
    BC组件编码器基类
    
    定义了BC编码器的统一接口，返回bc_component列表
    """
    
    @abstractmethod
    def forward(self, G: torch.Tensor) -> List[torch.Tensor]:
        """
        将G编码为BC组件列表
        
        Args:
            G: [B, H, W, G_channels] 边界条件张量
        
        Returns:
            bc_components: list of [B, H, W, bc_dim]
        """
        pass
    
    @property
    @abstractmethod
    def num_bc_components(self) -> int:
        """返回BC组件数量"""
        pass
    
    @property
    @abstractmethod
    def bc_dim(self) -> int:
        """返回BC组件输出维度"""
        pass


class SingleBCEncoder(BCComponentEncoderBase):
    """
    单BC编码器（第一阶段：当前数据格式）
    
    将整个G编码为1个bc_component
    适用于：当前数据格式，G包含合并后的边界条件信息
    """
    
    def __init__(
        self,
        G_channels: int = 4,
        bc_dim: int = 16,
        encoder_type: str = 'linear',
    ):
        """
        Args:
            G_channels: G的输入通道数（默认4）
            bc_dim: BC组件输出维度
            encoder_type: 编码器类型，'linear' 或 'cnn'
        """
        super().__init__()
        self._num_bc = 1
        self._bc_dim = bc_dim
        self.G_channels = G_channels
        self.encoder_type = encoder_type
        
        if encoder_type == 'linear':
            # 简单线性投影
            self.encoder = nn.Linear(G_channels, bc_dim)
        elif encoder_type == 'cnn':
            # 小型CNN编码器
            self.encoder = nn.Sequential(
                nn.Conv2d(G_channels, bc_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(bc_dim, bc_dim, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    def forward(self, G: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            G: [B, H, W, G_channels]
        
        Returns:
            [bc_component]: 长度1的列表，元素为 [B, H, W, bc_dim]
        """
        if self.encoder_type == 'linear':
            # [B, H, W, G_channels] -> [B, H, W, bc_dim]
            bc = self.encoder(G)
        else:
            # CNN需要 [B, C, H, W] 格式
            B, H, W, C = G.shape
            G_permuted = G.permute(0, 3, 1, 2)  # [B, C, H, W]
            bc = self.encoder(G_permuted)  # [B, bc_dim, H, W]
            bc = bc.permute(0, 2, 3, 1)  # [B, H, W, bc_dim]
        
        return [bc]
    
    @property
    def num_bc_components(self) -> int:
        return self._num_bc
    
    @property
    def bc_dim(self) -> int:
        return self._bc_dim


class MultiBCComponentEncoder(BCComponentEncoderBase):
    """
    多BC组件编码器（第二阶段：新数据格式）

    将每个 cooling window 的 sdf 单独编码为一个 bc_component，所有组件共享编码器权重。
    """
    
    def __init__(
        self,
        G_channels: int = 4,
        bc_dim: int = 16,
        num_bc_components: int = 1,
        encoder_type: str = 'linear',
    ):
        """
        Args:
            G_channels: 每个BC组件的输入通道数（_mc 数据下固定为1个 sdf 通道）
            bc_dim: BC组件输出维度
            num_bc_components: BC组件数量M（=散热窗数量）
            encoder_type: 编码器类型
        """
        super().__init__()
        self._num_bc = num_bc_components
        self._bc_dim = bc_dim
        self.G_channels = G_channels
        self.encoder_type = encoder_type
        
        # 共享编码器
        if encoder_type == 'linear':
            self.shared_encoder = nn.Linear(G_channels, bc_dim)
        elif encoder_type == 'cnn':
            self.shared_encoder = nn.Sequential(
                nn.Conv2d(G_channels, bc_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(bc_dim, bc_dim, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    def forward(self, G: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            G: [B, H, W, G_channels * num_bc_components]
        
        Returns:
            bc_components: list of [B, H, W, bc_dim]，长度M
        """
        # 1. 验证通道数
        expected_channels = self.G_channels * self._num_bc
        if G.shape[-1] != expected_channels:
            raise ValueError(
                f"G channels mismatch: expected {expected_channels}, "
                f"got {G.shape[-1]}. "
                f"G_channels={self.G_channels}, num_bc={self._num_bc}"
            )
        
        # 2. 将G拆分为M个部分
        G_split = torch.chunk(G, self._num_bc, dim=-1)
        
        # 3. 使用共享编码器分别编码
        bc_components = []
        for g_i in G_split:
            if self.encoder_type == 'linear':
                bc = self.shared_encoder(g_i)
            else:
                B, H, W, C = g_i.shape
                g_permuted = g_i.permute(0, 3, 1, 2)
                bc = self.shared_encoder(g_permuted)
                bc = bc.permute(0, 2, 3, 1)
            bc_components.append(bc)
        
        return bc_components
    
    @property
    def num_bc_components(self) -> int:
        return self._num_bc
    
    @property
    def bc_dim(self) -> int:
        return self._bc_dim


def create_bc_encoder(
    G_channels: int = 4,
    bc_dim: int = 16,
    num_bc_components: int = 1,
    encoder_type: str = 'linear',
) -> BCComponentEncoderBase:
    """
    工厂函数：根据参数创建对应的BC编码器
    
    Args:
        G_channels: G的输入通道数（单BC）或每个BC组件的通道数（多BC）
        bc_dim: BC组件输出维度
        num_bc_components: BC组件数量
        encoder_type: 编码器类型
    
    Returns:
        BCComponentEncoderBase: 对应的编码器实例
    """
    if num_bc_components == 1:
        return SingleBCEncoder(
            G_channels=G_channels,
            bc_dim=bc_dim,
            encoder_type=encoder_type,
        )
    else:
        return MultiBCComponentEncoder(
            G_channels=G_channels,
            bc_dim=bc_dim,
            num_bc_components=num_bc_components,
            encoder_type=encoder_type,
        )
