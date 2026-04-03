"""
BC组件化叠加预测器

将边界条件BC编码为特殊组件，与普通元件一起参与统一的分组、计算与叠加。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Tuple, Dict

from Demo.satellite_sup_2d.trains_satellite import split_satellite_g_inputs
from .bc_component_encoder import (
    BCComponentEncoderBase,
    SingleBCEncoder,
    MultiBCComponentEncoder,
    create_bc_encoder,
)
from .component_partition import (
    partition_components_with_bc,
    get_group_feature_list,
    split_U_to_components,
    ComponentGrouper,
)


class supredictor_bc_component(nn.Module):
    """
    BC组件化叠加预测器
    
    将BC编码为特殊组件，与普通元件一起参与分组、计算与叠加
    """
    
    def __init__(
        self,
        pred_net: nn.Module,
        super_net: nn.Module,
        channel_num: int = 16,
        G_channels: int = 1,
        bc_dim: int = 16,
        num_bc_components: int = 1,
        encoder_type: str = 'linear',
        partition_strategy: str = 'sequential',
        win_split: int = 1,
        g_meta: Optional[Dict] = None,
    ):
        super().__init__()
        self.pred_net = pred_net
        self.super_net = super_net
        self.channel_num = channel_num
        self.num_bc_components = num_bc_components
        self.partition_strategy = partition_strategy
        self.win_split = win_split
        self.g_meta = g_meta

        self.bc_encoder = create_bc_encoder(
            G_channels=G_channels,
            bc_dim=bc_dim,
            num_bc_components=num_bc_components,
            encoder_type=encoder_type,
        )
        self.bc_dim = bc_dim
        self.predictor_u_dim = getattr(pred_net, 'U_dim', None)
        if self.predictor_u_dim is not None:
            if self.channel_num != self.predictor_u_dim:
                raise ValueError(
                    f"channel_num({self.channel_num})必须与predictor U_dim({self.predictor_u_dim})一致。"
                )
            if self.bc_dim != self.predictor_u_dim:
                raise ValueError(
                    f"bc_dim({self.bc_dim})必须与predictor U_dim({self.predictor_u_dim})一致，"
                    "当前 BC 聚合实现依赖所有组件宽度一致。"
                )
    
    def _aggregate_group_features(self, group_features: List[torch.Tensor]) -> torch.Tensor:
        if len(group_features) == 0:
            raise ValueError("group_features不能为空。")

        aggregated = torch.stack(group_features, dim=0).mean(dim=0)
        if self.predictor_u_dim is not None and aggregated.shape[-1] != self.predictor_u_dim:
            raise ValueError(
                f"聚合后group特征通道数({aggregated.shape[-1]})与predictor U_dim({self.predictor_u_dim})不一致。"
            )
        return aggregated

    def forward(self, G: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        parsed_G = split_satellite_g_inputs(G, self.g_meta)
        global_G = parsed_G['global_G']
        bc_sdf = parsed_G['bc_sdf'] if parsed_G['bc_sdf'] is not None else G

        bc_features = self.bc_encoder(bc_sdf)
        component_features = split_U_to_components(U, self.channel_num)
        
        K = len(component_features)
        M = len(bc_features)
        num_groups = max(K, 1)
        
        groups = partition_components_with_bc(
            num_components=K,
            num_bc_components=M,
            num_groups=num_groups,
            strategy=self.partition_strategy,
        )
        
        field_list = []
        for group_indices in groups:
            group_features = get_group_feature_list(
                component_features, bc_features, group_indices
            )
            group_tensor = self._aggregate_group_features(group_features)
            pred = self.pred_net(global_G, group_tensor)
            field_list.append(pred)
        
        if len(field_list) == 1:
            return field_list[0]
        
        return self._superpose(field_list)
    
    def _superpose(self, field_list: List[torch.Tensor]) -> torch.Tensor:
        fields = torch.cat(field_list, dim=-1)
        return fields.mean(dim=-1, keepdim=True)
    
    def forward_c_only(self, G: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """C-only模式"""
        parsed_G = split_satellite_g_inputs(G, self.g_meta)
        global_G = parsed_G['global_G']
        bc_sdf = parsed_G['bc_sdf'] if parsed_G['bc_sdf'] is not None else G

        bc_features = self.bc_encoder(bc_sdf)
        component_features = split_U_to_components(U, self.channel_num)

        preds = []
        for comp in component_features:
            if self.predictor_u_dim is not None and comp.shape[-1] != self.predictor_u_dim:
                raise ValueError(
                    f"组件通道数({comp.shape[-1]})与predictor U_dim({self.predictor_u_dim})不一致。"
                )
            pred = self.pred_net(global_G, comp)
            preds.append(pred)
        
        if len(preds) == 0:
            return torch.zeros(G.shape[0], G.shape[1], G.shape[2], 1, device=G.device)
        return torch.stack(preds, dim=0).mean(dim=0)
