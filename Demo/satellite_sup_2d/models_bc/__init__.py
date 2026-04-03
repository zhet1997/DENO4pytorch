"""
BC组件化叠加模型模块

提供将边界条件BC编码为特殊组件的功能，与普通元件一起参与统一的分组、计算与叠加。

主要组件:
- SingleBCEncoder: 单BC编码器（将整个G编码为1个bc_component）
- MultiBCComponentEncoder: 多BC编码器（将G拆分为多个bc_component）
- supredictor_bc_component: BC组件化叠加预测器
- partition_components_with_bc: 组件分组函数
"""

from .bc_component_encoder import (
    BCComponentEncoderBase,
    SingleBCEncoder,
    MultiBCComponentEncoder,
)
from .supredictor_bc import supredictor_bc_component
from .component_partition import (
    partition_components_with_bc,
    get_group_tensors,
)

__all__ = [
    # 编码器
    'BCComponentEncoderBase',
    'SingleBCEncoder',
    'MultiBCComponentEncoder',
    # 模型
    'supredictor_bc_component',
    # 分组函数
    'partition_components_with_bc',
    'get_group_tensors',
]
