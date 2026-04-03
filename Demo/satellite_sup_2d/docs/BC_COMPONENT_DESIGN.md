# BC组件化叠加框架设计文档

## 概述

本文档描述了BC组件化叠加框架的设计，该框架将边界条件BC编码为特殊组件，与普通元件一起参与统一的分组、计算与叠加。

## 问题背景

### 原始叠加框架的问题

原始叠加框架隐含了一个很强的前提：**所有样本共享相同边界条件**。

当边界条件随样本变化时，叠加网络面对的子问题可能来自不同的边界条件可行域，导致叠加语义失效。

### 解决方案

将BC编码为特殊组件，加入组件集合后统一分组：

```
原: C = {c1, c2, ..., cK}        # 只有普通元件
新: C = {c1, c2, ..., cK, bc}    # 加入BC组件
```

---

## 架构设计

### 模块结构

```
models_bc/
├── __init__.py              # 模块导出
├── bc_component_encoder.py  # BC编码器
├── supredictor_bc.py        # BC组件化预测器
└── component_partition.py   # 组件分组逻辑
```

### 数据流

```
G [B,H,W,G_ch] ──────────────────────────────┐
                                              │
U [B,H,W,K*ch] → split → [c1,...,cK]         │
                          │                   │
                          ↓                   ↓
                    ┌─────────────────────────────┐
                    │   BC Encoder (shared)       │
                    │   G → [bc1,...,bcM]         │
                    └─────────────────────────────┘
                                   │
                                   ↓
                    [c1,...,cK, bc1,...,bcM]
                                   │
                                   ↓
                    ┌─────────────────────────────┐
                    │   Partition (统一分组)       │
                    │   每个BC只出现在一个group   │
                    └─────────────────────────────┘
                                   │
                                   ↓
                    ┌─────────────────────────────┐
                    │   pred_net + super_net      │
                    └─────────────────────────────┘
                                   │
                                   ↓
                            T [B,H,W,1]
```

---

## 核心组件

### 1. BC编码器

**SingleBCEncoder** - 单BC模式（当前数据格式）：
```python
# G: [B, H, W, 4]
# 输出: [bc_component] (长度1的列表)

bc_encoder = SingleBCEncoder(
    G_channels=4,
    bc_dim=16,
    encoder_type='linear'  # 或 'cnn'
)
```

**MultiBCComponentEncoder** - 多BC模式（新数据格式）：
```python
# G: [B, H, W, 4*M] (M=散热窗数量)
# 输出: [bc1, ..., bcM] (长度M的列表)

bc_encoder = MultiBCComponentEncoder(
    G_channels=4,           # 每个BC组件的通道数
    bc_dim=16,
    num_bc_components=M,    # BC组件数量
    encoder_type='linear'
)
```

### 2. 组件分组

**约束**：每个BC组件只能出现在一个group中

```python
groups = partition_components_with_bc(
    num_components=K,       # 普通组件数量
    num_bc_components=M,    # BC组件数量
    num_groups=K,           # 分组数量
    strategy='sequential',  # 或 'distribute'
)

# 返回: [[idx1, idx2, ...], ...]
# 索引 0~K-1 为普通组件，K~K+M-1 为BC组件
```

### 3. BC组件化预测器

```python
model = supredictor_bc_component(
    pred_net=pred_model,
    super_net=super_model,
    channel_num=16,
    G_channels=4,
    bc_dim=16,
    num_bc_components=1,        # 1=单BC, M=多BC
    encoder_type='linear',
    partition_strategy='sequential',
)
```

---

## 使用方法

### 单BC模式（当前数据格式）

```bash
python Demo/satellite_sup_2d/train_entry_bc.py \
    --work_name test_bc_single \
    --train_component_nums 1,2,3,4,5 \
    --super_train_mode S0 \
    --epochs 200 \
    --num_bc_components 1 \
    --bc_dim 16 \
    --bc_encoder_type linear
```

### 多BC模式（新数据格式）

当您准备好新数据格式（G包含每个散热窗的独立BC）后：

```bash
python Demo/satellite_sup_2d/train_entry_bc.py \
    --work_name test_bc_multi \
    --train_component_nums 1,2,3,4,5 \
    --super_train_mode S0 \
    --epochs 200 \
    --num_bc_components M \
    --bc_dim 16 \
    --bc_encoder_type linear
```

---

## 参数说明

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `--num_bc_components` | 1 | BC组件数量（1=单BC, M=多BC） |
| `--bc_dim` | 16 | BC组件输出维度 |
| `--bc_encoder_type` | linear | BC编码器类型（linear/cnn） |
| `--partition_strategy` | sequential | 分组策略（sequential/distribute） |

---

## 扩展指南

### 添加新的编码器类型

在 `bc_component_encoder.py` 中添加新的编码器类：

```python
class MyCustomEncoder(BCComponentEncoderBase):
    def __init__(self, G_channels, bc_dim, ...):
        super().__init__()
        # 你的编码器实现
    
    def forward(self, G):
        # 返回 bc_components 列表
        pass
    
    @property
    def num_bc_components(self):
        return self._num_bc
    
    @property
    def bc_dim(self):
        return self._bc_dim
```

### 修改分组策略

在 `component_partition.py` 中添加新的分组函数：

```python
def _my_partition(K, M, num_groups):
    """
    自定义分组策略
    
    约束：每个BC组件只能出现在一个group中
    """
    # 你的分组逻辑
    return groups
```

---

## 注意事项

1. **BC组件数量约束**：
   - 单BC模式：G通道数 = 4
   - 多BC模式：G通道数 = 4 * num_bc_components

2. **分组约束**：
   - 每个BC组件只能出现在一个group中
   - BC组件不能复制到多个group

3. **数据格式**：
   - 当前数据格式：G已合并所有散热窗BC
   - 新数据格式：需要为每个散热窗独立存储BC

---

## 版本历史

- v1.0 (2026-01): 初始实现，支持单BC模式
- 后续: 支持多BC模式
