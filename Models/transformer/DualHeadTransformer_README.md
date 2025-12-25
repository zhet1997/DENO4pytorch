# DualHeadFourierTransformer 使用说明

## 概述

`DualHeadFourierTransformer` 是一个双头条件化 Transformer 模型，专门设计用于处理包含条件场（G）和源项场（U）的物理场预测问题。

### 核心特性

- **双头输入**: 分别处理条件场 G 和源项场 U
- **条件化机制**: 通过 CondLayerNorm (FiLM) 用 G 调制 U 的表征
- **无交叉注意力**: 不使用 cross-attention，保持结构简洁
- **2D/3D 兼容**: 支持 2D 和 3D 空间数据

## 与 FourierTransformer 的区别

| 特性 | FourierTransformer | DualHeadFourierTransformer |
|------|-------------------|---------------------------|
| 输入方式 | `node`: 拼接后的单一输入 | `G`, `U`: 分离的双输入 |
| 配置参数 | `node_feats` | `G_dim`, `U_dim` |
| 前向调用 | `model(node)` | `model(G, U)` |
| G/U 交互 | 通过 attention 隐式交互 | 通过 CondLN 显式调制 |
| Pos 编码 | 可选的 positional encoding | Pos 已编码在 G 中 |

## 快速开始

### 1. 基本使用

```python
import torch
from transformer.DualHeadTransformer import DualHeadFourierTransformer

# 配置模型
config = dict(
    G_dim=2,           # G 的通道数 (如 x, y 坐标)
    U_dim=3,           # U 的通道数 (源项场)
    n_targets=1,       # 输出通道数
    n_hidden=96,
    num_encoder_layers=4,
    n_head=4,
    attention_type='fourier',
    decoder_type='pointwise',
    spacial_dim=2,
)

# 创建模型
model = DualHeadFourierTransformer(**config)

# 前向传播
batch_size = 4
H, W = 64, 64
G = torch.randn(batch_size, 2, H, W)  # 条件场
U = torch.randn(batch_size, 3, H, W)  # 源项场
T = model(G, U)                        # 输出: [4, 1, 64, 64]
```

### 2. 从 FourierTransformer 迁移

如果你已有 FourierTransformer 的代码，需要做以下修改：

#### 旧代码 (FourierTransformer)

```python
# 配置
config = dict(
    node_feats=5,      # G(2通道) + U(3通道)
    n_targets=1,
    ...
)

# 数据准备
G = data[:, :2, :, :]   # [B, 2, H, W]
U = data[:, 2:5, :, :]  # [B, 3, H, W]
node = torch.cat([G, U], dim=1)  # [B, 5, H, W]

# 模型
model = FourierTransformer(**config)
output = model(node)
```

#### 新代码 (DualHeadFourierTransformer)

```python
# 配置
config = dict(
    G_dim=2,          # 分离 G 的通道数
    U_dim=3,          # 分离 U 的通道数
    n_targets=1,
    ...
)

# 数据准备 (保持 G 和 U 分离)
G = data[:, :2, :, :]   # [B, 2, H, W]
U = data[:, 2:5, :, :]  # [B, 3, H, W]

# 模型
model = DualHeadFourierTransformer(**config)
output = model(G, U)    # 直接传入两个张量
```

### 3. 训练示例

完整的训练脚本请参考：
- `Demo/satellite_sup_2d/train_DualHead_satellite_example.py`

关键代码：

```python
for epoch in range(epochs):
    for x_batch, y_batch in train_loader:
        # 假设 x_batch 包含拼接的 [G, U]
        G = x_batch[:, :G_dim, :, :]
        U = x_batch[:, G_dim:G_dim+U_dim, :, :]
        
        # 前向传播
        pred = model(G, U)
        loss = criterion(pred, y_batch)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 配置参数详解

### 必需参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `G_dim` | int | G 的通道数（含 pos 信息） | `2` (x, y 坐标) |
| `U_dim` | int | U 的通道数 | `3` (源项场) |
| `n_targets` | int | 输出通道数 | `1` (温度场) |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `n_hidden` | int | `96` | Token 隐藏维度 |
| `num_encoder_layers` | int | `4` | Encoder 层数 |
| `n_head` | int | `4` | 注意力头数 |
| `dim_feedforward` | int | `2*n_hidden` | FeedForward 中间维度 |
| `attention_type` | str | `'fourier'` | 注意力类型: 'fourier', 'galerkin', 'linear', 'softmax' |
| `decoder_type` | str | `'pointwise'` | 解码器类型: 'pointwise', 'ifft' |
| `num_regressor_layers` | int | `2` | Regressor 层数 |
| `spacial_dim` | int | `2` | 空间维度: 2 (2D) 或 3 (3D) |
| `dropout` | float | `0.05` | Dropout 比例 |
| `activation_type` | str | `'silu'` | 激活函数: 'silu', 'relu', 'gelu', 'tanh' |
| `debug` | bool | `False` | 是否打印调试信息 |

## 架构详解

### 前向传播流程

```
G [B, G_dim, H, W] ──┐
                     ├─► Embeddings ─► Tokens [B, N, hidden]
U [B, U_dim, H, W] ──┘                    │
                                          │
                     g_tok [B, N, hidden] │ (条件)
                                          ↓
                     u_tok [B, N, hidden] ─► Conditional Encoder Layers
                                          │   (用 g_tok 调制 u_tok)
                                          │
                                          ↓
                     u_tok [B, N, hidden]
                                          │
                                          ↓
                     Reshape ─► [B, H, W, hidden]
                                          │
                                          ↓
                     Regressor
                                          │
                                          ↓
                     T [B, n_targets, H, W]
```

### ConditionalEncoderLayer 结构

每个 Conditional Encoder Layer 采用 PostNorm 风格：

```
x (u_tok)  ───┬─► Attention ─► Dropout ─┬─► CondLN1(·, cond=g_tok) ─┬─► FeedForward ─► Dropout ─┬─► CondLN2(·, cond=g_tok) ─► out
              │                          │                            │                           │
              └──────────────────────────┘                            └───────────────────────────┘
                  (残差连接)                                                (残差连接)
```

### CondLayerNorm 机制 (FiLM)

```python
# 1. 标准归一化
x_norm = LayerNorm(x, elementwise_affine=False)

# 2. 从条件预测 gamma 和 beta
gamma, beta = Linear(cond) -> split into 2

# 3. 调制
gamma = 1.0 + gamma  # 初始化为 1
output = gamma * x_norm + beta
```

## 常见问题

### Q1: 为什么会报错 "Required parameter 'G_dim' is missing"?

**原因**: 你可能使用了 FourierTransformer 的配置，它使用 `node_feats` 而不是 `G_dim` 和 `U_dim`。

**解决**: 将配置从 `node_feats` 拆分为 `G_dim` 和 `U_dim`。

### Q2: G 和 U 的通道数应该如何确定？

**G (条件场)**: 
- 包含位置信息（如 x, y 坐标）
- 可能包含物理尺度信息
- 不可叠加的场变量

**U (源项场)**:
- 可叠加的源项或输入场
- 问题的主要变量

### Q3: 如何处理已拼接的数据？

如果你的数据加载器返回拼接的数据 `[B, G_dim+U_dim, H, W]`：

```python
# 在训练循环中手动拆分
x_batch = next(train_loader)  # [B, 5, H, W]
G = x_batch[:, :2, :, :]      # [B, 2, H, W]
U = x_batch[:, 2:, :, :]      # [B, 3, H, W]
output = model(G, U)
```

### Q4: 3D 数据如何使用？

只需设置 `spacial_dim=3` 并传入 3D 张量：

```python
config = dict(
    G_dim=3,          # 3D pos (x, y, z)
    U_dim=4,
    n_targets=1,
    spacial_dim=3,    # 关键：设置为 3
    ...
)

G = torch.randn(B, 3, H, W, D)  # 3D 条件场
U = torch.randn(B, 4, H, W, D)  # 3D 源项场
T = model(G, U)                 # [B, 1, H, W, D]
```

## 性能建议

1. **批量大小**: 根据显存调整，典型值 4-16
2. **隐藏维度**: 96 (小规模) 到 256 (大规模)
3. **层数**: 4-8 层通常足够
4. **注意力类型**: 
   - `fourier`: 适合光滑场，计算高效
   - `galerkin`: 更灵活，适合复杂场
   - `softmax`: 经典 attention，计算较慢

## 相关文件

- **模型实现**: `Models/transformer/DualHeadTransformer.py`
- **使用示例**: `Demo/satellite_sup_2d/train_DualHead_satellite_example.py`
- **模块导出**: `Models/transformer/__init__.py`

## 引用

如果你使用了这个模型，请引用相关论文（待补充）。

