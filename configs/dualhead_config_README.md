# DualHeadFourierTransformer 配置文件使用说明

## 配置文件位置

`data/configs/dualhead_transformer_config_sate.yml`

## 可用配置

| 配置名称 | 用途 | 参数量 | 解码器 | 适用场景 |
|---------|------|--------|--------|---------|
| `DualHead_GUT_2d` | 基础 GUT 配置 | ~350K | ifft | 卫星数据，平衡性能 |
| `DualHead_Small_2d` | 小型模型 | ~26K | pointwise | 快速实验，资源受限 |
| `DualHead_Base_2d` | 标准模型 | ~429K | pointwise | 大多数应用，推荐 |
| `DualHead_Large_2d` | 大型模型 | ~11M | ifft | 追求最佳性能 |
| `DualHead_3D` | 3D 数据 | ~146K | pointwise | 三维场数据 |
| `DualHead_MultiChannel_2d` | 多通道 | ~429K | pointwise | 多输入多输出 |

## 使用方法

### 方法 1: Python 脚本加载

```python
import yaml
from transformer.DualHeadTransformer import DualHeadFourierTransformer

# 加载配置
with open('data/configs/dualhead_transformer_config_sate.yml', 'r') as f:
    configs = yaml.full_load(f)

# 选择配置
config = configs['DualHead_Base_2d']

# 创建模型
model = DualHeadFourierTransformer(**config)
```

### 方法 2: 使用示例训练脚本

```bash
cd /data/wqn/DENO4pytorch

# 使用默认配置 (DualHead_GUT_2d)
python Demo/satellite_sup_2d/train_DualHead_with_config.py

# 使用其他配置
python Demo/satellite_sup_2d/train_DualHead_with_config.py --config DualHead_Base_2d

# 自定义训练参数
python Demo/satellite_sup_2d/train_DualHead_with_config.py \
    --config DualHead_Small_2d \
    --batch_size 32 \
    --epochs 300 \
    --lr 1e-3
```

## 配置参数说明

### 必需参数

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `G_dim` | G 的通道数（条件场） | `4` (x, y, scale_x, scale_y) |
| `U_dim` | U 的通道数（源项场） | `1` |
| `n_targets` | 输出通道数 | `1` |

### 网络结构

| 参数 | 说明 | 典型值 | 备注 |
|------|------|--------|------|
| `n_hidden` | Token 隐藏维度 | 32-128 | 影响模型容量 |
| `num_encoder_layers` | Encoder 层数 | 2-6 | 越多越慢 |
| `n_head` | 注意力头数 | 2-8 | 需整除 n_hidden |
| `dim_feedforward` | FFN 中间维度 | n_hidden 的 2-4 倍 | |

### 注意力配置

| 参数 | 说明 | 可选值 | 推荐 |
|------|------|--------|------|
| `attention_type` | 注意力类型 | fourier, galerkin, linear, softmax | fourier (快速) 或 galerkin (灵活) |
| `xavier_init` | Xavier 初始化增益 | 0.01 | 默认即可 |
| `diagonal_weight` | 对角权重 | 0.01 | 默认即可 |

### 解码器配置

| 参数 | 说明 | 可选值 | 备注 |
|------|------|--------|------|
| `decoder_type` | 解码器类型 | pointwise, ifft | pointwise 更通用 |
| `num_regressor_layers` | Regressor 层数 | 1-3 | |
| `freq_dim` | 频域维度 (ifft) | 32-128 | 仅 ifft 需要 |
| `fourier_modes` | Fourier modes (ifft) | 8-16 | 须 < 输入尺寸 |

**重要**: 使用 `ifft` decoder 时：
- `fourier_modes` 必须小于输入的空间尺寸
- 例如: `modes=8` 适用于 16×16 输入，`modes=12` 需要 ≥32×32 输入
- 如果输入尺寸不确定，建议使用 `pointwise` decoder

### 空间配置

| 参数 | 说明 | 可选值 | 备注 |
|------|------|--------|------|
| `spacial_dim` | 空间维度 | 2 (2D), 3 (3D) | |
| `spacial_fc` | Regressor 使用 spatial FC | True, False | 通常 False (pos 已在 G 中) |

### 正则化

| 参数 | 说明 | 典型值 | 备注 |
|------|------|--------|------|
| `dropout` | 全局 dropout | 0.0-0.1 | 0.0 无 dropout |
| `encoder_dropout` | Encoder dropout | 0.0-0.1 | |
| `decoder_dropout` | Decoder dropout | 0.0-0.1 | |
| `ffn_dropout` | FFN dropout | 0.0-0.1 | |

### 激活函数

| 参数 | 说明 | 可选值 | 推荐 |
|------|------|--------|------|
| `activation_type` | Encoder 激活 | silu, relu, gelu, tanh | silu 或 gelu |
| `regressor_activation` | Regressor 激活 | silu, relu, gelu, tanh | 与 encoder 一致 |

### 其他

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `pos_dim` | 位置编码维度 | 通常 0 (pos 已在 G 中) |
| `return_latent` | 返回中间层特征 | True, False |
| `debug` | 打印调试信息 | True, False |

## 自定义配置

### 添加新配置

在 `dualhead_transformer_config_sate.yml` 中添加：

```yaml
MyCustom_2d:
  G_dim: 2              # 自定义 G 通道数
  U_dim: 4              # 自定义 U 通道数
  n_targets: 2          # 自定义输出通道数
  n_hidden: 64
  num_encoder_layers: 3
  n_head: 4
  dim_feedforward: 128
  attention_type: fourier
  decoder_type: pointwise
  num_regressor_layers: 2
  spacial_dim: 2
  spacial_fc: False
  dropout: 0.05
  encoder_dropout: 0.05
  decoder_dropout: 0.05
  ffn_dropout: 0.05
  activation_type: silu
  regressor_activation: silu
  pos_dim: 0
  xavier_init: 0.01
  diagonal_weight: 0.01
  symmetric_init: False
  norm_eps: 0.00001
  return_latent: False
  debug: False
```

### 使用自定义配置

```python
# 方法 1: 从文件加载
config = load_dualhead_config('MyCustom_2d')
model = DualHeadFourierTransformer(**config)

# 方法 2: 命令行
python train_DualHead_with_config.py --config MyCustom_2d
```

## 配置选择指南

### 根据资源选择

| 资源情况 | 推荐配置 | 参数量 | 备注 |
|---------|---------|--------|------|
| GPU < 4GB | DualHead_Small_2d | ~26K | 快速实验 |
| GPU 4-8GB | DualHead_Base_2d | ~429K | **推荐** |
| GPU > 8GB | DualHead_Large_2d | ~11M | 最佳性能 |

### 根据任务选择

| 任务类型 | 推荐配置 | 理由 |
|---------|---------|------|
| 2D 场预测 (标准) | DualHead_Base_2d | 平衡性能和速度 |
| 2D 场预测 (高精度) | DualHead_GUT_2d 或 Large | ifft decoder 更适合光滑场 |
| 3D 场预测 | DualHead_3D | 针对 3D 优化 |
| 多变量预测 | DualHead_MultiChannel_2d | 支持多输入多输出 |
| 快速原型 | DualHead_Small_2d | 训练快，迭代快 |

### 根据数据特征选择

| 数据特征 | 建议 |
|---------|------|
| 光滑、周期性场 | `attention_type: fourier`, `decoder_type: ifft` |
| 复杂、非光滑场 | `attention_type: galerkin`, `decoder_type: pointwise` |
| 小数据集 | 减少 `num_encoder_layers`, 增加 `dropout` |
| 大数据集 | 增加 `num_encoder_layers`, 减少 `dropout` |

## 常见问题

### Q1: 如何修改输入通道数？

修改 `G_dim` 和 `U_dim`：

```yaml
G_dim: 3    # 例如只有 x, y, scale
U_dim: 2    # 例如 2 个源项场
```

### Q2: 模型太大/太小怎么办？

调整 `n_hidden` 和 `num_encoder_layers`：

```yaml
# 减小模型
n_hidden: 32
num_encoder_layers: 2

# 增大模型
n_hidden: 128
num_encoder_layers: 6
```

### Q3: 训练不稳定怎么办？

1. 增加 dropout:
```yaml
dropout: 0.1
encoder_dropout: 0.1
decoder_dropout: 0.1
```

2. 降低学习率: `--lr 5e-4`

3. 增加 batch size: `--batch_size 32`

### Q4: ifft decoder 报错怎么办？

检查 `fourier_modes` 是否小于输入尺寸：

```yaml
# 对于 16x16 输入
fourier_modes: 8    # 必须 < 16

# 对于 32x32 输入
fourier_modes: 12   # 必须 < 32
```

或者改用 `pointwise` decoder（更通用）。

### Q5: 如何对比 FourierTransformer？

创建等效的配置：

```python
# FourierTransformer: node_feats=5 (G:2 + U:3)
fourier_config = {
    'node_feats': 5,
    'n_targets': 1,
    ...
}

# DualHeadFourierTransformer: 分离 G 和 U
dualhead_config = {
    'G_dim': 2,
    'U_dim': 3,
    'n_targets': 1,
    ...  # 其他参数保持一致
}
```

## 相关文件

- **配置文件**: `data/configs/dualhead_transformer_config_sate.yml`
- **使用示例**: `Demo/satellite_sup_2d/train_DualHead_with_config.py`
- **模型文档**: `Models/transformer/DualHeadTransformer_README.md`
- **模型代码**: `Models/transformer/DualHeadTransformer.py`

## 更新日志

- **2024/12/24**: 初始版本，包含 6 个预定义配置
  - 修复 `spacial_fc` 和 `fourier_modes` 的默认值
  - 所有配置测试通过

