# 卫星导热数据集加载工具使用说明

## 概述

`utilizes_satellite.py` 提供了卫星导热数据集的加载、预处理和DataLoader创建功能，专门用于叠加神经网络训练。

**支持格式**:
- ✅ V1格式 (6通道) - 合并元件SDF
- ✅ V2格式 (17通道) - 分层元件SDF（12个独立通道）

> 📖 **详细格式说明**: 参见 [DATASET_FORMATS.md](./DATASET_FORMATS.md)

## 数据集信息

### V1格式 (6通道) - 推荐用于快速训练

- **数据路径**: `/data/wqn/datasets/packaged_dataset20251011/heat_dataset.h5`
- **数据规模**: 5000个样本
- **输入维度**: (256, 256, 6) - 6个通道
  - Ch 0: `component_sdf` - 元件符号距离场（所有元件合并）
  - Ch 1: `component_power` - 元件功率密度
  - Ch 2: `cooling_sdf` - 散热窗符号距离场
  - Ch 3: `cooling_temp` - 散热窗温度（统一边界温度）
  - Ch 4: `coord_x` - X坐标
  - Ch 5: `coord_y` - Y坐标
- **输出维度**: (256, 256, 1) - 温度场

### V2格式 (17通道) - 推荐用于元件级分析

- **数据路径**: `/data/wqn/datasets/packaged_dataset_test/heat_dataset.h5`
- **数据规模**: 100+个样本
- **输入维度**: (256, 256, 17) - 17个通道
  - Ch 0: `cooling_sdf` - 散热窗符号距离场
  - Ch 1: `component_power` - 元件功率密度
  - Ch 2: `cooling_temp` - 散热窗温度（统一边界温度）
  - Ch 3: `coord_x` - X坐标
  - Ch 4: `coord_y` - Y坐标
  - Ch 5-16: `component_sdf_1~12` - 12个元件的独立SDF
- **输出维度**: (256, 256, 1) - 温度场

## 主要功能

### 1. `get_origin_satellite()` - 数据加载与划分（支持多格式）

从h5文件加载数据并划分训练/验证集，自动检测V1或V2格式。

```python
from utilizes_satellite import get_origin_satellite

# 基础用法（自动检测格式）
train_inputs, train_outputs, valid_inputs, valid_outputs, format_ver, num_comp = get_origin_satellite(
    h5_path=None,                    # 默认路径或指定路径
    train_num=4000,                  # 训练集样本数（从前面取）
    valid_num=500,                   # 验证集样本数（从后面取）
    shuffled=False,                  # 是否打乱数据
    num_component_channels=None,     # 元件通道数（验证用，None=不验证）
    dataset_format='auto'            # 格式：'auto'(自动), 'v1', 'v2'
)
```

**参数说明**:
- `h5_path`: h5文件路径，默认为V1格式数据集路径
- `train_num`: 训练集样本数，从索引0开始取
- `valid_num`: 验证集样本数，从末尾开始取
- `shuffled`: 是否在划分前打乱数据（使用固定seed=8905保证可复现）
- `num_component_channels`: 元件SDF通道数，用于验证格式（None=不验证）
- `dataset_format`: 数据集格式，'auto'(自动检测)、'v1'(6通道)、'v2'(17通道)

**返回值**:
- `train_inputs`: (train_num, 256, 256, C)，C=6或17
- `train_outputs`: (train_num, 256, 256, 1)
- `valid_inputs`: (valid_num, 256, 256, C) 或 None
- `valid_outputs`: (valid_num, 256, 256, 1) 或 None
- `format_version`: 检测到的格式 'v1' 或 'v2'
- `num_components`: 元件SDF通道数（v1为1，v2为12）

**向后兼容**:
```python
# 旧代码仍然有效（忽略格式信息）
train_x, train_y, valid_x, valid_y = get_origin_satellite(train_num=4000, valid_num=500)
```

### 2. `get_loader_satellite()` - 归一化与DataLoader创建

对数据进行每通道独立归一化并创建PyTorch DataLoader。

```python
from utilizes_satellite import get_loader_satellite

train_loader, valid_loader, x_normalizer, y_normalizer = get_loader_satellite(
    train_inputs, train_outputs,
    valid_inputs, valid_outputs,
    x_normalizer=None,      # 输入归一化器（可选）
    y_normalizer=None,      # 输出归一化器（可选）
    batch_size=32           # batch大小
)
```

**参数说明**:
- `train_x`, `train_y`: 训练集输入输出
- `valid_x`, `valid_y`: 验证集输入输出（可选）
- `x_normalizer`, `y_normalizer`: 归一化器（如果为None则自动创建）
- `batch_size`: DataLoader的batch大小

**归一化策略**:
- 使用mean-std归一化方法
- 每个通道独立计算统计量（axis=(0,1,2)）
- 输入归一化器统计量shape=(6,)
- 输出归一化器统计量shape=(1,)

**返回值**:
- `train_loader`: 训练集DataLoader（shuffle=True, drop_last=True）
- `valid_loader`: 验证集DataLoader（shuffle=False, drop_last=False）
- `x_normalizer`: 输入归一化器（DataNormer对象）
- `y_normalizer`: 输出归一化器（DataNormer对象）

## 完整使用示例

### V1格式使用示例（6通道）

```python
import sys
import os
sys.path.append('/data/wqn/DENO4pytorch')

from Demo.satellite_sup_2d.utilizes_satellite import get_origin_satellite, get_loader_satellite

# 1. 加载V1格式数据
train_inputs, train_outputs, valid_inputs, valid_outputs, fmt, nc = get_origin_satellite(
    train_num=4000,
    valid_num=500,
    shuffled=False
)

print(f"数据格式: {fmt}, 元件通道数: {nc}")  # v1, 1

# 2. 创建DataLoader
train_loader, valid_loader, x_norm, y_norm = get_loader_satellite(
    train_inputs, train_outputs,
    valid_inputs, valid_outputs,
    batch_size=32
)

# 3. 训练循环
for epoch in range(num_epochs):
    for batch_x, batch_y in train_loader:
        # batch_x: (32, 256, 256, 6)
        # batch_y: (32, 256, 256, 1)
        # 训练代码...
        pass

# 4. 反归一化预测结果
predictions_norm = model(batch_x)
predictions = y_norm.back(predictions_norm.detach().cpu().numpy())
```

### V2格式使用示例（17通道）

```python
from Demo.satellite_sup_2d.utilizes_satellite import get_origin_satellite, get_loader_satellite

# 1. 加载V2格式数据
train_inputs, train_outputs, valid_inputs, valid_outputs, fmt, nc = get_origin_satellite(
    h5_path='/data/wqn/datasets/packaged_dataset_test/heat_dataset.h5',
    train_num=80,
    valid_num=20,
    shuffled=False
)

print(f"数据格式: {fmt}, 元件通道数: {nc}")  # v2, 12

# 2. 创建DataLoader
train_loader, valid_loader, x_norm, y_norm = get_loader_satellite(
    train_inputs, train_outputs,
    valid_inputs, valid_outputs,
    batch_size=16
)

# 3. 使用V2格式的元件级信息
for batch_x, batch_y in train_loader:
    # batch_x: (16, 256, 256, 17)
    
    # 提取基础通道
    cooling_sdf = batch_x[:, :, :, 0]       # 散热窗SDF
    component_power = batch_x[:, :, :, 1]   # 元件功率
    cooling_temp = batch_x[:, :, :, 2]      # 散热窗温度
    coord_x = batch_x[:, :, :, 3]           # X坐标
    coord_y = batch_x[:, :, :, 4]           # Y坐标
    
    # 提取12个元件SDF
    component_sdfs = batch_x[:, :, :, 5:17]  # (16, 256, 256, 12)
    
    # 可以选择性使用部分元件
    first_6_components = batch_x[:, :, :, 5:11]  # 前6个元件
    
    # 训练代码...
```

## 测试脚本

运行以下命令测试数据加载功能：

```bash
cd /data/wqn/DENO4pytorch
python3 Demo/satellite_sup_2d/utilizes_satellite.py
```

测试输出将显示：
- 数据集基本信息
- 数据划分详情
- 归一化统计信息
- DataLoader信息
- 样本batch测试结果
- 反归一化测试结果

## 格式选择指南

### 何时使用V1格式（6通道）？

✅ **推荐场景**:
- 快速原型开发和模型验证
- 大规模训练（5000样本）
- 关注全局温度场分布
- 计算资源有限
- 需要与现有代码保持兼容

❌ **不适用场景**:
- 需要分析单个元件影响
- 需要元件级别的可解释性

### 何时使用V2格式（17通道）？

✅ **推荐场景**:
- 元件级分析和可解释性研究
- 渐进式学习（逐步增加元件）
- 注意力机制模型（关注特定元件）
- 迁移学习（少元件→多元件）
- 元件重要性排序研究

❌ **不适用场景**:
- 需要大量训练数据
- 简单的全局预测任务

### 格式对比表

| 特性 | V1格式 (6通道) | V2格式 (17通道) |
|------|---------------|----------------|
| 样本数量 | 5000 | 100+ |
| 通道数 | 6 | 17 |
| 元件表示 | 合并为1个通道 | 12个独立通道 |
| 训练速度 | 快 | 较慢 |
| 内存占用 | 低 | 较高 |
| 可解释性 | 一般 | 强 |
| 适用场景 | 全局优化 | 元件级分析 |

## 与PakB数据集的对比

| 特性 | PakB数据集 | Satellite V1 | Satellite V2 |
|------|-----------|-------------|-------------|
| 数据格式 | .mat文件 | .h5文件 | .h5文件 |
| 数据源 | 多个mat文件拼接 | 单个h5文件 | 单个h5文件 |
| 划分方式 | 从前面取全部 | train前valid后 | train前valid后 |
| 归一化策略 | 统一axis=(0,1,2,3) | 每通道独立 | 每通道独立 |
| 输入通道 | 变化 | 6通道 | 17通道 |
| 元件表示 | - | 合并 | 分层 |

## 注意事项

1. **路径设置**: 确保在导入前将项目根目录添加到`sys.path`
2. **内存管理**: 5000个样本占用约2.5GB内存，大batch size时注意显存
3. **数据划分**: train和valid从不同区域取样，确保没有重叠
4. **归一化器保存**: 训练后请保存归一化器用于推理阶段
5. **固定随机种子**: shuffled模式使用seed=8905保证可复现性

## 数据集来源

数据集由 `dataset_collection_new.py` 脚本生成，包含卫星散热仿真数据：
- 输入包含几何信息（SDF）、物理量（功率、温度）和坐标信息
- 输出为稳态温度场分布
- cooling_temp通道采用统一边界温度策略，避免稀疏数据问题

