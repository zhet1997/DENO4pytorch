# 卫星导热数据集格式说明文档

## 概述

本文档详细说明卫星导热数据集的两种格式版本，帮助用户理解格式差异并选择合适的数据集。

## 格式版本对比

### V1格式 (6通道) - 合并元件SDF

**数据路径**: `/data/wqn/datasets/packaged_dataset20251011/heat_dataset.h5`

**基本信息**:
- 样本数量: 5000
- 输入维度: (256, 256, 6)
- 输出维度: (256, 256, 1)
- 特点: 所有元件的SDF合并为单一通道

**通道布局**:
```
Ch 0: component_sdf    - 元件符号距离场（所有元件合并）
Ch 1: component_power  - 元件功率密度
Ch 2: cooling_sdf      - 散热窗符号距离场
Ch 3: cooling_temp     - 散热窗温度（统一边界温度）
Ch 4: coord_x          - X坐标（以中心为原点）
Ch 5: coord_y          - Y坐标（以中心为原点）
```

**设计思想**:
- 简化表示：将多个元件的几何信息合并为单一SDF通道
- 减少通道数：便于快速原型开发和模型训练
- 全局视角：强调整体布局而非单个元件

### V2格式 (17通道) - 分层元件SDF

**数据路径**: `/data/wqn/datasets/packaged_dataset_test/heat_dataset.h5`

**基本信息**:
- 样本数量: 100+
- 输入维度: (256, 256, 17)
- 输出维度: (256, 256, 1)
- 特点: 12个元件的SDF分别保存为独立通道

**通道布局**:
```
Ch 0:     cooling_sdf         - 散热窗符号距离场
Ch 1:     component_power     - 元件功率密度
Ch 2:     cooling_temp        - 散热窗温度（统一边界温度）
Ch 3:     coord_x             - X坐标（以中心为原点）
Ch 4:     coord_y             - Y坐标（以中心为原点）
Ch 5:     component_sdf_1     - 元件1的SDF
Ch 6:     component_sdf_2     - 元件2的SDF
Ch 7:     component_sdf_3     - 元件3的SDF
Ch 8:     component_sdf_4     - 元件4的SDF
Ch 9:     component_sdf_5     - 元件5的SDF
Ch 10:    component_sdf_6     - 元件6的SDF
Ch 11:    component_sdf_7     - 元件7的SDF
Ch 12:    component_sdf_8     - 元件8的SDF
Ch 13:    component_sdf_9     - 元件9的SDF
Ch 14:    component_sdf_10    - 元件10的SDF
Ch 15:    component_sdf_11    - 元件11的SDF
Ch 16:    component_sdf_12    - 元件12的SDF
```

**设计思想**:
- 精细表示：每个元件的几何信息独立保存
- 灵活操作：可以选择性地使用部分元件信息
- 局部控制：便于研究单个元件对温度场的影响

## 关键差异总结

| 特性 | V1格式 (6通道) | V2格式 (17通道) |
|------|---------------|----------------|
| 通道总数 | 6 | 17 |
| 元件SDF表示 | 1个合并通道 | 12个独立通道 |
| 元件SDF位置 | 第0通道 | 第5-16通道 |
| cooling_sdf位置 | 第2通道 | 第0通道 |
| 通道顺序 | component→cooling→coords | cooling→coords→components |
| 数据样本数 | 5000 | 100+ |
| 适用场景 | 全局优化、快速训练 | 精细控制、元件级分析 |

## 通道映射关系

### V1 → V2 映射

```python
# V1格式通道索引
v1_component_sdf = 0    # → V2的Ch5-16（分散为12个通道）
v1_component_power = 1  # → V2的Ch1
v1_cooling_sdf = 2      # → V2的Ch0
v1_cooling_temp = 3     # → V2的Ch2
v1_coord_x = 4          # → V2的Ch3
v1_coord_y = 5          # → V2的Ch4
```

### V2 → V1 映射

```python
# V2格式通道索引
v2_cooling_sdf = 0        # ← V1的Ch2
v2_component_power = 1    # ← V1的Ch1
v2_cooling_temp = 2       # ← V1的Ch3
v2_coord_x = 3            # ← V1的Ch4
v2_coord_y = 4            # ← V1的Ch5
v2_component_sdfs = 5:17  # ← V1的Ch0（需要合并操作）
```

## 使用场景建议

### 推荐使用V1格式的场景

1. **快速原型开发**: 通道数少，模型训练更快
2. **全局温度场预测**: 关注整体布局而非单个元件
3. **资源受限环境**: 内存或显存有限
4. **大规模数据集**: 5000个样本足够训练深度模型

### 推荐使用V2格式的场景

1. **元件级分析**: 需要研究单个元件的影响
2. **渐进式学习**: 可以逐步增加元件数量训练
3. **注意力机制**: 模型需要关注特定元件
4. **迁移学习**: 从少元件迁移到多元件场景
5. **可解释性研究**: 需要分析每个元件的贡献

## 代码使用示例

### 自动格式检测（推荐）

```python
from Demo.satellite_sup_2d.utilizes_satellite import get_origin_satellite, get_loader_satellite

# 自动检测格式
train_x, train_y, valid_x, valid_y, format_ver, num_comp = get_origin_satellite(
    h5_path='/path/to/heat_dataset.h5',  # 指定任一格式的数据集
    train_num=4000,
    valid_num=500
)

print(f"检测到格式: {format_ver}")
print(f"元件SDF通道数: {num_comp}")

# 创建DataLoader（归一化逻辑自动适配通道数）
train_loader, valid_loader, x_norm, y_norm = get_loader_satellite(
    train_x, train_y, valid_x, valid_y, batch_size=32
)
```

### 使用V1格式（6通道）

```python
# 加载V1格式数据
train_x, train_y, valid_x, valid_y, fmt, nc = get_origin_satellite(
    h5_path='/data/wqn/datasets/packaged_dataset20251011/heat_dataset.h5',
    train_num=4000,
    valid_num=500
)

# 验证格式
assert fmt == 'v1'
assert nc == 1

# 使用时的通道索引
batch_x, batch_y = next(iter(train_loader))
component_sdf = batch_x[:, :, :, 0]      # 合并的元件SDF
component_power = batch_x[:, :, :, 1]    # 元件功率
cooling_sdf = batch_x[:, :, :, 2]        # 散热窗SDF
cooling_temp = batch_x[:, :, :, 3]       # 散热窗温度
coord_x = batch_x[:, :, :, 4]            # X坐标
coord_y = batch_x[:, :, :, 5]            # Y坐标
```

### 使用V2格式（17通道）

```python
# 加载V2格式数据
train_x, train_y, valid_x, valid_y, fmt, nc = get_origin_satellite(
    h5_path='/data/wqn/datasets/packaged_dataset_test/heat_dataset.h5',
    train_num=80,
    valid_num=20
)

# 验证格式
assert fmt == 'v2'
assert nc == 12

# 使用时的通道索引
batch_x, batch_y = next(iter(train_loader))
cooling_sdf = batch_x[:, :, :, 0]          # 散热窗SDF
component_power = batch_x[:, :, :, 1]      # 元件功率
cooling_temp = batch_x[:, :, :, 2]         # 散热窗温度
coord_x = batch_x[:, :, :, 3]              # X坐标
coord_y = batch_x[:, :, :, 4]              # Y坐标

# 获取所有元件SDF
component_sdfs = batch_x[:, :, :, 5:17]    # 12个元件SDF，shape=(B, 256, 256, 12)

# 或者获取特定元件
component_1_sdf = batch_x[:, :, :, 5]      # 元件1的SDF
component_2_sdf = batch_x[:, :, :, 6]      # 元件2的SDF
# ... 依此类推
```

### V2格式的高级用法

#### 选择性使用元件

```python
# 只使用前6个元件
selected_channels = [0, 1, 2, 3, 4] + list(range(5, 11))  # cooling + coords + 前6个元件
train_x_selected = train_x[:, :, :, selected_channels]

# 创建DataLoader
train_loader, _, x_norm, _ = get_loader_satellite(
    train_x_selected, train_y, batch_size=32
)
```

#### 元件重要性分析

```python
# 逐个添加元件，观察性能变化
for num_components in range(1, 13):
    # 选择通道：基础通道(0-4) + 前num_components个元件SDF
    selected_channels = list(range(5)) + list(range(5, 5 + num_components))
    train_x_subset = train_x[:, :, :, selected_channels]
    
    # 训练和评估
    # ... (训练代码)
    
    print(f"使用 {num_components} 个元件，性能: ...")
```

#### 合并为V1格式风格

```python
import numpy as np

# 将12个元件SDF合并为单一通道（取最小值）
component_sdfs = train_x[:, :, :, 5:17]  # (N, 256, 256, 12)
component_sdf_merged = np.min(component_sdfs, axis=-1, keepdims=True)  # (N, 256, 256, 1)

# 重组为类似V1的布局
train_x_v1_style = np.concatenate([
    component_sdf_merged,     # Ch0: 合并的component_sdf
    train_x[:, :, :, 1:2],    # Ch1: component_power
    train_x[:, :, :, 0:1],    # Ch2: cooling_sdf
    train_x[:, :, :, 2:3],    # Ch3: cooling_temp
    train_x[:, :, :, 3:4],    # Ch4: coord_x
    train_x[:, :, :, 4:5],    # Ch5: coord_y
], axis=-1)  # (N, 256, 256, 6)
```

## 数据统计对比

### V1格式统计

```
输入通道统计（基于4000个训练样本）:
  Ch0 (component_sdf):   mean=3.34,    std=7.39
  Ch1 (component_power): mean=0.067,   std=0.253
  Ch2 (cooling_sdf):     mean=32.77,   std=64.76
  Ch3 (cooling_temp):    mean≈0,       std=0.130
  Ch4 (coord_x):         mean≈0,       std=0.130
  Ch5 (coord_y):         mean=0.010,   std=0.035

输出统计:
  温度范围: 268-275K
  平均温度: 273.15K
```

### V2格式统计

```
基础通道统计（基于100个样本）:
  Ch0 (cooling_sdf):     mean=5.75,    std=8.57
  Ch1 (component_power): mean=0.418,   std=0.251
  Ch2 (cooling_temp):    mean=272.26,  std=2.15
  Ch3 (coord_x):         mean≈0,       std=0.266
  Ch4 (coord_y):         mean≈0,       std=0.266

元件SDF通道 (Ch5-16):
  各通道值域: [-0.13, 0.52]
  100%非零覆盖
```

## 向后兼容性说明

### 现有代码兼容性

如果您的代码已经使用了`utilizes_satellite.py`，无需修改即可兼容：

```python
# 旧代码（仍然有效）
train_x, train_y, valid_x, valid_y = get_origin_satellite(
    train_num=4000, valid_num=500
)
# 格式信息被忽略，不影响使用

# 新代码（推荐）
train_x, train_y, valid_x, valid_y, fmt, nc = get_origin_satellite(
    train_num=4000, valid_num=500
)
# 显式接收格式信息
```

### 归一化兼容性

两种格式都使用相同的归一化策略：
- 方法：mean-std归一化
- 轴设置：axis=(0, 1, 2) - 每通道独立
- V1格式：6个独立的归一化统计量
- V2格式：17个独立的归一化统计量

## 常见问题

### Q1: 我应该使用哪种格式？

**A**: 
- 如果您需要大量数据（5000样本）且关注全局性能 → 使用V1格式
- 如果您需要元件级别的精细控制和分析 → 使用V2格式

### Q2: 两种格式可以混用吗？

**A**: 不建议直接混用。但您可以：
- 在V1格式上预训练，然后在V2格式上微调（需要网络结构调整）
- 将V2格式合并为V1格式风格使用（见上述示例代码）

### Q3: V2格式为什么通道顺序不同？

**A**: V2格式设计时考虑了以下因素：
- 将cooling_sdf放在最前面，与元件SDF在概念上分离
- 将12个元件SDF放在最后，便于批量处理和选择性使用
- 保持基础通道（功率、温度、坐标）的相对顺序

### Q4: 如何从V2格式转换为V1格式？

**A**: 使用"合并为V1格式风格"章节中的代码，核心是对12个元件SDF取最小值（或其他合并策略）。

### Q5: 归一化器可以跨格式使用吗？

**A**: 不可以。V1和V2格式的归一化器形状不同：
- V1: x_normalizer.mean.shape = (6,)
- V2: x_normalizer.mean.shape = (17,)

## 更新历史

- **2025-10-11**: 初始版本，定义V1和V2格式
- **2025-10-11**: 添加详细的使用示例和代码片段

## 参考资料

- `utilizes_satellite.py`: 数据加载工具实现
- `README_satellite_dataset.md`: 快速入门指南
- `dataset_collection_new.py`: V1格式数据集生成脚本

