# 卫星数据叠加消融实验框架

## 概述

本框架用于大规模卫星热传导数据集的叠加消融实验，支持：
- 按K桶（元件数量）分组的外推评估
- 多层叠加训练（S0/S01/S012）
- 动态通道压缩机制
- 自动化批量实验

## 文件结构

```
Demo/satellite_sup_2d/
├── data_loader_satellite.py      # 数据加载器（支持分桶/混合/下采样）
├── trainer_satellite.py           # 训练器（通道压缩/训练/验证/可视化）
├── train_entry.py                 # 主入口脚本
├── run_experiments.sh             # 批量实验脚本（18个实验）
├── test_single_exp.sh             # 测试脚本
└── README_ablation_framework.md   # 本文档
```

## 核心机制

### 1. 统一数据加载 + 动态通道压缩

**数据加载阶段**：
- 根据 `super_train_mode` 加载最大U通道数
  - S0 → 16通道
  - S01 → 32通道
  - S012 → 64通道

**训练阶段**：
- 单一 `train_loader`，动态压缩U通道
  - 训练S0: `compress_U(U_64, 16)` → 分组取min
  - 训练S1: `compress_U(U_64, 32)` → 分组取min
  - 训练S2: 直接使用 `U_64`

**优势**：
- 避免多次遍历dataloader
- 训练函数接口统一
- 通道压缩逻辑清晰

### 2. 数据划分策略

每个K桶独立划分：
- 训练集：前 `ntrain_perK` 个样本（默认800）
- 验证集：后 `nvalid_perK` 个样本（默认40）

### 3. 下采样

默认 `downsample=2`，即 256×256 → 128×128，在数据加载时执行。

## 快速开始

### 1. 测试单个实验

```bash
# 快速测试（3个epoch，小数据集）
bash Demo/satellite_sup_2d/test_single_exp.sh

# 完整训练单个实验
python Demo/satellite_sup_2d/train_entry.py \
  --work_name test_kmax10_S01 \
  --train_component_nums 1,2,3,4,5,6,7,8,9,10 \
  --super_train_mode S01 \
  --epochs 200 \
  --gpu 0
```

### 2. 批量实验

```bash
# 启动18个实验（6 Kmax × 3 modes）
bash Demo/satellite_sup_2d/run_experiments.sh

# 查看单个实验日志
tail -f runs/kmax10_S01/train.log
```

## 命令行参数

### 数据参数

- `--train_component_nums`: 训练用的K桶列表（逗号分隔）
- `--valid_component_nums`: 验证用的K桶列表（默认1-25）
- `--ntrain_perK`: 每个K桶的训练样本数（默认800）
- `--nvalid_perK`: 每个K桶的验证样本数（默认40）
- `--downsample`: 下采样因子，1=256, 2=128, 4=64（默认2）

### 叠加参数

- `--super_train_mode`: 训练模式 S0/S01/S012（必选）
- `--super_nums_eval`: 评估的super_num列表（默认"0,1,2"）
- `--target_U_channels`: 基础通道数（默认16）

### 训练参数

- `--epochs`: 训练轮数（默认200）
- `--batch_size`: 批大小（默认16）
- `--lr`: 学习率（默认1e-3）
- `--scheduler_step`: 学习率衰减步长（默认100）
- `--scheduler_gamma`: 学习率衰减系数（默认0.5）

### 运行参数

- `--work_name`: 实验名称（必选）
- `--save_dir`: 保存目录（默认runs）
- `--seed`: 随机种子（默认8905）
- `--gpu`: GPU编号（默认0）

## 输出文件

每个实验会在 `runs/<work_name>/` 目录下生成：

```
runs/<work_name>/
├── config.json              # 实验配置
├── metrics.json             # 逐epoch详细指标（完整嵌套结构）
├── metrics_summary.csv      # CSV摘要（包含所有分桶loss）
├── ckpt_best.pth            # 最佳模型（基于验证loss平均值）
├── ckpt_last.pth            # 最新模型
├── log_loss.svg             # 收敛曲线（所有super_num）
├── train_s0_sample*.jpg     # 训练样本可视化（每50epoch）
└── valid_s0_sample*.jpg     # 验证样本可视化（每50epoch）
```

### CSV格式说明

`metrics_summary.csv` 包含以下列：

```
epoch | train_s0 | train_s1 | train_s2 | 
      | valid_s0 | valid_s1 | valid_s2 | 
      | valid_s0_k1 | valid_s0_k2 | ... | valid_s0_k25 |
      | valid_s1_k1 | valid_s1_k2 | ... | valid_s1_k25 |
      | valid_s2_k1 | valid_s2_k2 | ... | valid_s2_k25
```

- `train_sX`: super_num=X的训练loss
- `valid_sX`: super_num=X的验证loss平均值（所有有效K桶的均值）
- `valid_sX_kY`: super_num=X在K桶=Y的验证loss（可能是NaN对于OOD桶）

**示例数据**：
```
epoch  train_s0  valid_s0  valid_s0_k1  valid_s0_k2  valid_s0_k3
0      0.965     0.817     0.422        1.112        0.916
1      0.855     0.759     0.254        1.056        0.966
```

## 实验矩阵

批量实验脚本会运行以下18个实验：

| Kmax | super_train_mode | 实验名称 | 训练K桶 |
|------|-----------------|---------|---------|
| 1    | S0              | kmax1_S0 | [1] |
| 1    | S01             | kmax1_S01 | [1] |
| 1    | S012            | kmax1_S012 | [1] |
| 2    | S0              | kmax2_S0 | [1,2] |
| ...  | ...             | ... | ... |
| 10   | S012            | kmax10_S012 | [1..10] |

验证集固定为K=1-25的所有桶。

## 验证指标

`metrics.json` 包含：

```json
{
  "epoch": 0,
  "train": {
    "0": 0.123,  // super_num=0的训练loss
    "1": 0.234   // super_num=1的训练loss
  },
  "per_super": {
    "0": 0.456,  // super_num=0的验证loss平均值
    "1": 0.567
  },
  "per_K": {
    "0": {       // super_num=0
      "1": 0.1,  // K=1桶的loss
      "2": 0.2,
      ...
      "25": 0.9
    },
    "1": {...}
  }
}
```

## 注意事项

1. **GPU内存**：S012模式下U通道达到64，确保GPU内存充足
2. **数据路径**：确保 `/data/wqn/datasets/dataset_20251218/heat_dataset_source_{K}.h5` 存在
3. **Conda环境**：脚本自动激活 `torch_py310` 环境
4. **并发控制**：批量脚本自动轮转GPU，限制并发数等于GPU数量

## 常见问题

### Q: 验证loss出现NaN？
A: 主要原因：**验证集包含训练集外的K桶（OOD）**

例如训练K=1,2，但验证K=1-5：
- K=1,2（ID桶）：loss正常
- K=3,4,5（OOD桶）：U通道大部分是填充值，模型产生极端预测 → NaN

**解决方案**：
1. **已修复**：`validate_all()` 现在会跳过NaN值计算平均
2. **建议**：在训练早期，OOD桶可能都是NaN，这是正常的
3. **验证**：查看 `metrics.json` 中的 `per_K`，ID桶应该有正常值

**示例**：
```json
"per_K": {
  "0": {
    "1": 0.133,  // ✓ ID桶，正常
    "2": 1.726,  // ✓ ID桶，正常  
    "3": NaN,    // ✗ OOD桶，模型未见过
    "4": NaN,
    "5": NaN
  }
}
```

建议：训练更多epoch后，OOD桶的泛化性能会改善。

### Q: 如何修改GPU列表？
A: 编辑 `run_experiments.sh`，修改 `GPUS=(0 1 2 3)` 数组。

### Q: 如何只跑部分实验？
A: 修改 `run_experiments.sh` 中的循环范围，或直接用 `train_entry.py` 单独运行。

## 代码风格

本框架遵循 Research Code Rules：
- 单文件 ≤ 300行
- 函数 ≤ 50行
- 显式参数，早返回
- 简洁优先，可调试优先

## 作者与日期

- 重构时间：2025-12-25
- 基于临时脚本：`train_Trans_base_satellite_GUT_sup.py`

