# 卫星2D模型评估脚本使用说明

## 功能概述

`eval_satellite_predict.py` 是一个简化的评估脚本，用于对训练好的 FNO 和 MLP 模型进行推理和性能评估。

## 主要特性

1. **统一接口**：支持 FNO 和 MLP 两种模型
2. **自动配置**：模型结构参数使用默认值，无需手动指定
3. **归一化器自动加载**：优先从检查点文件夹的 `normalizers.yaml` 加载，否则基于数据重新计算
4. **物理空间评估**：在物理空间（反归一化后）计算指标和绘图
5. **简洁设计**：最小化参数配置，专注于快速评估

## 使用方法

### 基本用法

```bash
python eval_satellite_predict.py \
    --model {fno|mlp} \
    --ckpt /path/to/checkpoint/folder/ \
    [其他可选参数]
```

### 参数说明

#### 必需参数
- `--ckpt`: 检查点文件夹路径（包含 `latest_model.pth` 和 `normalizers.yaml`）

#### 核心参数（带默认值）
- `--model`: 模型类型，可选 `fno` 或 `mlp`（默认：`fno`）
- `--data_path`: 数据集路径（默认：`/data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5`）
- `--device`: 设备类型（默认：`cuda`）
- `--cuda_index`: CUDA 设备索引（默认：`0`）
- `--batch_size`: 批次大小（默认：`4`）
- `--split`: 评估子集，可选 `train`/`valid`/`all`（默认：`valid`）
- `--save_dir`: 输出目录（默认：`work_satellite_eval`）
- `--max_plots`: 最大绘图数量（默认：`10`）

#### 数据划分参数
- `--ntrain`: 训练样本数（默认：None，自动按 9:1 划分）
- `--nvalid`: 验证样本数（默认：None，自动按 9:1 划分）

### 示例

#### 评估 MLP 模型

```bash
python eval_satellite_predict.py \
    --model mlp \
    --ckpt work_satellite_dssl_mlp_20251021/MLP_DSSL_n2000_20251021_225421/ \
    --split valid \
    --max_plots 20
```

#### 评估 FNO 模型

```bash
python eval_satellite_predict.py \
    --model fno \
    --ckpt work_satellite_dssl_fno_20251021/FNO_DSSL_n5000_20251021_225421/ \
    --split valid \
    --batch_size 8
```

## 输出结果

评估完成后，会在 `{save_dir}/eval_{model}_{timestamp}/` 目录下生成：

1. **metrics.json / metrics.txt**: 评估指标
   - avg_mse: 平均均方误差
   - avg_mae: 平均绝对误差
   - avg_r2: 平均 R² 分数
   - dataset_size: 评估样本数
   - height/width: 数据空间分辨率

2. **samples/**: 对比图（真值/预测/误差）
   - 格式：`{idx:06d}.png`
   - 数量由 `--max_plots` 控制

3. **config.json**: 运行配置记录

## 模型默认配置

### FNO 默认参数
- modes: (10, 10)
- width: 64
- depth: 3
- steps: 1
- padding: 8

### MLP 默认参数
- down: 4（下采样倍数，256→64）
- hidden: 1024
- layers: 4

## 注意事项

1. **检查点文件夹结构**：
   ```
   checkpoint_folder/
   ├── latest_model.pth
   └── normalizers.yaml  # 可选，不存在则重新计算
   ```

2. **数据一致性**：确保评估使用的数据集与训练时一致

3. **归一化器**：
   - 优先使用保存的归一化器（确保与训练时一致）
   - 若不存在，会基于当前数据集重新计算（可能与训练时不同）

4. **内存管理**：
   - FNO 使用完整 256x256 数据
   - MLP 下采样到 64x64（减少内存占用）

## 故障排查

### 常见问题

1. **找不到检查点文件**
   - 确认 `--ckpt` 路径正确，且包含 `latest_model.pth`

2. **形状不匹配错误**
   - 检查模型类型是否正确（FNO/MLP）
   - 确认数据集格式正确

3. **CUDA 内存不足**
   - 减小 `--batch_size`
   - 使用 `--device cpu`

## 更新日志

- 2025-10-24: 初始版本，支持 FNO 和 MLP 评估

