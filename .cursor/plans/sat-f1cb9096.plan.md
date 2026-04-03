---
name: 卫星2D模型推理与可视化评估脚本设计
overview: ""
todos:
  - id: d608f52d-31f1-43b4-852a-00083d86fb76
    content: 创建 Demo/satellite_2d_base/eval_satellite_models.py 脚本骨架
    status: pending
  - id: 11b83115-370b-4bf1-abb2-14912ca21d3a
    content: 实现CLI参数解析（模型/数据/权重/划分/结构/下采样/输出）
    status: pending
  - id: 5a9d7d24-ca06-4869-8fdc-11fbf6e40c4f
    content: 复用load_satellite_data并实现9:1或给定ntrain/nvalid划分与下采样
    status: pending
  - id: 8b1a5b8b-e344-4153-801f-59f1c35f8409
    content: 按训练范式构建DataNormer并应用于评估子集，输出反归一化结果
    status: pending
  - id: d54c692f-eb0e-4408-afba-9dbaa2ade8d2
    content: 按--model重建网络并导入对应feature_transform，实现权重加载与设备放置
    status: pending
  - id: de7dc7ae-777a-49b1-8d37-555111bd56fa
    content: 批量前向推理并缓存逐批预测与标签用于指标与可视化
    status: pending
  - id: 8a5ead09-0347-4e41-9bb1-f228475c28f9
    content: 计算MSE/MAE(可选R2)，保存metrics.json与文本摘要
    status: pending
  - id: 04891185-a15c-4208-98c6-6f1bf02b97bf
    content: 复用MatplotlibVision绘制truth/pred/error三列并批量保存
    status: pending
  - id: 76338493-ce0d-492e-9bc7-7fa2e882ca23
    content: 组织输出目录结构与配置落盘（samples/、metrics、config）
    status: pending
---

# 卫星2D模型推理与可视化评估脚本设计

## 文件与位置

- 新增脚本：`Demo/satellite_2d_base/eval_satellite_models.py`
- 功能：统一评估 FNO/UNet/Transformer/MLP 四类模型在指定数据集上的预测表现，保存指标与大批量可视化对比图。

## CLI 参数（与训练脚本风格一致）

- 基本
- `--model {fno,unet,trans,mlp}` 模型类型
- `--data_path PATH` 数据集 H5 路径
- `--ckpt PATH` 训练后权重路径（支持保存为`latest_model.pth`的字典或纯state_dict）
- `--device {cuda,cpu}`，`--cuda_index INT`
- `--batch_size INT`，默认与训练脚本相近
- `--save_dir PATH` 评估输出目录（默认 `work_satellite/eval_{model}_{ts}`）
- `--max_plots INT` 保存对比图的样本上限，避免海量图片过多（默认 200）
- 数据划分与下采样（与训练范式一致）
- `--ntrain INT`，`--nvalid INT`；若均为空则按 9:1 自动分割
- `--split {valid,train,all}` 评估子集（默认 valid）
- `--down INT` 下采样（Transformer/MLP 用；默认 8，与训练一致）
- 各模型结构超参（需与训练保持一致以匹配权重）
- FNO：`--modes_x` `--modes_y` `--width` `--depth` `--steps` `--padding`
- UNet：`--width` `--depth` `--steps` `--dropout`
- Transformer：`--hidden` `--nhead` `--enc_layers` `--ffn` `--decoder` `--fourier_modes` `--dropout` `--down`
- 以及：`--config_path PATH`，`--use_config`（同训练脚本，若提供则优先从yaml载入并强制校正与数据维度相关字段）
- MLP：`--hidden` `--layers`（配合`--down`计算输入/输出维度）

## 数据加载与划分

- 复用：`Demo/satellite_2d_base/dataset_satellite.py::load_satellite_data`
- 流程：
1) 读取 inputs(N,256,256,6) 与 outputs(N,256,256,1)
2) 若未指定 ntrain/nvalid，则按 9:1 划分（与训练保持一致）
3) `--split` 控制评估数据范围：train/valid/all
4) 对 Transformer/MLP：按 `--down` 做空间下采样，与训练范式一致

## 归一化与反归一化

- 与训练保持一致：
- 基于“训练划分”的数据计算 `x_normalizer/y_normalizer = DataNormer(..., method='mean-std')`
- 对评估子集应用相同 normalizer；推理输出通过 `y_normalizer.back(...)` 反归一化，用于绘图与指标

## 模型构建与权重加载

- 统一入口根据 `--model` 分派：
- FNO：`from fno.FNOs import FNO2d`；同时`from Demo.satellite_2d_base.run_FNO_satellite import feature_transform`
- UNet：`from cnn.ConvNets import UNet2d`；同时`from Demo.satellite_2d_base.run_UNet_satellite import feature_transform`
- Transformer：`from transformer.Transformers import FourierTransformer`；同时`from Demo.satellite_2d_base.run_Trans_satellite import feature_transform`
- MLP：`from Demo.satellite_2d_base.run_MLP_satellite import MLP`（直接复用该脚本中的定义）
- 以 CLI 超参重建网络（默认值与训练脚本一致）；随后：
- 加载权重：优先读取 `torch.load(ckpt)` 中的 `['net_model']`；若不存在则尝试直接当作 state_dict 加载
- 设备放置与 `cuda_index` 设置同训练风格

## 前向推理

- 批处理 DataLoader（`drop_last=False`）
- 根据模型类型处理输入：
- FNO/UNet/Trans：生成 `gd = feature_transform(xx)` 并调用 `net(xx, gd)`
- MLP：使用下采样后的展平向量（与训练一致），直接 `net(vec)`
- 反归一化得到物理量尺度结果，用于指标与绘图

## 指标计算

- 对评估子集计算：
- MSE、MAE（逐样本与整体平均），可选 R2
- 输出：将汇总指标保存为 `metrics.json` 与 `metrics.txt`

## 可视化（复用 Utilizes/visual_data.py）

- 复用 `MatplotlibVision`：`from Utilizes.visual_data import MatplotlibVision`
- 每个样本保存一张三列对比图（truth/pred/error）：
- 构造 `(H,W,1)` 的 `real` 与 `pred` 数组
- 建立 `fig, axs = plt.subplots(1, 3, ...)`，传入 `vision.plot_fields_grid(fig, axs, real, pred, titles=['truth','predicted','error'])`
- 保存到 `save_dir/samples/{idx:06d}.png`
- 通过 `--max_plots` 控制最多保存的样本数

## 输出目录结构

- `save_dir/`
- `metrics.json`、`metrics.txt`
- `samples/` 大量三列对比图
- `config.yml`（可选，记录本次评估关键参数与模型结构）

## 兼容性与健壮性

- 容错：
- 权重键名兼容（带/不带`net_model`）
- 若 `--use_config` 与 CLI 冲突：以 config 为主，必要字段按数据维度强制校正
- 当 `--split=all` 时，指标基于全部集；绘图仍受 `--max_plots` 限制

## 性能考虑

- `num_workers` 可选参数（默认0），避免在受限环境下引发多进程问题
- 大图片批量保存时控制分辨率与频率，防止 I/O 过载