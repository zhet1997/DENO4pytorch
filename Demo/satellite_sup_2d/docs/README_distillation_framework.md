# 循环式叠加蒸馏训练框架

## 概述

本框架实现了"循环式叠加蒸馏（Alternating Superposition + Self-Distillation）"训练方法，用于训练支持多源项叠加的神经网络算子。

### 核心特性

1. **叠加训练**：支持按源项数量动态叠加训练
2. **多路径一致性**：使用K=8条不同叠加路径训练超分辨率网络
3. **自蒸馏**：基于teacher合格判定的动态蒸馏策略
4. **分层评估**：按源项数量n分层评估模型性能
5. **离线伪标签**：HDF5压缩存储，支持版本管理

---

## 框架架构

```
训练流程（每个Round）:

监督训练 (C+S) 
    ↓
一致性训练 (冻结C, 更新S)
    ↓
评估 m_C(n) 与 m_CS(n)
    ↓
判定Teacher合格集合T与区间[A,B]
    ↓
生成伪标签 (K路径均值)
    ↓
蒸馏训练 (混合真伪标签)
    ↓
清理旧Shards
    ↓
停止判定 (n*推进/停滞)
```

---

## 模块说明

### 1. 一致性训练模块 (`consistency_modules.py`)

**PathSampler**
- 生成K=8条不同叠加路径
- 策略：通道随机重排 + 分组顺序变化

**ConsistencyTrainer**
- 冻结predictor (C网络)
- 只更新super_net (S网络)
- 损失：K条路径输出的一致性（到均值的方差）

### 2. 分层评估模块 (`evaluation_modules.py`)

**AnchorEvaluator**
- 按component_num (n) 分层评估RMSE
- 支持C-only模式 (m_C) 和C+S模式 (m_CS)

**EligibilityGate**
- 判定teacher合格集合T
- 条件：m_CS(n) ≤ τ 且 (m_C - m_CS) ≥ Δ
- 计算可靠前沿n*

**IntervalScheduler**
- 动态调整蒸馏区间[A,B]
- 策略：以n*为中心，宽度随推进/停滞动态调整

### 3. 伪标签生成模块 (`pseudo_label_modules.py`)

**UnlabeledSampler**
- 抽取无标签(G,U,n)样本
- 当前版本：复用现有数据集（忽略T）
- 支持后续切换到大规模无标签库

**TeacherManager**
- 保存/加载teacher模型快照
- 每个round独立teacher

**PseudoLabelBuilder**
- 使用teacher生成伪标签
- 伪标签=K条路径输出的均值
- 写入HDF5 shard（gzip压缩）

**ShardManager**
- 管理HDF5 shards版本
- 自动清理旧shards（保留最近N轮）

### 4. 蒸馏训练模块 (`distill_modules.py`)

**DistillTrainer**
- 混合真标签与伪标签训练
- 损失：L = (1-λ)*L_real + λ*L_pseudo

**StopController**
- 监控n*推进情况
- 停止条件：n*连续patience轮停滞

**RoundLogger**
- 记录每个round的指标
- 生成训练摘要

---

## 使用方法

### 快速开始

#### 1. 阶段1：一致性训练测试

测试监督训练+一致性训练：

```bash
python train_consistency.py \
    --epochs 100 \
    --batch_size 16 \
    --lr 1e-3 \
    --consistency_epochs 5 \
    --K 8 \
    --work_name test_consistency
```

**验收标准**：
- ✓ consistency_loss 稳定下降
- ✓ anchor集RMSE不恶化
- ✓ 路径输出标准差下降（稳定性提升）

#### 2. 阶段2：分层评估测试

评估模块测试（在train_consistency.py基础上）：

```python
from Demo.satellite_sup_2d.evaluation_modules import create_anchor_set, AnchorEvaluator, EligibilityGate

# 创建anchor集
anchor_dict = create_anchor_set(
    component_nums=[1,2,3,4,5,6,7,8,9,10],
    samples_per_n=50
)

# 评估m_C和m_CS
evaluator = AnchorEvaluator(model, anchor_dict, normalizers, device)
m_C = evaluator.evaluate_by_n(c_only=True)
m_CS = evaluator.evaluate_by_n(c_only=False)

# 判定T和n*
gate = EligibilityGate(tau_rmse=0.05, delta_rmse=0.01)
T, n_star, metrics = gate.compute_eligible_set(m_C, m_CS)

print(f"可蒸馏集合T: {T}")
print(f"可靠前沿n*: {n_star}")
```

#### 3. 阶段3：完整蒸馏循环

运行完整的循环蒸馏训练：

```bash
python train_distillation_loop.py \
    --max_rounds 10 \
    --supervised_epochs 50 \
    --consistency_epochs 10 \
    --distill_epochs 20 \
    --batch_size 16 \
    --lr 1e-3 \
    --lr_consistency 1e-4 \
    --lr_distill 5e-4 \
    --K 8 \
    --tau_rmse 0.05 \
    --delta_rmse 0.01 \
    --lambda_distill 0.5 \
    --samples_per_n 1000 \
    --patience 3 \
    --work_name distill_full
```

---

## 参数说明

### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ntrain` | 8000 | 训练样本数 |
| `--nvalid` | 500 | 验证样本数 |
| `--batch_size` | 16 | 批次大小 |
| `--component_nums` | [1,2,...,10] | 元件数量列表 |

### Round循环参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_rounds` | 10 | 最大round数 |
| `--supervised_epochs` | 50 | 每round监督训练epochs |
| `--consistency_epochs` | 10 | 每round一致性训练epochs |
| `--distill_epochs` | 20 | 每round蒸馏训练epochs |

### 学习率参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 1e-3 | 监督训练学习率 |
| `--lr_consistency` | 1e-4 | 一致性训练学习率 |
| `--lr_distill` | 5e-4 | 蒸馏训练学习率 |

### Teacher合格判定参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tau_rmse` | 0.05 | C+S绝对RMSE阈值 |
| `--delta_rmse` | 0.01 | C+S相比C的最小改进量 |

### 蒸馏参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lambda_distill` | 0.5 | 伪标签损失权重 (0~1) |
| `--samples_per_n` | 1000 | 每个n的伪标签样本数 |
| `--K` | 8 | 多路径数量 |

### 停止条件参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--patience` | 3 | n*停滞容忍轮数 |

---

## 输出文件结构

训练完成后，工作目录结构如下：

```
work_satellite/distill_full/
├── distillation_loop.log          # 主日志
├── round_summary.log               # Round摘要日志
├── training_summary.json           # 训练摘要（JSON格式）
├── final_model.pth                 # 最终模型
├── model_round_0.pth               # 各round模型快照
├── model_round_1.pth
├── ...
├── teachers/                       # Teacher快照
│   ├── teacher_round_0.pth
│   ├── teacher_round_1.pth
│   └── ...
└── pseudo_shards/                  # 伪标签HDF5 shards
    ├── pseudo_round_0.h5
    ├── pseudo_round_1.h5
    └── ...
```

---

## 监控指标

### Round级别指标

每个round记录以下指标：

```python
{
    'supervised_train_loss': float,     # 监督训练损失
    'supervised_valid_loss': float,     # 监督验证损失
    'consistency_loss': float,          # 一致性损失
    'consistency_std': float,           # 路径输出标准差（稳定性）
    'm_C': {n: rmse},                   # C-only RMSE（按n）
    'm_CS': {n: rmse},                  # C+S RMSE（按n）
    'T': [n1, n2, ...],                 # 可蒸馏集合
    'n_star': int,                      # 可靠前沿
    'interval_A': int,                  # 蒸馏区间下界
    'interval_B': int,                  # 蒸馏区间上界
    'pseudo_samples': int,              # 伪标签样本数
    'shard_size_mb': float,             # Shard文件大小
    'distill_total_loss': float,        # 蒸馏总损失
    'distill_real_loss': float,         # 蒸馏真标签损失
    'distill_pseudo_loss': float,       # 蒸馏伪标签损失
    'round_time': float                 # Round耗时（秒）
}
```

### 关键可视化指标

1. **RMSE分层曲线**：m_C(n) vs m_CS(n)
2. **n*推进曲线**：追踪可靠前沿变化
3. **一致性稳定性**：路径输出标准差
4. **蒸馏损失**：真标签vs伪标签损失对比

---

## 故障排查

### 问题1：一致性loss不下降

**原因**：
- S网络学习率过大/过小
- K条路径差异不足

**解决方案**：
```bash
# 调整一致性学习率
--lr_consistency 5e-5  # 减小学习率

# 增加路径数量
--K 16  # 尝试更多路径
```

### 问题2：T集合始终为空

**原因**：
- tau_rmse阈值过严
- delta_rmse要求过高
- 模型精度不足

**解决方案**：
```bash
# 放宽阈值
--tau_rmse 0.1 --delta_rmse 0.005

# 增加监督训练epochs
--supervised_epochs 100
```

### 问题3：n*停滞不推进

**原因**：
- 蒸馏样本数不足
- 伪标签质量差
- lambda_distill权重不当

**解决方案**：
```bash
# 增加伪标签样本数
--samples_per_n 2000

# 调整蒸馏权重
--lambda_distill 0.3  # 降低伪标签权重，更依赖真标签

# 增加蒸馏epochs
--distill_epochs 30
```

### 问题4：磁盘空间不足

**原因**：
- Shard文件累积过多

**解决方案**：
- ShardManager会自动清理旧shards（保留最近2轮）
- 如需手动清理：

```python
from Demo.satellite_sup_2d.pseudo_label_modules import ShardManager

shard_mgr = ShardManager('work_satellite/distill_full', keep_last_n=1)
shard_mgr.cleanup_old_shards(current_round=5)
```

---

## 实验建议

### 初次运行建议

1. **小规模测试**（验证流程）：
```bash
python train_distillation_loop.py \
    --max_rounds 3 \
    --supervised_epochs 20 \
    --consistency_epochs 5 \
    --distill_epochs 10 \
    --batch_size 8 \
    --samples_per_n 200 \
    --component_nums 1 2 3 4 5 \
    --work_name test_small
```

2. **中等规模实验**（调参）：
```bash
python train_distillation_loop.py \
    --max_rounds 5 \
    --supervised_epochs 50 \
    --consistency_epochs 10 \
    --distill_epochs 20 \
    --batch_size 16 \
    --samples_per_n 500 \
    --work_name test_medium
```

3. **完整实验**（最终训练）：
```bash
python train_distillation_loop.py \
    --max_rounds 10 \
    --supervised_epochs 100 \
    --consistency_epochs 20 \
    --distill_epochs 30 \
    --batch_size 16 \
    --samples_per_n 1000 \
    --work_name final_run
```

---

## 扩展与定制

### 添加新的路径采样策略

修改`PathSampler._get_group_order()`：

```python
def _get_group_order(self, path_id: int) -> str:
    # 添加新的分组顺序策略
    orders = ['sequential', 'reverse', 'custom_strategy_1', ...]
    return orders[path_id % len(orders)]
```

### 调整Teacher合格判定条件

修改`EligibilityGate`的判定逻辑：

```python
gate = EligibilityGate(
    tau_rmse=0.05,           # 绝对阈值
    delta_rmse=0.01,         # 绝对改进
    relative_improvement=0.1  # 相对改进（可选）
)
```

### 自定义停止条件

修改`StopController.should_stop()`添加额外条件：

```python
# 例如：添加RMSE停滞检测
if m_CS_dict[n_star] - prev_m_CS > threshold:
    return True, "RMSE不再下降"
```

---

## 引用与参考

如果本框架对您的研究有帮助，请引用：

```bibtex
@software{distillation_framework_2026,
  title={Alternating Superposition and Self-Distillation Framework},
  author={Your Name},
  year={2026},
  url={https://github.com/...}
}
```

---

## 更新日志

### v1.0 (2026-01-12)
- ✅ 初始版本发布
- ✅ 一致性训练模块
- ✅ 分层评估模块
- ✅ 伪标签生成与HDF5管理
- ✅ 循环蒸馏主流程
- ✅ 完整文档与示例

---

## 联系方式

如有问题或建议，请联系：
- Email: your.email@example.com
- Issues: https://github.com/.../issues


