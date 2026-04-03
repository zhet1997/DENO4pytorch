# 循环式叠加蒸馏框架 - 快速上手指南

## 🚀 5分钟快速开始

### 方式1：使用快速启动脚本（推荐）

```bash
cd /data/wqn/Code/DENO4pytorch/Demo/satellite_sup_2d

# 快速测试（15分钟）
bash quick_start.sh quick

# 中等规模（1小时）
bash quick_start.sh medium

# 完整训练（3-5小时）
bash quick_start.sh full

# 仅一致性训练（30分钟）
bash quick_start.sh consistency_only
```

### 方式2：手动运行

```bash
cd /data/wqn/Code/DENO4pytorch

# 阶段1：一致性训练测试
python Demo/satellite_sup_2d/train_consistency.py \
    --epochs 100 \
    --consistency_epochs 5 \
    --work_name test_consistency

# 阶段3：完整蒸馏循环
python Demo/satellite_sup_2d/train_distillation_loop.py \
    --max_rounds 5 \
    --supervised_epochs 50 \
    --consistency_epochs 10 \
    --distill_epochs 20 \
    --work_name test_distillation
```

---

## 📊 查看结果

训练完成后，结果保存在 `work_satellite/[work_name]/` 目录：

```bash
# 查看训练日志
tail -f work_satellite/test_distillation/distillation_loop.log

# 查看round摘要
cat work_satellite/test_distillation/round_summary.log

# 查看JSON摘要
python -m json.tool work_satellite/test_distillation/training_summary.json

# 检查磁盘使用
du -sh work_satellite/test_distillation/pseudo_shards/
```

---

## 🔍 验证指标

### 一致性训练成功标志
- ✅ `consistency_loss` 持续下降
- ✅ `consistency_std`（路径标准差）下降
- ✅ `valid_supervised_loss` 不恶化

### 蒸馏训练成功标志
- ✅ `n_star` 逐渐推进（或稳定在高值）
- ✅ `m_CS[n]` < `m_C[n]`（C+S优于C）
- ✅ `T`集合逐渐扩大
- ✅ `[A,B]`区间RMSE下降

---

## ⚠️ 常见问题

### 问题1：JSON序列化错误
**已修复** ✅ - numpy类型已自动转换为Python原生类型

### 问题2：T集合始终为空
**解决方案**：放宽阈值
```bash
--tau_rmse 0.08 --delta_rmse 0.005
```

### 问题3：n*停滞不推进
**解决方案**：增加样本数或调整权重
```bash
--samples_per_n 2000 --lambda_distill 0.3
```

### 问题4：磁盘空间不足
**自动清理**：ShardManager会保留最近2轮，自动删除旧shards

---

## 📚 详细文档

- **完整文档**: [`README_distillation_framework.md`](README_distillation_framework.md)
- **实施报告**: [`IMPLEMENTATION_COMPLETE.md`](IMPLEMENTATION_COMPLETE.md)
- **参数说明**: 见README第4节

---

## 🎯 推荐实验流程

### 第1步：快速验证（15分钟）
```bash
bash quick_start.sh quick
```
**目标**: 验证流程能跑通，无报错

### 第2步：调参实验（1小时）
```bash
bash quick_start.sh medium
```
**目标**: 观察n*推进，调整tau_rmse和delta_rmse

### 第3步：完整训练（3-5小时）
```bash
bash quick_start.sh full
```
**目标**: 获得最终模型，评估性能

---

## 📝 实验记录模板

```
实验名称: _________________
运行时间: _________________
参数设置:
  - max_rounds: _____
  - tau_rmse: _____
  - delta_rmse: _____
  - lambda_distill: _____

结果:
  - 最终n*: _____
  - T集合: _____
  - 最优RMSE: _____
  - 停止原因: _____

观察:
  - ___________________________
  - ___________________________

改进方向:
  - ___________________________
  - ___________________________
```

---

## ✅ 验收检查清单

运行完成后，检查以下项目：

- [ ] 训练日志无ERROR
- [ ] `training_summary.json`能正常加载
- [ ] `n_star` > 0
- [ ] `T`集合非空
- [ ] `pseudo_shards/`目录存在且有文件
- [ ] `teachers/`目录存在且有快照
- [ ] 磁盘空间合理（自动清理生效）
- [ ] 能加载最终模型进行推理

---

## 🛠️ 故障排查

### 检查GPU使用
```bash
nvidia-smi
# 应看到train_distillation_loop.py进程
```

### 检查日志
```bash
# 实时查看
tail -f work_satellite/test_distillation/distillation_loop.log

# 搜索错误
grep ERROR work_satellite/test_distillation/distillation_loop.log
```

### 检查进度
```bash
# 查看当前round
grep "Round " work_satellite/test_distillation/distillation_loop.log | tail -5

# 查看n*演化
grep "n_star" work_satellite/test_distillation/round_summary.log
```

---

## 📞 获取帮助

1. 查阅完整文档：[`README_distillation_framework.md`](README_distillation_framework.md)
2. 检查实施报告：[`IMPLEMENTATION_COMPLETE.md`](IMPLEMENTATION_COMPLETE.md)
3. 运行测试脚本：`python test_json_fix_simple.py`

---

**祝实验顺利！** 🎉

