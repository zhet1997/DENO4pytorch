#!/bin/bash

###############################################################################
# 单个实验测试脚本
#
# 功能：快速测试训练流程，验证：
#   1. 数据加载和通道压缩
#   2. 训练和验证循环
#   3. 日志和可视化输出
#
# 使用方法：
#   bash test_single_exp.sh
###############################################################################

echo "=========================================="
echo "测试单个实验"
echo "=========================================="

# 测试配置（小规模，快速验证）
WORK_NAME="test_kmax3_S"
TRAIN_NUMS="1,2,3"
VALID_NUMS="1,2,3,4,5,6"
NTRAIN_PER_K=800  # 减少样本数
NVALID_PER_K=40
EPOCHS=300          # 只跑3个epoch
BATCH_SIZE=16      # 减小batch size
GPU=1

echo "实验名称: $WORK_NAME"
echo "训练K桶: $TRAIN_NUMS"
echo "验证K桶: $VALID_NUMS"
echo "Epochs: $EPOCHS"
echo ""

# 清理旧的测试结果
if [ -d "runs/$WORK_NAME" ]; then
  echo "清理旧的测试结果..."
  rm -rf "runs/$WORK_NAME"
fi

# 运行测试
echo "开始测试..."
echo ""

# 激活conda环境
source /data/conda/etc/profile.d/conda.sh
conda activate torch_py310

python Demo/satellite_sup_2d/train_entry.py \
  --work_name $WORK_NAME \
  --train_component_nums $TRAIN_NUMS \
  --valid_component_nums $VALID_NUMS \
  --ntrain_perK $NTRAIN_PER_K \
  --nvalid_perK $NVALID_PER_K \
  --super_train_mode S012 \
  --super_nums_eval "0,1,2" \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr 1e-3 \
  --target_U_channels 8 \
  --downsample 2 \
  --gpu $GPU

# 检查结果
echo ""
echo "=========================================="
echo "测试完成，检查输出文件..."
echo "=========================================="

if [ -d "runs/$WORK_NAME" ]; then
  echo "✓ 工作目录已创建: runs/$WORK_NAME"
  
  if [ -f "runs/$WORK_NAME/config.json" ]; then
    echo "✓ 配置文件: config.json"
  else
    echo "✗ 缺少: config.json"
  fi
  
  if [ -f "runs/$WORK_NAME/metrics.json" ]; then
    echo "✓ 指标文件: metrics.json"
    echo "  最后一行:"
    tail -n 3 "runs/$WORK_NAME/metrics.json"
  else
    echo "✗ 缺少: metrics.json"
  fi
  
  if [ -f "runs/$WORK_NAME/metrics_summary.csv" ]; then
    echo "✓ CSV摘要: metrics_summary.csv"
    echo "  内容:"
    cat "runs/$WORK_NAME/metrics_summary.csv"
  else
    echo "✗ 缺少: metrics_summary.csv"
  fi
  
  if [ -f "runs/$WORK_NAME/ckpt_best.pth" ]; then
    echo "✓ 最佳模型: ckpt_best.pth"
  else
    echo "✗ 缺少: ckpt_best.pth"
  fi
  
  if [ -f "runs/$WORK_NAME/ckpt_last.pth" ]; then
    echo "✓ 最新模型: ckpt_last.pth"
  else
    echo "✗ 缺少: ckpt_last.pth"
  fi
  
  if [ -f "runs/$WORK_NAME/log_loss.svg" ]; then
    echo "✓ 损失曲线: log_loss.svg"
  else
    echo "✗ 缺少: log_loss.svg"
  fi
  
  # 检查可视化图片
  sample_count=$(ls runs/$WORK_NAME/train_s*.jpg 2>/dev/null | wc -l)
  if [ $sample_count -gt 0 ]; then
    echo "✓ 可视化样本: $sample_count 张图片"
  else
    echo "✗ 缺少可视化样本"
  fi
  
else
  echo "✗ 工作目录未创建"
fi

echo ""
echo "=========================================="
echo "测试总结"
echo "=========================================="
echo "如果所有项目都显示 ✓，说明训练流程正常"
echo "查看完整日志: cat runs/$WORK_NAME/metrics.json"
echo "=========================================="

