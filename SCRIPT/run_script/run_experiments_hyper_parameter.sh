#!/bin/bash

# 训练脚本：串行运行不同 loss_gap 的 MLP_DSSL 和 FNO_DSSL 实验（超参数搜索）
# 创建时间: 2025-11-12
# 用法: bash SCRIPT/batch_script/run_experiments_hyper_parameter.sh

# 获取脚本所在目录并切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_SCRIPT_DIR="$PROJECT_ROOT/SCRIPT/run_script"

cd "$RUN_SCRIPT_DIR" || exit 1
echo "工作目录: $(pwd)"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=7

# 设置日志文件
LOG_FILE="experiments_lossgap_$(date +%Y%m%d_%H%M%S).log"
echo "超参数实验开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# 定义 loss_gap 列表
LOSS_GAP_LIST=(0.05 0.1 1 5)

# 固定训练样本数量
NTRAIN=2500

# 自监督数据目录
SELFSUP_DIR="/data/wqn/datasets/packaged_dataset20251017_6c_sim1/"

# 循环遍历不同的 loss_gap
for loss_gap in "${LOSS_GAP_LIST[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "开始训练 loss_gap=$loss_gap (ntrain=$NTRAIN)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # 训练 MLP_DSSL (自监督)
    echo "" | tee -a "$LOG_FILE"
    echo "[MLP_DSSL] loss_gap=$loss_gap 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    python run_MLP_satellite_selfsup_new.py \
        --ntrain $NTRAIN \
        --cuda_index 0 \
        --work_dir work_satellite_dssl_mlp_lossgap \
        --selfsup_dir "$SELFSUP_DIR" \
        --self_batch_size 32 \
        --self_lr_final 2e-5 \
        --self_lr_start 2e-6 \
        --loss_gap $loss_gap
    
    if [ $? -eq 0 ]; then
        echo "[MLP_DSSL] loss_gap=$loss_gap 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [成功]" | tee -a "$LOG_FILE"
    else
        echo "[MLP_DSSL] loss_gap=$loss_gap 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [失败]" | tee -a "$LOG_FILE"
    fi
    
    # 训练 FNO_DSSL (自监督)
    echo "" | tee -a "$LOG_FILE"
    echo "[FNO_DSSL] loss_gap=$loss_gap 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    python run_FNO_satellite_selfsup_new.py \
        --ntrain $NTRAIN \
        --cuda_index 0 \
        --work_dir work_satellite_dssl_fno_lossgap \
        --selfsup_dir "$SELFSUP_DIR" \
        --self_batch_size 32 \
        --self_lr_final 1e-4 \
        --loss_gap $loss_gap
    
    if [ $? -eq 0 ]; then
        echo "[FNO_DSSL] loss_gap=$loss_gap 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [成功]" | tee -a "$LOG_FILE"
    else
        echo "[FNO_DSSL] loss_gap=$loss_gap 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [失败]" | tee -a "$LOG_FILE"
    fi
    
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "所有超参数实验完成时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"


