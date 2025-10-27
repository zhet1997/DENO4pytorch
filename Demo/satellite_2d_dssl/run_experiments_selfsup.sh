#!/bin/bash

# 训练脚本：串行运行不同样本量的 MLP_DSSL 和 FNO_DSSL 实验（自监督学习）
# 创建时间: 2025-10-18
# 用法: 在项目根目录执行 bash Demo/satellite_2d_dssl/run_experiments_selfsup.sh

# 获取脚本所在目录并切换到该目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
echo "工作目录: $(pwd)"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=6

# 设置日志文件
LOG_FILE="experiments_selfsup_$(date +%Y%m%d_%H%M%S).log"
echo "自监督实验开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# 定义训练样本数量列表
NTRAIN_LIST=(100 500 1000 2000 4000 8000)

# 自监督数据目录（根据实际情况修改）
SELFSUP_DIR="/data/wqn/datasets/packaged_dataset20251017_6c_sim1/"

# 循环遍历不同的训练样本数
for ntrain in "${NTRAIN_LIST[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "开始训练 ntrain=$ntrain (自监督)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # 训练 MLP_DSSL (自监督)
    echo "" | tee -a "$LOG_FILE"
    echo "[MLP_DSSL] ntrain=$ntrain 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    python run_MLP_satellite_selfsup.py \
        --ntrain $ntrain \
        --cuda_index 0 \
        --work_dir work_satellite_dssl_mlp \
        --selfsup_dir "$SELFSUP_DIR" \
        --self_batch_size 32 \
        --self_lr_final 2e-5 \
        --self_lr_start 2e-6
    
    if [ $? -eq 0 ]; then
        echo "[MLP_DSSL] ntrain=$ntrain 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [成功]" | tee -a "$LOG_FILE"
    else
        echo "[MLP_DSSL] ntrain=$ntrain 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [失败]" | tee -a "$LOG_FILE"
    fi
    
    # 训练 FNO_DSSL (自监督)
    echo "" | tee -a "$LOG_FILE"
    echo "[FNO_DSSL] ntrain=$ntrain 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    python run_FNO_satellite_selfsup.py \
        --ntrain $ntrain \
        --cuda_index 0 \
        --work_dir work_satellite_dssl_fno \
        --selfsup_dir "$SELFSUP_DIR" \
        --self_batch_size 32 \
        --self_lr_final 1e-4
    
    if [ $? -eq 0 ]; then
        echo "[FNO_DSSL] ntrain=$ntrain 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [成功]" | tee -a "$LOG_FILE"
    else
        echo "[FNO_DSSL] ntrain=$ntrain 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [失败]" | tee -a "$LOG_FILE"
    fi
    
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "所有自监督实验完成时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"


