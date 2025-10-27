#!/bin/bash

# 训练脚本：串行运行不同样本量的 MLP 和 FNO 实验
# 创建时间: 2025-10-18
# 用法: 在项目根目录执行 bash Demo/satellite_2d_base/run_experiments.sh

# 获取脚本所在目录并切换到该目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1
echo "工作目录: $(pwd)"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=6

# 设置日志文件
LOG_FILE="experiments_$(date +%Y%m%d_%H%M%S).log"
echo "实验开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# 定义训练样本数量列表
NTRAIN_LIST=(100 500 1000 2000 4000 8000)

# 循环遍历不同的训练样本数
for ntrain in "${NTRAIN_LIST[@]}"; do
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "开始训练 ntrain=$ntrain" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    
    # 训练 MLP
    echo "" | tee -a "$LOG_FILE"
    echo "[MLP] ntrain=$ntrain 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    python run_MLP_satellite.py \
        --ntrain $ntrain \
        --cuda_index 0 \
        --work_dir work_satellite_mlp
    
    if [ $? -eq 0 ]; then
        echo "[MLP] ntrain=$ntrain 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [成功]" | tee -a "$LOG_FILE"
    else
        echo "[MLP] ntrain=$ntrain 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [失败]" | tee -a "$LOG_FILE"
    fi
    
    # 训练 FNO
    echo "" | tee -a "$LOG_FILE"
    echo "[FNO] ntrain=$ntrain 开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
    python run_FNO_satellite.py \
        --ntrain $ntrain \
        --cuda_index 0 \
        --work_dir work_satellite_fno
    
    if [ $? -eq 0 ]; then
        echo "[FNO] ntrain=$ntrain 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [成功]" | tee -a "$LOG_FILE"
    else
        echo "[FNO] ntrain=$ntrain 结束时间: $(date '+%Y-%m-%d %H:%M:%S') [失败]" | tee -a "$LOG_FILE"
    fi
    
    echo "----------------------------------------" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "所有实验完成时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

