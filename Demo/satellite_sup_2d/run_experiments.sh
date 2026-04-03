#!/bin/bash

###############################################################################
# 批量实验启动脚本 - 卫星数据叠加消融实验
#
# 实验矩阵：
#   - Kmax: 1, 2, 4, 6, 8, 10 (6种)
#   - super_train_mode: S0, S01, S012 (3种)
#   - 总计: 6 × 3 = 18 个实验
#
# 使用方法：
#   bash run_experiments.sh
#
# 注意：
#   - 修改 GPUS 数组以匹配可用GPU
#   - 修改 BASE_DIR 为实际项目路径
###############################################################################

# 配置
BASE_DIR="Demo/satellite_sup_2d"
GPUS=(1 6 7)  # 可用GPU列表
gpu_idx=0

# 固定参数
VALID_NUMS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
NTRAIN_PER_K=800
NVALID_PER_K=40
EPOCHS=1000
BATCH_SIZE=16
LR=1e-3
TARGET_U_CHANNELS=16
DOWNSAMPLE=2

echo "=========================================="
echo "批量实验启动"
echo "=========================================="
echo "GPU列表: ${GPUS[@]}"
echo "实验数量: 18 (6 Kmax × 3 modes)"
echo ""

# 激活conda环境
source /data/conda/etc/profile.d/conda.sh
conda activate torch_py310
echo "Conda环境已激活: torch_py310"
echo ""

# 遍历实验矩阵
for Kmax in 1 2 4 6 8 10; do
  for mode in S0 S01 S012; do
    # 生成实验名称
    work_name="kmax${Kmax}_${mode}"
    
    # 生成训练K列表
    if [ $Kmax -eq 1 ]; then
      train_nums="1"
    else
      train_nums=$(seq -s',' 1 $Kmax)
    fi
    
    # 根据训练模式设置评估的super_nums
    if [ "$mode" = "S0" ]; then
      eval_nums="0"
    elif [ "$mode" = "S01" ]; then
      eval_nums="0,1"
    else  # S012
      eval_nums="0,1,2"
    fi
    
    # 选择GPU（轮转）
    gpu=${GPUS[$gpu_idx]}
    
    # 创建输出目录
    mkdir -p runs/${work_name}
    
    # 启动训练（后台运行）
    python ${BASE_DIR}/train_entry.py \
      --work_name $work_name \
      --train_component_nums $train_nums \
      --valid_component_nums $VALID_NUMS \
      --ntrain_perK $NTRAIN_PER_K \
      --nvalid_perK $NVALID_PER_K \
      --super_train_mode $mode \
      --super_nums_eval "$eval_nums" \
      --epochs $EPOCHS \
      --batch_size $BATCH_SIZE \
      --lr $LR \
      --target_U_channels $TARGET_U_CHANNELS \
      --downsample $DOWNSAMPLE \
      --gpu $gpu \
      &> runs/${work_name}/train.log &
    
    # 记录PID
    pid=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动实验: $work_name on GPU$gpu (PID: $pid)"
    
    # 更新GPU索引（轮转）
    gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
    
    # 限制并发数（等于GPU数量）
    if [ $(jobs -r | wc -l) -ge ${#GPUS[@]} ]; then
      echo "  等待GPU资源..."
      wait -n  # 等待任意一个后台任务完成
    fi
    
    # 短暂延迟，避免同时启动过多进程
    sleep 2
  done
done

echo ""
echo "=========================================="
echo "所有实验已启动，等待完成..."
echo "=========================================="

# 等待所有后台任务完成
wait

echo ""
echo "=========================================="
echo "所有实验完成！"
echo "=========================================="
echo "结果保存在: runs/"
echo ""

# 生成简单的汇总报告
echo "实验列表:"
for Kmax in 1 2 4 6 8 10; do
  for mode in S0 S01 S012; do
    work_name="kmax${Kmax}_${mode}"
    if [ -f "runs/${work_name}/ckpt_best.pth" ]; then
      echo "  ✓ $work_name"
    else
      echo "  ✗ $work_name (未完成或失败)"
    fi
  done
done

echo ""
echo "查看单个实验日志: tail -f runs/<work_name>/train.log"
echo "=========================================="

