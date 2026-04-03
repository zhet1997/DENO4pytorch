#!/bin/bash

# BC组件化批量实验启动脚本
# 实验矩阵: Kmax x mode = 18个实验
# 结果保存在 runs_bc/ 目录下

BASE_DIR="Demo/satellite_sup_2d"
GPUS=(1 6 7)
gpu_idx=0
SAVE_DIR="runs_bc"

VALID_NUMS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
NTRAIN_PER_K=800
NVALID_PER_K=40
EPOCHS=1000
BATCH_SIZE=16
LR=1e-3
TARGET_U_CHANNELS=16
DOWNSAMPLE=2
NUM_BC_COMPONENTS=1
BC_DIM=16

echo "=========================================="
echo "BC组件化批量实验启动"
echo "=========================================="
echo "GPU列表: ${GPUS[@]}"
echo "结果保存: ${SAVE_DIR}/"

mkdir -p ${SAVE_DIR}
source /data/conda/etc/profile.d/conda.sh
conda activate torch_py310

for Kmax in 1 2 4 6 8 10; do
  for mode in S0 S01 S012; do
    work_name="kmax${Kmax}_${mode}_bc"
    
    if [ $Kmax -eq 1 ]; then
      train_nums="1"
    else
      train_nums=$(seq -s',' 1 $Kmax)
    fi
    
    if [ "$mode" = "S0" ]; then
      eval_nums="0"
    elif [ "$mode" = "S01" ]; then
      eval_nums="0,1"
    else
      eval_nums="0,1,2"
    fi
    
    gpu=${GPUS[$gpu_idx]}
    mkdir -p ${SAVE_DIR}/${work_name}
    
    python ${BASE_DIR}/train_entry_bc.py \
      --work_name $work_name \
      --save_dir $SAVE_DIR \
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
      --num_bc_components $NUM_BC_COMPONENTS \
      --bc_dim $BC_DIM \
      --gpu $gpu \
      &> ${SAVE_DIR}/${work_name}/train.log &
    
    pid=$!
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动: $work_name on GPU$gpu (PID: $pid)"
    
    gpu_idx=$(( (gpu_idx + 1) % ${#GPUS[@]} ))
    
    if [ $(jobs -r | wc -l) -ge ${#GPUS[@]} ]; then
      wait -n
    fi
    sleep 2
  done
done

echo "等待所有实验完成..."
wait

echo "=========================================="
echo "所有实验完成！结果保存在: ${SAVE_DIR}/"
echo "=========================================="

for Kmax in 1 2 4 6 8 10; do
  for mode in S0 S01 S012; do
    work_name="kmax${Kmax}_${mode}_bc"
    if [ -f "${SAVE_DIR}/${work_name}/ckpt_best.pth" ]; then
      echo "  ✓ $work_name"
    else
      echo "  ✗ $work_name"
    fi
  done
done

echo "对比: 原方法(runs/) vs BC组件化(runs_bc/)"
