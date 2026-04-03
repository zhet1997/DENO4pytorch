#!/bin/bash

# 修改后 BC module 的单实验完整训练入口
# 用法:
#   bash Demo/satellite_sup_2d/run_train_bc_full.sh
#   bash Demo/satellite_sup_2d/run_train_bc_full.sh --epochs 300 --gpu 1

BASE_DIR="Demo/satellite_sup_2d"
WORK_NAME="bc_full_kmax10_S012"
SAVE_DIR="runs_bc_full"
TRAIN_COMPONENT_NUMS="1,2,3,4,5,6,7,8,9,10"
VALID_COMPONENT_NUMS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25"
NTRAIN_PER_K=800
NVALID_PER_K=40
SUPER_TRAIN_MODE="S012"
SUPER_NUMS_EVAL="0,1,2"
EPOCHS=1000
BATCH_SIZE=16
LR=1e-3
TARGET_U_CHANNELS=16
DOWNSAMPLE=2
NUM_BC_COMPONENTS=2
BC_DIM=16
BC_ENCODER_TYPE="linear"
PARTITION_STRATEGY="sequential"
BASE_PATH="/data/wqn/datasets/dataset_20251218_mc"
GPU=0
SEED=8905

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work_name)
      WORK_NAME="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --save_dir)
      SAVE_DIR="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --train_component_nums)
      TRAIN_COMPONENT_NUMS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --valid_component_nums)
      VALID_COMPONENT_NUMS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --ntrain_perK)
      NTRAIN_PER_K="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --nvalid_perK)
      NVALID_PER_K="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --super_train_mode)
      SUPER_TRAIN_MODE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --super_nums_eval)
      SUPER_NUMS_EVAL="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --lr)
      LR="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --target_U_channels)
      TARGET_U_CHANNELS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --downsample)
      DOWNSAMPLE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --num_bc_components)
      NUM_BC_COMPONENTS="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --bc_dim)
      BC_DIM="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --bc_encoder_type)
      BC_ENCODER_TYPE="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --partition_strategy)
      PARTITION_STRATEGY="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --base_path)
      BASE_PATH="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --gpu)
      GPU="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    --seed)
      SEED="$2"
      EXTRA_ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

WORK_PATH="${SAVE_DIR}/${WORK_NAME}"
LOG_PATH="${WORK_PATH}/train.log"

mkdir -p "${WORK_PATH}"

source /data/conda/etc/profile.d/conda.sh
conda activate torch_py310

export PYTHONPATH="/data/wqn/Code/DENO4pytorch:/data/wqn/Code/DENO4pytorch/Models:/data/wqn/Code/DENO4pytorch/Utilizes:${PYTHONPATH}"

echo "======================================================================"
echo "修改后 BC module 完整训练"
echo "======================================================================"
echo "work_name: ${WORK_NAME}"
echo "save_dir: ${SAVE_DIR}"
echo "base_path: ${BASE_PATH}"
echo "train_component_nums: ${TRAIN_COMPONENT_NUMS}"
echo "valid_component_nums: ${VALID_COMPONENT_NUMS}"
echo "super_train_mode: ${SUPER_TRAIN_MODE}"
echo "super_nums_eval: ${SUPER_NUMS_EVAL}"
echo "num_bc_components: ${NUM_BC_COMPONENTS}"
echo "bc_dim: ${BC_DIM}"
echo "bc_encoder_type: ${BC_ENCODER_TYPE}"
echo "partition_strategy: ${PARTITION_STRATEGY}"
echo "gpu: ${GPU}"
echo "log_path: ${LOG_PATH}"
echo "======================================================================"

python "${BASE_DIR}/train_entry_bc.py" \
  --work_name "${WORK_NAME}" \
  --save_dir "${SAVE_DIR}" \
  --train_component_nums "${TRAIN_COMPONENT_NUMS}" \
  --valid_component_nums "${VALID_COMPONENT_NUMS}" \
  --ntrain_perK "${NTRAIN_PER_K}" \
  --nvalid_perK "${NVALID_PER_K}" \
  --super_train_mode "${SUPER_TRAIN_MODE}" \
  --super_nums_eval "${SUPER_NUMS_EVAL}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --target_U_channels "${TARGET_U_CHANNELS}" \
  --downsample "${DOWNSAMPLE}" \
  --num_bc_components "${NUM_BC_COMPONENTS}" \
  --bc_dim "${BC_DIM}" \
  --bc_encoder_type "${BC_ENCODER_TYPE}" \
  --partition_strategy "${PARTITION_STRATEGY}" \
  --base_path "${BASE_PATH}" \
  --gpu "${GPU}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
