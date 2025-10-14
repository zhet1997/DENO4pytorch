#!/usr/bin/env bash
# Robust serial benchmark on 3 datasets × 4 models. GPU index is fixed to 6.

set -u -o pipefail

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1

ROOT=/data/wqn/DENO4pytorch
GPU=4
EPOCHS=${EPOCHS:-1000}

timestamp=$(date +%Y%m%d_%H%M%S)
LOG_DIR=${ROOT}/work_satellite/logs_${timestamp}
mkdir -p "${LOG_DIR}"

DATASETS=(
  "/data/wqn/datasets/packaged_dataset20251010/heat_dataset.h5 8000 100"
)

run_with_retries() {
  local name="$1"; shift
  local log_file="$1"; shift

  local tries=("$@")
  local i=0
  for cmd in "${tries[@]}"; do
    i=$((i+1))
    echo "[RUN ${name}] Try ${i}: ${cmd}" | tee -a "${log_file}"
    bash -lc "${cmd}" >>"${log_file}" 2>&1 && {
      echo "[RUN ${name}] Success on try ${i}" | tee -a "${log_file}"
      return 0
    }
    echo "[RUN ${name}] Failed on try ${i}" | tee -a "${log_file}"
  done
  echo "[RUN ${name}] All retries failed" | tee -a "${log_file}"
  return 1
}

for item in "${DATASETS[@]}"; do
  read -r DATA NTRAIN NVALID <<<"${item}"
  base=$(basename "${DATA}")
  tag="${base%.*}_nt${NTRAIN}_nv${NVALID}"


# FNO
  fno_log="${LOG_DIR}/fno_${tag}.log"
  fno_cmds=(
    "python ${ROOT}/Demo/satellite_2d_base/run_FNO_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 1e-4 --modes_x 10 --modes_y 10 --width 64 --depth 3"
    "python ${ROOT}/Demo/satellite_2d_base/run_FNO_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 1e-4 --modes_x 10 --modes_y 10 --width 64 --depth 3"
    "python ${ROOT}/Demo/satellite_2d_base/run_FNO_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 1e-4 --modes_x 10 --modes_y 10 --width 64 --depth 3"
  )
  run_with_retries "FNO:${tag}" "${fno_log}" "${fno_cmds[@]}"

  # Transformer（使用配置；无下采样，专用学习率）
  trans_log="${LOG_DIR}/trans_${tag}.log"
  trans_cmds=(
    "python ${ROOT}/Demo/satellite_2d_base/run_Trans_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 5e-4 --down 1 --use_config"
    "python ${ROOT}/Demo/satellite_2d_base/run_Trans_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 5e-4 --down 1 --use_config"
    "python ${ROOT}/Demo/satellite_2d_base/run_Trans_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 5e-4 --down 1 --use_config"
  )
  run_with_retries "Trans:${tag}" "${trans_log}" "${trans_cmds[@]}"

  # MLP（下采样=4）
  mlp_log="${LOG_DIR}/mlp_${tag}.log"
  mlp_cmds=(
    "python ${ROOT}/Demo/satellite_2d_base/run_MLP_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 1e-3 --down 4 --hidden 1024 --layers 4"
    "python ${ROOT}/Demo/satellite_2d_base/run_MLP_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 1e-3 --down 4 --hidden 1024 --layers 4"
    "python ${ROOT}/Demo/satellite_2d_base/run_MLP_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 1e-3 --down 4 --hidden 1024 --layers 4"
  )
  run_with_retries "MLP:${tag}" "${mlp_log}" "${mlp_cmds[@]}"


  # UNet
  unet_log="${LOG_DIR}/unet_${tag}.log"
  unet_cmds=(
    "python ${ROOT}/Demo/satellite_2d_base/run_UNet_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 1e-3"
    "python ${ROOT}/Demo/satellite_2d_base/run_UNet_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 1e-3"
    "python ${ROOT}/Demo/satellite_2d_base/run_UNet_satellite.py --data_path ${DATA} --device cuda --cuda_index ${GPU} --ntrain ${NTRAIN} --nvalid ${NVALID} --batch_size 32 --epochs ${EPOCHS} --lr 1e-3"
  )
  run_with_retries "UNet:${tag}" "${unet_log}" "${unet_cmds[@]}"

done

echo "All benchmarks completed. Logs at: ${LOG_DIR}"


