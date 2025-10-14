#!/usr/bin/env bash
set -e

DATA=/data/wqn/turbine_uq/data_post/heat_dataset_780.h5
GPU=4

echo "[UNet]"
python /data/wqn/DENO4pytorch/Demo/satellite_2d_base/run_UNet_satellite.py \
  --data_path ${DATA} --device cuda --cuda_index ${GPU} \
  --ntrain 700 --nvalid 79 --batch_size 2 --epochs 1 --lr 1e-3

echo "[FNO]"
python /data/wqn/DENO4pytorch/Demo/satellite_2d_base/run_FNO_satellite.py \
  --data_path ${DATA} --device cuda --cuda_index ${GPU} \
  --ntrain 700 --nvalid 79 --batch_size 2 --epochs 1 --lr 1e-3 \
  --modes_x 4 --modes_y 4 --width 32 --depth 3

echo "[Trans]"
python /data/wqn/DENO4pytorch/Demo/satellite_2d_base/run_Trans_satellite.py \
  --data_path ${DATA} --device cuda --cuda_index ${GPU} \
  --ntrain 700 --nvalid 79 --batch_size 1 --epochs 1 --lr 1e-3 \
  --down 16 --use_config

echo "[MLP]"
python /data/wqn/DENO4pytorch/Demo/satellite_2d_base/run_MLP_satellite.py \
  --data_path ${DATA} --device cuda --cuda_index ${GPU} \
  --ntrain 700 --nvalid 79 --batch_size 4 --epochs 1 --lr 1e-3 \
  --down 8 --hidden 512 --layers 4

echo "Done."


