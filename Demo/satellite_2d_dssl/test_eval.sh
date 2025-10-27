#!/bin/bash
# 测试评估脚本（MLP模型）

cd /data/wqn/DENO4pytorch/Demo/satellite_2d_dssl

python eval_satellite_predict.py \
    --model mlp \
    --data_path /data/wqn/datasets/packaged_dataset20251017_6c/heat_dataset.h5 \
    --ckpt /data/wqn/DENO4pytorch/work_satellite_dssl_mlp_20251021/MLP_DSSL_n2000_20251021_225421/ \
    --device cuda \
    --cuda_index 6 \
    --batch_size 8 \
    --split valid \
    --max_plots 5

