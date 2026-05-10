#!/usr/bin/env bash
# Smoke run: 1 epoch, batch 4, 200/200 train/val with shuffle.
# Verifies the full training -> eval pipeline end-to-end.
set -euo pipefail

cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}

python3 train.py \
    --index data/nuscenes_ped_intent_seq3_v2.pkl \
    --epochs 1 --batch_size 4 \
    --train_limit 200 --val_limit 200 --shuffle \
    --version smoke
