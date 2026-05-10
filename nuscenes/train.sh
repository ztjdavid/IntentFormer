#!/usr/bin/env bash
# Full training with required CUDA env. Args forwarded to train.py.
set -euo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.8/dist-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH:-}
exec python3 train.py "$@"
