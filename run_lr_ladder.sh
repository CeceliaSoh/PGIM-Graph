#!/usr/bin/env bash
set -euo pipefail

# Stage 1/3: learning-rate sweep at num_hops=0, hidden_dim=64
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 0 --hidden-dim 64 --learning-rate 1e-4 --weight-decay 1e-4 --dropout 0.1 --epochs 100
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 0 --hidden-dim 64 --learning-rate 3e-4 --weight-decay 1e-4 --dropout 0.1 --epochs 100
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 0 --hidden-dim 64 --learning-rate 1e-3 --weight-decay 1e-4 --dropout 0.1 --epochs 100
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 0 --hidden-dim 64 --learning-rate 3e-3 --weight-decay 1e-4 --dropout 0.1 --epochs 100
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 0 --hidden-dim 64 --learning-rate 1e-2 --weight-decay 1e-4 --dropout 0.1 --epochs 100

# Stage 2/3: validate the likely best learning rates on larger hop counts
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 3 --hidden-dim 64 --learning-rate 3e-4 --weight-decay 1e-4 --dropout 0.1 --epochs 100
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 3 --hidden-dim 64 --learning-rate 1e-3 --weight-decay 1e-4 --dropout 0.1 --epochs 100
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 7 --hidden-dim 64 --learning-rate 3e-4 --weight-decay 1e-4 --dropout 0.1 --epochs 100
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 7 --hidden-dim 64 --learning-rate 1e-3 --weight-decay 1e-4 --dropout 0.1 --epochs 100

# Stage 3/3: compare model capacity after choosing the best learning rate
# Replace 1e-3 below if Stage 1-2 shows a better learning rate.
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 3 --hidden-dim 64 --ff-hidden-dim 32  --learning-rate 1e-3 --weight-decay 1e-4 --dropout 0.1 --epochs 100
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 3 --hidden-dim 64 --ff-hidden-dim 64  --learning-rate 1e-3 --weight-decay 1e-4 --dropout 0.1 --epochs 100
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 3 --hidden-dim 64 --ff-hidden-dim 128 --learning-rate 1e-3 --weight-decay 1e-4 --dropout 0.1 --epochs 100
