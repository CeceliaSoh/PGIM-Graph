CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 0 --learning-rate 1e-2
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 1 --learning-rate 1e-2
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 3 --learning-rate 1e-2
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 5 --learning-rate 1e-2
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 7 --learning-rate 1e-2

CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 0 --learning-rate 1e-2 --hidden-dim 32 --ff-hidden-dim 32
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 1 --learning-rate 1e-2 --hidden-dim 32 --ff-hidden-dim 32
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 3 --learning-rate 1e-2 --hidden-dim 32 --ff-hidden-dim 32
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 5 --learning-rate 1e-2 --hidden-dim 32 --ff-hidden-dim 32
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 7 --learning-rate 1e-2 --hidden-dim 32 --ff-hidden-dim 32

CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 1 --learning-rate 1e-2 --hidden-dim 32 --ff-hidden-dim 64
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 3 --learning-rate 1e-2 --hidden-dim 32 --ff-hidden-dim 64
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 5 --learning-rate 1e-2 --hidden-dim 32 --ff-hidden-dim 64
CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 7 --learning-rate 1e-2 --hidden-dim 32 --ff-hidden-dim 64