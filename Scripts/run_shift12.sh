CUDA_VISIBLE_DEVICES=7 python /home/cecelia/project/PGIM-Graph/main.py \
  --root /home/cecelia/project/PGIM-Graph/dataset/ccr \
  --feature feature.npy \
  --egde-file graph_link_250m/links.txt \
  --num-hops 3 \
  --window-size 12 \
  --feat-norm true \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --batch-size 256 \
  --dropout 0.2 \
  --shift 12 \
  --predict-last true \
  --target-mask-mode train_allow_interpolated \
  --wandb-project pgim-gnn-regressor-shift12-exp

CUDA_VISIBLE_DEVICES=7 python /home/cecelia/project/PGIM-Graph/main.py \
  --root /home/cecelia/project/PGIM-Graph/dataset/ccr \
  --feature feature.npy \
  --egde-file graph_link_250m/links.txt \
  --num-hops 5 \
  --window-size 12 \
  --feat-norm true \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --batch-size 256 \
  --dropout 0.2 \
  --shift 12 \
  --predict-last true \
  --target-mask-mode train_allow_interpolated \
  --wandb-project pgim-gnn-regressor-shift12-exp

CUDA_VISIBLE_DEVICES=7 python /home/cecelia/project/PGIM-Graph/main.py \
  --root /home/cecelia/project/PGIM-Graph/dataset/ccr \
  --feature feature.npy \
  --egde-file graph_link_250m/links.txt \
  --num-hops 7 \
  --window-size 12 \
  --feat-norm true \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --batch-size 256 \
  --dropout 0.2 \
  --shift 12 \
  --predict-last true \
  --target-mask-mode train_allow_interpolated \
  --wandb-project pgim-gnn-regressor-shift12-exp