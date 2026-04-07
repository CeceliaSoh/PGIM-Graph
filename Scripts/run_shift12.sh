CUDA_VISIBLE_DEVICES=7 python /home/cecelia/project/PGIM-Graph/main.py \
  --root /home/cecelia/project/PGIM-Graph/dataset/ccr \
  --feature feature.npy \
  --egde-file graph_link_250m/links.txt \
  --num-hops 0 \
  --window-size 12 \
  --feat-norm true \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --batch-size 256 \
  --dropout 0.2 \
  --shift 12 \
  --predict-last false \
  --target-mask-mode train_allow_interpolated \
  --wandb-project pgim-mask-trans-shift12-exp

CUDA_VISIBLE_DEVICES=7 python /home/cecelia/project/PGIM-Graph/main.py \
  --root /home/cecelia/project/PGIM-Graph/dataset/ccr \
  --feature feature.npy \
  --egde-file graph_link_250m/links.txt \
  --num-hops 1 \
  --window-size 12 \
  --feat-norm true \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --batch-size 256 \
  --dropout 0.2 \
  --shift 12 \
  --predict-last false \
  --target-mask-mode train_allow_interpolated \
  --wandb-project pgim-mask-trans-shift12-exp

CUDA_VISIBLE_DEVICES=7 python /home/cecelia/project/PGIM-Graph/main.py \
  --root /home/cecelia/project/PGIM-Graph/dataset/ccr \
  --feature feature.npy \
  --egde-file graph_link_250m/links.txt \
  --num-hops 2 \
  --window-size 12 \
  --feat-norm true \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --batch-size 256 \
  --dropout 0.2 \
  --shift 12 \
  --predict-last false \
  --target-mask-mode train_allow_interpolated \
  --wandb-project pgim-mask-trans-shift12-exp