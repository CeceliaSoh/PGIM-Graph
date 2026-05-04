CUDA_VISIBLE_DEVICES=4 python /home/cecelia/project/PGIM-Graph/main.py \
  --root /home/cecelia/project/PGIM-Graph/dataset/ccr \
  --feature feature_timesfm.npy \
  --egde-file graph_link_300m/links.txt \
  --num-hops 3 \
  --window-size 12 \
  --feat-norm true \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --batch-size 256 \
  --dropout 0.2 \
  --shift 1 \
  --predict-last false \
  --target-mask-mode train_allow_interpolated \
  --wandb-project pgim-mask-trans-shift1-exp \
  --run-name-sufix dist300_timesfm

CUDA_VISIBLE_DEVICES=6 python /home/cecelia/project/PGIM-Graph/main.py \
  --root /home/cecelia/project/PGIM-Graph/dataset/ccr \
  --feature feature.npy \
  --egde-file graph_link_300m/links.txt \
  --num-hops 3 \
  --window-size 12 \
  --feat-norm true \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --batch-size 256 \
  --dropout 0.2 \
  --shift 1 \
  --predict-last false \
  --target-mask-mode train_allow_interpolated \
  --wandb-project pgim-mask-trans-shift1-exp \
  --run-name-sufix dist300

CUDA_VISIBLE_DEVICES=6 python /home/cecelia/project/PGIM-Graph/main.py \
  --root /home/cecelia/project/PGIM-Graph/dataset/ccr \
  --feature feature.npy \
  --egde-file graph_link_250m_distance_or_mrt_radius/links.txt \
  --num-hops 3 \
  --window-size 12 \
  --feat-norm true \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --batch-size 256 \
  --dropout 0.2 \
  --shift 1 \
  --predict-last false \
  --target-mask-mode train_allow_interpolated \
  --wandb-project pgim-mask-trans-shift1-exp \
  --run-name-sufix dist250mrt