CUDA_VISIBLE_DEVICES=7 python /home/cecelia/project/PGIM-Graph/main.py \
  --root /home/cecelia/project/PGIM-Graph/dataset/ccr \
  --feature feature.npy \
  --egde-file graph_link_250m/links.txt \
  --num-hops 2 \
  --window-size 12 \
  --batch-size 32 \
  --feat-norm true \
  --learning-rate 1e-4 \
  --shift 12