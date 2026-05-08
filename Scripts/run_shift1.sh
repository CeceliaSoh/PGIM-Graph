CUDA_VISIBLE_DEVICES=7 python main.py \
  --root "database_v3/Graph_Size" \
  --ccr true \
  --nodes-dir nodes \
  --edges-dir edges \
  --macro-file macro_data_v1_processed.csv \
  --ccr-node-file node_id_ccr.csv \
  --graph-edge-files \
    dist_250.csv \
    mrt_cir_500.csv \
    mrt_nearest_dist_eps_1.csv \
    same_condo_age_2026.csv \
  --ts-test 25 \
  --shift 1 \
  --num-hops 3 \
  --window-size 12 \
  --predict-last false \
  --target-mask-mode observed_only \
  --feat-norm false \
  --batch-size 256 \
  --hidden-dim 64 \
  --mlp-layers 2 \
  --num-layers 2 \
  --num-heads 4 \
  --dropout 0.2 \
  --epochs 100 \
  --eval-interval 1 \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --tracked-indices 0 6 11 \
  --wandb-project pgim-graph-v3-shift1_new_size \
  --run-name-sufix ccr_multigraph_macro_lag1_73_obs \
  --ts-test 73

CUDA_VISIBLE_DEVICES=7 python main.py \
  --root "database_v3/Graph_Size" \
  --ccr true \
  --nodes-dir nodes \
  --edges-dir edges \
  --macro-file macro_data_v1_processed.csv \
  --ccr-node-file node_id_ccr.csv \
  --graph-edge-files \
    dist_250.csv \
  --ts-test 25 \
  --shift 1 \
  --num-hops 3 \
  --window-size 12 \
  --predict-last false \
  --target-mask-mode train_allow_interpolated \
  --feat-norm false \
  --batch-size 256 \
  --hidden-dim 64 \
  --mlp-layers 2 \
  --num-layers 2 \
  --num-heads 4 \
  --dropout 0.2 \
  --epochs 100 \
  --eval-interval 1 \
  --learning-rate 1e-4 \
  --weight-decay 1e-2 \
  --tracked-indices 0 6 11 \
  --wandb-project pgim-graph-v3-shift1_new \
  --run-name-sufix ccr_multigraph_macro_lag1_73_1Graph \
  --ts-test 73
# GPU=7 Scripts/run_shift1.sh
