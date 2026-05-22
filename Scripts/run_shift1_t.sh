#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
PROJECT="pgim-rphgnn-shift1"
SHIFT=12

CUDA_VISIBLE_DEVICES=6 python main.py \
  graph.edges.same_planning_area.enable=false \
  graph.edges.same_age.enable=false \
  data.target_shift=1 \
  data.ts_test=73 \
  data.target_mask_mode=observed_only \
  data.feat_norm=norm1 \
  data.num_hops=2 \
  data.window_size=12 \
  data.rp_dim=16 \
  training.batch_size=128 \
  training.epochs=100 \
  training.eval_interval=1 \
  training.predict_last=false \
  training.run_name_sufix=shift1_tiny_h16_hop2_rp16_do05_wd2e1_noplanarea_noage \
  model.hidden_dim=16 \
  model.mlp_layers=1 \
  model.num_layers=1 \
  model.num_heads=4 \
  model.dropout=0.5 \
  model.conv_filters=1 \
  model.merge_mode=mean \
  optimizer.learning_rate=1e-4 \
  optimizer.weight_decay=2e-1 \
  logging.wandb_project="pgim-rphgnn-shift1"

CUDA_VISIBLE_DEVICES=6 python main.py \

  data.target_shift=1 \
  data.ts_test=73 \
  data.target_mask_mode=observed_only \
  data.feat_norm=norm1 \
  data.num_hops=2 \
  data.window_size=18 \
  data.rp_dim=16 \
  training.batch_size=128 \
  training.epochs=60 \
  training.eval_interval=1 \
  training.predict_last=false \
  training.run_name_sufix=shift1_small_w18_h32_rp16_do05_wd1e1 \
  model.hidden_dim=32 \
  model.mlp_layers=1 \
  model.num_layers=1 \
  model.num_heads=4 \
  model.dropout=0.5 \
  model.conv_filters=1 \
  model.merge_mode=mean \
  optimizer.learning_rate=5e-5 \
  optimizer.weight_decay=1e-1 \
  logging.wandb_project="pgim-rphgnn-shift1"

CUDA_VISIBLE_DEVICES=6 python main.py \
  data.target_shift=1 \
  data.ts_test=73 \
  data.target_mask_mode=observed_only \
  data.feat_norm=norm1 \
  data.num_hops=1 \
  data.window_size=24 \
  data.rp_dim=16 \
  training.batch_size=128 \
  training.epochs=60 \
  training.eval_interval=1 \
  training.predict_last=false \
  training.run_name_sufix=shift1_long_w24_h16_hop1_do06_wd2e1 \
  model.hidden_dim=16 \
  model.mlp_layers=1 \
  model.num_layers=1 \
  model.num_heads=4 \
  model.dropout=0.6 \
  model.conv_filters=1 \
  model.merge_mode=mean \
  optimizer.learning_rate=3e-5 \
  optimizer.weight_decay=2e-1 \
  logging.wandb_project="pgim-rphgnn-shift1"

# GPU=6 Scripts/run_shift1.sh
