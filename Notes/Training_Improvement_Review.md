# Model Training Improvement Review

Review date: 2026-05-26

This note reviews the current model-training code path:

```text
main.py
  -> src/data/pgim.py:get_dataloaders()
  -> src/models/gnn_regressor.py:GNNRegressor
  -> src/trainer/trainer.py:Trainer
  -> src/utils/loss.py:MaskedMSELoss
```

The current training stack is workable and fairly compact, but several improvements would make experiments more trustworthy, faster to run, and easier to compare.

## Highest Priority

| Area | Current code | Why it matters | Suggested improvement |
|---|---|---|---|
| Test set is used for checkpoint selection and early stopping | `Trainer.train()` evaluates on `test_loader`, then `_track_best_checkpoint()` saves the best model by test loss. See `src/trainer/trainer.py:64` and `src/trainer/trainer.py:268`. | This makes the reported final test score optimistic because the test set influences model selection. It is effectively a validation set, not a held-out test set. | Split data into train, validation, and test by time. Use validation loss for checkpointing and early stopping, then evaluate the selected checkpoint once on test. |
| Same-timestep target leakage risk in size features | `infer_feature_columns()` includes every numeric column not in `target_columns` (`src/data/pgim.py:58`). Current Hydra config excludes `rent_per_sqft`, `rent_per_sqft_feat`, and `y_mask`, but not `rent_per_sqft_imp` (`src/config/data/pgim.yaml:14`). The generated size table contains `rent_per_sqft_imp`. | With `target_col: rent_per_sqft`, `rent_per_sqft_imp` can become an input feature for the same target timestep. That can make training metrics look good for the wrong reason. | Add `rent_per_sqft_imp` to `data.target_columns` or explicitly define allowed feature columns. If rent history is intended, use lagged features only. |
| No separate validation horizon | `build_window_indices()` creates only train and test sample lists (`src/data/pgim.py:394`). | Hyperparameter sweeps, early stopping, and architecture choices all need a validation set. Without it, experiment iteration leaks into test performance. | Add `ts_val` or explicit date cutoffs. Return `train_loader`, `val_loader`, and `test_loader`. |
| Config options are present but not honored | `data.root`, `data.prepare_graph`, and `data.graph_edge_files` exist in `src/config/data/pgim.yaml`, but the current `get_dataloaders()` always runs node/edge processors and loads enabled edges via processors (`src/data/pgim.py:440`). | Configs become misleading. Users may think a run is using a different root or edge set when it is not. This hurts reproducibility. | Either implement these options fully or remove them from active config. Log resolved root, edge files, feature columns, target column, and split cutoffs at run start. |

## Training Loop

| Area | Current code | Why it matters | Suggested improvement |
|---|---|---|---|
| No learning-rate scheduler | `main.py` creates AdamW only (`main.py:123`). | Fixed LR can be unstable or leave performance on the table, especially with Transformers. | Add configurable schedulers such as cosine decay, ReduceLROnPlateau, and warmup. Log current LR every epoch. |
| No gradient clipping | Training calls `loss.backward()` and `optimizer.step()` directly (`src/trainer/trainer.py:91`). | Attention models can produce occasional gradient spikes. One bad step can damage a run. | Add optional `training.grad_clip_norm`, defaulting to a conservative value such as 1.0 or disabled by config. |
| Epoch loss is averaged per batch, not per active target count | `train_one_epoch()` sums `loss.item()` and divides by number of batches (`src/trainer/trainer.py:95`). | `MaskedMSELoss` normalizes each batch by active mask count. Averaging batch losses equally gives small sparse batches the same weight as dense batches. | Accumulate total squared error and total active mask count for exact epoch loss, or return numerator/count from the loss helper. |
| Empty-mask batches are not specially handled | `MaskedMSELoss` divides by `mask.sum() + eps` (`src/utils/loss.py:22`). | If a batch has no active targets, it contributes a zero loss and a gradientless step. Window filtering reduces this risk but does not make it impossible after `predict_last` slicing. | Skip optimizer steps when active mask count is zero. Also assert that train/eval loaders have nonzero active target counts. |
| W&B lifecycle is not protected by `try/finally` | `wandb.finish()` is called only after `trainer.train()` returns (`main.py:144`). | Exceptions can leave runs in a partial state and hide useful failure metadata. | Wrap training in `try/finally`, and log resolved config and failure details where practical. |

## Data Loader And Feature Handling

| Area | Current code | Why it matters | Suggested improvement |
|---|---|---|---|
| Feature selection is implicit | Numeric columns are inferred from CSVs (`src/data/pgim.py:58`). | A regenerated dataset can silently add or remove model inputs. Target-like columns can slip in. | Write and load explicit feature manifests for project and size nodes. Fail fast if unexpected columns appear. |
| Random projection is fixed and non-learned | `random_project_features()` projects features before training (`src/data/pgim.py:217`). | It compresses useful signal before the model can learn feature weights. It also makes feature importance difficult to inspect. | Compare against a learned linear projection per node type. Keep random projection as a fast baseline, not the only path. |
| Feature normalization uses full saved stats | `feature_normalize()` loads stats generated during preprocessing (`src/data/pgim.py:103`). | If those stats were computed on the full time range, train features may depend on future/test distribution. | Fit normalization on train timesteps only inside the loader, or save train-only stats during preprocessing. |
| Context precompute runs every training invocation | `rphgnn_precompute_contexts()` is called during `get_dataloaders()` (`src/data/pgim.py:506`). | Sweeps with many hyperparameters repeatedly pay the same preprocessing cost. | Cache contexts by graph config, feature manifest, normalization mode, `rp_dim`, `num_hops`, and seed. |
| Test windows can overlap the train/test boundary | `build_window_indices()` allows test windows whose target window partly includes train timesteps, then masks train positions (`src/data/pgim.py:415`). | This is valid for rolling forecasts, but it should be explicit because it changes what "test sample" means. | Log split dates and consider separate modes: strict post-split windows vs boundary-overlap windows. |

## Model Architecture

| Area | Current code | Why it matters | Suggested improvement |
|---|---|---|---|
| Static relation aggregation is not learned | `rphgnn_precompute_contexts()` mean-aggregates relation messages before the model sees them (`src/data/pgim.py:299`). | The model cannot learn edge weights, attention over neighbors, or time-varying relation importance during graph propagation. | Add an optional learned graph encoder path, or at least relation/hop attention after precompute. |
| Hop and relation mixing is shallow | `RpHGNNSpatialEncoder` uses `Conv1d(kernel_size=1)` over hop dimension and then flattens (`src/models/gnn_regressor.py:116`). | It learns a compact hop combination, but does not model richer hop order interactions. | Try hop attention, relation attention, or a small Transformer/MLP over `(relation, hop)` tokens. |
| Positional encoding is fixed and recreated every forward | `_sinusoidal_pe()` is called in `forward()` (`src/models/gnn_regressor.py:221`). | Minor overhead, and fixed encodings may not fit monthly seasonality or calendar effects. | Cache positional encodings as a buffer or use learned/month-aware temporal embeddings. |
| Model predicts all timesteps by default | `training.predict_last: false` and trainer keeps all output positions (`src/trainer/trainer.py:146`). | For forecasting, users often care about a specific horizon. Predicting every position can blur objectives across easy near-history and harder future targets. | Make horizon objective explicit: last step only, all future steps, or multi-horizon weighted loss. |

## Evaluation And Experiment Quality

| Area | Current code | Why it matters | Suggested improvement |
|---|---|---|---|
| Metrics are global only plus fixed window indices | Overall metrics flatten all valid positions, and index metrics use fixed positions `0, 5, 11` (`src/trainer/trainer.py:123`, `src/config/evaluation/default.yaml:1`). | It is hard to know which horizon, district, size tier, or time period is failing. | Add metrics by forecast horizon, calendar year/month, size tier, project group, and observation count bucket. |
| No baseline models in the training script | Training always instantiates `GNNRegressor` (`main.py:107`). | Without naive and classical baselines, it is hard to tell if the graph model is adding value. | Add baseline evaluation: last observed rent, moving average, district median, and linear/XGBoost-style tabular baseline. |
| Checkpoints do not include feature schema metadata | `_save_checkpoint()` stores model/optimizer/config only (`src/trainer/trainer.py:290`). | A checkpoint may be impossible to reproduce if generated feature columns change. | Store feature column lists, group names, split cutoffs, context cache key, and dataset file fingerprints. |
| Limited run-time diagnostics | Logs include losses and a few metrics, but not data shape/fill/leakage checks in W&B. | Silent data changes can dominate results. | Log train/val/test sample counts, active mask counts, feature counts, target stats, and top missing-feature rates. |

## Suggested Implementation Order

1. Fix feature leakage first: exclude `rent_per_sqft_imp` and move to explicit feature manifests.
2. Add a validation split and select checkpoints by validation loss, not test loss.
3. Implement exact masked loss aggregation and skip empty-mask batches.
4. Add scheduler, gradient clipping, and richer run diagnostics.
5. Cache precomputed RpHGNN contexts for faster sweeps.
6. Run controlled ablations: no graph, graph without project-project edges, learned projection vs random projection, last-step loss vs all-step loss.

## Quick Checks To Add

These checks would catch many accidental experiment bugs before a long run starts:

```text
- Assert target columns are absent from feature columns.
- Assert train/validation/test split dates are ordered and logged.
- Assert active target count is nonzero for every split.
- Print and log project/size feature column lists.
- Print and log target mean/std by split.
- Save feature schema and split metadata into every checkpoint.
```
