# Current Model Architecture Notes

This note describes the model currently used by `main.py` for training. The short version is that the training pipeline builds RpHGNN-style graph context tensors in the dataloader, then trains a compact spatial-temporal regressor on sliding windows of those tensors.

The model code lives in:

- `src/models/gnn_regressor.py`
- `src/data/pgim.py`
- `src/trainer/trainer.py`
- `src/config/model/gnn_regressor.yaml`
- `src/config/data/pgim.yaml`

## Executive Summary

The current architecture is:

```text
Generated node/edge CSVs
        |
        v
Loader builds lightweight heterograph
        |
        v
Feature normalization + random projection
        |
        v
RpHGNN context precompute
        |
        v
Sliding window dataset
        |
        v
RpHGNNSpatialEncoder
        |
        v
Causal Transformer
        |
        v
Linear regression head
```

The model predicts condo rent for size-level nodes. Each sample is a time window for one size node, not the entire graph at once.

## Active Defaults

Current model defaults from `src/config/model/gnn_regressor.yaml`:

| Setting | Value | Meaning |
|---|---:|---|
| `hidden_dim` | 64 | Internal embedding size after spatial encoding. |
| `mlp_layers` | 2 | Number of hidden layers in each relation-group MLP. |
| `num_layers` | 2 | Number of causal Transformer layers. |
| `num_heads` | 4 | Multi-head attention heads. |
| `dropout` | 0.1 | Dropout in spatial MLPs and Transformer layers. |
| `conv_filters` | 2 | Number of 1D convolution filters per relation group. |
| `merge_mode` | `concat` | Concatenate relation-group embeddings before fusion. |

Current data/model-shape defaults from `src/config/data/pgim.yaml`:

| Setting | Value | Meaning |
|---|---:|---|
| `num_hops` | 2 | Number of graph propagation hops used by the context precompute. |
| `window_size` | 12 | Number of timesteps per training sample. |
| `target_shift` | 1 | Inputs are shifted one timestep before the target window. |
| `rp_dim` | 32 | Random-projected feature dimension. |
| `target_col` | `rent_per_sqft` | Target value. |
| `mask_col` | `y_mask` | Mask used by the loss. |
| `target_mask_mode` | `observed_only` | Train/evaluate only where the target is observed. |
| `ts_test` | 73 | Number of final timesteps used for test windows. |

## Input Tensor Shape

`src/data/pgim.py` precomputes one context tensor for every size-level node:

```text
(num_size_nodes, num_timesteps, num_groups, num_hops + 1, rp_dim)
```

The dataset then slices this into per-sample windows:

```text
(window_size, num_groups, num_hops + 1, rp_dim)
```

After batching, the model input is:

```text
(batch, T, num_groups, group_size, feat_dim)
```

With current defaults:

```text
T = 12
group_size = num_hops + 1 = 3
feat_dim = rp_dim = 32
```

`num_groups` depends on the loaded edge relations. It always includes a `self` group, plus incoming relation groups for size nodes.

## Graph Context Precompute

The loader does most of the graph work before the neural model runs.

1. Project-level and size-level node tables are loaded from `dataset/database_260519`.
2. Numeric feature columns are inferred after excluding ID and target columns.
3. Features are normalized using saved preprocessing stats.
4. Project and size features are random-projected to `rp_dim`.
5. Edge CSVs are loaded into a lightweight heterograph representation.
6. Reverse edges are added for each relation.
7. Mean aggregation is run for `num_hops` rounds.
8. The result is stored as RpHGNN context tensors.

For each timestep, the precompute keeps:

- the target size node's own representation in the `self` group;
- incoming relation-specific messages into size nodes;
- hop-wise representations from hop `0` through `num_hops`.

This means the model does not run message passing during each forward pass. It consumes precomputed graph neighborhoods as dense tensors.

## Spatial Encoder

The first learned part of the model is `RpHGNNSpatialEncoder`.

For every timestep independently, it processes each relation group:

```text
(group_size, feat_dim)
        |
        v
Conv1d(group_size -> conv_filters, kernel_size=1)
        |
        v
Flatten
        |
        v
MLP(conv_filters * feat_dim -> hidden_dim -> hidden_dim)
```

With current defaults:

```text
group_size = 3
feat_dim = 32
conv_filters = 2
MLP input = 2 * 32 = 64
MLP output = 64
```

Each relation group becomes one `hidden_dim` vector. With `merge_mode: concat`, the encoder concatenates all group vectors and applies a final fusion MLP:

```text
hidden_dim * num_groups -> hidden_dim
```

The output of the spatial encoder is:

```text
(batch, T, hidden_dim)
```

With current defaults:

```text
(batch, 12, 64)
```

## Temporal Encoder

After spatial encoding, the model adds sinusoidal positional encoding to the time dimension.

The sequence is then passed through `num_layers` causal Transformer layers. Each layer contains:

- multi-head self-attention;
- a causal attention mask;
- residual connection and layer norm after attention;
- feed-forward block `hidden_dim -> hidden_dim * 4 -> hidden_dim`;
- residual connection and layer norm after the feed-forward block.

The causal mask prevents timestep `t` from attending to future timesteps in the input window.

With current defaults:

```text
Transformer input:  (batch, 12, 64)
Transformer layers: 2
Attention heads:    4
Feed-forward dim:   256
Transformer output: (batch, 12, 64)
```

## Prediction Head

The final prediction layer is a single linear projection:

```text
hidden_dim -> 1
```

So the final model output is:

```text
(batch, T, 1)
```

With current defaults:

```text
(batch, 12, 1)
```

Each value is a predicted rent target for one timestep in the sample window.

## Training Objective

Training uses `MaskedMSELoss`:

```text
loss = sum(((prediction - target) ** 2) * mask) / sum(mask)
```

Only positions where `y_mask` is active contribute to the loss. With the current `target_mask_mode: observed_only`, missing or imputed-unobserved target positions are ignored during loss and metrics.

By default, `training.predict_last` is `false`, so the loss is applied to all valid positions in the output window. If `predict_last` is set to `true`, the trainer keeps only the final timestep:

```text
outputs = outputs[:, -1:, :]
targets = targets[:, -1:, :]
mask = mask[:, -1:, :]
```

## Training Loop

`main.py` wires the system together:

1. Load Hydra config from `src/config/config.yaml`.
2. Build train/test dataloaders with `get_dataloaders(cfg)`.
3. Inspect `train_loader.dataset.contexts.shape` to determine:
   - `num_graphs`, used as `num_groups`;
   - `num_hops`, actually `num_hops + 1` from the context tensor;
   - `feat_dim`.
4. Instantiate `GNNRegressor`.
5. Train with AdamW and `MaskedMSELoss`.
6. Evaluate every `eval_interval` epochs.
7. Save the best checkpoint by test loss.
8. Log metrics and checkpoint metadata to Weights & Biases.

Current optimizer defaults:

| Setting | Value |
|---|---:|
| `learning_rate` | `1e-3` |
| `weight_decay` | `1e-4` |

Current training defaults:

| Setting | Value |
|---|---:|
| `batch_size` | 32 |
| `epochs` | 10 |
| `eval_interval` | 1 |
| `predict_last` | `false` |

## Important Interpretation

Despite the class name `GNNRegressor`, the neural model is not doing dynamic graph message passing during `forward()`. The graph aggregation happens first in `src/data/pgim.py`, and the model learns how to combine those precomputed relation/hop contexts over time.

A concise description is:

> A precomputed RpHGNN-context encoder with per-relation hop convolution, MLP-based spatial fusion, causal Transformer temporal modeling, and a linear rent regression head.

