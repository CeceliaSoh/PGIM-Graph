# Current Data Flow Notes

This note describes the code that is currently on disk after the Hydra training config migration. The main thing to keep in mind is that the repository now has two different configuration layers:

- `src/config/graph/V260519.yaml` configures preprocessing and edge generation.
- `src/config/config.yaml` and its config groups configure training.

Those two layers are related by files on disk, not by a shared config object. The graph config creates `dataset/database_260519`; the training config points the dataloader at that generated folder.

## Executive Summary

The current pipeline is:

1. `src/graph/form_node.py` reads `src/config/graph/V260519.yaml`.
2. It reads raw tabular data from `dataset/V260519.csv`.
3. It standardizes identifiers and target columns, creates project-level and size-level node tables, reindexes all nodes, and writes CSVs under `dataset/database_260519`.
4. `src/graph/form_edge.py` reads the same graph config plus the node metadata and creates graph edge CSVs under `dataset/database_260519/edges`.
5. `main.py` reads Hydra training config from `src/config/config.yaml`.
6. `src/data/pgim.py:get_dataloaders()` loads the generated node and edge CSVs, creates a lightweight heterograph, random-projects node features, precomputes RpHGNN context tensors, builds sliding windows, and returns PyTorch dataloaders.
7. `main.py` trains `GNNRegressor` on those windows and logs metrics/checkpoints.

## Config Layers

### Preprocessing Config

File: `src/config/graph/V260519.yaml`

This config is used only by:

- `python src/graph/form_node.py`
- `python src/graph/form_edge.py`

Important keys:

| Config key | Used by | Meaning |
|---|---|---|
| `io_paths.input_path` | `form_node.py` | Raw input CSV, currently `dataset/V260519.csv`. |
| `io_paths.output_folder` | `form_node.py`, `form_edge.py` | Generated dataset root, currently `dataset/database_260519`. |
| `io_paths.full_project_location_path` | `form_node.py` | Source for latitude/longitude when missing from raw data. |
| `io_paths.mrt_info_path` | `form_edge.py` | MRT station data for MRT-related edges. |
| `io_node_paths.*` | `form_node.py`, `form_edge.py` | Filenames for generated node and metadata tables. |
| `rename` | `form_node.py` | Source-to-standard column mapping. `null` means "fill/create this standard column." |
| `standard_cols` | `form_node.py` | Required columns after rename/checking. |
| `form_project_node` | `form_node.py` | How to aggregate raw size-level rows into project-level rows. |
| `reindex_node_id` | `form_node.py` | How to assign final graph `node_id` values. |
| `size_node` | `form_node.py` | How to select/process the size-level node table. |
| `project_node` | `form_node.py` | How to select/process the project-level node table. |
| `edges` | `form_edge.py` | Which edge files to create and what thresholds/epsilons to use. |

### Training Config

Root file: `src/config/config.yaml`

Current groups:

- `data: pgim`
- `model: gnn_regressor`
- `training: default`
- `optimizer: adamw`
- `logging: wandb`
- `evaluation: default`
- `paths: default`

The training config does not rebuild node or edge CSVs. It only controls how `main.py` calls the loader/model/trainer.

Example override:

```bash
WANDB_MODE=offline python main.py data.num_hops=7 optimizer.learning_rate=1e-4 training.epochs=1
```

## Generated Dataset Layout

Current generated files found under `dataset/database_260519`:

| File | Current role |
|---|---|
| `node_id.csv` | Maps `(project_id, size_tier)` to graph `node_id`. Project-level nodes have `size_tier == 0`; size-level nodes have `size_tier != 0`. |
| `project_info.csv` | One row per project with location, age, planning area, and school-distance metadata. Used by edge generation. |
| `timestep_info.csv` | Maps integer timesteps to dates. Not used by the current training loader. |
| `project_level_node.csv` | Time/node feature table for project-level nodes. |
| `size_level_node.csv` | Time/node feature table for size-level nodes, including target and mask columns. |
| `nodes/project_level_node.csv` | Duplicate/cached location of the project node table. |
| `nodes/size_level_node.csv` | Duplicate/cached location of the size node table. |
| `project_level_node/*.json` | Preprocessing stats. Not used by the current training loader. |
| `size_level_node/*.json` | Preprocessing stats. Not used by the current training loader. |
| `edges/*.csv` | Graph relation files loaded by `src/data/pgim.py`. |

The current `node_id.csv` contains 804 project-level nodes and 2,259 size-level nodes.

## Node Preprocessing Flow

Entry point: `python src/graph/form_node.py`

Default Hydra graph config:

```bash
python src/graph/form_node.py
```

The default config is `V260519`; use Hydra overrides such as `--config-name V260506` to run another graph config.

Flow:

1. `FormNodeProcessor.__init__()`
   - Reads input/output paths from `io_paths` and `io_node_paths`.
   - Creates `dataset/database_260519`.
   - Creates metadata folders such as `project_level_node/` and `size_level_node/`.

2. `process()`
   - If every configured output path already exists, it loads cached `project_level_node.csv` and `size_level_node.csv` and exits early.
   - Otherwise it reads `dataset/V260519.csv`.

3. `rename()`
   - Applies the `rename` mapping.
   - For config values set to `null`, it fills/creates fields such as `node_id`, `project_id`, `timestep`, `longitude`, and `latitude`.

4. `check_std()`
   - Validates standard columns.
   - Rebuilds `project_id` from sorted `project_name`.
   - Rebuilds `timestep` into contiguous integer order.
   - Converts `size_tier` from `SZ1` to `SZ5` or integer 1 to 5.
   - Converts `y_mask` to 0/1.

5. `save_meta_info()`
   - Writes `project_info.csv`.
   - Writes `timestep_info.csv`.

6. `prepare_projlv_df()`
   - Aggregates rows by `["project_id", "timestep"]`.
   - Uses mean `rent_per_sqft`.
   - Sets project-level `size_tier = 0`.
   - Sets project-level `y_mask` based on whether the aggregated rent is present.

7. `reindex_nodes_id()`
   - Combines project-level rows and original size-level rows.
   - Groups by `["project_id", "size_tier"]`.
   - Sorts by `["project_id", "size_tier"]`.
   - Assigns final integer graph `node_id`.
   - Writes `node_id.csv`.

8. `process_n_store_size_node()`
   - One-hot encodes `size_tier`.
   - Duplicates configured columns as feature columns, currently `project_id_feat`, `timestep_feat`, and `rent_per_sqft_feat`.
   - Writes `size_level_node.csv`.
   - Writes normalization/stat JSON files.

9. `process_n_store_project_node()`
   - Max-clips/fills configured distance-like columns.
   - One-hot encodes configured categoricals.
   - Duplicates `timestep`, `node_id`, and `rent_per_sqft` as feature columns.
   - Writes `project_level_node.csv`.
   - Writes normalization/stat JSON files.

## Edge Preprocessing Flow

Entry point: `python src/graph/form_edge.py`

Default Hydra graph config:

```bash
python src/graph/form_edge.py
```

The default config is `V260519`; use Hydra overrides such as `--config-name V260506` to run another graph config.

Flow:

1. Load `project_info.csv`, `node_id.csv`, and `Rail_Transport.csv`.
2. For each enabled edge config under `edges`, skip the edge if its output CSV already exists.
3. Create edge CSVs with columns including `node_id` and `neig_node_id`.
4. Write files under `dataset/database_260519/edges`.

Current edge files used by the training loader:

| Edge file | Interpreted as |
|---|---|
| `dist_250.csv` | Project-project relation. |
| `same_age.csv` | Project-project relation. |
| `same_mrt_250.csv` | Project-project relation. |
| `same_mrt_dist_eps_5.csv` | Project-project relation. |
| `same_planning_area.csv` | Project-project relation. |
| `same_school_dist_eps_0p01.csv` | Project-project relation. |
| `size_project.csv` | Size-project relation. |

`src/data/pgim.py` automatically adds reverse edges for every loaded edge file.

## Training Loader Flow

Entry point from training: `main.py -> get_dataloaders()`

Important behavior in `src/data/pgim.py`:

1. `get_dataloaders(config)` takes one config object. In training this is the full Hydra `cfg`; in standalone/debug use it can be a plain data-loader dict.

2. The loader normalizes config internally. If the full Hydra config is passed, it reads data settings from `cfg.data`, batch size from `cfg.training.batch_size`, and random seed from `cfg.seed`.

3. The normalized loader config starts from `graph_config_path`, currently `src/config/graph/V260519.yaml`.

4. If `prepare_graph` is true, the loader runs `FormNodeProcessor(graph_config).process()` and then `FormEdgeProcessor(graph_config).process()` before reading CSVs. Existing outputs are mostly reused/skipped by those processors.

5. If `root` is null, the loader derives it from `graph_config["io_paths"]["output_folder"]`.

6. If `graph_edge_files` is null, the loader derives enabled edge filenames from the graph config. The hard-coded default remains only as a fallback:

```python
DEFAULT_GRAPH_EDGE_FILES = (
    "dist_250.csv",
    "same_age.csv",
    "same_mrt_250.csv",
    "same_mrt_dist_eps_5.csv",
    "same_planning_area.csv",
    "same_school_dist_eps_0p01.csv",
    "size_project.csv",
)
```

7. `node_id.csv` is loaded from `root / "node_id.csv"`.

8. Project node IDs are `size_tier == 0`; size node IDs are all nonzero `size_tier`.

9. `project_level_node.csv` and `size_level_node.csv` are loaded by `load_node_table()`:
   - It first tries `root / "nodes" / filename`.
   - If that does not exist, it tries `root / filename`.

10. Timesteps come from `size_level_node.csv`, not from `timestep_info.csv`.

11. `ts_test` means number of final timesteps reserved for test. It is not a timestep id.

12. Feature columns are inferred from numeric columns by excluding ID and target columns.

13. Feature values are arranged into tensors:
    - Project features: `(T, num_project_nodes, project_feature_dim)`
    - Size features: `(T, num_size_nodes, size_feature_dim)`

14. If `feat_norm` is true, train-period min/max scaling maps features to `[-1, 1]`.

15. Project and size feature tensors are separately random-projected to `data.rp_dim`.

16. Edge CSVs are loaded from `root / "edges"`.

17. The heterograph is represented by `HeteroGraphSpec`, not DGL.

18. `rphgnn_precompute_contexts()` precomputes relation context tensors shaped:

```text
(num_size_nodes, num_timesteps, groups, num_hops + 1, rp_dim)
```

19. `build_target_tensor()` builds targets and masks only for size-level nodes.

20. `build_window_indices()` creates sliding windows:
    - Train windows have `end <= train_steps`.
    - Test windows overlap the train/test boundary if any post-train part has active mask.

21. Each dataset sample returns:

```text
inputs:  (window_size, groups, num_hops + 1, rp_dim)
targets: (window_size, 1)
masks:   (window_size, 1)
node_idx
start_timestep_index
```

After batching, `main.py` gives the model:

```text
(batch, window_size, groups, num_hops + 1, rp_dim)
```

## Training Flow

Entry point: `python main.py`

Hydra loads `src/config/config.yaml`.

Important config-to-code mapping:

| Hydra config | Main usage |
|---|---|
| Full `cfg` | Passed as the single argument to `get_dataloaders(cfg)`. |
| `data.graph_config_path` | Loader source of generated dataset root and enabled edge names. |
| `data.root` | Optional override for the generated dataset root. If null, use graph config `io_paths.output_folder`. |
| `data.prepare_graph` | Controls whether the loader runs node/edge processors before loading CSVs. |
| `data.num_hops` | Loader hop count; also included in run name. |
| `data.window_size` | Loader window length; also included in run name. |
| `data.rp_dim` | Random projection dimension in loader. |
| `training.batch_size` | Read by loader from the full config. |
| `training.predict_last` | If true, train/eval loss only uses last timestep in each window. |
| `model.*` | Passed to `GNNRegressor`. |
| `optimizer.*` | Passed to `torch.optim.AdamW`. |
| `evaluation.tracked_indices` | Extra metrics by timestep index inside each test window. |
| `paths.checkpoint_dir` | Checkpoint root. |
| `logging.*` | Weights & Biases project/entity. |

The model output shape is:

```text
(batch, window_size, 1)
```

`MaskedMSELoss` uses `y_mask` to ignore inactive target positions.

## Current Config Mismatches and Suspicious Points

These are the places that look wrong or at least misleading from the current code.

| Suspicious item | Where | What actually happens |
|---|---|---|
| `graph_edge_files: null` | `src/config/data/pgim.yaml` | Means "derive enabled edge files from `data.graph_config_path`." |
| `target_mask_mode` default mismatch | `src/config/data/pgim.yaml` vs `src/data/pgim.py` signature | Hydra default is `observed_only`; function default is `train_all_test_observed`. Training via `main.py` uses the Hydra value. Direct `python src/data/pgim.py` uses the function default because its debug parser does not expose mask mode. |
| Preprocessing `overwrite: false` | `src/config/graph/V260519.yaml` | The processors mainly skip existing outputs based on path existence. There is no complete overwrite policy wired through the code. |
| Preprocessing config is Hydra-managed in both paths | `data.graph_config_path`, `form_node.py`, `form_edge.py` | `main.py` passes the graph config path into `get_dataloaders()`, which can run both processors. The standalone preprocessing scripts also use Hydra with default config `V260519`. |
| Normalization JSON files | Generated under node table folders | Current loader ignores them and does its own train-period min/max scaling plus random projection. |
| `rent_per_sqft_feat` | Generated node tables | This duplicates target-like rent information as an input feature. It may be intentional as a historical/current covariate, but it is a leakage risk if the prediction target is same-timestep rent. |
| `target_col=rent_per_sqft_imp` | Hydra data config | Targets train against imputed rent by default. If `target_mask_mode=observed_only`, loss/metrics only use observed positions even though target values come from the imputed column. |
| Edge direction naming | `size_project.csv` | The file stores `node_id` as size node and `neig_node_id` as project node. The loader then adds both size-to-project and project-to-size edges. |

## How To Reason About a Training Run

For a default training run:

```bash
python main.py
```

Use this mental model:

1. Hydra composes training config from `src/config/config.yaml`.
2. `data.graph_config_path` resolves to `src/config/graph/V260519.yaml`.
3. If `data.prepare_graph` is true, the loader calls the node and edge processors. Existing generated CSVs are reused/skipped by the processors.
4. `data.root` is null by default, so the loader uses `io_paths.output_folder` from the graph config: `dataset/database_260519`.
5. Feature columns are inferred from numeric columns in `project_level_node.csv` and `size_level_node.csv`.
6. The loader excludes these target-like columns from model inputs:
   - `rent_per_sqft`
   - `rent_per_sqft_imp`
   - `rent_per_sqft_feat`
   - `y_mask`
7. The loader excludes ID-like columns from model inputs:
   - `node_id`
   - `project_id`
   - `timestep`
   - `project_id_feat`
   - `node_id_feat`
   - `timestep_feat`
8. Remaining numeric columns become features.
9. The train/test split is temporal, based on the final `data.ts_test` timesteps.
10. The graph is static across time; only node features/targets vary by timestep.
11. Test windows can include train-period context positions, but evaluation masks zero out train-period target positions inside mixed windows.

## Practical Checks Before Trusting a Run

Use these checks when config feels wrong:

```bash
python main.py --cfg job
```

Confirms the composed Hydra training config.

```bash
python src/data/pgim.py --graph-config src/config/graph/V260519.yaml --skip-processors --num-hops 2 --window-size 12 --batch-size 8 --rp-dim 32
```

Confirms the loader can build graph contexts and window batches.

```bash
python -c "import pandas as pd; print(pd.read_csv('dataset/database_260519/node_id.csv')['size_tier'].value_counts().sort_index())"
```

Confirms project and size node counts.

```bash
python -c "import pandas as pd; print(pd.read_csv('dataset/database_260519/size_level_node.csv', nrows=1).columns.tolist())"
```

Confirms the generated size node schema.

```bash
python -c "import pandas as pd; print(pd.read_csv('dataset/database_260519/project_level_node.csv', nrows=1).columns.tolist())"
```

Confirms the generated project node schema.

## Recommended Cleanup Direction

The smallest remaining cleanup would be to add explicit overwrite or freshness behavior for processor outputs. Right now `prepare_graph=true` reuses many existing outputs because the processors skip based on path existence.

The next conceptual cleanup would be to make processor overwrite/freshness behavior explicit in the graph config and code.
