# PGIM-Graph Agent Notes

## Project Shape
- Python/PyTorch research code for condo rent graph time-series modeling.
- Main training entry point: `main.py`.
- Current data loader: `src/data/pgim.py`.
- Current model: `src/models/gnn_regressor.py`.
- Node preprocessing: `src/graph/form_node.py`.
- Edge preprocessing: `src/graph/form_edge.py`.
- Dataset documentation: `dataset/README_V260519.md`.
- Main graph config: `src/config/graph/V260519.yaml`.
- Older/experimental code still exists in `data/preprocessing.py`, `src/models/regressor.py`, `src/models/transformer.py`, and notebooks.

## Current Pipeline
- Raw tabular input is configured as `dataset/V260519.csv`.
- `src/graph/form_node.py` reads `src/config/graph/V260519.yaml` and creates `dataset/database_260519` node tables and metadata.
- `src/graph/form_edge.py` reads the same config and creates edge CSVs under `dataset/database_260519/edges`.
- `src/data/pgim.py` loads `dataset/database_260519/node_id.csv`, `project_level_node.csv`, `size_level_node.csv`, and graph edge CSVs.
- The loader builds a lightweight heterograph spec, random-projects project/size features, precomputes RpHGNN relation context tensors, and returns windowed PyTorch dataloaders.
- `main.py` trains `GNNRegressor`, logs metrics to Weights & Biases, and writes best checkpoints under `checkpoints/<wandb-project>/`.

## Expected Dataset Layout
- Default training root: `dataset/database_260519`.
- Required node files:
  - `node_id.csv`
  - `project_level_node.csv`
  - `size_level_node.csv`
- Required edge directory:
  - `edges/`
- Default edge files in `src/data/pgim.py`:
  - `dist_250.csv`
  - `same_age.csv`
  - `same_mrt_250.csv`
  - `same_mrt_dist_eps_5.csv`
  - `same_planning_area.csv`
  - `same_school_dist_eps_0p01.csv`
  - `size_project.csv`
- `node_id.csv` uses `size_tier == 0` for project-level nodes and nonzero `size_tier` for size-level nodes.

## Important Conventions
- The CLI argument is intentionally misspelled as `--egde-file` in `main.py`; keep that spelling unless refactoring every call site.
- Several legacy CLI arguments are still accepted by `main.py` but ignored inside `get_dataloaders`: `egde_file`, `shift`, `ccr`, `nodes_dir`, `ccr_node_file`, `edges_dir`, and `macro_file`.
- Feature tables are time/node tables keyed by `timestep` and `node_id`.
- Target tensors are built only for size-level nodes.
- Default target column is `rent_per_sqft_imp`; default mask column is `y_mask`.
- Supported target mask modes are `all`, `observed_only`, and `train_all_test_observed`.
- `GNNRegressor` expects input shaped `(batch, T, num_groups, group_size, feat_dim)` and returns `(batch, T, out_dim)`.
- The current loader's per-sample input shape is `(window_size, groups, K+1, rp_dim)`.
- `MaskedMSELoss` divides squared error by the active mask count.

## Common Commands
- Build node tables:
  `python src/graph/form_node.py`
- Build edge tables:
  `python src/graph/form_edge.py`
- Debug dataloader construction:
  `python src/data/pgim.py --root dataset/database_260519 --num-hops 2 --window-size 12 --batch-size 8 --rp-dim 32`
- Train with defaults:
  `python main.py --root dataset/database_260519`
- Train a small smoke run:
  `WANDB_MODE=offline python main.py --root dataset/database_260519 --epochs 1 --batch-size 8 --eval-interval 1`

## Configuration Notes
- `src/config/graph/V260519.yaml` is the active graph preprocessing config.
- `overwrite: false` is present in the config, but the current processors mostly skip existing outputs based on path existence rather than a full overwrite flow.
- Node preprocessing saves normalization/stat JSON files beside each node table stem.
- Edge output names are generated from each edge config name plus threshold/epsilon values, for example `dist_250.csv` and `same_mrt_dist_eps_5.csv`.

## Scripts and Stale Paths
- `Scripts/run_shift1.sh` and `Scripts/run_shift12.sh` are useful as historical sweep examples but currently reference older roots such as `database_v3/Graph_Size`.
- Those scripts also pass some arguments or mask modes that do not match the current `get_dataloaders` implementation.
- Prefer checking `main.py` and `src/data/pgim.py` before trusting script defaults.

## Dependencies
- No `requirements.txt` or `pyproject.toml` is currently present.
- Imports indicate dependencies including `numpy`, `pandas`, `torch`, `scikit-learn`, `tqdm`, `wandb`, and `pyyaml`.
- Current graph loading code does not require DGL at runtime, despite older graph assets and naming in the repo.

## Working Tree Notes
- This repo often contains generated CSVs, PNG graph previews, notebooks, checkpoints, and `wandb/` outputs.
- Large data lives under `dataset/`; do not rewrite or delete it unless explicitly asked.
- Avoid reverting user changes. Worktree state may already be dirty.
- Keep edits focused to the requested files or the specific module being fixed.
- When changing preprocessing, be careful about generated outputs in `dataset/database_260519`; code changes should usually be separate from regenerated data unless the user asks for both.

## Validation
- There is no obvious automated test suite.
- For preprocessing changes, prefer narrow import/syntax checks and small pandas smoke checks before regenerating full data.
- For loader/model changes, prefer `python src/data/pgim.py ...` or a one-epoch offline `main.py` smoke run when the local dataset is available.
- Full training can be expensive and may require GPU and Weights & Biases configuration.
