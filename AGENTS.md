# PGIM-Graph Agent Notes

## Project Shape
- Python/PyTorch research code for rental graph time-series modeling.
- Main training entry point: `main.py`.
- Core data loader: `src/data/pgim.py`.
- Main model: `src/models/gnn_regressor.py`.
- Preprocessing work is mostly in notebooks named `Preprocess-*.ipynb` plus helper scripts in `data/`.
- Large datasets live under `dataset/`; avoid rewriting or deleting them unless explicitly asked.

## Important Conventions
- The graph data loader expects a root such as `dataset/ccr`, a feature tensor such as `feature.npy`, and an edge text file such as `graph_link_250m/links.txt`.
- The CLI argument is currently spelled `--egde-file` in `main.py` and `src/data/pgim.py`; use that spelling unless refactoring both call sites.
- Feature tensors are expected as `(N, T, D)`, with the last column used as `y_mask` and the second-last column used as the target when shifting.
- `WindowedNodeDataset` returns tensors shaped `(K+1, window_size, feat_dim)` for each sample.
- `GNNRegressor` expects input shaped `(batch, num_hops, T, feat_dim)` and returns `(batch, T, out_dim)`.

## Common Commands
- Train/debug main model:
  `python main.py --root dataset/ccr --feature feature.npy --egde-file graph_link_250m/links.txt`
- Debug dataloader construction:
  `python src/data/pgim.py --root dataset/ccr --feature feature.npy --egde-file graph_link_250m/links.txt`
- Existing sweep examples are in `Scripts/run_shift1.sh` and `Scripts/run_shift12.sh`, but they contain Linux absolute paths and CUDA device settings.

## Dependencies
- No `requirements.txt` or `pyproject.toml` is currently present.
- Imports indicate dependencies including `numpy`, `pandas`, `torch`, `dgl`, `scikit-learn`, `tqdm`, `wandb`, and notebook tooling.

## Working Tree Notes
- This repo often has many modified/untracked notebooks and generated data files. Do not revert notebook or dataset changes unless the user explicitly asks.
- Prefer focused edits to Python modules or the specific notebook requested.
- When editing notebooks, keep JSON valid and avoid unnecessary output churn.

## Validation
- There is no obvious automated test suite.
- For code changes, run the narrowest import or CLI smoke test that does not require expensive training or unavailable data.
- Full training runs can be expensive and may require GPU, DGL, and Weights & Biases configuration.
