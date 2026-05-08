import argparse
from collections import defaultdict
from pathlib import Path

import dgl
import dgl.function as fn
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


def compute_minmax_stats(features: np.ndarray):
    min_vals = features.min(axis=(0, 1), keepdims=True)
    max_vals = features.max(axis=(0, 1), keepdims=True)
    return min_vals, max_vals


def normalize_to_minus1_1(features: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray):
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    return 2 * (features - min_vals) / range_vals - 1


def sign_precompute(x: torch.Tensor, g: dgl.DGLGraph, num_hops: int):
    """
    SIGN-style pre-computation for one timestamp.

    Args:
        x: Node features of shape (N, feat_dim).
        g: Static graph over the N nodes.
        num_hops: Number of graph propagation hops.

    Returns:
        List of propagated features [A x, A^2 x, ..., A^K x].
    """
    with torch.no_grad():
        x = x.to("cpu").float()
        g = g.to("cpu")
        h = x
        h_list = []

        in_degs = g.in_degrees().float().clamp(min=1)
        out_degs = g.out_degrees().float().clamp(min=1)
        src_norm = torch.pow(out_degs, -0.5)
        dst_norm = torch.pow(in_degs, -0.5)

        with g.local_scope():
            g.ndata["src_norm"] = src_norm
            g.ndata["dst_norm"] = dst_norm
            g.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_w"))

            for _ in range(num_hops):
                g.ndata["h"] = h
                g.update_all(fn.u_mul_e("h", "gcn_w", "m"), fn.sum("m", "h"))
                h = g.ndata.pop("h")
                h_list.append(h.clone())

    return h_list


DEFAULT_GRAPH_EDGE_FILES = (
    "dist_250.csv",
    "mrt_cir_500.csv",
    "mrt_nearest_dist_eps_2.csv",
    "same_condo_age_2026.csv",
)

DEFAULT_FEATURE_EXCLUDE_COLS = {"timestep", "node_id", "project_id", "Project Name"}
DEFAULT_MACRO_FILE = "macro_data_processed.csv"


def load_graph_from_edges(edge_path, num_nodes):
    """Load a DGL graph from a text file with one `src,dst` edge per line."""
    edges = np.loadtxt(edge_path, dtype=np.int64, delimiter=",")
    if edges.ndim == 1:
        edges = edges.reshape(1, -1)
    return dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=num_nodes)


def load_edge_table(edge_path: Path) -> pd.DataFrame:
    edges = pd.read_csv(edge_path)
    if {"project_id", "neig_project_id"}.issubset(edges.columns):
        edge_key = "project_id"
        edges = edges.rename(columns={"project_id": "src", "neig_project_id": "dst"})
    elif {"node_id", "neig_id"}.issubset(edges.columns):
        edge_key = "node_id"
        edges = edges.rename(columns={"node_id": "src", "neig_id": "dst"})
    elif {"src", "dst"}.issubset(edges.columns):
        edge_key = "node_id"
        edges = edges[["src", "dst"]].copy()
    else:
        raise ValueError(f"Unsupported edge schema in {edge_path}: {list(edges.columns)}")

    edges = edges[["src", "dst"]].copy()
    edges["src"] = pd.to_numeric(edges["src"], errors="coerce")
    edges["dst"] = pd.to_numeric(edges["dst"], errors="coerce")
    edges = edges.dropna().astype({"src": int, "dst": int}).drop_duplicates()
    edges = edges.reset_index(drop=True)
    edges.attrs["edge_key"] = edge_key
    return edges


def load_ccr_node_ids(ccr_path: Path):
    ccr_nodes = pd.read_csv(ccr_path)
    if "node_id" not in ccr_nodes.columns:
        raise ValueError(f"Expected a node_id column in {ccr_path}")
    return set(pd.to_numeric(ccr_nodes["node_id"], errors="coerce").dropna().astype(int).tolist())


def discover_node_files(nodes_dir: Path, ccr: bool, ccr_path: Path):
    node_files = sorted(nodes_dir.glob("*.csv"))
    if not ccr:
        return node_files

    allowed_nodes = load_ccr_node_ids(ccr_path)
    return [
        path
        for path in node_files
        if int(path.stem) in allowed_nodes
    ]


def infer_feature_columns(sample_frame: pd.DataFrame, target_col: str, mask_col: str, no_feat_col=None):
    excluded = set(no_feat_col or [])
    excluded.add(target_col)
    excluded.add(mask_col)
    numeric_cols = [
        col
        for col in sample_frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(sample_frame[col])
    ]
    if target_col not in sample_frame.columns:
        raise ValueError(f"Expected target column '{target_col}' in node CSVs")
    if mask_col not in sample_frame.columns:
        raise ValueError(f"Expected mask column '{mask_col}' in node CSVs")
    if not pd.api.types.is_numeric_dtype(sample_frame[target_col]):
        raise ValueError(f"Target column '{target_col}' must be numeric")
    return numeric_cols


def load_macro_features(macro_path: Path, feature_cols):
    macro_frame = pd.read_csv(macro_path)
    if "timestep" not in macro_frame.columns:
        raise ValueError(f"Expected a timestep column in {macro_path}")

    macro_frame["timestep"] = pd.to_numeric(macro_frame["timestep"], errors="coerce")
    macro_frame = macro_frame.dropna(subset=["timestep"]).copy()
    macro_frame["timestep"] = macro_frame["timestep"].astype(int)
    macro_feature_cols = [
        col
        for col in feature_cols
        if col in macro_frame.columns and pd.api.types.is_numeric_dtype(macro_frame[col])
    ]
    min_macro_timestep = int(macro_frame["timestep"].min())
    macro_lookup = (
        macro_frame.set_index("timestep")[macro_feature_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(np.float32)
        .to_dict("index")
    )
    return macro_lookup, macro_feature_cols, min_macro_timestep


def apply_macro_shift(node_frames, macro_lookup, macro_feature_cols, shift, min_macro_timestep):
    if shift <= 0 or not macro_feature_cols:
        return

    missing_count = 0
    for frame in tqdm(node_frames.values(), desc=f"Applying macro lag shift={shift}"):
        for row_idx, timestep in frame["timestep"].astype(int).items():
            macro_timestep = max(timestep - shift, min_macro_timestep)
            macro_values = macro_lookup.get(macro_timestep)
            if macro_values is None:
                missing_count += 1
                continue
            for col in macro_feature_cols:
                frame.at[row_idx, col] = macro_values[col]

    if missing_count:
        print(f"Macro lag shift={shift}: kept original macro values for {missing_count} rows with missing macro data.")


def normalize_window_features(windows: np.ndarray, train_count: int):
    train_features = windows[:train_count]
    min_vals = train_features.min(axis=(0, 1), keepdims=True)
    max_vals = train_features.max(axis=(0, 1), keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    return 2 * (windows - min_vals) / range_vals - 1


def build_hop_tensor(features: np.ndarray, g: dgl.DGLGraph, num_hops: int) -> np.ndarray:
    """
    Args:
        features: (N, T, F)

    Returns:
        (N, T, K+1, F)
    """
    stacked_hops = []

    for feat_t in features.transpose(1, 0, 2):
        x_t = torch.from_numpy(feat_t)
        h_list = sign_precompute(x_t, g, num_hops)
        hops_t = torch.stack([x_t] + h_list, dim=0)
        stacked_hops.append(hops_t)

    all_hops = torch.stack(stacked_hops, dim=0).numpy()
    return all_hops.transpose(2, 0, 1, 3)


def collect_valid_positions(mask: np.ndarray, offset: int = 0, min_time_idx: int = 0):
    nodes, times = np.where(mask > 0)
    global_times = times + offset
    valid = global_times >= min_time_idx
    return list(zip(nodes[valid].tolist(), global_times[valid].tolist()))


class WindowedNodeDataset(Dataset):
    def __init__(
        self,
        hop_features: np.ndarray,
        targets: np.ndarray,
        masks: np.ndarray,
        valid_positions,
        window_size: int,
    ):
        self.hop_features = hop_features.astype(np.float32)  # (N, T, K+1, F)
        self.targets = targets.astype(np.float32)  # (N, T)
        self.masks = masks.astype(np.float32)  # (N, T)
        self.valid_positions = list(valid_positions)
        self.window_size = window_size
        self.num_nodes, self.total_steps, self.num_hops, self.feat_dim = self.hop_features.shape

    def __len__(self):
        return len(self.valid_positions)

    def __getitem__(self, idx):
        node_idx, time_idx = self.valid_positions[idx]
        start = time_idx - self.window_size + 1
        if start < 0:
            raise IndexError(
                f"Encountered incomplete history window for node={node_idx}, time_idx={time_idx}, "
                f"window_size={self.window_size}. Valid positions should ensure time_idx >= {self.window_size - 1}."
            )

        window = self.hop_features[node_idx, start : time_idx + 1]
        y_window = self.targets[node_idx, start : time_idx + 1]
        mask_window = self.masks[node_idx, start : time_idx + 1]

        x = window.transpose(1, 0, 2)  # (K+1, window_size, feat_dim)
        y = y_window[:, None]  # (window_size, 1)
        mask = mask_window[:, None]  # (window_size, 1)
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask),
            torch.tensor(node_idx, dtype=torch.long),
            torch.tensor(time_idx, dtype=torch.long),
        )


class MultiGraphWindowDataset(Dataset):
    def __init__(
        self,
        hop_lookup,
        targets: np.ndarray,
        masks: np.ndarray,
        node_ids,
        start_times,
        window_size: int,
        graph_names,
        num_hops: int,
        feat_dim: int,
        train_cutoff: int,
    ):
        self.hop_lookup = hop_lookup
        self.targets = targets.astype(np.float32)  # (N', T)
        self.masks = masks.astype(np.float32)  # (N', T)
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.start_times = np.asarray(start_times, dtype=np.int64)
        self.window_size = window_size
        self.graph_names = list(graph_names)
        self.num_hops = num_hops
        self.feat_dim = feat_dim
        self.train_cutoff = train_cutoff
        self.zero_hops = np.zeros((len(self.graph_names), num_hops + 1, feat_dim), dtype=np.float32)

    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, idx):
        node_id = int(self.node_ids[idx])
        start_time = int(self.start_times[idx])
        x = np.zeros(
            (self.window_size, len(self.graph_names), self.num_hops + 1, self.feat_dim),
            dtype=np.float32,
        )
        for offset in range(self.window_size):
            timestep = start_time + offset
            x[offset] = self.hop_lookup.get((timestep, node_id), self.zero_hops)
        y = self.targets[idx, :, None]
        mask = self.masks[idx, :, None]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask),
            torch.tensor(self.node_ids[idx], dtype=torch.long),
            torch.tensor(self.start_times[idx], dtype=torch.long),
        )


def prepare_node_frames(node_files, feature_cols, target_col, mask_col):
    frames = {}
    for node_file in tqdm(node_files, desc="Loading node CSVs"):
        frame = pd.read_csv(node_file)
        if frame.empty:
            continue
        frame["timestep"] = pd.to_numeric(frame["timestep"], errors="coerce")
        frame["node_id"] = pd.to_numeric(frame["node_id"], errors="coerce")
        frame["project_id"] = pd.to_numeric(frame["project_id"], errors="coerce")
        frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
        frame[mask_col] = pd.to_numeric(frame[mask_col], errors="coerce").fillna(0)
        for col in feature_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)

        frame = frame.dropna(subset=["timestep", "node_id", "project_id"])
        frame = frame.sort_values("timestep").drop_duplicates("timestep", keep="last")
        if frame.empty:
            continue
        node_id = int(frame["node_id"].iloc[0])
        frames[node_id] = frame.reset_index(drop=True)
    return frames


def compute_feature_stats(node_frames, feature_cols, train_cutoff):
    train_parts = [
        frame.loc[frame["timestep"] < train_cutoff, feature_cols]
        for frame in node_frames.values()
        if (frame["timestep"] < train_cutoff).any()
    ]
    if not train_parts:
        train_parts = [frame[feature_cols] for frame in node_frames.values()]
    train_values = pd.concat(train_parts, ignore_index=True).to_numpy(dtype=np.float32)
    min_vals = train_values.min(axis=0, keepdims=True)
    max_vals = train_values.max(axis=0, keepdims=True)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    return min_vals, max_vals, range_vals


def normalize_frame_features(node_frames, feature_cols, min_vals, range_vals):
    for frame in tqdm(node_frames.values(), desc="Normalizing node features"):
        values = frame[feature_cols].to_numpy(dtype=np.float32)
        frame.loc[:, feature_cols] = 2 * (values - min_vals) / range_vals - 1


def build_timestep_hop_lookup(node_frames, edge_tables, feature_cols, num_hops):
    timestep_rows = defaultdict(list)
    for node_id, frame in tqdm(node_frames.items(), desc="Indexing timestep rows", total=len(node_frames)):
        for row in frame.to_dict("records"):
            timestep_rows[int(row["timestep"])].append(row)

    graph_names = list(edge_tables.keys())
    num_graphs = len(graph_names)
    feat_dim = len(feature_cols)
    hop_lookup = {}

    print(f"\nRunning multi-graph sign_precompute (K={num_hops}) across {len(timestep_rows)} timestamps ...")
    for timestep in tqdm(sorted(timestep_rows), desc="Precomputing graph hops"):
        rows = timestep_rows[timestep]
        node_ids = [int(row["node_id"]) for row in rows]
        project_ids = [int(row["project_id"]) for row in rows]
        node_to_local = {node_id: idx for idx, node_id in enumerate(node_ids)}
        project_to_local = {project_id: idx for idx, project_id in enumerate(project_ids)}
        x = torch.tensor(
            [[row[col] for col in feature_cols] for row in rows],
            dtype=torch.float32,
        )
        per_graph_hops = []
        for graph_name in graph_names:
            edges = edge_tables[graph_name]
            edge_key = edges.attrs.get("edge_key", "node_id")
            key_to_local = project_to_local if edge_key == "project_id" else node_to_local
            active_keys = set(key_to_local.keys())
            valid_edges = edges[
                edges["src"].isin(active_keys) & edges["dst"].isin(active_keys)
            ]
            if valid_edges.empty:
                edge_index = (torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
            else:
                src = valid_edges["src"].map(key_to_local).to_numpy(dtype=np.int64)
                dst = valid_edges["dst"].map(key_to_local).to_numpy(dtype=np.int64)
                edge_index = (torch.from_numpy(src), torch.from_numpy(dst))
            g = dgl.graph(edge_index, num_nodes=len(rows))
            g = dgl.remove_self_loop(g)
            g = dgl.add_reverse_edges(g)
            g = dgl.to_simple(g)
            g = dgl.add_self_loop(g)

            h_list = sign_precompute(x, g, num_hops)
            per_graph_hops.append(torch.stack([x] + h_list, dim=1))

        stacked = torch.stack(per_graph_hops, dim=1).numpy()  # (active_nodes, G, K+1, F)
        for local_idx, node_id in enumerate(node_ids):
            hop_lookup[(timestep, node_id)] = stacked[local_idx].astype(np.float32)

    return hop_lookup, graph_names, num_graphs, feat_dim


def build_node_window_arrays(
    node_frames,
    hop_lookup,
    graph_names,
    feat_dim,
    num_hops,
    window_size,
    train_cutoff,
    target_col,
    mask_col,
    target_mask_mode,
):
    train_y, train_masks, train_nodes, train_starts = [], [], [], []
    test_y, test_masks, test_nodes, test_starts = [], [], [], []

    for node_id, frame in tqdm(node_frames.items(), desc="Building node windows", total=len(node_frames)):
        timesteps = frame["timestep"].astype(int).tolist()
        if not timesteps:
            continue
        target_by_time = {
            int(row["timestep"]): float(row[target_col])
            for row in frame.to_dict("records")
        }
        observed_by_time = {
            int(row["timestep"]): bool(row[mask_col])
            for row in frame.to_dict("records")
        }

        for start_time in range(min(timesteps), max(timesteps) + 1):
            end_time = start_time + window_size - 1
            y_window = np.zeros(window_size, dtype=np.float32)
            observed_mask = np.zeros(window_size, dtype=np.float32)
            interpolated_mask = np.zeros(window_size, dtype=np.float32)

            for offset in range(window_size):
                timestep = start_time + offset
                if timestep in target_by_time:
                    target_value = target_by_time[timestep]
                    y_window[offset] = target_value
                    observed_mask[offset] = float(observed_by_time.get(timestep, False))
                    interpolated_mask[offset] = float(target_value != 0)

            if end_time < train_cutoff:
                mode = "allow_interpolated" if target_mask_mode == "train_allow_interpolated" else target_mask_mode
                mask_window = observed_mask if mode == "observed_only" else interpolated_mask
                bucket = (train_y, train_masks, train_nodes, train_starts)
            else:
                mode = "observed_only" if target_mask_mode == "train_allow_interpolated" else target_mask_mode
                mask_window = observed_mask if mode == "observed_only" else interpolated_mask
                bucket = (test_y, test_masks, test_nodes, test_starts)

            if mask_window.sum() <= 0:
                continue
            if end_time >= train_cutoff:
                for offset in range(window_size):
                    if start_time + offset < train_cutoff:
                        mask_window[offset] = 0.0
                if mask_window.sum() <= 0:
                    continue
            bucket[0].append(y_window)
            bucket[1].append(mask_window)
            bucket[2].append(node_id)
            bucket[3].append(start_time)

    def stack_or_empty(parts, shape_tail):
        if parts:
            return np.stack(parts, axis=0)
        return np.empty((0, *shape_tail), dtype=np.float32)

    train_arrays = (
        hop_lookup,
        stack_or_empty(train_y, (window_size,)),
        stack_or_empty(train_masks, (window_size,)),
        train_nodes,
        train_starts,
        window_size,
        graph_names,
        num_hops,
        feat_dim,
        train_cutoff,
    )
    test_arrays = (
        hop_lookup,
        stack_or_empty(test_y, (window_size,)),
        stack_or_empty(test_masks, (window_size,)),
        test_nodes,
        test_starts,
        window_size,
        graph_names,
        num_hops,
        feat_dim,
        train_cutoff,
    )
    return train_arrays, test_arrays


def get_dataloaders(
    root="database_v3/Graph_Size",
    egde_file="graph_link_200m/links.txt",
    ts_test=25,
    k=2,
    batch_size=32,
    shift=1,
    no_feat_col=None,
    target_col="rent_per_sqft",
    mask_col="y_mask",
    feat_norm=False,
    window_size=12,
    target_mask_mode="observed_only",
    ccr=True,
    nodes_dir="nodes",
    ccr_node_file="node_id_ccr.csv",
    edges_dir="edges",
    graph_edge_files=None,
    macro_file=DEFAULT_MACRO_FILE,
):
    del egde_file

    root_path = Path(root)
    nodes_path = root_path / nodes_dir
    ccr_path = root_path / ccr_node_file
    edge_path = root_path / edges_dir
    macro_path = root_path.parent / macro_file

    graph_edge_files = list(graph_edge_files or DEFAULT_GRAPH_EDGE_FILES)
    node_files = discover_node_files(nodes_path, ccr=ccr, ccr_path=ccr_path)
    if not node_files:
        raise ValueError(f"No node CSV files found under {nodes_path}")

    sample_frame = pd.read_csv(node_files[0])
    default_no_feat_col = set(DEFAULT_FEATURE_EXCLUDE_COLS)
    if no_feat_col is None:
        no_feat_col = default_no_feat_col
    else:
        no_feat_col = set(no_feat_col).union(default_no_feat_col)
    feature_cols = infer_feature_columns(sample_frame, target_col, mask_col, no_feat_col=no_feat_col)
    excluded_cols = set(no_feat_col).union({target_col, mask_col})
    print(
        "Using node CSV feature columns "
        f"(excluded={sorted(excluded_cols)}, count={len(feature_cols)})."
    )

    node_frames = prepare_node_frames(node_files, feature_cols, target_col, mask_col)
    if not node_frames:
        raise ValueError(f"No non-empty node CSV files found under {nodes_path}")

    macro_lookup, macro_feature_cols, min_macro_timestep = load_macro_features(macro_path, feature_cols)
    print(
        f"Macro features: {len(macro_feature_cols)} columns from {macro_path} | "
        f"macro_lag_shift={shift} | earliest_macro_timestep={min_macro_timestep}"
    )
    apply_macro_shift(node_frames, macro_lookup, macro_feature_cols, shift, min_macro_timestep)

    all_timesteps = sorted({int(t) for frame in node_frames.values() for t in frame["timestep"]})
    total_steps = len(all_timesteps)
    if ts_test <= 0 or ts_test >= total_steps:
        raise ValueError(f"ts_test must be in [1, T-1], got ts_test={ts_test}, T={total_steps}")
    train_cutoff = all_timesteps[-ts_test]

    if feat_norm:
        min_vals, _, range_vals = compute_feature_stats(node_frames, feature_cols, train_cutoff)
        normalize_frame_features(node_frames, feature_cols, min_vals, range_vals)

    edge_tables = {}
    for edge_file in tqdm(graph_edge_files, desc="Loading graph edge CSVs"):
        edge_tables[Path(edge_file).stem] = load_edge_table(edge_path / edge_file)
    print(
        f"Condos: {len(node_frames)} | Timestamps: {total_steps} | feat_dim: {len(feature_cols)} | "
        f"Graphs: {len(edge_tables)} | CCR: {ccr}"
    )
    for name, edges in edge_tables.items():
        print(f"  Graph {name}: {len(edges)} global edges")

    hop_lookup, graph_names, _, feat_dim = build_timestep_hop_lookup(
        node_frames=node_frames,
        edge_tables=edge_tables,
        feature_cols=feature_cols,
        num_hops=k,
    )
    train_arrays, test_arrays = build_node_window_arrays(
        node_frames=node_frames,
        hop_lookup=hop_lookup,
        graph_names=graph_names,
        feat_dim=feat_dim,
        num_hops=k,
        window_size=window_size,
        train_cutoff=train_cutoff,
        target_col=target_col,
        mask_col=mask_col,
        target_mask_mode=target_mask_mode,
    )

    train_ds = MultiGraphWindowDataset(*train_arrays)
    test_ds = MultiGraphWindowDataset(*test_arrays)

    print(
        f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)} | "
        f"Per-sample input: (window={window_size}, graphs={len(graph_names)}, K+1={k + 1}, feat_dim={feat_dim})"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug PGIM dataloader construction")
    parser.add_argument("--root", type=str, default="database_v3/Graph_Size")
    parser.add_argument("--egde-file", type=str, default="graph_link_200m/links.txt")
    parser.add_argument("--ts-test", type=int, default=25)
    parser.add_argument("--k", type=int, default=2, help="Number of SIGN hops")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shift", type=int, default=1)
    parser.add_argument("--ccr", type=str2bool, default=True)
    parser.add_argument(
        "--target-mask-mode",
        type=str,
        choices=("observed_only", "allow_interpolated", "train_allow_interpolated"),
        default="observed_only",
        help="Whether to count only observed targets, allow interpolated targets for both train/test, or allow them only in train.",
    )
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument(
        "--feat-norm",
        action="store_true",
        help="Apply train-set min-max normalization to [-1, 1]. Disabled by default because node CSV features are already normalized.",
    )
    args = parser.parse_args()

    train_loader, test_loader = get_dataloaders(
        root=args.root,
        egde_file=args.egde_file,
        ts_test=args.ts_test,
        k=args.k,
        batch_size=args.batch_size,
        shift=args.shift,
        ccr=args.ccr,
        target_mask_mode=args.target_mask_mode,
        feat_norm=args.feat_norm,
        window_size=args.window_size,
    )

    print("\nDataloader summary:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    train_batch = next(iter(train_loader), None)
    test_batch = next(iter(test_loader), None)

    if train_batch is not None:
        x_train, y_train, mask_train, node_train, time_train = train_batch
        print("\nFirst train batch:")
        print(f"  x shape: {tuple(x_train.shape)}")
        print(f"  y shape: {tuple(y_train.shape)}")
        print(f"  mask shape: {tuple(mask_train.shape)}")
        print(f"  node_idx shape: {tuple(node_train.shape)}")
        print(f"  time_idx shape: {tuple(time_train.shape)}")
        print(f"  x dtype: {x_train.dtype}")
        print(f"  y dtype: {y_train.dtype}")

    if test_batch is not None:
        x_test, y_test, mask_test, node_test, time_test = test_batch
        print("\nFirst test batch:")
        print(f"  x shape: {tuple(x_test.shape)}")
        print(f"  y shape: {tuple(y_test.shape)}")
        print(f"  mask shape: {tuple(mask_test.shape)}")
        print(f"  node_idx shape: {tuple(node_test.shape)}")
        print(f"  time_idx shape: {tuple(time_test.shape)}")
        print(f"  x dtype: {x_test.dtype}")
        print(f"  y dtype: {y_test.dtype}")


    print(len(train_loader.dataset), len(test_loader.dataset))
