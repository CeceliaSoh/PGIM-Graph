import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


DEFAULT_GRAPH_EDGE_FILES = (
    "dist_250.csv",
    "same_age.csv",
    "same_mrt_250.csv",
    "same_mrt_dist_eps_5.csv",
    "same_planning_area.csv",
    "same_school_dist_eps_0p01.csv",
    "size_project.csv",
)

ID_COLUMNS = {
    "node_id",
    "project_id",
    "timestep",
    "project_id_feat",
    "node_id_feat",
    "timestep_feat",
}
TARGET_COLUMNS = {
    "rent_per_sqft",
    "rent_per_sqft_imp",
    "rent_per_sqft_feat",
    "y_mask",
}


@dataclass
class HeteroGraphSpec:
    edge_index: dict
    num_nodes: dict

    @property
    def canonical_etypes(self):
        return list(self.edge_index.keys())

    @property
    def ntypes(self):
        return list(self.num_nodes.keys())

    def num_nodes_of(self, ntype):
        return self.num_nodes[ntype]

    def __repr__(self):
        edge_counts = {
            f"{src}:{etype}:{dst}": int(edges[0].numel())
            for (src, etype, dst), edges in self.edge_index.items()
        }
        return f"HeteroGraphSpec(num_nodes={self.num_nodes}, edge_counts={edge_counts})"


def infer_feature_columns(frame: pd.DataFrame, extra_exclude=None):
    excluded = set(extra_exclude or set()).union(ID_COLUMNS).union(TARGET_COLUMNS)
    return [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def load_node_table(root: Path, filename: str):
    node_path = root / "nodes" / filename
    if not node_path.exists():
        node_path = root / filename
    if not node_path.exists():
        raise FileNotFoundError(f"Could not find {filename} under {root} or {root / 'nodes'}")
    return pd.read_csv(node_path)


def train_minmax_scale(features: np.ndarray, train_steps: int):
    train_values = features[:train_steps].reshape(-1, features.shape[-1])
    min_vals = np.nanmin(train_values, axis=0, keepdims=True)
    max_vals = np.nanmax(train_values, axis=0, keepdims=True)
    min_vals = np.nan_to_num(min_vals, nan=0.0)
    max_vals = np.nan_to_num(max_vals, nan=0.0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    scaled = 2.0 * (features - min_vals.reshape(1, 1, -1)) / ranges.reshape(1, 1, -1) - 1.0
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_time_node_tensor(frame, node_ids, timesteps, feature_cols):
    frame = frame.copy()
    for col in ["node_id", "timestep", *feature_cols]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.sort_values(["timestep", "node_id"]).drop_duplicates(["timestep", "node_id"], keep="last")

    time_index = {int(t): i for i, t in enumerate(timesteps)}
    node_index = {int(n): i for i, n in enumerate(node_ids)}
    tensor = np.zeros((len(timesteps), len(node_ids), len(feature_cols)), dtype=np.float32)

    rows = frame[["timestep", "node_id", *feature_cols]].copy()
    rows["time_idx"] = rows["timestep"].map(time_index)
    rows["node_idx"] = rows["node_id"].map(node_index)
    rows = rows.dropna(subset=["time_idx", "node_idx"])
    if rows.empty:
        return tensor

    time_idx = rows["time_idx"].to_numpy(dtype=np.int64)
    node_idx = rows["node_idx"].to_numpy(dtype=np.int64)
    values = rows[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    tensor[time_idx, node_idx] = values
    return tensor


def build_target_tensor(size_frame, size_node_ids, timesteps, target_col, mask_col, target_mask_mode, train_steps):
    frame = size_frame.copy()
    for col in ["node_id", "timestep", target_col, mask_col]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.sort_values(["timestep", "node_id"]).drop_duplicates(["timestep", "node_id"], keep="last")

    time_index = {int(t): i for i, t in enumerate(timesteps)}
    node_index = {int(n): i for i, n in enumerate(size_node_ids)}
    targets = np.zeros((len(size_node_ids), len(timesteps)), dtype=np.float32)
    observed = np.zeros_like(targets)

    rows = frame[["timestep", "node_id", target_col, mask_col]].copy()
    rows["time_idx"] = rows["timestep"].map(time_index)
    rows["node_idx"] = rows["node_id"].map(node_index)
    rows = rows.dropna(subset=["time_idx", "node_idx"])
    time_idx = rows["time_idx"].to_numpy(dtype=np.int64)
    node_idx = rows["node_idx"].to_numpy(dtype=np.int64)
    targets[node_idx, time_idx] = rows[target_col].fillna(0.0).to_numpy(dtype=np.float32)
    observed[node_idx, time_idx] = rows[mask_col].fillna(0.0).to_numpy(dtype=np.float32)

    if target_mask_mode == "all":
        masks = np.ones_like(targets, dtype=np.float32)
    elif target_mask_mode == "observed_only":
        masks = observed
    elif target_mask_mode == "train_all_test_observed":
        masks = np.ones_like(targets, dtype=np.float32)
        masks[:, train_steps:] = observed[:, train_steps:]
    else:
        raise ValueError(f"Unknown target_mask_mode={target_mask_mode}")
    return targets, masks


def random_project_features(features: np.ndarray, out_dim: int, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.from_numpy(features.astype(np.float32))
    if x.shape[-1] != out_dim:
        scale = np.sqrt(2.0 / max(x.shape[-1] + out_dim, 1))
        kernel = torch.randn((x.shape[-1], out_dim), generator=generator, dtype=torch.float32) * scale
        x = x @ kernel
    x = torch.nn.functional.normalize(x, dim=-1)
    return x.numpy().astype(np.float32)


def edge_pairs(edge_path: Path):
    edges = pd.read_csv(edge_path)
    if not {"node_id", "neig_node_id"}.issubset(edges.columns):
        raise ValueError(f"Expected node_id/neig_node_id columns in {edge_path}")
    edges = edges[["node_id", "neig_node_id"]].copy()
    edges["node_id"] = pd.to_numeric(edges["node_id"], errors="coerce")
    edges["neig_node_id"] = pd.to_numeric(edges["neig_node_id"], errors="coerce")
    return edges.dropna().astype({"node_id": int, "neig_node_id": int}).drop_duplicates()


def build_heterograph(root, graph_edge_files, project_node_ids, size_node_ids):
    edge_root = Path(root) / "edges"
    project_index = {int(node_id): i for i, node_id in enumerate(project_node_ids)}
    size_index = {int(node_id): i for i, node_id in enumerate(size_node_ids)}
    graph_data = {}

    def add_edges(src_type, etype, dst_type, src, dst):
        src_tensor = torch.tensor(src, dtype=torch.int64)
        dst_tensor = torch.tensor(dst, dtype=torch.int64)
        graph_data[(src_type, etype, dst_type)] = (src_tensor, dst_tensor)
        graph_data[(dst_type, f"rev_{etype}", src_type)] = (dst_tensor, src_tensor)

    for edge_file in graph_edge_files:
        pairs = edge_pairs(edge_root / edge_file)
        stem = Path(edge_file).stem
        if stem == "size_project":
            src = pairs["node_id"].map(size_index)
            dst = pairs["neig_node_id"].map(project_index)
            valid = src.notna() & dst.notna()
            add_edges("size", stem, "project", src[valid].astype(int).tolist(), dst[valid].astype(int).tolist())
        else:
            src = pairs["node_id"].map(project_index)
            dst = pairs["neig_node_id"].map(project_index)
            valid = src.notna() & dst.notna()
            add_edges("project", stem, "project", src[valid].astype(int).tolist(), dst[valid].astype(int).tolist())

    return HeteroGraphSpec(
        edge_index=graph_data,
        num_nodes={"project": len(project_node_ids), "size": len(size_node_ids)},
    )


def mean_aggregate(src_h, src_idx, dst_idx, num_dst):
    out = torch.zeros((num_dst, src_h.shape[-1]), dtype=src_h.dtype)
    counts = torch.zeros((num_dst, 1), dtype=src_h.dtype)
    if src_idx.numel() == 0:
        return out
    out.index_add_(0, dst_idx, src_h[src_idx])
    counts.index_add_(0, dst_idx, torch.ones((dst_idx.numel(), 1), dtype=src_h.dtype))
    return out / counts.clamp(min=1.0)


def rphgnn_precompute_contexts(graph, project_features, size_features, num_hops):
    target_type = "size"
    incoming_target_etypes = [etype for etype in graph.canonical_etypes if etype[-1] == target_type]
    group_names = ["self", *[etype[1] for etype in incoming_target_etypes]]
    group_index = {name: idx for idx, name in enumerate(group_names)}

    num_timesteps, num_size_nodes, rp_dim = size_features.shape
    contexts = np.zeros(
        (num_size_nodes, num_timesteps, len(group_names), num_hops + 1, rp_dim),
        dtype=np.float32,
    )

    for t in tqdm(range(num_timesteps), desc="RpHGNN precompute"):
        h_dict = {
            "project": torch.from_numpy(project_features[t]).float(),
            "size": torch.from_numpy(size_features[t]).float(),
        }
        contexts[:, t, group_index["self"], 0, :] = size_features[t]

        for hop in range(1, num_hops + 1):
            next_parts = defaultdict(list)
            for etype in graph.canonical_etypes:
                src_type, _, dst_type = etype
                src_idx, dst_idx = graph.edge_index[etype]
                msg = mean_aggregate(
                    h_dict[src_type],
                    src_idx,
                    dst_idx,
                    graph.num_nodes_of(dst_type),
                )
                next_parts[dst_type].append(msg)
                if dst_type == target_type:
                    contexts[:, t, group_index[etype[1]], hop, :] = msg.numpy()

            next_h = {}
            for ntype in graph.ntypes:
                if next_parts[ntype]:
                    mixed = torch.stack(next_parts[ntype], dim=0).mean(dim=0)
                    next_h[ntype] = torch.nn.functional.normalize(0.5 * h_dict[ntype] + 0.5 * mixed, dim=-1)
                else:
                    next_h[ntype] = h_dict[ntype]
            h_dict = next_h
            contexts[:, t, group_index["self"], hop, :] = h_dict[target_type].numpy()

    return contexts, group_names


class RpHGNNWindowDataset(Dataset):
    def __init__(self, contexts, targets, masks, node_indices, start_times, window_size):
        self.contexts = contexts.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.masks = masks.astype(np.float32)
        self.node_indices = np.asarray(node_indices, dtype=np.int64)
        self.start_times = np.asarray(start_times, dtype=np.int64)
        self.window_size = window_size

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, idx):
        node_idx = int(self.node_indices[idx])
        start = int(self.start_times[idx])
        end = start + self.window_size
        return (
            torch.from_numpy(self.contexts[node_idx, start:end]),
            torch.from_numpy(self.targets[node_idx, start:end, None]),
            torch.from_numpy(self.masks[node_idx, start:end, None]),
            torch.tensor(node_idx, dtype=torch.long),
            torch.tensor(start, dtype=torch.long),
        )


def build_window_indices(masks, window_size, train_steps):
    train_nodes, train_starts, test_nodes, test_starts = [], [], [], []
    num_nodes, num_timesteps = masks.shape
    for node_idx in range(num_nodes):
        for start in range(0, num_timesteps - window_size + 1):
            end = start + window_size
            mask_window = masks[node_idx, start:end]
            if end <= train_steps:
                if mask_window.sum() > 0:
                    train_nodes.append(node_idx)
                    train_starts.append(start)
            else:
                eval_mask = mask_window.copy()
                eval_mask[: max(train_steps - start, 0)] = 0.0
                if eval_mask.sum() > 0:
                    test_nodes.append(node_idx)
                    test_starts.append(start)
    return (train_nodes, train_starts), (test_nodes, test_starts)


def get_dataloaders(
    root="dataset/database_260519",
    egde_file=None,
    ts_test=25,
    k=2,
    batch_size=32,
    shift=1,
    no_feat_col=None,
    target_col="rent_per_sqft_imp",
    mask_col="y_mask",
    feat_norm=True,
    window_size=12,
    target_mask_mode="train_all_test_observed",
    ccr=True,
    nodes_dir=None,
    ccr_node_file=None,
    edges_dir=None,
    graph_edge_files=None,
    macro_file=None,
    rp_dim=32,
    random_seed=42,
):
    del egde_file, shift, ccr, nodes_dir, ccr_node_file, edges_dir, macro_file

    root = Path(root)
    graph_edge_files = list(graph_edge_files or DEFAULT_GRAPH_EDGE_FILES)

    node_map = pd.read_csv(root / "node_id.csv")
    project_node_ids = node_map.loc[node_map["size_tier"].eq(0), "node_id"].astype(int).sort_values().tolist()
    size_node_ids = node_map.loc[~node_map["size_tier"].eq(0), "node_id"].astype(int).sort_values().tolist()

    project_frame = load_node_table(root, "project_level_node.csv")
    size_frame = load_node_table(root, "size_level_node.csv")
    timesteps = sorted(size_frame["timestep"].dropna().astype(int).unique().tolist())
    if ts_test <= 0 or ts_test >= len(timesteps):
        raise ValueError(f"ts_test must be in [1, T-1], got ts_test={ts_test}, T={len(timesteps)}")
    train_steps = len(timesteps) - ts_test

    extra_exclude = set(no_feat_col or [])
    project_feature_cols = infer_feature_columns(project_frame, extra_exclude)
    size_feature_cols = infer_feature_columns(size_frame, extra_exclude)
    if not project_feature_cols or not size_feature_cols:
        raise ValueError("Project and size node tables must each have at least one numeric feature column.")

    print(
        "Using database_260519 RpHGNN loader: "
        f"project_nodes={len(project_node_ids)}, size_nodes={len(size_node_ids)}, "
        f"timesteps={len(timesteps)}, project_feat={len(project_feature_cols)}, "
        f"size_feat={len(size_feature_cols)}, rp_dim={rp_dim}"
    )

    project_features = build_time_node_tensor(project_frame, project_node_ids, timesteps, project_feature_cols)
    size_features = build_time_node_tensor(size_frame, size_node_ids, timesteps, size_feature_cols)
    if feat_norm:
        project_features = train_minmax_scale(project_features, train_steps)
        size_features = train_minmax_scale(size_features, train_steps)
    else:
        project_features = np.nan_to_num(project_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        size_features = np.nan_to_num(size_features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    project_features = random_project_features(project_features, rp_dim, random_seed + 17)
    size_features = random_project_features(size_features, rp_dim, random_seed + 29)

    graph = build_heterograph(root, graph_edge_files, project_node_ids, size_node_ids)
    print(f"Heterograph: {graph}")
    contexts, group_names = rphgnn_precompute_contexts(graph, project_features, size_features, k)
    print(f"RpHGNN groups: {group_names}")

    targets, masks = build_target_tensor(
        size_frame=size_frame,
        size_node_ids=size_node_ids,
        timesteps=timesteps,
        target_col=target_col,
        mask_col=mask_col,
        target_mask_mode=target_mask_mode,
        train_steps=train_steps,
    )
    (train_nodes, train_starts), (test_nodes, test_starts) = build_window_indices(masks, window_size, train_steps)

    train_ds = RpHGNNWindowDataset(contexts, targets, masks, train_nodes, train_starts, window_size)
    test_ds = RpHGNNWindowDataset(contexts, targets, masks, test_nodes, test_starts, window_size)
    print(
        f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)} | "
        f"Per-sample input: (window={window_size}, groups={contexts.shape[2]}, "
        f"K+1={contexts.shape[3]}, rp_dim={contexts.shape[4]})"
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug database_260519 RpHGNN dataloader construction")
    parser.add_argument("--root", type=str, default="dataset/database_260519")
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rp-dim", type=int, default=32)
    args = parser.parse_args()
    train_loader, test_loader = get_dataloaders(
        root=args.root,
        k=args.num_hops,
        window_size=args.window_size,
        batch_size=args.batch_size,
        rp_dim=args.rp_dim,
    )
    batch = next(iter(train_loader))
    print([item.shape if hasattr(item, "shape") else item for item in batch])
