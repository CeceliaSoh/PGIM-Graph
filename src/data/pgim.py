import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.graph.form_edge import FormEdgeProcessor
from src.graph.form_node import FormNodeProcessor

DEFAULT_GRAPH_CONFIG_PATH = "src/config/graph/V260519.yaml"
logger = logging.getLogger(__name__)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


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


def infer_feature_columns(frame: pd.DataFrame, id_columns=None, target_columns=None, extra_exclude=None):
    excluded = set(extra_exclude or set()).union(id_columns or set()).union(target_columns or set())
    return [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def to_plain_config(config):
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    if config is None:
        return {}
    return dict(config)


def normalize_loader_config(config):
    config = to_plain_config(config)
    if "data" in config:
        data_config = dict(config["data"])
        training_config = config.get("training", {})
        data_config.setdefault("batch_size", training_config.get("batch_size", 32))
        data_config.setdefault("random_seed", config.get("seed", 42))
    else:
        data_config = config

    data_config.setdefault("graph_config_path", DEFAULT_GRAPH_CONFIG_PATH)
    data_config.setdefault("ts_test", 25)
    data_config.setdefault("num_hops", 2)
    data_config.setdefault("batch_size", 32)
    data_config.setdefault("target_col", "rent_per_sqft_imp")
    data_config.setdefault("mask_col", "y_mask")
    data_config.setdefault("feat_norm", True)
    data_config.setdefault("window_size", 12)
    data_config.setdefault("target_shift", 1)
    data_config.setdefault("target_mask_mode", "train_all_test_observed")
    data_config.setdefault("id_columns", ["node_id", "project_id", "timestep"])
    data_config.setdefault("target_columns", ["y_mask", "rent_per_sqft", "rent_per_sqft_imp"])
    data_config.setdefault("extra_exclude_columns", [])
    data_config.setdefault("rp_dim", 32)
    data_config.setdefault("random_seed", 42)
    return data_config


def feature_normalize(frame: pd.DataFrame, mode, stats_folder, feature_cols):
    """Apply saved preprocessing normalization statistics to selected feature columns."""
    if mode in (None, False, "false", "False", "none", "None", "off", "raw"):
        return frame.copy()

    mode = "norm1" if mode is True else str(mode)
    stats_filename = {
        "norm0": "norm0.json",
        "norm1": "norm1.json",
        "stand": "stand.json",
        "standard": "stand.json",
        "standardize": "stand.json",
        "standardise": "stand.json",
    }.get(mode)
    if stats_filename is None:
        raise ValueError(
            "feat_norm must be one of norm0, norm1, stand, false/raw, "
            f"got {mode!r}."
        )

    stats_path = Path(stats_folder) / stats_filename
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing normalization stats file: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    normalized = frame.copy()
    missing_stats = [col for col in feature_cols if col not in stats]
    if missing_stats:
        raise KeyError(
            f"Missing {mode} stats for feature columns in {stats_path}: {missing_stats}"
        )

    for col in feature_cols:
        values = pd.to_numeric(normalized[col], errors="coerce").astype(float)
        col_stats = stats[col]
        if stats_filename == "stand.json":
            std = float(col_stats.get("std", 1.0)) or 1.0
            normalized[col] = (values - float(col_stats["mean"])) / std
        else:
            denominator = float(col_stats.get("denominator", 1.0)) or 1.0
            target_min = float(col_stats.get("target_min", 0.0))
            target_max = float(col_stats.get("target_max", 1.0))
            normalized[col] = (
                (values - float(col_stats["min"]))
                / denominator
                * (target_max - target_min)
                + target_min
            )

    normalized[feature_cols] = normalized[feature_cols].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return normalized


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


def edge_pairs_from_frame(edges: pd.DataFrame, source):
    if not {"node_id", "neig_node_id"}.issubset(edges.columns):
        raise ValueError(f"Expected node_id/neig_node_id columns in {source}")
    edges = edges[["node_id", "neig_node_id"]].copy()
    edges["node_id"] = pd.to_numeric(edges["node_id"], errors="coerce")
    edges["neig_node_id"] = pd.to_numeric(edges["neig_node_id"], errors="coerce")
    return edges.dropna().astype({"node_id": int, "neig_node_id": int}).drop_duplicates()


def build_heterograph_from_edge_frames(edge_frames, project_node_ids, size_node_ids):
    project_index = {int(node_id): i for i, node_id in enumerate(project_node_ids)}
    size_index = {int(node_id): i for i, node_id in enumerate(size_node_ids)}
    graph_data = {}

    def add_edges(src_type, etype, dst_type, src, dst):
        src_tensor = torch.tensor(src, dtype=torch.int64)
        dst_tensor = torch.tensor(dst, dtype=torch.int64)
        graph_data[(src_type, etype, dst_type)] = (src_tensor, dst_tensor)
        graph_data[(dst_type, f"rev_{etype}", src_type)] = (dst_tensor, src_tensor)

    for edge_name, edge_frame in edge_frames.items():
        pairs = edge_pairs_from_frame(edge_frame, edge_name)
        stem = Path(edge_name).stem
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
    def __init__(
        self,
        contexts,
        targets,
        masks,
        node_indices,
        target_start_times,
        window_size,
        target_shift=0,
        min_target_time=None,
    ):
        self.contexts = contexts.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.masks = masks.astype(np.float32)
        self.node_indices = np.asarray(node_indices, dtype=np.int64)
        self.target_start_times = np.asarray(target_start_times, dtype=np.int64)
        self.window_size = window_size
        self.target_shift = target_shift
        self.min_target_time = min_target_time

    def __len__(self):
        return len(self.node_indices)

    def __getitem__(self, idx):
        node_idx = int(self.node_indices[idx])
        target_start = int(self.target_start_times[idx])
        target_end = target_start + self.window_size
        input_start = target_start - self.target_shift
        input_end = input_start + self.window_size

        valid_input_start = max(input_start, 0)
        valid_input_end = min(input_end, self.contexts.shape[1])
        if valid_input_end < valid_input_start:
            valid_input_end = valid_input_start

        input_window = self.contexts[node_idx, valid_input_start:valid_input_end]
        left_pad = min(self.window_size, max(-input_start, 0))
        right_pad = self.window_size - left_pad - input_window.shape[0]
        if left_pad > 0:
            input_window = np.concatenate(
                [
                    np.zeros((left_pad, *self.contexts.shape[2:]), dtype=np.float32),
                    input_window,
                ],
                axis=0,
            )
        if right_pad > 0:
            input_window = np.concatenate(
                [
                    input_window,
                    np.zeros((right_pad, *self.contexts.shape[2:]), dtype=np.float32),
                ],
                axis=0,
            )

        target_mask_window = self.masks[node_idx, target_start:target_end].copy()
        if self.min_target_time is not None:
            target_mask_window[: max(int(self.min_target_time) - target_start, 0)] = 0.0
        return (
            torch.from_numpy(input_window),
            torch.from_numpy(self.targets[node_idx, target_start:target_end, None]),
            torch.from_numpy(target_mask_window[:, None]),
            torch.tensor(node_idx, dtype=torch.long),
            torch.tensor(target_start, dtype=torch.long),
        )


def build_window_indices(masks, window_size, train_steps, target_shift=0):
    train_nodes, train_target_starts, test_nodes, test_target_starts = [], [], [], []
    num_nodes, num_timesteps = masks.shape
    max_target_start = num_timesteps - window_size + 1
    if max_target_start <= 0:
        raise ValueError(
            f"window_size must be <= number of timesteps, got "
            f"window_size={window_size}, T={num_timesteps}."
        )

    for node_idx in range(num_nodes):
        for target_start in range(max_target_start):
            target_end = target_start + window_size
            mask_window = masks[node_idx, target_start:target_end]
            if target_end <= train_steps:
                if target_start - target_shift < 0:
                    continue
                if mask_window.sum() > 0:
                    train_nodes.append(node_idx)
                    train_target_starts.append(target_start)
            else:
                eval_mask = mask_window.copy()
                eval_mask[: max(train_steps - target_start, 0)] = 0.0
                if eval_mask.sum() > 0:
                    test_nodes.append(node_idx)
                    test_target_starts.append(target_start)
    return (train_nodes, train_target_starts), (test_nodes, test_target_starts)


def get_dataloaders(config):
    loader_config = normalize_loader_config(config)
    ts_test = loader_config["ts_test"]
    num_hops = loader_config["num_hops"]
    batch_size = loader_config["batch_size"]
    target_col = loader_config["target_col"]
    mask_col = loader_config["mask_col"]
    feat_norm = loader_config["feat_norm"]
    window_size = loader_config["window_size"]
    target_shift = loader_config["target_shift"]
    target_mask_mode = loader_config["target_mask_mode"]
    rp_dim = loader_config["rp_dim"]
    random_seed = loader_config["random_seed"]

    if target_shift < 0:
        raise ValueError(f"target_shift must be >= 0, got {target_shift}.")

    node_processor = FormNodeProcessor(config.graph)
    edge_processor = FormEdgeProcessor(config.graph)
    node_processor.process()
    edge_processor.process()
    
    node_map, project_frame, size_frame = node_processor.load_node_tables()
    edge_dfs = edge_processor.load_enabled_edge_dfs()

    project_node_ids = node_map.loc[node_map["size_tier"].eq(0), "node_id"].astype(int).sort_values().tolist()
    size_node_ids = node_map.loc[~node_map["size_tier"].eq(0), "node_id"].astype(int).sort_values().tolist()
    logger.info(
        "Project nodes in each timestep: %s, Size nodes: %s",
        len(project_node_ids),
        len(size_node_ids),
    )
    timesteps = sorted(size_frame["timestep"].dropna().astype(int).unique().tolist())
    logger.info("Total Timesteps: %s", len(timesteps))
    if ts_test <= 0 or ts_test >= len(timesteps):
        raise ValueError(f"ts_test must be in [1, T-1], got ts_test={ts_test}, T={len(timesteps)}")
    train_steps = len(timesteps) - ts_test
    logger.info("Train steps: %s, Test steps: %s", train_steps, ts_test)

    extra_exclude = loader_config["extra_exclude_columns"]
    target_cols = loader_config["target_columns"]
    id_cols = loader_config["id_columns"]
    project_feature_cols = infer_feature_columns(project_frame, id_cols, target_cols, extra_exclude)
    size_feature_cols = infer_feature_columns(size_frame, id_cols, target_cols, extra_exclude)

    if not project_feature_cols or not size_feature_cols:
        raise ValueError("Project and size node tables must each have at least one numeric feature column.")

    logger.info(
        "Using database_260519 RpHGNN loader: "
        "project_nodes=%s, size_nodes=%s, timesteps=%s, project_feat=%s, "
        "size_feat=%s, rp_dim=%s, target_shift=%s",
        len(project_node_ids),
        len(size_node_ids),
        len(timesteps),
        len(project_feature_cols),
        len(size_feature_cols),
        rp_dim,
        target_shift,
    )

    project_frame = feature_normalize(
        project_frame,
        feat_norm,
        node_processor.out_paths["project_node_folder"],
        project_feature_cols,
    )
    size_frame = feature_normalize(
        size_frame,
        feat_norm,
        node_processor.out_paths["size_node_folder"],
        size_feature_cols,
    )

    project_features = build_time_node_tensor(project_frame, project_node_ids, timesteps, project_feature_cols)
    size_features = build_time_node_tensor(size_frame, size_node_ids, timesteps, size_feature_cols)


    project_features = random_project_features(project_features, rp_dim, random_seed + 17)
    size_features = random_project_features(size_features, rp_dim, random_seed + 29)

    graph = build_heterograph_from_edge_frames(edge_dfs, project_node_ids, size_node_ids)
    logger.info("Heterograph: %s", graph)
    contexts, group_names = rphgnn_precompute_contexts(graph, project_features, size_features, num_hops)
    logger.info("RpHGNN groups: %s", group_names)

    targets, masks = build_target_tensor(
        size_frame=size_frame,
        size_node_ids=size_node_ids,
        timesteps=timesteps,
        target_col=target_col,
        mask_col=mask_col,
        target_mask_mode=target_mask_mode,
        train_steps=train_steps,
    )
    (train_nodes, train_target_starts), (test_nodes, test_target_starts) = build_window_indices(
        masks,
        window_size,
        train_steps,
        target_shift,
    )

    train_ds = RpHGNNWindowDataset(
        contexts,
        targets,
        masks,
        train_nodes,
        train_target_starts,
        window_size,
        target_shift=target_shift,
    )
    test_ds = RpHGNNWindowDataset(
        contexts,
        targets,
        masks,
        test_nodes,
        test_target_starts,
        window_size,
        target_shift=target_shift,
        min_target_time=train_steps,
    )
    logger.info(
        "Train samples: %s | Test samples: %s | "
        "Per-sample input: (window=%s, groups=%s, K+1=%s, rp_dim=%s), target_shift=%s",
        len(train_ds),
        len(test_ds),
        window_size,
        contexts.shape[2],
        contexts.shape[3],
        contexts.shape[4],
        target_shift,
    )
    generator = torch.Generator()
    generator.manual_seed(random_seed)
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
            worker_init_fn=seed_worker,
        ),
        DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            worker_init_fn=seed_worker,
        ),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = argparse.ArgumentParser(description="Debug database_260519 RpHGNN dataloader construction")
    parser.add_argument("--graph-config", type=str, default=DEFAULT_GRAPH_CONFIG_PATH)
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--target-shift", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rp-dim", type=int, default=32)
    args = parser.parse_args()
    graph_config = OmegaConf.load(args.graph_config)
    cfg = OmegaConf.create(
        {
            "graph": graph_config,
            "data": {
                "graph_config_path": args.graph_config,
                "num_hops": args.num_hops,
                "window_size": args.window_size,
                "target_shift": args.target_shift,
                "rp_dim": args.rp_dim,
            },
            "training": {
                "batch_size": args.batch_size,
            },
            "seed": 42,
        }
    )
    train_loader, test_loader = get_dataloaders(cfg)
    batch = next(iter(train_loader))
    logger.info("First train batch: %s", [item.shape if hasattr(item, "shape") else item for item in batch])
