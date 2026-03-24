from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TemporalWindowDataset(Dataset):
    """Node-centric sliding-window dataset with time-based splits."""

    def __init__(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        split_type: str,
        window_size: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        if split_type not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split_type: {split_type}")
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        if features.ndim != 3:
            raise ValueError(f"Expected features with shape [N, T, D], got {tuple(features.shape)}")
        if targets.ndim != 2:
            raise ValueError(f"Expected targets with shape [N, T], got {tuple(targets.shape)}")

        self.features = features
        self.targets = targets
        self.target_mask = target_mask
        self.split_type = split_type
        self.window_size = window_size

        self.num_nodes, self.num_timesteps, _ = features.shape
        train_end = int(self.num_timesteps * train_ratio)
        val_end = train_end + int(self.num_timesteps * val_ratio)

        split_ranges = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, self.num_timesteps),
        }
        self.start_t, self.end_t = split_ranges[split_type]
        self.num_windows_per_node = max(0, (self.end_t - self.start_t) - window_size + 1)

    def __len__(self) -> int:
        return self.num_nodes * self.num_windows_per_node

    def __getitem__(self, idx: int):
        if self.num_windows_per_node == 0:
            raise IndexError(f"No windows available for split '{self.split_type}'")

        node_idx = idx // self.num_windows_per_node
        window_idx = idx % self.num_windows_per_node
        window_start = self.start_t + window_idx
        window_end = window_start + self.window_size

        return {
            "x": self.features[node_idx, window_start:window_end],
            "y": self.targets[node_idx, window_start:window_end],
            "mask": self.target_mask[node_idx, window_start:window_end],
            "node_idx": node_idx,
            "time_start": window_start,
            "time_end": window_end,
        }


class TemporalGraphDatasetManager:
    """Load graph data, create A^kX features, and build time-based window loaders."""

    def __init__(self, dataset_paths: Dict[str, Path]):
        self.dataset_paths = dataset_paths
        self.features: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.target_mask: Optional[torch.Tensor] = None
        self.graph = None
        self.adjacency: Optional[torch.Tensor] = None

    def _select_graph_variant(self, graph, graph_variant: Optional[str]):
        if not isinstance(graph, dict) or "graphs" not in graph:
            return graph

        graph_variants = graph.get("graphs") or {}
        if not graph_variants:
            return graph

        selected_name = graph_variant or graph.get("default_graph_name")
        if selected_name is None:
            return graph
        if selected_name not in graph_variants:
            available = ", ".join(sorted(graph_variants))
            raise ValueError(
                f"Unknown graph_variant '{selected_name}'. Available variants: {available}"
            )

        selected_graph = graph_variants[selected_name]
        resolved_graph = dict(graph)
        resolved_graph["edge_index"] = selected_graph["edge_index"]
        if "edge_attr" in selected_graph:
            resolved_graph["edge_attr"] = selected_graph["edge_attr"]
        resolved_graph["selected_graph_name"] = selected_name
        resolved_graph["selected_graph_description"] = selected_graph.get("description")
        return resolved_graph

    @staticmethod
    def _extract_edge_index(graph) -> torch.Tensor:
        if hasattr(graph, "edge_index"):
            return graph.edge_index
        if isinstance(graph, dict) and "edge_index" in graph:
            return graph["edge_index"]
        raise ValueError("Could not find edge_index in graph object")

    @staticmethod
    def _extract_num_nodes(graph, edge_index: torch.Tensor) -> int:
        if hasattr(graph, "num_nodes") and graph.num_nodes is not None:
            return int(graph.num_nodes)
        if isinstance(graph, dict) and graph.get("num_nodes") is not None:
            return int(graph["num_nodes"])
        return int(edge_index.max().item()) + 1

    @staticmethod
    def _ensure_tnd_layout(features: np.ndarray, num_nodes: int) -> np.ndarray:
        if features.ndim != 3:
            raise ValueError(f"Expected feature.npy with shape [T, N, D] or [N, T, D], got {features.shape}")

        if features.shape[1] == num_nodes:
            return features
        if features.shape[0] == num_nodes:
            return np.transpose(features, (1, 0, 2))
        raise ValueError(
            "Could not align features with graph nodes. "
            f"Feature shape {features.shape}, num_nodes {num_nodes}"
        )

    @staticmethod
    def _ensure_target_tn_layout(targets: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if targets.ndim != 2:
            raise ValueError(f"Expected targets with shape [T, N] or [N, T], got {tuple(targets.shape)}")

        if targets.shape[1] == num_nodes:
            return targets
        if targets.shape[0] == num_nodes:
            return targets.transpose(0, 1)
        raise ValueError(
            "Could not align targets with graph nodes. "
            f"Target shape {tuple(targets.shape)}, num_nodes {num_nodes}"
        )

    @staticmethod
    def _sanitize_numpy_features(features: np.ndarray, clip_value: float = 5000.0) -> np.ndarray:
        features = features.astype(np.float32, copy=True)
        features = np.nan_to_num(features, nan=0.0, posinf=clip_value, neginf=-clip_value)
        features = np.clip(features, -clip_value, clip_value)

        feature_mean = np.nanmean(features, axis=(0, 1), keepdims=True)
        feature_std = np.nanstd(features, axis=(0, 1), keepdims=True)

        feature_mean = np.nan_to_num(feature_mean, nan=0.0, posinf=0.0, neginf=0.0)
        feature_std = np.nan_to_num(feature_std, nan=1.0, posinf=1.0, neginf=1.0)
        feature_std[feature_std < 1e-6] = 1.0

        features = (features - feature_mean) / feature_std
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features.astype(np.float32, copy=False)

    @staticmethod
    def _build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        edge_index = edge_index.long()
        values = torch.ones(edge_index.shape[1], dtype=torch.float32)
        return torch.sparse_coo_tensor(
            edge_index,
            values,
            size=(num_nodes, num_nodes),
        ).coalesce()

    @staticmethod
    def _aggregate_features(features: torch.Tensor, adjacency: torch.Tensor, num_hops: int) -> torch.Tensor:
        if num_hops < 0:
            raise ValueError("num_hops must be non-negative")

        aggregated_parts = [features]
        current = features

        for _ in range(num_hops):
            hop_outputs = [torch.sparse.mm(adjacency, timestamp_features) for timestamp_features in current]
            current = torch.stack(hop_outputs, dim=0)
            aggregated_parts.append(current)

        return torch.cat(aggregated_parts, dim=-1)

    @staticmethod
    def _standardize_torch_features(features: torch.Tensor) -> torch.Tensor:
        features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        feature_mean = features.mean(dim=(0, 1), keepdim=True)
        feature_std = features.std(dim=(0, 1), keepdim=True)
        feature_std = torch.where(feature_std < 1e-6, torch.ones_like(feature_std), feature_std)
        normalized = (features - feature_mean) / feature_std
        return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)

    def load_data(
        self,
        num_hops: int = 0,
        graph_variant: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]:
        features_path = Path(self.dataset_paths["features"])
        graph_path = Path(self.dataset_paths["graph"])

        raw_features = np.load(features_path).astype(np.float32, copy=False)
        self.graph = torch.load(graph_path, weights_only=False)
        self.graph = self._select_graph_variant(self.graph, graph_variant)

        edge_index = self._extract_edge_index(self.graph)
        num_nodes = self._extract_num_nodes(self.graph, edge_index)
        raw_features = self._ensure_tnd_layout(raw_features, num_nodes)
        raw_features = self._sanitize_numpy_features(raw_features)

        targets = self.graph.get("y") if isinstance(self.graph, dict) else getattr(self.graph, "y", None)
        if targets is None:
            raise ValueError("graph.pt does not contain target tensor 'y'")
        targets = self._ensure_target_tn_layout(targets.float(), num_nodes)

        target_mask = self.graph.get("y_mask") if isinstance(self.graph, dict) else getattr(self.graph, "y_mask", None)
        if target_mask is None:
            target_mask = torch.ones_like(targets, dtype=torch.bool)
        else:
            target_mask = self._ensure_target_tn_layout(target_mask.bool(), num_nodes)

        base_features = torch.from_numpy(raw_features)
        self.adjacency = self._build_adjacency(edge_index, num_nodes)
        aggregated_features = self._aggregate_features(base_features, self.adjacency, num_hops)
        aggregated_features = self._standardize_torch_features(aggregated_features)

        self.features = aggregated_features.permute(1, 0, 2).contiguous()
        self.targets = targets.transpose(0, 1).contiguous()
        self.target_mask = target_mask.transpose(0, 1).contiguous()

        if verbose:
            print(f"Loaded features from {features_path}")
            print(f"  Base feature shape [T, N, D]: {tuple(base_features.shape)}")
            print(f"Loaded graph from {graph_path}")
            print(f"  Nodes: {num_nodes}")
            print(f"  Edges: {edge_index.shape[1]}")
            if isinstance(self.graph, dict) and self.graph.get("selected_graph_name") is not None:
                print(f"  Selected graph variant: {self.graph['selected_graph_name']}")
            print(f"Applied adjacency aggregation with k={num_hops}")
            print(f"  Node-major feature shape [N, T, D*(k+1)]: {tuple(self.features.shape)}")
            print(f"  Target shape [N, T]: {tuple(self.targets.shape)}")

        return self.features, self.targets, self.target_mask, self.graph

    def create_dataloaders(
        self,
        window_size: int,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        num_workers: int = 0,
        shuffle_train: bool = True,
    ) -> Dict[str, DataLoader]:
        if self.features is None or self.targets is None or self.target_mask is None:
            raise RuntimeError("Data not loaded. Call load_data() first.")

        dataloaders: Dict[str, DataLoader] = {}
        for split in ("train", "val", "test"):
            dataset = TemporalWindowDataset(
                features=self.features,
                targets=self.targets,
                target_mask=self.target_mask,
                split_type=split,
                window_size=window_size,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle_train and split == "train",
                num_workers=num_workers,
            )

        return dataloaders
