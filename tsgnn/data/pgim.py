import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

def build_row_normalized_adj(graph_data, add_self_loop=True):
    edge_index = graph_data["edge_index"]   # [2, E]
    num_nodes = len(graph_data["node_names"])

    # Step 1: build adjacency
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    # make undirected
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0

    # Step 2: add self-loops (recommended)
    if add_self_loop:
        adj.fill_diagonal_(1.0)

    # Step 3: compute degree
    deg = adj.sum(dim=1)  # [N]

    # Step 4: row normalization: D^{-1} A
    deg_inv = deg.clamp(min=1e-8).reciprocal()  # avoid divide by 0
    adj_norm = deg_inv.unsqueeze(1) * adj       # [N, N]

    return adj_norm

def compute_minmax_stats(features: np.ndarray):
    """
    Compute per-feature min/max stats for arrays of shape [N, T, D].
    """
    min_vals = features.min(axis=(0, 1), keepdims=True)  # [1, 1, D]
    max_vals = features.max(axis=(0, 1), keepdims=True)  # [1, 1, D]
    return min_vals, max_vals


def normalize_to_minus1_1(features: np.ndarray, min_vals=None, max_vals=None):
    """
    Normalize features of shape [N, T, D] to [-1, 1] per feature dimension.

    If min/max stats are not provided, they are computed from `features`.
    """
    if min_vals is None or max_vals is None:
        min_vals, max_vals = compute_minmax_stats(features)

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0
    normalized = 2 * (features - min_vals) / range_vals - 1

    return normalized, min_vals.squeeze(), max_vals.squeeze()

class PGIMDataset(Dataset):
    """
    Dataset for data with shape:
        features: [N, T, D]
        y:        [N, T]
        y_mask:   [N, T]

    Each sample corresponds to one (node, timestep).
    The dataset returns a temporal window for that node:
        x_window:    [window_size, D]
        y_window:    [window_size]
        mask_window: [window_size]

    If timestep t is smaller than window_size, the window is left-padded.
    """

    def __init__(self, data_paths, window_size=50, split="train", test_pad=True, num_hops=7, feat_norm=True):
        self.train_path = data_paths["feat_train"]
        self.test_path = data_paths["feat_test"]
        self.graph_path = data_paths["graph"]
        self.feat_norm = feat_norm

        self.window_size = window_size
        self.num_hops = num_hops
        self.test_pad = test_pad
        self.split = split

        if num_hops > 0:
            self._add_hop_features()

        feat_train = np.load(self.train_path)

        if split == "train":
            data = feat_train
        else:
            data = np.load(self.test_path)

        train_features = feat_train[:, :, 1:-2]
        train_min = train_max = None
        if self.feat_norm:
            train_min, train_max = compute_minmax_stats(train_features)

        # data: [N, T, D_total]
        self.features = data[:, :, 1:-2]   # [N, T, D]
        self.y = data[:, :, -2]           # [N, T]
        self.y_mask = data[:, :, -1]      # [N, T]

        if self.feat_norm:
            self.features, _, _ = normalize_to_minus1_1(self.features, train_min, train_max)


        self.N, self.T, self.D = self.features.shape

        # For test-time padding from the end of training set
        if split != "train" and self.test_pad:
            self.y_last_window = feat_train[:, -self.window_size:, -2]       # [N, w]
            self.y_last_window_mask = np.zeros_like(
                feat_train[:, -self.window_size:, -1]
            )  # [N, w]
            feat_train = train_features
            if self.feat_norm:
                feat_train, _, _ = normalize_to_minus1_1(feat_train, train_min, train_max)
            
            self.feat_last_window = feat_train[:, -self.window_size:, :]   # [N, w, D]
            

        # print("features:", self.features.shape)
        # print("y:", self.y.shape)
        # print("y_mask:", self.y_mask.shape)

    def _add_hop_features(self):
        train_path = Path(self.train_path)
        test_path = Path(self.test_path)

        train_feat_hop = train_path.with_name(f"{train_path.stem}_hop_{self.num_hops}_{self.feat_norm}.npy")
        test_feat_hop = test_path.with_name(f"{test_path.stem}_hop_{self.num_hops}_{self.feat_norm}.npy")

        if os.path.exists(train_feat_hop) and os.path.exists(test_feat_hop):
            self.train_path = train_feat_hop
            self.test_path = test_feat_hop
            return

        feat_train = np.load(train_path)   # [N, T, D_total]
        feat_test = np.load(test_path)     # [N, T, D_total]

        x_train = feat_train[:, :, 1:-2]    # [N, T, D]
        yt_train = feat_train[:, :, -2:]   # [N, T, 2]

        x_test = feat_test[:, :, 1:-2]      # [N, T, D]
        yt_test = feat_test[:, :, -2:]     # [N, T, 2]

        graph_data = torch.load(self.graph_path)
        adj_norm = build_row_normalized_adj(graph_data, add_self_loop=True)  # [N, N]

        x_train_t = torch.from_numpy(x_train).float()
        x_test_t = torch.from_numpy(x_test).float()

        # collect [X, AX, A^2X, ..., A^kX]
        train_hops = [x_train_t]
        test_hops = [x_test_t]

        cur_train = x_train_t
        cur_test = x_test_t

        for _ in range(self.num_hops):
            cur_train = torch.einsum("nm,mtd->ntd", adj_norm, cur_train)
            cur_test = torch.einsum("nm,mtd->ntd", adj_norm, cur_test)

            train_hops.append(cur_train)
            test_hops.append(cur_test)

        # concatenate along feature dimension
        x_train_hop = torch.cat(train_hops, dim=-1)   # [N, T, (k+1)D]
        x_test_hop = torch.cat(test_hops, dim=-1)     # [N, T, (k+1)D]

        feat_train_hop = np.concatenate(
            [x_train_hop.numpy(), yt_train],
            axis=-1
        )  # [N, T, (k+1)D + 2]

        feat_test_hop = np.concatenate(
            [x_test_hop.numpy(), yt_test],
            axis=-1
        )  # [N, T, (k+1)D + 2]

        np.save(train_feat_hop, feat_train_hop)
        np.save(test_feat_hop, feat_test_hop)

        self.train_path = train_feat_hop
        self.test_path = test_feat_hop

    def __len__(self):
        # one sample for each (node, timestep)
        return self.N * self.T

    def _idx_to_nt(self, idx):
        n = idx // self.T
        t = idx % self.T
        return n, t

    def __getitem__(self, idx):
        n, t = self._idx_to_nt(idx)

        # We want a window ending at timestep t (inclusive)
        # so the desired range is [t - window_size + 1, ..., t]
        start = t - self.window_size + 1

        if start >= 0:
            x_window = self.features[n, start:t + 1, :]   # [window_size, D]
            y_window = self.y[n, start:t + 1]             # [window_size]
            mask_window = self.y_mask[n, start:t + 1]     # [window_size]
        else:
            pad_len = -start

            # available part from current sequence
            x_cur = self.features[n, 0:t + 1, :]          # [t+1, D]
            y_cur = self.y[n, 0:t + 1]                    # [t+1]
            mask_cur = self.y_mask[n, 0:t + 1]            # [t+1]

            if self.split != "train" and self.test_pad:
                # Pad from the last training window of the same node
                x_pad = self.feat_last_window[n, -pad_len:, :]   # [pad_len, D]
                y_pad = self.y_last_window[n, -pad_len:]         # [pad_len]
                mask_pad = self.y_last_window_mask[n, -pad_len:] # [pad_len]
            else:
                # Zero padding
                x_pad = np.zeros((pad_len, self.D), dtype=self.features.dtype)
                y_pad = np.zeros((pad_len,), dtype=self.y.dtype)
                mask_pad = np.zeros((pad_len,), dtype=self.y_mask.dtype)

            x_window = np.concatenate([x_pad, x_cur], axis=0)
            y_window = np.concatenate([y_pad, y_cur], axis=0)
            mask_window = np.concatenate([mask_pad, mask_cur], axis=0)

        return (
            torch.from_numpy(x_window).float(),
            torch.from_numpy(y_window).float(),
            torch.from_numpy(mask_window).float(),
            torch.tensor(n, dtype=torch.long),
            torch.tensor(t, dtype=torch.long),
        )

def get_pgim_dataloader(data_paths, feat_norm=True, window_size=12, batch_size=32, num_hops=7, num_workers=0):
    """
    Utility to get a DataLoader for the PGIM dataset.
    """
    dataset_train = PGIMDataset(data_paths, window_size, 'train', num_hops=num_hops, feat_norm=feat_norm)
    dataset_test = PGIMDataset(data_paths, window_size, 'test', num_hops=num_hops, feat_norm=feat_norm)
    train_dataloader = DataLoader(
        dataset_train, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )
    test_dataloader = DataLoader(
        dataset_test, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    # Example usage
    data_paths = {
        'feat_train': 'dataset/feature_ccr_v3.1_within_250m_train.npy',
        'feat_test': 'dataset/feature_ccr_v3.1_within_250m_test.npy',
        'graph': 'dataset/graph_ccr_v3.1_within_250m.pt'
    }
    train_loader, test_loader = get_pgim_dataloader(data_paths, window_size=12, batch_size=32, num_hops=7, num_workers=0)
    print(len(train_loader.dataset), len(test_loader.dataset))
    print(train_loader.dataset.features.shape[-1])
    for x, y, mask, n, t in train_loader:
        print(f"Batch features shape: {x.shape}, Batch targets shape: {y.shape}")
        break
