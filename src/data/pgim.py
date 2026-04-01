import os
import argparse

import dgl
import dgl.function as fn
import numpy as np
import torch
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


def load_graph_from_edges(edge_path, num_nodes):
    """Load a DGL graph from a text file with one `src,dst` edge per line."""
    edges = np.loadtxt(edge_path, dtype=np.int64, delimiter=",")
    if edges.ndim == 1:
        edges = edges.reshape(1, -1)
    return dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=num_nodes)


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


def collect_valid_positions(mask: np.ndarray, offset: int = 0):
    nodes, times = np.where(mask > 0)
    return list(zip(nodes.tolist(), (times + offset).tolist()))


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

        if start >= 0:
            window = self.hop_features[node_idx, start : time_idx + 1]
            y_window = self.targets[node_idx, start : time_idx + 1]
            mask_window = self.masks[node_idx, start : time_idx + 1]
        else:
            pad_len = -start
            pad = np.zeros((pad_len, self.num_hops, self.feat_dim), dtype=self.hop_features.dtype)
            cur = self.hop_features[node_idx, 0 : time_idx + 1]
            window = np.concatenate([pad, cur], axis=0)
            y_pad = np.zeros((pad_len,), dtype=self.targets.dtype)
            mask_pad = np.zeros((pad_len,), dtype=self.masks.dtype)
            y_cur = self.targets[node_idx, 0 : time_idx + 1]
            mask_cur = self.masks[node_idx, 0 : time_idx + 1]
            y_window = np.concatenate([y_pad, y_cur], axis=0)
            mask_window = np.concatenate([mask_pad, mask_cur], axis=0)

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


def get_dataloaders(
    root="dataset/ccr",
    feature="feature.npy",
    egde_file="graph_link_200m/links.txt",
    ts_test=25,
    k=2,
    batch_size=32,
    shift=1,
    no_feat_col=None,
    target_col="rent_per_sqft",
    mask_col="y_mask",
    feat_norm=True,
    window_size=12,
):
    del no_feat_col, target_col, mask_col

    feat_npy = os.path.join(root, feature)
    feat = np.load(feat_npy).astype(np.float32)

    if feat.ndim != 3:
        raise ValueError(f"Expected feature tensor with shape (N, T, D), got {feat.shape}")

    if shift > 0:
        X = feat[:, :, :-1]
        Y = feat[:, :, -2]
        mask = feat[:, :, -1].astype(bool)

        total_steps = Y.shape[1]
        if shift >= total_steps:
            raise ValueError(f"shift={shift} must be smaller than T={total_steps}")

        Y_shift = np.zeros_like(Y)
        mask_shift = np.zeros_like(mask, dtype=bool)

        Y_shift[:, :-shift] = Y[:, shift:]
        mask_shift[:, :-shift] = mask[:, shift:]

        Y = Y_shift
        mask = np.logical_and(mask, mask_shift)
    else:
        X = feat[:, :, 1:-2]
        Y = feat[:, :, -2]
        mask = feat[:, :, -1].astype(bool)

    num_nodes, total_steps, feat_dim = X.shape
    print(f"Condos: {num_nodes} | Timestamps: {total_steps} | feat_dim: {feat_dim}")

    if ts_test <= 0 or ts_test >= total_steps:
        raise ValueError(f"ts_test must be in [1, T-1], got ts_test={ts_test}, T={total_steps}")

    train_steps = total_steps - ts_test
    x_train = X[:, :train_steps, :]
    x_test = X[:, train_steps:, :]
    y_train = Y[:, :train_steps]
    y_test = Y[:, train_steps:]
    mask_train = mask[:, :train_steps]
    mask_test = mask[:, train_steps:]

    if feat_norm:
        train_min, train_max = compute_minmax_stats(x_train)
        x_train = normalize_to_minus1_1(x_train, train_min, train_max)
        x_test = normalize_to_minus1_1(x_test, train_min, train_max)

    x_full = np.concatenate([x_train, x_test], axis=1)
    y_full = np.concatenate([y_train, y_test], axis=1)

    edge_path = os.path.join(root, egde_file)
    print(f"\nLoading graph from {edge_path} ...")
    g = load_graph_from_edges(edge_path, num_nodes=num_nodes)

    print(f"num_edges (raw): {g.num_edges()}")

    g = dgl.remove_self_loop(g)
    print(f"num_edges (after removing self loop): {g.num_edges()}")

    g = dgl.add_reverse_edges(g)
    print(f"num_edges (after adding reverse edges): {g.num_edges()}")

    g = dgl.to_simple(g)
    print(f"num_edges (after converting to simple graph): {g.num_edges()}")

    g = dgl.add_self_loop(g)
    print(f"num_edges (after adding self loop): {g.num_edges()}")

    print(f"\nRunning sign_precompute (K={k}) across {total_steps} timestamps ...")
    full_hops = build_hop_tensor(x_full, g, k)
    train_hops = full_hops[:, :train_steps]

    train_positions = collect_valid_positions(mask_train)
    test_positions = collect_valid_positions(mask_test, offset=train_steps)

    train_ds = WindowedNodeDataset(train_hops, y_train, mask_train, train_positions, window_size)
    test_ds = WindowedNodeDataset(full_hops, y_full, mask, test_positions, window_size)

    print(
        f"Train samples: {len(train_ds)} | Test samples: {len(test_ds)} | "
        f"Per-sample input: (K+1={k + 1}, window={window_size}, feat_dim={feat_dim})"
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug PGIM dataloader construction")
    parser.add_argument("--root", type=str, default="dataset/ccr")
    parser.add_argument("--feature", type=str, default="feature.npy")
    parser.add_argument("--egde-file", type=str, default="graph_link_200m/links.txt")
    parser.add_argument("--ts-test", type=int, default=25)
    parser.add_argument("--k", type=int, default=2, help="Number of SIGN hops")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shift", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument(
        "--no-feat-norm",
        action="store_true",
        help="Disable train-set min-max normalization to [-1, 1]",
    )
    args = parser.parse_args()

    train_loader, test_loader = get_data_loader(
        root=args.root,
        feature=args.feature,
        egde_file=args.egde_file,
        ts_test=args.ts_test,
        k=args.k,
        batch_size=args.batch_size,
        shift=args.shift,
        feat_norm=not args.no_feat_norm,
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