import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import TemporalTransformerRegressor
from training_framework import TemporalGraphDatasetManager


def truncate_float(value: float, decimals: int = 4) -> float:
    factor = 10 ** decimals
    return math.trunc(value * factor) / factor


def print_final_metrics(mse: float, mae: float, rmse: float, r2: float, nrmse: float) -> None:
    print("\nFinal Evaluation Metrics (on Test Set):")
    print(f"  MSE:   {truncate_float(mse, 6):.6f}")
    print(f"  MAE:   {truncate_float(mae, 6):.6f}")
    print(f"  RMSE:  {truncate_float(rmse, 6):.6f}")
    print(f"  R2:    {truncate_float(r2, 6):.6f}")
    print(f"  NRMSE: {truncate_float(nrmse, 6):.6f}")


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not mask.any():
        return pred.new_tensor(0.0)
    return F.mse_loss(pred[mask], target[mask])


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def run_epoch(model, dataloader, device: torch.device, optimizer=None, desc: str = "Epoch") -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_batches = 0

    progress = tqdm(dataloader, desc=desc, leave=False)
    for batch in progress:
        batch = move_batch_to_device(batch, device)
        x = batch["x"]
        y = batch["y"]
        mask = batch["mask"].bool()

        if not mask.any():
            continue

        if is_training:
            optimizer.zero_grad()

        pred = model(x)
        loss = masked_mse_loss(pred, y, mask)
        if not torch.isfinite(loss):
            continue

        if is_training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        progress.set_postfix(mse=f"{truncate_float(loss.item(), 4):.4f}")

    if total_batches == 0:
        return 0.0
    return total_loss / total_batches


def evaluate_on_dataloader(
    model: TemporalTransformerRegressor,
    dataloader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Test Eval", leave=False):
            batch = move_batch_to_device(batch, device)
            x = batch["x"]
            y = batch["y"]
            mask = batch["mask"].bool()

            pred = model(x)
            if mask.any():
                predictions.append(pred[mask].cpu())
                targets.append(y[mask].cpu())

    if not predictions:
        raise ValueError("No valid target values found in the evaluation dataloader.")

    pred_values = torch.cat(predictions).numpy()
    target_values = torch.cat(targets).numpy()

    errors = pred_values - target_values
    mse = float(np.mean(errors ** 2))
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(mse))

    target_mean = float(np.mean(target_values))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((target_values - target_mean) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    target_range = float(np.max(target_values) - np.min(target_values))
    nrmse = rmse / target_range if target_range > 0.0 else 0.0

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "nrmse": nrmse,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build node-wise temporal windows and a transformer regressor.",
    )
    parser.add_argument("--features-path", type=Path, default=Path("dataset/feature.npy"))
    parser.add_argument("--graph-path", type=Path, default=Path("dataset/graph.pt"))
    parser.add_argument("--graph-variant", type=str, default="distance_le_0_5km")
    parser.add_argument("--num-hops", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")

    dataset_paths = {
        "features": args.features_path,
        "graph": args.graph_path,
    }
    dataset_manager = TemporalGraphDatasetManager(dataset_paths)
    features, targets, target_mask, graph = dataset_manager.load_data(
        num_hops=args.num_hops,
        graph_variant=args.graph_variant,
    )
    dataloaders = dataset_manager.create_dataloaders(
        window_size=args.window_size,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
    )

    model = TemporalTransformerRegressor(
        input_dim=features.shape[-1],
        hidden_dim=args.hidden_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        window_size=args.window_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\nFramework ready.")
    print(f"Device: {device}")
    print(f"Node-major features [N, T, D]: {tuple(features.shape)}")
    print(f"Targets [N, T]: {tuple(targets.shape)}")
    print(f"Target mask [N, T]: {tuple(target_mask.shape)}")

    if hasattr(graph, "num_nodes"):
        print(f"Graph nodes: {graph.num_nodes}")
    elif isinstance(graph, dict) and "num_nodes" in graph:
        print(f"Graph nodes: {graph['num_nodes']}")

    for split_name, dataloader in dataloaders.items():
        print(f"{split_name} batches: {len(dataloader)}")

    print(f"Training epochs: {args.epochs}")
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            dataloaders["train"],
            device=device,
            optimizer=optimizer,
            desc=f"Train {epoch:03d}",
        )
        val_loss = run_epoch(
            model,
            dataloaders["val"],
            device=device,
            optimizer=None,
            desc=f"Val   {epoch:03d}",
        )
        print(
            f"Epoch {epoch:03d} | "
            f"train_mse: {truncate_float(train_loss, 6):.6f} | "
            f"val_mse: {truncate_float(val_loss, 6):.6f}"
        )

    first_batch = next(iter(dataloaders["train"]))
    first_batch = move_batch_to_device(first_batch, device)
    x = first_batch["x"]
    y = first_batch["y"]
    mask = first_batch["mask"]

    with torch.no_grad():
        pred = model(x)

    print(f"Train batch x shape [B, t, D]: {tuple(x.shape)}")
    print(f"Train batch y shape [B, t]: {tuple(y.shape)}")
    print(f"Train batch mask shape [B, t]: {tuple(mask.shape)}")
    print(f"Model prediction shape [B, t]: {tuple(pred.shape)}")

    metrics = evaluate_on_dataloader(model, dataloaders["test"], device=device)
    print_final_metrics(
        mse=metrics["mse"],
        mae=metrics["mae"],
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        nrmse=metrics["nrmse"],
    )


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=3 python main.py
