import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.pgim import get_dataloaders
from src.models.gnn_regressor import GNNRegressor
from src.utils.loss import MaskedMSELoss


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return device


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def build_model(checkpoint_args: dict, sample_inputs: torch.Tensor, device: torch.device) -> GNNRegressor:
    _, num_hops, _, feat_dim = sample_inputs.shape
    model = GNNRegressor(
        num_hops=num_hops,
        feat_dim=feat_dim,
        hidden_dim=checkpoint_args["hidden_dim"],
        out_dim=1,
        mlp_layers=checkpoint_args["mlp_layers"],
        num_transformer_layers=checkpoint_args["num_layers"],
        heads=checkpoint_args["num_heads"],
        dropout=checkpoint_args["dropout"],
    ).to(device)
    return model


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    predict_last=False,
    accuracy_tolerance=0.10,
):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, targets, y_mask, _, _ in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            y_mask = y_mask.to(device).float()

            outputs = model(inputs)
            if predict_last:
                outputs = outputs[:, -1:, :]
                targets = targets[:, -1:, :]
                y_mask = y_mask[:, -1:, :]

            loss = criterion(outputs, targets, y_mask)
            total_loss += loss.item()

            valid_mask = y_mask > 0
            valid_outputs = outputs[valid_mask]
            valid_targets = targets[valid_mask]

            if valid_outputs.numel() > 0:
                all_preds.append(valid_outputs.cpu().numpy().reshape(-1, 1))
                all_targets.append(valid_targets.cpu().numpy().reshape(-1, 1))

    avg_loss = total_loss / len(dataloader)

    if len(all_preds) == 0:
        return {
            "loss": avg_loss,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "r2": np.nan,
            "acc_within_tol": np.nan,
            "num_samples": 0,
        }

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mse)

    nonzero_mask = np.abs(all_targets) > 1e-8
    if np.any(nonzero_mask):
        ape = np.abs((all_targets[nonzero_mask] - all_preds[nonzero_mask]) / all_targets[nonzero_mask])
        mape = np.mean(ape) * 100.0
        acc_within_tol = np.mean(ape <= accuracy_tolerance) * 100.0
    else:
        mape = np.nan
        acc_within_tol = np.nan

    if len(all_targets) > 1 and np.unique(all_targets).size > 1:
        r2 = r2_score(all_targets, all_preds)
    else:
        r2 = np.nan

    return {
        "loss": avg_loss,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "acc_within_tol": acc_within_tol,
        "num_samples": int(all_targets.shape[0]),
    }


def resolve_checkpoints(checkpoint_path: Optional[Path], checkpoint_dir: Optional[Path]):
    if checkpoint_path is not None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return [checkpoint_path]

    if checkpoint_dir is None:
        checkpoint_dir = Path("checkpoints")

    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints found in {checkpoint_dir}")
    return checkpoints


def build_test_loader(train_args: dict, eval_window_size: int):
    train_loader, test_loader = get_dataloaders(
        root=str(train_args["root"]),
        feature=train_args["feature"],
        egde_file=train_args["egde_file"],
        ts_test=train_args["ts_test"],
        k=train_args["num_hops"],
        batch_size=train_args["batch_size"],
        shift=train_args["shift"],
        target_mask_mode=train_args["target_mask_mode"],
        feat_norm=train_args["feat_norm"],
        window_size=eval_window_size,
    )
    return train_loader, test_loader


def format_metric(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "nan"
    return f"{value:.6f}"


def evaluate_checkpoint(
    checkpoint_path: Path,
    eval_window_size: int,
    device_arg: str,
    accuracy_tolerance: float,
):
    device = resolve_device(device_arg)
    checkpoint = load_checkpoint(checkpoint_path, device)
    train_args = checkpoint["args"]

    train_loader, test_loader = build_test_loader(train_args, eval_window_size)
    sample_inputs, _, _, _, _ = next(iter(train_loader))

    model = build_model(train_args, sample_inputs, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = MaskedMSELoss().to(device)
    metrics = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
        predict_last=train_args.get("predict_last", False),
        accuracy_tolerance=accuracy_tolerance,
    )

    return {
        "checkpoint": str(checkpoint_path),
        "trained_window_size": train_args.get("window_size"),
        "eval_window_size": eval_window_size,
        "shift": train_args.get("shift"),
        "num_hops": train_args.get("num_hops"),
        "epoch": checkpoint.get("epoch"),
        **metrics,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Load trained checkpoint(s) and run inference on the PGIM test set with a different evaluation window size.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Path to a single checkpoint to evaluate.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory of checkpoints to evaluate when --checkpoint-path is not provided.")
    parser.add_argument("--eval-window-size", type=int, default=6, help="Window size used for test-set inference. Default is 6 months.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument(
        "--accuracy-tolerance",
        type=float,
        default=0.10,
        help="Tolerance used for Acc@tol. Example: 0.10 means prediction is counted correct when relative error <= 10%%.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.eval_window_size <= 0:
        raise ValueError(f"eval-window-size must be positive, got {args.eval_window_size}")
    if args.accuracy_tolerance < 0:
        raise ValueError(f"accuracy-tolerance must be non-negative, got {args.accuracy_tolerance}")

    checkpoints = resolve_checkpoints(args.checkpoint_path, args.checkpoint_dir)
    results = []

    print(f"Evaluating {len(checkpoints)} checkpoint(s) with eval window size = {args.eval_window_size}")
    print(f"Acc@tol uses tolerance = {args.accuracy_tolerance * 100:.2f}% relative error\n")

    for checkpoint_path in checkpoints:
        result = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            eval_window_size=args.eval_window_size,
            device_arg=args.device,
            accuracy_tolerance=args.accuracy_tolerance,
        )
        results.append(result)

        print(f"Checkpoint: {result['checkpoint']}")
        print(
            "  trained_window={trained} | eval_window={eval_window} | shift={shift} | hops={hops} | epoch={epoch}".format(
                trained=result["trained_window_size"],
                eval_window=result["eval_window_size"],
                shift=result["shift"],
                hops=result["num_hops"],
                epoch=result["epoch"],
            )
        )
        print(
            "  loss={loss} | RMSE={rmse} | MAE={mae} | MAPE(%)={mape} | R2={r2} | Acc@tol(%)={acc} | N={n}".format(
                loss=format_metric(result["loss"]),
                rmse=format_metric(result["rmse"]),
                mae=format_metric(result["mae"]),
                mape=format_metric(result["mape"]),
                r2=format_metric(result["r2"]),
                acc=format_metric(result["acc_within_tol"]),
                n=format_metric(result["num_samples"]),
            )
        )
        print()

    if len(results) > 1:
        ranked = sorted(results, key=lambda item: (np.inf if np.isnan(item["rmse"]) else item["rmse"]))
        best = ranked[0]
        print("Best checkpoint by RMSE:")
        print(
            "  {path} | RMSE={rmse} | MAE={mae} | MAPE(%)={mape} | R2={r2} | Acc@tol(%)={acc}".format(
                path=best["checkpoint"],
                rmse=format_metric(best["rmse"]),
                mae=format_metric(best["mae"]),
                mape=format_metric(best["mape"]),
                r2=format_metric(best["r2"]),
                acc=format_metric(best["acc_within_tol"]),
            )
        )


if __name__ == "__main__":
    main()
