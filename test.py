import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

from src.data.pgim import WindowedNodeDataset, get_dataloaders
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
    shift=0,
    test_start_idx=0,
    project_names=None,
    timestamps=None,
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


def collect_prediction_rows(
    model,
    dataloader,
    device,
    shift,
    test_start_idx,
    project_names,
    timestamps,
):
    model.eval()
    prediction_rows = []

    with torch.no_grad():
        for inputs, targets, y_mask, node_indices, time_indices in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            y_mask = y_mask.to(device).float()

            outputs = model(inputs)
            outputs = outputs[:, -1:, :]
            targets = targets[:, -1:, :]
            y_mask = y_mask[:, -1:, :]

            outputs_cpu = outputs.detach().cpu()
            targets_cpu = targets.detach().cpu()
            valid_mask_cpu = (y_mask > 0).detach().cpu()
            node_indices_cpu = node_indices.detach().cpu()
            time_indices_cpu = time_indices.detach().cpu()
            window_size = inputs.shape[2]

            for batch_idx in range(outputs_cpu.shape[0]):
                end_time_idx = int(time_indices_cpu[batch_idx].item())
                window_start_idx = end_time_idx - window_size + 1
                target_time_idx = end_time_idx + shift
                if 0 <= target_time_idx < len(timestamps):
                    target_timestamp = timestamps[int(target_time_idx)]
                else:
                    target_timestamp = add_months(timestamps[end_time_idx], shift)
                has_ground_truth = bool(valid_mask_cpu[batch_idx, 0, 0].item())
                split = "train" if end_time_idx < test_start_idx else "test"

                prediction_rows.append({
                    "split": split,
                    "project_name": project_names[int(node_indices_cpu[batch_idx].item())],
                    "window_start_timestamp": timestamps[window_start_idx],
                    "window_end_timestamp": timestamps[end_time_idx],
                    "target_timestamp": target_timestamp,
                    "prediction": float(outputs_cpu[batch_idx, 0, 0].item()),
                    "ground_truth": float(targets_cpu[batch_idx, 0, 0].item()) if has_ground_truth else "",
                })

    return prediction_rows


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


def build_prediction_loader(test_loader, batch_size: int):
    base_dataset = test_loader.dataset
    min_time_idx = base_dataset.window_size - 1
    all_positions = [
        (node_idx, time_idx)
        for time_idx in range(min_time_idx, base_dataset.total_steps)
        for node_idx in range(base_dataset.num_nodes)
    ]
    prediction_dataset = WindowedNodeDataset(
        hop_features=base_dataset.hop_features,
        targets=base_dataset.targets,
        masks=base_dataset.masks,
        valid_positions=all_positions,
        window_size=base_dataset.window_size,
    )
    return DataLoader(prediction_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def format_metric(value):
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return "nan"
    return f"{value:.6f}"


def resolve_metadata_dir(root: str, feature: str) -> Path:
    feature_stem = Path(feature).stem
    if not feature_stem.startswith("feature"):
        return Path(root) / "feature_metadata"
    suffix = feature_stem[len("feature") :]
    return Path(root) / f"feature_metadata{suffix}"


def load_dataset_labels(root: str, feature: str):
    metadata_dir = resolve_metadata_dir(root, feature)
    project_names_path = metadata_dir / "project_names.txt"
    timestamps_path = metadata_dir / "timestamps.txt"

    if not project_names_path.is_file():
        raise FileNotFoundError(f"Project names metadata not found: {project_names_path}")
    if not timestamps_path.is_file():
        raise FileNotFoundError(f"Timestamps metadata not found: {timestamps_path}")

    project_names = project_names_path.read_text(encoding="utf-8").splitlines()
    timestamps = timestamps_path.read_text(encoding="utf-8").splitlines()
    return project_names, timestamps


def add_months(timestamp_str: str, months: int) -> str:
    base = datetime.strptime(timestamp_str, "%Y-%m-%d")
    total_month = (base.year * 12 + (base.month - 1)) + months
    year = total_month // 12
    month = total_month % 12 + 1
    return f"{year:04d}-{month:02d}-01"


def save_predictions(prediction_rows, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deduped_with_target = {}
    deduped_without_target = {}
    for row in prediction_rows:
        if row["target_timestamp"]:
            key = (row["project_name"], row["target_timestamp"])
            existing = deduped_with_target.get(key)
            if existing is None or row["window_end_timestamp"] > existing["window_end_timestamp"]:
                deduped_with_target[key] = row
        else:
            key = (row["project_name"], row["window_end_timestamp"])
            existing = deduped_without_target.get(key)
            if existing is None:
                deduped_without_target[key] = row

    rows_to_write = sorted(
        [*deduped_with_target.values(), *deduped_without_target.values()],
        key=lambda row: (
            row["project_name"],
            row["target_timestamp"] if row["target_timestamp"] else "9999-99-99",
            row["window_start_timestamp"],
            row["window_end_timestamp"],
        ),
    )

    fieldnames = [
        "split",
        "project_name",
        "window_start_timestamp",
        "window_end_timestamp",
        "target_timestamp",
        "prediction",
        "ground_truth",
    ]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_to_write)
    return rows_to_write


def evaluate_checkpoint(
    checkpoint_path: Path,
    eval_window_size: int,
    device_arg: str,
    accuracy_tolerance: float,
    output_dir: Optional[Path],
):
    device = resolve_device(device_arg)
    checkpoint = load_checkpoint(checkpoint_path, device)
    train_args = checkpoint["args"]
    project_names, timestamps = load_dataset_labels(
        root=str(train_args["root"]),
        feature=train_args["feature"],
    )

    train_loader, test_loader = build_test_loader(train_args, eval_window_size)
    prediction_loader = build_prediction_loader(test_loader, batch_size=int(train_args["batch_size"]))
    sample_inputs, _, _, _, _ = next(iter(train_loader))
    total_steps = test_loader.dataset.total_steps
    test_start_idx = total_steps - int(train_args["ts_test"])

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
        shift=int(train_args.get("shift", 0) or 0),
        test_start_idx=test_start_idx,
        project_names=project_names,
        timestamps=timestamps,
    )
    prediction_rows = collect_prediction_rows(
        model=model,
        dataloader=prediction_loader,
        device=device,
        shift=int(train_args.get("shift", 0) or 0),
        test_start_idx=test_start_idx,
        project_names=project_names,
        timestamps=timestamps,
    )

    predictions_path = None
    if output_dir is not None:
        predictions_path = output_dir / f"{checkpoint_path.stem}_predictions.csv"
        saved_rows = save_predictions(prediction_rows, predictions_path)
    else:
        saved_rows = prediction_rows

    missing_ground_truth_rows = [
        row for row in saved_rows if row["target_timestamp"] and row["ground_truth"] == ""
    ]

    return {
        "checkpoint": str(checkpoint_path),
        "trained_window_size": train_args.get("window_size"),
        "eval_window_size": eval_window_size,
        "shift": train_args.get("shift"),
        "num_hops": train_args.get("num_hops"),
        "epoch": checkpoint.get("epoch"),
        "predictions_path": None if predictions_path is None else str(predictions_path),
        "missing_ground_truth_rows": missing_ground_truth_rows,
        "num_prediction_rows": len(saved_rows),
        **metrics,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Load trained checkpoint(s) and run inference on the PGIM test set with a different evaluation window size.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Path to a single checkpoint to evaluate.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"), help="Directory of checkpoints to evaluate when --checkpoint-path is not provided.")
    parser.add_argument("--eval-window-size", type=int, default=12, help="Window size used for test-set inference. Default is 6 months.")
    parser.add_argument("--device", type=str, default="auto", help="Device to run inference on: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("predictions"),
        help="Directory where per-sample test-set predictions are saved as CSV files.",
    )
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
            output_dir=args.output_dir,
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
        if result["predictions_path"] is not None:
            print(f"  saved_predictions={result['predictions_path']}")
        if result["missing_ground_truth_rows"]:
            print("  target_timestamp rows without ground_truth:")
            for row in result["missing_ground_truth_rows"]:
                print(
                    "    project={project} | window=[{start} -> {end}] | target={target} | prediction={prediction:.6f}".format(
                        project=row["project_name"],
                        start=row["window_start_timestamp"],
                        end=row["window_end_timestamp"],
                        target=row["target_timestamp"],
                        prediction=row["prediction"],
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

# python test.py --checkpoint-path /home/cecelia/project/PGIM-Graph/checkpoints/pgim-mask-trans-shift1-exp/hop3_lr0.0001_wd0.01_head4_layer2_ts12_dist300_20260408_121456_best.pt

# python test.py --checkpoint-path /home/cecelia/project/PGIM-Graph/checkpoints/pgim-mask-trans-shift12-exp/hop3_lr0.0001_wd0.01_head4_layer2_ts12_dist300_20260408_112238_best.pt
