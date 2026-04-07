import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import wandb
from datetime import datetime

from src.data.pgim import get_dataloaders
from src.models.gnn_regressor import GNNRegressor
from src.utils.loss import MaskedMSELoss
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "t", "yes", "y", "1"}:
        return True
    if value in {"false", "f", "no", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got '{value}'.")


def train_one_epoch(model, dataloader, criterion, optimizer, device, predict_last=False):
    model.train()
    total_loss = 0.0
    for inputs, targets, y_mask, _, _ in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        y_mask = y_mask.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        if predict_last:
            outputs = outputs[:, -1:, :]
            targets = targets[:, -1:, :]
            y_mask = y_mask[:, -1:, :]
        loss = criterion(outputs, targets, y_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def compute_regression_metrics(preds: np.ndarray, targets: np.ndarray):
    if preds.size == 0 or targets.size == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "r2": np.nan,
            "num_samples": 0,
        }

    preds = preds.reshape(-1, 1)
    targets = targets.reshape(-1, 1)

    mse = mean_squared_error(targets, preds)
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mse)
    nonzero_mask = np.abs(targets) > 1e-8
    if np.any(nonzero_mask):
        mape = np.mean(
            np.abs((targets[nonzero_mask] - preds[nonzero_mask]) / targets[nonzero_mask])
        ) * 100.0
    else:
        mape = np.nan

    if len(targets) > 1 and np.unique(targets).size > 1:
        r2 = r2_score(targets, preds)
    else:
        r2 = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "num_samples": int(targets.shape[0]),
    }


def evaluate(model, dataloader, criterion, device, predict_last=False, selected_indices=None):
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    index_buffers = {}

    if selected_indices is None:
        selected_indices = []
    selected_indices = sorted(set(selected_indices))

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
            outputs_cpu = outputs.detach().cpu()
            targets_cpu = targets.detach().cpu()
            valid_mask_cpu = valid_mask.detach().cpu()

            valid_outputs = outputs_cpu[valid_mask_cpu]
            valid_targets = targets_cpu[valid_mask_cpu]

            if valid_outputs.numel() > 0:
                all_preds.append(valid_outputs.numpy().reshape(-1, 1))
                all_targets.append(valid_targets.numpy().reshape(-1, 1))

            for idx in selected_indices:
                if idx >= outputs_cpu.shape[1]:
                    continue
                idx_mask = valid_mask_cpu[:, idx, :]
                idx_outputs = outputs_cpu[:, idx, :][idx_mask]
                idx_targets = targets_cpu[:, idx, :][idx_mask]
                if idx_outputs.numel() == 0:
                    continue
                bucket = index_buffers.setdefault(idx, {"preds": [], "targets": []})
                bucket["preds"].append(idx_outputs.numpy().reshape(-1, 1))
                bucket["targets"].append(idx_targets.numpy().reshape(-1, 1))

    avg_loss = total_loss / len(dataloader)

    if len(all_preds) == 0:
        overall_metrics = {
            "loss": avg_loss,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "r2": np.nan,
            "num_samples": 0,
        }
    else:
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        overall_metrics = {
            "loss": avg_loss,
            **compute_regression_metrics(all_preds, all_targets),
        }

    index_metrics = {}
    for idx in selected_indices:
        bucket = index_buffers.get(idx)
        if bucket is None or len(bucket["preds"]) == 0:
            index_metrics[idx] = compute_regression_metrics(
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )
            continue
        idx_preds = np.vstack(bucket["preds"])
        idx_targets = np.vstack(bucket["targets"])
        index_metrics[idx] = compute_regression_metrics(idx_preds, idx_targets)

    return overall_metrics, index_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a GNNRegressor on the PGIM dataset with SIGN-style precomputed hop features.",
    )
    parser.add_argument("--root", type=Path, default=Path("dataset/ccr"))
    parser.add_argument("--feature", type=str, default="feature.npy")
    parser.add_argument("--egde-file", type=str, default="graph_link_200m/links.txt")
    parser.add_argument("--ts-test", type=int, default=25)
    parser.add_argument("--shift", type=int, default=1)
    parser.add_argument(
        "--target-mask-mode",
        type=str,
        choices=("observed_only", "allow_interpolated", "train_allow_interpolated"),
        default="observed_only",
        help="Whether to use only observed targets, allow interpolated targets for both train/test, or allow them only in train.",
    )
    parser.add_argument("--feat-norm", type=str2bool, default=True, help="Whether to apply train-set min-max scaling to [-1, 1].")
    parser.add_argument("--num-hops", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument(
        "--predict-last",
        type=str2bool,
        default=False,
        help="Whether to train/evaluate only on the last timestep in each input window.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--mlp-layers", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2, help="Number of causal transformer layers.")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--tracked-indices",
        type=int,
        nargs="+",
        default=[0, 6, 11],
        help="Temporal indices within each evaluation window for which to report separate test metrics.",
    )
    parser.add_argument("--wandb-project", type=str, default="pgim-gnn-regressor-exp")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.hidden_dim % args.num_heads != 0:
        raise ValueError(
            f"hidden_dim ({args.hidden_dim}) must be divisible by num_heads ({args.num_heads})."
        )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"hop{args.num_hops}_"
        f"lr{args.learning_rate}_"
        f"wd{args.weight_decay}_"
        f"head{args.num_heads}_"
        f"layer{args.num_layers}_"
        f"ts{args.window_size}_"
        f"{timestamp}"
    )
    checkpoint_dir = args.checkpoint_dir / args.wandb_project
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / f"{run_name}_best.pt"

    wandb.init(
        project=args.wandb_project,
        entity="ccliasub-national-university-of-singapore",
        name=run_name,
        config={
            "num_hops": args.num_hops,
            "learning_rate": args.learning_rate,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "mlp_layers": args.mlp_layers,
            "window_size": args.window_size,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "best_model_path": str(best_model_path),
        },
    )

    train_loader, test_loader = get_dataloaders(
        root=str(args.root),
        feature=args.feature,
        egde_file=args.egde_file,
        ts_test=args.ts_test,
        k=args.num_hops,
        batch_size=args.batch_size,
        shift=args.shift,
        target_mask_mode=args.target_mask_mode,
        feat_norm=args.feat_norm,
        window_size=args.window_size,
    )

    sample_inputs, _, _, _, _ = next(iter(train_loader))
    _, num_hops, _, feat_dim = sample_inputs.shape

    model = GNNRegressor(
        num_hops=num_hops,
        feat_dim=feat_dim,
        hidden_dim=args.hidden_dim,
        out_dim=1,
        mlp_layers=args.mlp_layers,
        num_transformer_layers=args.num_layers,
        heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    criterion = MaskedMSELoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_test_loss = float("inf")
    best_epoch = -1
    best_metrics = None
    tracked_indices = sorted(set(args.tracked_indices))
    tracked_index_metrics = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            predict_last=args.predict_last,
        )
        log_dict = {
            "Training/epoch": epoch,
            "Training/train_loss": train_loss,
        }
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            eval_metrics, _ = evaluate(
                model,
                test_loader,
                criterion,
                device,
                predict_last=args.predict_last,
                selected_indices=tracked_indices,
            )
            test_loss = eval_metrics["loss"]
            rmse = eval_metrics["rmse"]
            mae = eval_metrics["mae"]
            mape = eval_metrics["mape"]
            r2 = eval_metrics["r2"]
            print(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | "
                f"Test Loss: {test_loss:.6f} | RMSE: {rmse:.4f} | "
                f"MAE: {mae:.4f} | MAPE(%): {mape:.2f} | R2: {r2:.4f}"
            )
            log_dict.update({
                "Training/test_loss": test_loss,
                # "test_rmse": rmse,
                "Training/test_mae": mae,
                # "test_mape": mape,
                "Training/test_r2": r2,
            })

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                best_metrics = {
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "r2": r2,
                    "num_samples": eval_metrics["num_samples"],
                }
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_test_loss": best_test_loss,
                        "best_metrics": best_metrics,
                        "args": vars(args),
                    },
                    best_model_path,
                )
                print(f"Saved best model to {best_model_path}")

                # log_dict.update({
                #     "best_epoch": best_epoch,
                #     "best_test_loss": best_test_loss,
                #     "best_test_rmse": best_metrics["rmse"],
                #     "best_test_mae": best_metrics["mae"],
                #     "best_test_mape": best_metrics["mape"],
                #     "best_test_r2": best_metrics["r2"],
                #     "best_model_path": str(best_model_path),
                # })

        wandb.log(log_dict)

    if best_metrics is not None:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_epoch = checkpoint["epoch"]
        final_test_metrics, tracked_index_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            predict_last=args.predict_last,
            selected_indices=tracked_indices,
        )
    else:
        final_test_metrics = {
            "loss": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "mape": np.nan,
            "r2": np.nan,
            "num_samples": 0,
        }
        tracked_index_metrics = {
            idx: compute_regression_metrics(
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )
            for idx in tracked_indices
        }

    print("\nFinal Evaluation Metrics:")
    print(f"  Loss:  {final_test_metrics['loss']:.6f}")
    print(f"  RMSE:  {final_test_metrics['rmse']:.6f}")
    print(f"  MAE:   {final_test_metrics['mae']:.6f}")
    print(f"  MAPE(%): {final_test_metrics['mape']:.6f}")
    print(f"  R2:    {final_test_metrics['r2']:.6f}")
    for idx in tracked_indices:
        idx_metrics = tracked_index_metrics[idx]
        print(
            f"  Index {idx}: RMSE={idx_metrics['rmse']:.6f} | "
            f"MAE={idx_metrics['mae']:.6f} | MAPE(%)={idx_metrics['mape']:.6f} | "
            f"R2={idx_metrics['r2']:.6f} | N={idx_metrics['num_samples']}"
        )
    if best_metrics is not None:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_test_loss"] = best_test_loss
        wandb.summary["best_test_rmse"] = best_metrics["rmse"]
        wandb.summary["best_test_mae"] = best_metrics["mae"]
        wandb.summary["best_test_mape"] = best_metrics["mape"]
        wandb.summary["best_test_r2"] = best_metrics["r2"]
        wandb.summary["best_model_path"] = str(best_model_path)
        wandb.summary["Test/loss"] = final_test_metrics["loss"]
        wandb.summary["Test/rmse"] = final_test_metrics["rmse"]
        wandb.summary["Test/mae"] = final_test_metrics["mae"]
        wandb.summary["Test/mape"] = final_test_metrics["mape"]
        wandb.summary["Test/r2"] = final_test_metrics["r2"]
        wandb.summary["Test/num_samples"] = final_test_metrics["num_samples"]

        test_log = {
            "Test/loss": final_test_metrics["loss"],
            "Test/rmse": final_test_metrics["rmse"],
            "Test/mae": final_test_metrics["mae"],
            "Test/mape": final_test_metrics["mape"],
            "Test/r2": final_test_metrics["r2"],
            "Test/num_samples": final_test_metrics["num_samples"],
            "Test/best_epoch": best_epoch,
        }
        for idx in tracked_indices:
            idx_metrics = tracked_index_metrics[idx]
            prefix = f"Test_index_{idx}"
            wandb.summary[f"{prefix}/rmse"] = idx_metrics["rmse"]
            wandb.summary[f"{prefix}/mae"] = idx_metrics["mae"]
            wandb.summary[f"{prefix}/mape"] = idx_metrics["mape"]
            wandb.summary[f"{prefix}/r2"] = idx_metrics["r2"]
            wandb.summary[f"{prefix}/num_samples"] = idx_metrics["num_samples"]

            test_log[f"{prefix}/rmse"] = idx_metrics["rmse"]
            test_log[f"{prefix}/mae"] = idx_metrics["mae"]
            test_log[f"{prefix}/mape"] = idx_metrics["mape"]
            test_log[f"{prefix}/r2"] = idx_metrics["r2"]
            test_log[f"{prefix}/num_samples"] = idx_metrics["num_samples"]

        wandb.log(test_log)
    else:
        print("No best model was saved because validation did not produce a finite improvement.")
        wandb.summary["best_epoch"] = None
        wandb.summary["best_test_loss"] = None
        wandb.summary["best_model_path"] = None

    wandb.finish()


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=7 python main.py --num-hops 7 --learning-rate 1e-4
# rent/sqft
# <=2023, 2024,2025,2026-01
