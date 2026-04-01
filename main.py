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


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for inputs, targets, y_mask, _, _ in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        y_mask = y_mask.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets, y_mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
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
        return avg_loss, np.nan, np.nan, np.nan, np.nan

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    nonzero_mask = np.abs(all_targets) > 1e-8
    if np.any(nonzero_mask):
        mape = np.mean(
            np.abs((all_targets[nonzero_mask] - all_preds[nonzero_mask]) / all_targets[nonzero_mask])
        ) * 100.0
    else:
        mape = np.nan

    if len(all_targets) > 1 and np.unique(all_targets).size > 1:
        r2 = r2_score(all_targets, all_preds)
    else:
        r2 = np.nan

    return avg_loss, rmse, mae, mape, r2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a GNNRegressor on the PGIM dataset with SIGN-style precomputed hop features.",
    )
    parser.add_argument("--root", type=Path, default=Path("dataset/ccr"))
    parser.add_argument("--feature", type=str, default="feature.npy")
    parser.add_argument("--egde-file", type=str, default="graph_link_200m/links.txt")
    parser.add_argument("--ts-test", type=int, default=25)
    parser.add_argument("--shift", type=int, default=1)
    parser.add_argument("--feat-norm", type=str2bool, default=True, help="Whether to apply train-set min-max scaling to [-1, 1].")
    parser.add_argument("--num-hops", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=12)
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
    checkpoint_dir = args.checkpoint_dir
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
            "checkpoint_dir": str(checkpoint_dir),
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_test_loss = float("inf")
    best_epoch = -1
    best_metrics = None
    rmse = mae = mape = r2 = np.nan
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
        }
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            test_loss, rmse, mae, mape, r2 = evaluate(model, test_loader, criterion, device)
            print(
                f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | "
                f"Test Loss: {test_loss:.6f} | RMSE: {rmse:.4f} | "
                f"MAE: {mae:.4f} | MAPE(%): {mape:.2f} | R2: {r2:.4f}"
            )
            log_dict.update({
                "test_loss": test_loss,
                "test_rmse": rmse,
                "test_mae": mae,
                "test_mape": mape,
                "test_r2": r2,
            })

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                best_metrics = {
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "r2": r2,
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

                log_dict.update({
                    "best_epoch": best_epoch,
                    "best_test_loss": best_test_loss,
                    "best_test_rmse": best_metrics["rmse"],
                    "best_test_mae": best_metrics["mae"],
                    "best_test_mape": best_metrics["mape"],
                    "best_test_r2": best_metrics["r2"],
                    "best_model_path": str(best_model_path),
                })

        wandb.log(log_dict)

    print("\nFinal Evaluation Metrics:")
    print(f"  RMSE:  {rmse:.6f}")
    print(f"  MAE:   {mae:.6f}")
    print(f"  MAPE(%): {mape:.6f}")
    print(f"  R2:    {r2:.6f}")
    if best_metrics is not None:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_test_loss"] = best_test_loss
        wandb.summary["best_test_rmse"] = best_metrics["rmse"]
        wandb.summary["best_test_mae"] = best_metrics["mae"]
        wandb.summary["best_test_mape"] = best_metrics["mape"]
        wandb.summary["best_test_r2"] = best_metrics["r2"]
        wandb.summary["best_model_path"] = str(best_model_path)
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
