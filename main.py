import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import wandb
from datetime import datetime

from tsgnn.data.pgim import get_pgim_dataloader
from tsgnn.models.regressor import TransformerRegressor
from tsgnn.utils.loss import MaskedMSELoss
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
    total_loss = 0
    for inputs, targets, y_mask, _, _ in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        y_mask = y_mask.to(device).float()
        
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(f"outputs shape: {outputs.shape}, targets shape: {targets.shape}, y_mask shape: {y_mask.shape}")
        loss = criterion(outputs.squeeze(-1), targets, y_mask)
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

            # masked loss
            outputs = outputs.squeeze(-1)  # [B, T]
            loss = criterion(outputs, targets, y_mask)
            total_loss += loss.item()

            # keep only valid positions for metric computation
            valid_mask = y_mask > 0

            valid_outputs = outputs[valid_mask]
            valid_targets = targets[valid_mask]

            if valid_outputs.numel() > 0:
                all_preds.append(valid_outputs.cpu().numpy().reshape(-1, 1))
                all_targets.append(valid_targets.cpu().numpy().reshape(-1, 1))

    avg_loss = total_loss / len(dataloader)

    # Handle case where all entries are masked out
    if len(all_preds) == 0:
        return avg_loss, np.nan, np.nan, np.nan, np.nan, np.nan

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    mse = mean_squared_error(all_targets, all_preds)
    mae = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mse)

    # r2 requires at least 2 samples and non-constant targets
    if len(all_targets) > 1 and np.unique(all_targets).size > 1:
        r2 = r2_score(all_targets, all_preds)
    else:
        r2 = np.nan

    target_range = np.max(all_targets) - np.min(all_targets)
    nrmse = rmse / target_range if target_range != 0 else rmse

    return avg_loss, mse, mae, rmse, r2, nrmse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build node-wise temporal windows and a transformer regressor.",
    )
    parser.add_argument("--feat-train", type=Path, default=Path("dataset/feature_ccr_v3.1.1_within_250m_train.npy"))
    parser.add_argument("--feat-test", type=Path, default=Path("dataset/feature_ccr_v3.1.1_within_250m_test.npy"))
    parser.add_argument("--feat-norm", type=str2bool, default=True, help="Whether to apply feature normalization (standardization) to the input features.")
    parser.add_argument("--graph-path", type=Path, default=Path("dataset/graph_ccr_v3.1_within_250m.pt"))
    parser.add_argument("--num-hops", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--ff-hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wandb-project", type=str, default="pgim-transformer-exp")
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
            "window_size": args.window_size,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "ff_hidden_dim": args.ff_hidden_dim,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "checkpoint_dir": str(checkpoint_dir),
        },
    )
    dataset_paths = {
        "feat_train": args.feat_train,
        "feat_test": args.feat_test,
        "graph": args.graph_path,
    }
    train_loader, test_loader = get_pgim_dataloader(dataset_paths, feat_norm=args.feat_norm, window_size=args.window_size, batch_size=args.batch_size, num_hops=args.num_hops, num_workers=args.num_workers)

    model = TransformerRegressor(
        in_channels=train_loader.dataset.features.shape[-1], 
        out_channels=1, 
        units=args.hidden_dim, 
        ff_hidden_dim=args.ff_hidden_dim,
        len_ts=args.window_size, 
        num_layers=args.num_layers, 
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
    mse = mae = rmse = r2 = nrmse = np.nan
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
        }
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            test_loss, mse, mae, rmse, r2, nrmse = evaluate(model, test_loader, criterion, device)
            print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | R2: {r2:.4f} | NRMSE: {nrmse:.4f}")
            log_dict.update({
                "test_loss": test_loss,
                "test_mse": mse,
                "test_mae": mae,
                "test_rmse": rmse,
                "test_r2": r2,
                "test_nrmse": nrmse,
            })

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                best_metrics = {
                    "mse": mse,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "nrmse": nrmse,
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
                    "best_test_mse": best_metrics["mse"],
                    "best_test_mae": best_metrics["mae"],
                    "best_test_rmse": best_metrics["rmse"],
                    "best_test_r2": best_metrics["r2"],
                    "best_test_nrmse": best_metrics["nrmse"],
                    "best_model_path": str(best_model_path),
                })

        wandb.log(log_dict)
        
    print("\nFinal Evaluation Metrics (on Synthetic Test Set):")
    print(f"  MSE:   {mse:.6f}")
    print(f"  MAE:   {mae:.6f}")
    print(f"  RMSE:  {rmse:.6f}")
    print(f"  R2:    {r2:.6f}")
    print(f"  NRMSE: {nrmse:.6f}")
    if best_metrics is not None:
        wandb.summary["best_epoch"] = best_epoch
        wandb.summary["best_test_loss"] = best_test_loss
        wandb.summary["best_test_mse"] = best_metrics["mse"]
        wandb.summary["best_test_mae"] = best_metrics["mae"]
        wandb.summary["best_test_rmse"] = best_metrics["rmse"]
        wandb.summary["best_test_r2"] = best_metrics["r2"]
        wandb.summary["best_test_nrmse"] = best_metrics["nrmse"]
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
