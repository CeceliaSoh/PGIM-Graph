from __future__ import annotations

from pathlib import Path

import numpy as np
import wandb


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

    residuals = targets - preds
    mse = float(np.mean(np.square(residuals)))
    mae = float(np.mean(np.abs(residuals)))
    rmse = np.sqrt(mse)
    nonzero_mask = np.abs(targets) > 1e-8
    if np.any(nonzero_mask):
        mape = np.mean(
            np.abs((targets[nonzero_mask] - preds[nonzero_mask]) / targets[nonzero_mask])
        ) * 100.0
    else:
        mape = np.nan

    if len(targets) > 1 and np.unique(targets).size > 1:
        ss_res = float(np.sum(np.square(residuals)))
        ss_tot = float(np.sum(np.square(targets - np.mean(targets))))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        r2 = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "num_samples": int(targets.shape[0]),
    }


def empty_metrics():
    return {
        "loss": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "r2": np.nan,
        "num_samples": 0,
    }


def empty_index_metrics(tracked_indices):
    return {
        idx: compute_regression_metrics(
            np.empty((0, 1), dtype=np.float32),
            np.empty((0, 1), dtype=np.float32),
        )
        for idx in tracked_indices
    }


def print_final_metrics(final_test_metrics, tracked_index_metrics, tracked_indices):
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


def log_final_metrics(
    best_epoch: int,
    best_test_loss: float,
    best_metrics: dict,
    best_model_path: Path,
    final_test_metrics: dict,
    tracked_index_metrics: dict,
    tracked_indices,
):
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
