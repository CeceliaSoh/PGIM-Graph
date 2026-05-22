from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
import wandb


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
        for inputs, targets, y_mask, _, _ in tqdm(dataloader, desc="Evaluating", leave=False):
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


def _empty_metrics():
    return {
        "loss": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
        "mape": np.nan,
        "r2": np.nan,
        "num_samples": 0,
    }


def _print_final_metrics(final_test_metrics, tracked_index_metrics, tracked_indices):
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


def _log_final_metrics(
    best_epoch,
    best_test_loss,
    best_metrics,
    best_model_path,
    final_test_metrics,
    tracked_index_metrics,
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


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    device,
    *,
    epochs: int,
    eval_interval: int,
    early_stopping_patience: int | None,
    predict_last: bool,
    tracked_indices,
    best_model_path: Path,
    checkpoint_config: dict[str, Any],
):
    best_test_loss = float("inf")
    best_epoch = -1
    best_metrics = None
    tracked_indices = sorted(set(tracked_indices))
    tracked_index_metrics = {}
    stale_eval_count = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            predict_last=predict_last,
        )
        log_dict = {
            "Training/epoch": epoch,
            "Training/train_loss": train_loss,
        }
        if epoch % eval_interval == 0 or epoch == epochs:
            eval_metrics, _ = evaluate(
                model,
                test_loader,
                criterion,
                device,
                predict_last=predict_last,
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
                stale_eval_count = 0
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
                        "config": checkpoint_config,
                    },
                    best_model_path,
                )
                print(f"Saved best model to {best_model_path}")
            else:
                stale_eval_count += 1

            log_dict["Training/early_stopping_stale_evals"] = stale_eval_count

            if early_stopping_patience is not None and stale_eval_count >= early_stopping_patience:
                stopped_early = True
                print(
                    f"Early stopping at epoch {epoch}: test loss did not improve for "
                    f"{stale_eval_count} evaluation checks."
                )

        wandb.log(log_dict)
        if stopped_early:
            break

    if best_metrics is not None:
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_epoch = checkpoint["epoch"]
        final_test_metrics, tracked_index_metrics = evaluate(
            model,
            test_loader,
            criterion,
            device,
            predict_last=predict_last,
            selected_indices=tracked_indices,
        )
    else:
        final_test_metrics = _empty_metrics()
        tracked_index_metrics = {
            idx: compute_regression_metrics(
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )
            for idx in tracked_indices
        }

    _print_final_metrics(final_test_metrics, tracked_index_metrics, tracked_indices)
    if best_metrics is not None:
        _log_final_metrics(
            best_epoch,
            best_test_loss,
            best_metrics,
            best_model_path,
            final_test_metrics,
            tracked_index_metrics,
            tracked_indices,
        )
        wandb.summary["early_stopped"] = stopped_early
        wandb.summary["stopped_epoch"] = epoch if stopped_early else None
    else:
        print("No best model was saved because validation did not produce a finite improvement.")
        wandb.summary["best_epoch"] = None
        wandb.summary["best_test_loss"] = None
        wandb.summary["best_model_path"] = None
        wandb.summary["early_stopped"] = stopped_early
        wandb.summary["stopped_epoch"] = epoch if stopped_early else None

    return {
        "best_epoch": best_epoch,
        "best_test_loss": best_test_loss if best_metrics is not None else None,
        "best_metrics": best_metrics,
        "final_test_metrics": final_test_metrics,
        "tracked_index_metrics": tracked_index_metrics,
        "early_stopped": stopped_early,
        "stopped_epoch": epoch if stopped_early else None,
    }
