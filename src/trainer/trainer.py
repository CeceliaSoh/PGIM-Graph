from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
import wandb

from src.utils.training import (
    compute_regression_metrics,
    empty_index_metrics,
    empty_metrics,
    log_final_metrics,
    print_final_metrics,
)


class Trainer:
    def __init__(
        self,
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
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.eval_interval = eval_interval
        self.early_stopping_patience = early_stopping_patience
        self.predict_last = predict_last
        self.tracked_indices = sorted(set(tracked_indices))
        self.best_model_path = best_model_path
        self.checkpoint_config = checkpoint_config

        self.best_test_loss = float("inf")
        self.best_epoch = -1
        self.best_metrics = None
        self.stale_eval_count = 0
        self.stopped_early = False
        self.stopped_epoch = None

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch()
            log_dict = self._training_log(epoch, train_loss)

            if self._should_evaluate(epoch):
                eval_metrics, _ = self.evaluate()
                self._print_epoch_metrics(epoch, train_loss, eval_metrics)
                log_dict.update(self._evaluation_log(eval_metrics))
                self._track_best_checkpoint(epoch, eval_metrics)
                log_dict["Training/early_stopping_stale_evals"] = self.stale_eval_count

                if self._should_stop_early():
                    self.stopped_early = True
                    self.stopped_epoch = epoch
                    self._print_early_stopping(epoch)

            wandb.log(log_dict)
            if self.stopped_early:
                break

        return self._finalize()

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        for inputs, targets, y_mask, _, _ in tqdm(self.train_loader, desc="Training"):
            inputs, targets, y_mask = self._move_batch_to_device(inputs, targets, y_mask)
            outputs, targets, y_mask = self._forward_for_loss(inputs, targets, y_mask)
            loss = self.criterion(outputs, targets, y_mask)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, selected_indices=None):
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        index_buffers = {}
        selected_indices = self._normalize_indices(selected_indices)

        with torch.no_grad():
            for inputs, targets, y_mask, _, _ in tqdm(
                self.test_loader,
                desc="Evaluating",
                leave=False,
            ):
                inputs, targets, y_mask = self._move_batch_to_device(inputs, targets, y_mask)
                outputs, targets, y_mask = self._forward_for_loss(inputs, targets, y_mask)
                loss = self.criterion(outputs, targets, y_mask)
                total_loss += loss.item()

                outputs_cpu, targets_cpu, valid_mask_cpu = self._detach_eval_tensors(
                    outputs,
                    targets,
                    y_mask,
                )
                self._collect_overall_predictions(
                    outputs_cpu,
                    targets_cpu,
                    valid_mask_cpu,
                    all_preds,
                    all_targets,
                )
                self._collect_index_predictions(
                    outputs_cpu,
                    targets_cpu,
                    valid_mask_cpu,
                    selected_indices,
                    index_buffers,
                )

        avg_loss = total_loss / len(self.test_loader)
        overall_metrics = self._overall_metrics(avg_loss, all_preds, all_targets)
        index_metrics = self._index_metrics(selected_indices, index_buffers)
        return overall_metrics, index_metrics

    def _move_batch_to_device(self, inputs, targets, y_mask):
        return inputs.to(self.device), targets.to(self.device), y_mask.to(self.device).float()

    def _forward_for_loss(self, inputs, targets, y_mask):
        outputs = self.model(inputs)
        if self.predict_last:
            outputs = outputs[:, -1:, :]
            targets = targets[:, -1:, :]
            y_mask = y_mask[:, -1:, :]
        return outputs, targets, y_mask

    def _normalize_indices(self, selected_indices):
        if selected_indices is None:
            selected_indices = self.tracked_indices
        return sorted(set(selected_indices))

    @staticmethod
    def _detach_eval_tensors(outputs, targets, y_mask):
        return (
            outputs.detach().cpu(),
            targets.detach().cpu(),
            (y_mask > 0).detach().cpu(),
        )

    @staticmethod
    def _collect_overall_predictions(
        outputs_cpu,
        targets_cpu,
        valid_mask_cpu,
        all_preds,
        all_targets,
    ):
        valid_outputs = outputs_cpu[valid_mask_cpu]
        valid_targets = targets_cpu[valid_mask_cpu]
        if valid_outputs.numel() == 0:
            return

        all_preds.append(valid_outputs.numpy().reshape(-1, 1))
        all_targets.append(valid_targets.numpy().reshape(-1, 1))

    @staticmethod
    def _collect_index_predictions(
        outputs_cpu,
        targets_cpu,
        valid_mask_cpu,
        selected_indices,
        index_buffers,
    ):
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

    @staticmethod
    def _overall_metrics(avg_loss, all_preds, all_targets):
        if len(all_preds) == 0:
            return {
                "loss": avg_loss,
                "rmse": np.nan,
                "mae": np.nan,
                "mape": np.nan,
                "r2": np.nan,
                "num_samples": 0,
            }

        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        return {
            "loss": avg_loss,
            **compute_regression_metrics(all_preds, all_targets),
        }

    @staticmethod
    def _index_metrics(selected_indices, index_buffers):
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
        return index_metrics

    @staticmethod
    def _training_log(epoch, train_loss):
        return {
            "Training/epoch": epoch,
            "Training/train_loss": train_loss,
        }

    @staticmethod
    def _evaluation_log(eval_metrics):
        return {
            "Training/test_loss": eval_metrics["loss"],
            "Training/test_mae": eval_metrics["mae"],
            "Training/test_r2": eval_metrics["r2"],
        }

    def _should_evaluate(self, epoch):
        return epoch % self.eval_interval == 0 or epoch == self.epochs

    @staticmethod
    def _print_epoch_metrics(epoch, train_loss, eval_metrics):
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | "
            f"Test Loss: {eval_metrics['loss']:.6f} | RMSE: {eval_metrics['rmse']:.4f} | "
            f"MAE: {eval_metrics['mae']:.4f} | MAPE(%): {eval_metrics['mape']:.2f} | "
            f"R2: {eval_metrics['r2']:.4f}"
        )

    def _track_best_checkpoint(self, epoch, eval_metrics):
        if eval_metrics["loss"] < self.best_test_loss:
            self.best_test_loss = eval_metrics["loss"]
            self.best_epoch = epoch
            self.stale_eval_count = 0
            self.best_metrics = self._best_metrics(eval_metrics)
            self._save_checkpoint(epoch)
            print(f"Saved best model to {self.best_model_path}")
            return

        self.stale_eval_count += 1

    @staticmethod
    def _best_metrics(eval_metrics):
        return {
            "rmse": eval_metrics["rmse"],
            "mae": eval_metrics["mae"],
            "mape": eval_metrics["mape"],
            "r2": eval_metrics["r2"],
            "num_samples": eval_metrics["num_samples"],
        }

    def _save_checkpoint(self, epoch):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_test_loss": self.best_test_loss,
                "best_metrics": self.best_metrics,
                "config": self.checkpoint_config,
            },
            self.best_model_path,
        )

    def _should_stop_early(self):
        return (
            self.early_stopping_patience is not None
            and self.stale_eval_count >= self.early_stopping_patience
        )

    def _print_early_stopping(self, epoch):
        print(
            f"Early stopping at epoch {epoch}: test loss did not improve for "
            f"{self.stale_eval_count} evaluation checks."
        )

    def _finalize(self):
        final_test_metrics, tracked_index_metrics = self._load_best_and_evaluate()
        print_final_metrics(final_test_metrics, tracked_index_metrics, self.tracked_indices)
        self._log_training_summary(final_test_metrics, tracked_index_metrics)

        return {
            "best_epoch": self.best_epoch,
            "best_test_loss": self.best_test_loss if self.best_metrics is not None else None,
            "best_metrics": self.best_metrics,
            "final_test_metrics": final_test_metrics,
            "tracked_index_metrics": tracked_index_metrics,
            "early_stopped": self.stopped_early,
            "stopped_epoch": self.stopped_epoch,
        }

    def _load_best_and_evaluate(self):
        if self.best_metrics is None:
            return self._empty_final_metrics()

        checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_epoch = checkpoint["epoch"]
        return self.evaluate(selected_indices=self.tracked_indices)

    def _empty_final_metrics(self):
        return empty_metrics(), empty_index_metrics(self.tracked_indices)

    def _log_training_summary(self, final_test_metrics, tracked_index_metrics):
        if self.best_metrics is None:
            print("No best model was saved because validation did not produce a finite improvement.")
            wandb.summary["best_epoch"] = None
            wandb.summary["best_test_loss"] = None
            wandb.summary["best_model_path"] = None
            wandb.summary["early_stopped"] = self.stopped_early
            wandb.summary["stopped_epoch"] = self.stopped_epoch
            return

        log_final_metrics(
            self.best_epoch,
            self.best_test_loss,
            self.best_metrics,
            self.best_model_path,
            final_test_metrics,
            tracked_index_metrics,
            self.tracked_indices,
        )
        wandb.summary["early_stopped"] = self.stopped_early
        wandb.summary["stopped_epoch"] = self.stopped_epoch
