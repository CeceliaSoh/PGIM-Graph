from pathlib import Path
import os
import random

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from datetime import datetime

from src.data.pgim import get_dataloaders
from src.models.gnn_regressor import GNNRegressor
from src.trainer import Trainer
from src.utils.loss import MaskedMSELoss


def set_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_path(path_value) -> Path:
    return Path(hydra.utils.to_absolute_path(str(path_value)))


def _validate_config(cfg: DictConfig) -> None:
    valid_mask_modes = {"all", "observed_only", "train_all_test_observed"}
    if cfg.data.target_mask_mode not in valid_mask_modes:
        raise ValueError(
            f"data.target_mask_mode must be one of {sorted(valid_mask_modes)}, "
            f"got {cfg.data.target_mask_mode!r}."
        )

    valid_merge_modes = {"concat", "mean"}
    if cfg.model.merge_mode not in valid_merge_modes:
        raise ValueError(
            f"model.merge_mode must be one of {sorted(valid_merge_modes)}, "
            f"got {cfg.model.merge_mode!r}."
        )

    if cfg.model.hidden_dim % cfg.model.num_heads != 0:
        raise ValueError(
            f"model.hidden_dim ({cfg.model.hidden_dim}) must be divisible by "
            f"model.num_heads ({cfg.model.num_heads})."
        )

    if cfg.training.early_stopping_patience is not None and cfg.training.early_stopping_patience <= 0:
        raise ValueError(
            "training.early_stopping_patience must be a positive integer or null, "
            f"got {cfg.training.early_stopping_patience}."
        )


@hydra.main(config_path="src/config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    _validate_config(cfg)
    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = (
        f"hop{cfg.data.num_hops}_"
        f"lr{cfg.optimizer.learning_rate}_"
        f"wd{cfg.optimizer.weight_decay}_"
        f"ts{cfg.data.window_size}_"
        f"shift{cfg.data.target_shift}_"
        f"{cfg.training.run_name_sufix}_"
        f"{timestamp}"
    )
    
    checkpoint_dir = _resolve_path(cfg.paths.checkpoint_dir) / cfg.logging.wandb_project
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = checkpoint_dir / f"{run_name}_best.pt"

    wandb.init(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.entity,
        name=run_name,
        config={
            **OmegaConf.to_container(cfg, resolve=True),
            "best_model_path": str(best_model_path),
        },
    )

    train_loader, test_loader = get_dataloaders(cfg)

    _, _, num_graphs, num_hops, feat_dim = train_loader.dataset.contexts.shape

    model = GNNRegressor(
        num_hops=num_hops,
        num_graphs=num_graphs,
        feat_dim=feat_dim,
        hidden_dim=cfg.model.hidden_dim,
        out_dim=1,
        mlp_layers=cfg.model.mlp_layers,
        num_transformer_layers=cfg.model.num_layers,
        heads=cfg.model.num_heads,
        dropout=cfg.model.dropout,
        conv_filters=cfg.model.conv_filters,
        merge_mode=cfg.model.merge_mode,
    ).to(device)

    criterion = MaskedMSELoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
    )

    trainer = Trainer(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        epochs=cfg.training.epochs,
        eval_interval=cfg.training.eval_interval,
        early_stopping_patience=cfg.training.early_stopping_patience,
        predict_last=cfg.training.predict_last,
        tracked_indices=cfg.evaluation.tracked_indices,
        best_model_path=best_model_path,
        checkpoint_config=OmegaConf.to_container(cfg, resolve=True),
    )
    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=7 python main.py data.num_hops=7 optimizer.learning_rate=1e-4
# rent/sqft
# <=2023, 2024,2025,2026-01
