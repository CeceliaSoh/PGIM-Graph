from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


SOURCE_TIMESTEP_DIR = Path("dataset/ccr/timesteps")
DEFAULT_OUTPUT_DIR = Path("dataset/ccr/timesteps_timesfm")
DEFAULT_EXAMPLE_OUTPUT_PATH = Path("dataset/ccr/timesfm_example_project.csv")
DEFAULT_HF_REPO_ID = "google/timesfm-1.0-200m-pytorch"

PROJECT_COL = "Project Name"
TIME_COL = "Lease Commencement Date"
TARGET_COL = "rent_per_sqft"
MASK_COL = "y_mask"

# TimesFM v1.0-200m was trained with context length up to 512.
DEFAULT_CONTEXT_LEN = 512
DEFAULT_MIN_CONTEXT_POINTS = 12
DEFAULT_BATCH_SIZE = 32
DEFAULT_FREQ = 1  # monthly data


@dataclass
class GapFillStats:
    total_projects: int = 0
    total_gaps: int = 0
    total_points_filled_by_model: int = 0
    total_points_filled_by_linear_fallback: int = 0
    total_points_left_unchanged: int = 0
    boundary_points_preserved: int = 0


def log(message: str) -> None:
    print(f"[timesfm] {message}", flush=True)


def list_timestep_files(time_step_dir: Path) -> list[Path]:
    files = sorted(time_step_dir.glob("data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No timestep CSV files found in {time_step_dir}")
    return files


def load_timestep_frames(time_step_dir: Path) -> list[pd.DataFrame]:
    files = list_timestep_files(time_step_dir)
    frames = [pd.read_csv(path) for path in files]

    expected_columns = list(frames[0].columns)
    expected_projects = frames[0][PROJECT_COL].astype(str).tolist()

    for path, frame in zip(files, frames):
        if list(frame.columns) != expected_columns:
            raise ValueError(f"Inconsistent columns in {path}")

        project_names = frame[PROJECT_COL].astype(str).tolist()
        if project_names != expected_projects:
            raise ValueError(f"Inconsistent project ordering in {path}")

        unique_dates = pd.to_datetime(frame[TIME_COL], errors="coerce").dt.strftime("%Y-%m-%d").unique().tolist()
        if len(unique_dates) != 1:
            raise ValueError(f"Expected exactly one timestamp in {path}, got {unique_dates}")

    return frames


def build_target_matrix(frames: list[pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    projects = frames[0][PROJECT_COL].astype(str).tolist()
    targets = np.stack(
        [pd.to_numeric(frame[TARGET_COL], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) for frame in frames],
        axis=0,
    )
    mask = np.stack(
        [pd.to_numeric(frame[MASK_COL], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32) > 0 for frame in frames],
        axis=0,
    )
    return targets, mask, projects


def find_missing_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return only interior missing runs bounded by observed points on both sides."""
    runs: list[tuple[int, int]] = []
    observed_idx = np.flatnonzero(mask)
    if observed_idx.size < 2:
        return runs

    start_bound = int(observed_idx[0])
    end_bound = int(observed_idx[-1])
    cursor = start_bound
    while cursor <= end_bound:
        if mask[cursor]:
            cursor += 1
            continue
        run_start = cursor
        while cursor <= end_bound and not mask[cursor]:
            cursor += 1
        runs.append((run_start, cursor))
    return runs


def count_boundary_missing_points(mask: np.ndarray) -> int:
    """Count missing points before the first observation and after the last one."""
    observed_idx = np.flatnonzero(mask)
    if observed_idx.size == 0:
        return int(mask.size)
    first_obs = int(observed_idx[0])
    last_obs = int(observed_idx[-1])
    leading_missing = first_obs
    trailing_missing = (mask.size - 1) - last_obs
    return leading_missing + trailing_missing


def build_linear_fallback(series: np.ndarray, observed_mask: np.ndarray) -> np.ndarray:
    interpolated = (
        pd.Series(series)
        .mask(~observed_mask)
        .interpolate(method="linear", limit_area="inside")
        .to_numpy(dtype=np.float32)
    )
    return interpolated


def _coerce_forecast_array(raw_forecast: Any) -> np.ndarray:
    forecast = np.asarray(raw_forecast, dtype=np.float32)
    if forecast.ndim == 3:
        forecast = forecast[:, :, 0]
    elif forecast.ndim == 1:
        forecast = forecast[None, :]
    if forecast.ndim != 2:
        raise ValueError(f"Unexpected forecast shape: {forecast.shape}")
    return forecast


def import_timesfm_package() -> Any:
    script_dir = str(Path(__file__).resolve().parent)
    cwd_dir = str(Path.cwd().resolve())
    removed_entries: list[tuple[int, str]] = []

    for index in range(len(sys.path) - 1, -1, -1):
        entry = sys.path[index]
        resolved = cwd_dir if entry == "" else str(Path(entry).resolve())
        if resolved in {script_dir, cwd_dir}:
            removed_entries.append((index, entry))
            sys.path.pop(index)

    try:
        return importlib.import_module("timesfm")
    finally:
        for index, entry in sorted(removed_entries, key=lambda item: item[0]):
            sys.path.insert(index, entry)


class TimesFMForecaster:
    def __init__(
        self,
        repo_id: str,
        context_len: int,
        per_core_batch_size: int,
        horizon_len: int,
        backend: str = "cpu",
    ) -> None:
        try:
            timesfm = import_timesfm_package()
        except ImportError as exc:
            raise ImportError(
                "The `timesfm` package is not installed. Install it first, for example:\n"
                "  pip install timesfm torch\n"
                f"Then rerun this script with `--hf-repo-id {repo_id}`."
            ) from exc

        self._timesfm = timesfm
        self._context_len = context_len
        self._model = self._build_model(
            repo_id=repo_id,
            context_len=context_len,
            per_core_batch_size=per_core_batch_size,
            horizon_len=horizon_len,
            backend=backend,
        )

    def _build_model(
        self,
        repo_id: str,
        context_len: int,
        per_core_batch_size: int,
        horizon_len: int,
        backend: str,
    ) -> Any:
        hparams_candidates = [
            {
                "backend": backend,
                "per_core_batch_size": per_core_batch_size,
                "horizon_len": horizon_len,
                "context_len": context_len,
            },
            {
                "backend": backend,
                "per_core_batch_size": per_core_batch_size,
                "horizon_len": horizon_len,
            },
        ]

        checkpoint_candidates = [
            {"huggingface_repo_id": repo_id},
            {"hugginface_repo_id": repo_id},
            {"repo_id": repo_id},
        ]

        last_error: Exception | None = None
        for hparam_kwargs in hparams_candidates:
            try:
                hparams = self._timesfm.TimesFmHparams(**hparam_kwargs)
            except TypeError as exc:
                last_error = exc
                continue

            for checkpoint_kwargs in checkpoint_candidates:
                try:
                    checkpoint = self._timesfm.TimesFmCheckpoint(**checkpoint_kwargs)
                    return self._timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
                except TypeError as exc:
                    last_error = exc
                    continue

        raise RuntimeError(f"Unable to initialize TimesFM model: {last_error}")

    def forecast(self, contexts: list[list[float]], freq: list[int]) -> np.ndarray:
        raw = self._model.forecast(inputs=contexts, freq=freq)
        forecast = raw[0] if isinstance(raw, tuple) else raw
        return _coerce_forecast_array(forecast)


def fill_with_timesfm(
    original_targets: np.ndarray,
    observed_mask: np.ndarray,
    forecaster: TimesFMForecaster,
    context_len: int,
    min_context_points: int,
    model_horizon_len: int,
    freq: int,
) -> tuple[np.ndarray, GapFillStats]:
    filled = original_targets.astype(np.float32).copy()
    linear_fallback = build_linear_fallback(original_targets, observed_mask)
    stats = GapFillStats(total_projects=1)
    stats.boundary_points_preserved = count_boundary_missing_points(observed_mask)

    gap_runs = find_missing_runs(observed_mask)
    stats.total_gaps = len(gap_runs)

    for gap_start, gap_end in gap_runs:
        gap_len = gap_end - gap_start
        history = filled[:gap_start]
        usable_history = history[np.isfinite(history)]

        can_use_model = usable_history.size >= min_context_points
        if can_use_model:
            context = usable_history[-context_len:].astype(np.float32).tolist()
            remaining = gap_len
            cursor = gap_start
            while remaining > 0:
                horizon = min(model_horizon_len, remaining)
                forecast = forecaster.forecast([context], freq=[freq])[0][:horizon]
                filled[cursor : cursor + horizon] = forecast
                context.extend(forecast.astype(np.float32).tolist())
                context = context[-context_len:]
                cursor += horizon
                remaining -= horizon
            stats.total_points_filled_by_model += gap_len
            continue

        fallback_values = linear_fallback[gap_start:gap_end]
        fallback_mask = np.isfinite(fallback_values)
        if np.any(fallback_mask):
            filled_slice = filled[gap_start:gap_end]
            filled_slice[fallback_mask] = fallback_values[fallback_mask]
            filled[gap_start:gap_end] = filled_slice
            stats.total_points_filled_by_linear_fallback += int(np.count_nonzero(fallback_mask))
            stats.total_points_left_unchanged += int(gap_len - np.count_nonzero(fallback_mask))
        else:
            stats.total_points_left_unchanged += gap_len

    return filled, stats


def write_output_frames(
    source_frames: list[pd.DataFrame],
    output_dir: Path,
    filled_targets: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(source_frames):
        out_frame = frame.copy()
        out_frame[TARGET_COL] = filled_targets[index]
        file_date = pd.to_datetime(out_frame[TIME_COL].iloc[0]).strftime("%Y%m%d")
        out_frame.to_csv(output_dir / f"data_{file_date}.csv", index=False)


def resolve_example_project_index(
    requested_project: str | None,
    project_names: list[str],
    original_targets: np.ndarray,
    filled_targets: np.ndarray,
) -> int:
    if requested_project is not None:
        try:
            return project_names.index(requested_project)
        except ValueError as exc:
            raise ValueError(f"Project not found for example export: {requested_project}") from exc

    changed_mask = np.any(~np.isclose(original_targets, filled_targets), axis=0)
    changed_indices = np.flatnonzero(changed_mask)
    if changed_indices.size > 0:
        return int(changed_indices[0])

    return 0


def write_example_project_csv(
    source_frames: list[pd.DataFrame],
    output_path: Path,
    project_name: str,
    project_index: int,
    filled_targets: np.ndarray,
) -> None:
    timestamps = [
        pd.to_datetime(frame[TIME_COL].iloc[0]).strftime("%Y-%m-%d")
        for frame in source_frames
    ]
    example_df = pd.DataFrame(
        {
            TIME_COL: timestamps,
            TARGET_COL: filled_targets[:, project_index].astype(np.float32),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example_df.to_csv(output_path, index=False)
    log(f"Wrote example project price series for '{project_name}' to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a TimesFM-imputed version of dataset/ccr/timesteps. "
            "Observed values keep their original rent_per_sqft; only interior "
            "y_mask=0 gaps bounded by observations on both sides are filled."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=SOURCE_TIMESTEP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-repo-id", type=str, default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--backend", type=str, default="gpu")
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--min-context-points", type=int, default=DEFAULT_MIN_CONTEXT_POINTS)
    parser.add_argument(
        "--model-horizon-len",
        type=int,
        default=12,
        help="Forecast horizon per TimesFM call. Long gaps are filled chunk by chunk.",
    )
    parser.add_argument("--per-core-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--freq", type=int, default=DEFAULT_FREQ)
    parser.add_argument(
        "--example-output",
        type=Path,
        default=DEFAULT_EXAMPLE_OUTPUT_PATH,
        help="CSV path for one example project's price series across all timesteps.",
    )
    parser.add_argument(
        "--example-project",
        type=str,
        default=None,
        help="Optional project name to export as the example CSV. Defaults to the first changed project.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log(f"Loading timestep CSVs from {args.input_dir} ...")
    source_frames = load_timestep_frames(args.input_dir)
    log(f"Loaded {len(source_frames)} timestep files.")

    targets, observed_mask, projects = build_target_matrix(source_frames)
    log(
        "Built target matrix with "
        f"{targets.shape[1]} projects x {targets.shape[0]} timesteps."
    )
    log("Only interior y_mask=0 gaps will be imputed; leading and trailing missing spans are preserved.")

    log(
        "Initializing TimesFM model "
        f"({args.hf_repo_id}, backend={args.backend}, context_len={args.context_len}, "
        f"horizon_len={args.model_horizon_len}) ..."
    )
    forecaster = TimesFMForecaster(
        repo_id=args.hf_repo_id,
        context_len=args.context_len,
        per_core_batch_size=args.per_core_batch_size,
        horizon_len=args.model_horizon_len,
        backend=args.backend,
    )
    log("TimesFM model is ready. Starting gap filling ...")

    filled_targets = targets.copy()
    total_stats = GapFillStats(total_projects=len(projects))
    progress_iter = enumerate(projects)
    if tqdm is not None:
        progress_iter = enumerate(
            tqdm(
                projects,
                total=len(projects),
                desc="Imputing projects",
                unit="project",
            )
        )

    for project_idx, project_name in progress_iter:
        filled_series, stats = fill_with_timesfm(
            original_targets=targets[:, project_idx],
            observed_mask=observed_mask[:, project_idx],
            forecaster=forecaster,
            context_len=args.context_len,
            min_context_points=args.min_context_points,
            model_horizon_len=args.model_horizon_len,
            freq=args.freq,
        )
        filled_targets[:, project_idx] = filled_series
        total_stats.total_gaps += stats.total_gaps
        total_stats.total_points_filled_by_model += stats.total_points_filled_by_model
        total_stats.total_points_filled_by_linear_fallback += stats.total_points_filled_by_linear_fallback
        total_stats.total_points_left_unchanged += stats.total_points_left_unchanged
        total_stats.boundary_points_preserved += stats.boundary_points_preserved

        if tqdm is None and ((project_idx + 1) % 25 == 0 or project_idx + 1 == len(projects)):
            log(
                f"Processed {project_idx + 1}/{len(projects)} projects. "
                f"Last project: {project_name}"
            )

    log(f"Writing imputed timestep CSVs to {args.output_dir} ...")
    write_output_frames(source_frames=source_frames, output_dir=args.output_dir, filled_targets=filled_targets)
    log("Finished writing output files.")

    example_project_index = resolve_example_project_index(
        requested_project=args.example_project,
        project_names=projects,
        original_targets=targets,
        filled_targets=filled_targets,
    )
    write_example_project_csv(
        source_frames=source_frames,
        output_path=args.example_output,
        project_name=projects[example_project_index],
        project_index=example_project_index,
        filled_targets=filled_targets,
    )

    print(f"Input timesteps: {args.input_dir}")
    print(f"Output timesteps: {args.output_dir}")
    print(f"Example project CSV: {args.example_output}")
    print(f"Projects: {total_stats.total_projects}")
    print(f"Gap runs inside observed spans: {total_stats.total_gaps}")
    print(f"Filled by TimesFM: {total_stats.total_points_filled_by_model}")
    print(f"Filled by linear fallback: {total_stats.total_points_filled_by_linear_fallback}")
    print(f"Left unchanged: {total_stats.total_points_left_unchanged}")
    print(f"Boundary missing points preserved: {total_stats.boundary_points_preserved}")


if __name__ == "__main__":
    main()
