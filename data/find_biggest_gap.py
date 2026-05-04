from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TIMESTEP_DIR = Path("dataset/ccr/timesteps")
DEFAULT_OUTPUT_PATH = Path("dataset/ccr/biggest_gaps.csv")

PROJECT_COL = "Project Name"
TIME_COL = "Lease Commencement Date"
MASK_COL = "y_mask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find the largest interior y_mask=0 gap for each project and the overall "
            "largest gap in a timestep folder."
        )
    )
    parser.add_argument("--timestep-dir", type=Path, default=DEFAULT_TIMESTEP_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def list_timestep_files(timestep_dir: Path) -> list[Path]:
    files = sorted(timestep_dir.glob("data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No timestep CSV files found in {timestep_dir}")
    return files


def load_mask_matrix(files: list[Path]) -> tuple[np.ndarray, list[str], list[str]]:
    frames = [pd.read_csv(path, usecols=[TIME_COL, PROJECT_COL, MASK_COL]) for path in files]
    project_names = frames[0][PROJECT_COL].astype(str).tolist()
    timestamps = [pd.to_datetime(frame[TIME_COL].iloc[0]).strftime("%Y-%m-%d") for frame in frames]
    mask = np.stack(
        [pd.to_numeric(frame[MASK_COL], errors="coerce").fillna(0).to_numpy(dtype=np.float32) > 0 for frame in frames],
        axis=0,
    )
    return mask, project_names, timestamps


def find_biggest_interior_gap(mask: np.ndarray) -> tuple[int, int, int] | None:
    observed_idx = np.flatnonzero(mask)
    if observed_idx.size < 2:
        return None

    start_bound = int(observed_idx[0])
    end_bound = int(observed_idx[-1])
    best: tuple[int, int, int] | None = None
    cursor = start_bound
    while cursor <= end_bound:
        if mask[cursor]:
            cursor += 1
            continue
        gap_start = cursor
        while cursor <= end_bound and not mask[cursor]:
            cursor += 1
        gap_end = cursor
        gap_len = gap_end - gap_start
        if best is None or gap_len > best[2]:
            best = (gap_start, gap_end, gap_len)
    return best


def main() -> None:
    args = parse_args()
    files = list_timestep_files(args.timestep_dir)
    mask_matrix, project_names, timestamps = load_mask_matrix(files)

    rows: list[dict[str, object]] = []
    overall_best: dict[str, object] | None = None

    for project_idx, project_name in enumerate(project_names):
        gap = find_biggest_interior_gap(mask_matrix[:, project_idx])
        if gap is None:
            continue

        gap_start, gap_end, gap_len = gap
        row = {
            PROJECT_COL: project_name,
            "gap_start": timestamps[gap_start],
            "gap_end": timestamps[gap_end - 1],
            "gap_length": gap_len,
        }
        rows.append(row)
        if overall_best is None or int(row["gap_length"]) > int(overall_best["gap_length"]):
            overall_best = row

    result_df = pd.DataFrame(rows).sort_values(["gap_length", PROJECT_COL], ascending=[False, True])
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(args.output_path, index=False)

    print(f"Timestep directory: {args.timestep_dir}")
    print(f"Output CSV: {args.output_path}")
    print(f"Projects with interior gaps: {len(result_df)}")
    if overall_best is None:
        print("No interior gaps found.")
    else:
        print(
            "Biggest gap overall: "
            f"{overall_best[PROJECT_COL]} | "
            f"{overall_best['gap_start']} to {overall_best['gap_end']} | "
            f"length={overall_best['gap_length']}"
        )


if __name__ == "__main__":
    main()
