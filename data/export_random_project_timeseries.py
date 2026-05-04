from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd


DEFAULT_TIMESTEP_DIR = Path("dataset/ccr/timesteps_timesfm")
DEFAULT_OUTPUT_PATH = Path("dataset/ccr/timesfm_random_project_timeseries.csv")

PROJECT_COL = "Project Name"
TIME_COL = "Lease Commencement Date"
TARGET_COL = "rent_per_sqft"
MASK_COL = "y_mask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly choose one project from a timestep folder and export its full "
            "timeseries with rent_per_sqft and y_mask."
        )
    )
    parser.add_argument("--timestep-dir", type=Path, default=DEFAULT_TIMESTEP_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Optional project name. If omitted, choose one randomly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when project-name is not provided.",
    )
    return parser.parse_args()


def list_timestep_files(timestep_dir: Path) -> list[Path]:
    files = sorted(timestep_dir.glob("data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No timestep CSV files found in {timestep_dir}")
    return files


def choose_project_name(files: list[Path], requested_project: str | None, seed: int) -> str:
    first_frame = pd.read_csv(files[0], usecols=[PROJECT_COL])
    project_names = first_frame[PROJECT_COL].astype(str).tolist()
    if not project_names:
        raise ValueError(f"No project names found in {files[0]}")

    if requested_project is not None:
        if requested_project not in project_names:
            raise ValueError(f"Project '{requested_project}' not found in {files[0]}")
        return requested_project

    rng = random.Random(seed)
    return rng.choice(project_names)


def export_project_timeseries(files: list[Path], project_name: str, output_path: Path) -> None:
    records: list[dict[str, object]] = []

    for file_path in files:
        frame = pd.read_csv(file_path, usecols=[TIME_COL, PROJECT_COL, TARGET_COL, MASK_COL])
        row = frame.loc[frame[PROJECT_COL].astype(str) == project_name, [TIME_COL, TARGET_COL, MASK_COL]]
        if row.empty:
            raise ValueError(f"Project '{project_name}' missing in {file_path}")
        if len(row) != 1:
            raise ValueError(f"Expected one row for project '{project_name}' in {file_path}, got {len(row)}")

        records.append(
            {
                PROJECT_COL: project_name,
                TIME_COL: pd.to_datetime(row.iloc[0][TIME_COL]).strftime("%Y-%m-%d"),
                TARGET_COL: float(row.iloc[0][TARGET_COL]),
                MASK_COL: int(row.iloc[0][MASK_COL]),
            }
        )

    output_df = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"Timestep directory: {files[0].parent}")
    print(f"Project name: {project_name}")
    print(f"Output CSV: {output_path}")
    print(f"Rows written: {len(output_df)}")


def main() -> None:
    args = parse_args()
    files = list_timestep_files(args.timestep_dir)
    project_name = choose_project_name(files, args.project_name, args.seed)
    export_project_timeseries(files, project_name, args.output_path)


if __name__ == "__main__":
    main()
