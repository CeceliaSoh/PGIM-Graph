from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_PATH = Path("dataset/ccr/timesfm_example_project.csv")
DEFAULT_TIMESTEP_DIR = Path("dataset/ccr/timesteps")
DEFAULT_OUTPUT_PATH = Path("dataset/ccr/timesfm_example_project_with_mask.csv")

PROJECT_COL = "Project Name"
TIME_COL = "Lease Commencement Date"
TARGET_COL = "rent_per_sqft"
MASK_COL = "y_mask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add y_mask to an example project CSV by looking up the specified project "
            "across timestep files."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--timestep-dir", type=Path, default=DEFAULT_TIMESTEP_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--project-name",
        type=str,
        required=True,
        help="Project name to look up in the timestep CSV files.",
    )
    return parser.parse_args()


def load_example_csv(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Example CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    required = {TIME_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Example CSV missing required columns: {sorted(missing)}")
    return df


def collect_masks_for_project(timestep_dir: Path, project_name: str) -> pd.DataFrame:
    files = sorted(timestep_dir.glob("data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No timestep CSV files found in {timestep_dir}")

    records: list[dict[str, object]] = []
    found_any = False
    for file_path in files:
        frame = pd.read_csv(file_path, usecols=[TIME_COL, PROJECT_COL, MASK_COL])
        row = frame.loc[frame[PROJECT_COL].astype(str) == project_name, [TIME_COL, MASK_COL]]
        if row.empty:
            continue
        if len(row) != 1:
            raise ValueError(f"Expected one row for project '{project_name}' in {file_path}, got {len(row)}")
        found_any = True
        records.append(
            {
                TIME_COL: pd.to_datetime(row.iloc[0][TIME_COL]).strftime("%Y-%m-%d"),
                MASK_COL: float(row.iloc[0][MASK_COL]),
            }
        )

    if not found_any:
        raise ValueError(f"Project '{project_name}' was not found in timestep directory {timestep_dir}")

    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()

    example_df = load_example_csv(args.input_path).copy()
    example_df[TIME_COL] = pd.to_datetime(example_df[TIME_COL]).dt.strftime("%Y-%m-%d")

    mask_df = collect_masks_for_project(args.timestep_dir, args.project_name)
    merged = example_df.merge(mask_df, on=TIME_COL, how="left", validate="one_to_one")

    if merged[MASK_COL].isna().any():
        missing_dates = merged.loc[merged[MASK_COL].isna(), TIME_COL].tolist()
        raise ValueError(
            "Could not find y_mask for some example timestamps. "
            f"Missing dates: {missing_dates[:10]}"
        )

    merged.insert(0, PROJECT_COL, args.project_name)
    merged[mask_col := MASK_COL] = merged[mask_col].astype(int)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_path, index=False)

    print(f"Input example CSV: {args.input_path}")
    print(f"Timestep directory: {args.timestep_dir}")
    print(f"Project name: {args.project_name}")
    print(f"Output CSV: {args.output_path}")
    print(f"Rows written: {len(merged)}")


if __name__ == "__main__":
    main()
