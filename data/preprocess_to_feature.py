from pathlib import Path

import numpy as np
import pandas as pd


TIME_STEP_DIR = Path("dataset/ccr/timesteps")
OUTPUT_PATH = Path("dataset/ccr/feature.npy")
METADATA_DIR = Path("dataset/ccr/feature_metadata")
FEATURE_COLUMNS_PATH = METADATA_DIR / "feature_columns.txt"
PROJECT_NAMES_PATH = METADATA_DIR / "project_names.txt"
TIMESTAMPS_PATH = METADATA_DIR / "timestamps.txt"

PROJECT_COL = "Project Name"
TIME_COL = "Lease Commencement Date"
TARGET_COL = "rent_per_sqft"
TARGET_MASK_COL = "y_mask"


def list_timestep_files(time_step_dir: Path) -> list[Path]:
    files = sorted(time_step_dir.glob("data_*.csv"))
    if not files:
        raise FileNotFoundError(f"No timestep CSV files found in {time_step_dir}")
    return files


def ordered_feature_columns(columns: list[str]) -> list[str]:
    excluded = {PROJECT_COL, TIME_COL, TARGET_COL, TARGET_MASK_COL}
    feature_columns = [column for column in columns if column not in excluded]
    feature_columns.extend([TARGET_COL, TARGET_MASK_COL])
    return feature_columns


def validate_timestep_frame(df: pd.DataFrame, expected_columns: list[str] | None = None) -> None:
    missing_columns = {PROJECT_COL, TIME_COL, TARGET_COL, TARGET_MASK_COL} - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if expected_columns is not None and list(df.columns) != expected_columns:
        raise ValueError("Inconsistent columns across timestep files")

    if list(df.columns)[-2:] != [TARGET_COL, TARGET_MASK_COL]:
        raise ValueError(
            f"Expected last two columns to be [{TARGET_COL}, {TARGET_MASK_COL}], "
            f"got {list(df.columns)[-2:]}"
        )

    if df[PROJECT_COL].duplicated().any():
        duplicated_projects = df.loc[df[PROJECT_COL].duplicated(), PROJECT_COL].tolist()
        raise ValueError(f"Duplicate projects found within a timestep file: {duplicated_projects[:10]}")


def build_feature_tensor(time_step_dir: Path) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    files = list_timestep_files(time_step_dir)
    first_df = pd.read_csv(files[0])
    validate_timestep_frame(first_df)

    expected_columns = list(first_df.columns)
    feature_columns = ordered_feature_columns(expected_columns)
    project_list = sorted(first_df[PROJECT_COL].astype(str).tolist())
    project_to_idx = {project: index for index, project in enumerate(project_list)}
    timestamp_list = []

    num_nodes = len(project_list)
    num_timesteps = len(files)
    num_features = len(feature_columns)
    feature_array = np.zeros((num_nodes, num_timesteps, num_features), dtype=np.float32)

    for time_index, file_path in enumerate(files):
        df = pd.read_csv(file_path)
        validate_timestep_frame(df, expected_columns=expected_columns)

        df = df.copy()
        df[PROJECT_COL] = df[PROJECT_COL].astype(str)
        timestamp_value = pd.to_datetime(df[TIME_COL], errors="coerce")
        if timestamp_value.isna().any():
            raise ValueError(f"Invalid timestamp values found in {file_path}")
        unique_timestamps = timestamp_value.dt.strftime("%Y-%m-%d").unique().tolist()
        if len(unique_timestamps) != 1:
            raise ValueError(f"Expected one timestamp per file in {file_path}, got {unique_timestamps}")
        timestamp_list.append(unique_timestamps[0])

        file_projects = set(df[PROJECT_COL].tolist())
        expected_projects = set(project_list)
        if file_projects != expected_projects:
            missing_projects = sorted(expected_projects - file_projects)
            extra_projects = sorted(file_projects - expected_projects)
            raise ValueError(
                f"Project set mismatch in {file_path}. "
                f"Missing={missing_projects[:10]}, extra={extra_projects[:10]}"
            )

        aligned_df = df.set_index(PROJECT_COL).loc[project_list].reset_index()
        values = aligned_df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
        feature_array[:, time_index, :] = values

        # Sanity-check the alignment for a few rows per timestep.
        for row_index in (0, num_nodes // 2, num_nodes - 1):
            project_name = aligned_df.iloc[row_index][PROJECT_COL]
            if project_to_idx[project_name] != row_index:
                raise ValueError(f"Project alignment mismatch for {project_name} in {file_path}")

    return feature_array, feature_columns, project_list, timestamp_list


def save_metadata(feature_columns: list[str], project_list: list[str], timestamp_list: list[str]) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_COLUMNS_PATH.write_text("\n".join(feature_columns) + "\n", encoding="utf-8")
    PROJECT_NAMES_PATH.write_text("\n".join(project_list) + "\n", encoding="utf-8")
    TIMESTAMPS_PATH.write_text("\n".join(timestamp_list) + "\n", encoding="utf-8")


def main() -> None:
    feature_array, feature_columns, project_list, timestamp_list = build_feature_tensor(TIME_STEP_DIR)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, feature_array)
    save_metadata(feature_columns, project_list, timestamp_list)

    print(f"Saved features to: {OUTPUT_PATH}")
    print(f"Feature shape [N, T, D]: {feature_array.shape}")
    print(f"Projects (N): {len(project_list)}")
    print(f"Timesteps (T): {len(timestamp_list)}")
    print(f"Feature dimension (D): {len(feature_columns)}")
    print(f"Last two columns: {feature_columns[-2:]}")
    print(f"First timestep: {timestamp_list[0]}")
    print(f"Last timestep: {timestamp_list[-1]}")


if __name__ == "__main__":
    main()
