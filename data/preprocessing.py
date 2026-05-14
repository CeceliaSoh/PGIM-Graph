from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


STANDARD_COLUMNS = ["timestep", "node_id", "project_id", "rent_per_sqft", "y_mask"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess rental forecasting data.")
    parser.add_argument(
        "--config",
        default=Path("data/config.yaml"),
        type=Path,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return config


def setup_logging(output_folder: Path) -> logging.Logger:
    output_folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprocessing")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(output_folder / "preprocessing.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def record_warning(message: str, logger: logging.Logger, warning_messages: list[str]) -> None:
    warnings.warn(message, stacklevel=2)
    logger.warning(message)
    warning_messages.append(message)


def load_input_csv(input_path: Path, logger: logging.Logger) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    logger.info("Loaded input CSV from %s with shape %s", input_path, df.shape)
    return df


def validate_required_columns(df: pd.DataFrame, config: dict[str, Any]) -> None:
    missing = []
    for standard_name in STANDARD_COLUMNS:
        source_name = config.get(standard_name)
        if source_name is None:
            missing.append(f"{standard_name} (missing config key)")
        elif source_name not in df.columns:
            missing.append(f"{standard_name}: {source_name}")
    if missing:
        raise ValueError("Missing required configured columns before renaming: " + ", ".join(missing))


def rename_standard_columns(
    df: pd.DataFrame, config: dict[str, Any], logger: logging.Logger
) -> tuple[pd.DataFrame, dict[str, str]]:
    validate_required_columns(df, config)
    rename_map = {
        config[standard_name]: standard_name
        for standard_name in STANDARD_COLUMNS
        if config[standard_name] != standard_name
    }
    df = df.rename(columns=rename_map)
    logger.info("Renamed columns: %s", rename_map)
    logger.info("Shape after renaming columns: %s", df.shape)
    return df, rename_map


def is_integer_series(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.notna().all() and np.isclose(numeric % 1, 0).all()


def ensure_integer_node_id(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if df["node_id"].isna().any():
        raise ValueError("node_id contains missing values before integer conversion.")
    if not is_integer_series(df["node_id"]):
        raise ValueError("Configured node_id column must contain integer-like values.")
    df["node_id"] = pd.to_numeric(df["node_id"], errors="raise").astype(int)
    logger.info("Ensured node_id is integer.")
    logger.info("Shape after node_id validation: %s", df.shape)
    return df


def ensure_integer_timestep(
    df: pd.DataFrame, output_folder: Path, logger: logging.Logger
) -> tuple[pd.DataFrame, Path | None]:
    mapping_path = None
    if df["timestep"].isna().any():
        raise ValueError("timestep contains missing values before integer conversion.")

    parsed_dates = pd.to_datetime(df["timestep"], errors="coerce")
    if is_integer_series(df["timestep"]):
        df["timestep"] = pd.to_numeric(df["timestep"], errors="raise").astype(int)
        if "date" in df.columns:
            mapping = (
                df[["timestep", "date"]]
                .drop_duplicates()
                .sort_values("timestep")
                .rename(columns={"date": "date"})
            )
            mapping_path = output_folder / "timestep.csv"
            mapping.to_csv(mapping_path, index=False)
            logger.info("Saved timestep-date mapping to %s", mapping_path)
    elif parsed_dates.notna().all():
        unique_dates = pd.Series(parsed_dates.drop_duplicates().sort_values().to_numpy())
        date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
        original_dates = parsed_dates.copy()
        df["timestep"] = original_dates.map(date_to_idx).astype(int)
        mapping = pd.DataFrame(
            {
                "timestep": range(len(unique_dates)),
                "date": unique_dates.dt.strftime("%Y-%m-%d"),
            }
        )
        mapping_path = output_folder / "timestep.csv"
        mapping.to_csv(mapping_path, index=False)
        logger.info("Converted date-like timestep to integer IDs.")
        logger.info("Saved timestep-date mapping to %s", mapping_path)
    else:
        raise ValueError("timestep must be integer-like or date-like with no unknown date values.")

    logger.info("Ensured timestep is integer.")
    logger.info("Shape after timestep validation: %s", df.shape)
    return df, mapping_path


def ensure_integer_project_id(
    df: pd.DataFrame, output_folder: Path, logger: logging.Logger
) -> tuple[pd.DataFrame, Path | None]:
    mapping_path = None
    if df["project_id"].isna().any():
        raise ValueError("project_id contains missing values before integer conversion.")

    if is_integer_series(df["project_id"]):
        df["project_id"] = pd.to_numeric(df["project_id"], errors="raise").astype(int)
        logger.info("Ensured project_id is integer.")
    else:
        unique_ids = sorted(df["project_id"].astype(str).unique())
        id_to_idx = {project_id: idx for idx, project_id in enumerate(unique_ids)}
        mapping = pd.DataFrame({"project_id": range(len(unique_ids)), "original_project_id": unique_ids})
        df["project_id"] = df["project_id"].astype(str).map(id_to_idx).astype(int)
        mapping_path = output_folder / "project_id.csv"
        mapping.to_csv(mapping_path, index=False)
        logger.info("Converted project_id to integer IDs and saved mapping to %s", mapping_path)

    logger.info("Shape after project_id validation: %s", df.shape)
    return df, mapping_path


def convert_y_mask(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    mapping = {
        True: 1,
        False: 0,
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "1": 1,
        "0": 0,
        1: 1,
        0: 0,
        1.0: 1,
        0.0: 0,
    }

    def convert(value: Any) -> int | float:
        if pd.isna(value):
            return np.nan
        if isinstance(value, str):
            key = value.strip().lower()
        else:
            key = value
        return mapping.get(key, np.nan)

    converted = df["y_mask"].map(convert)
    if converted.isna().any():
        unknown = sorted(df.loc[converted.isna(), "y_mask"].dropna().astype(str).unique())
        raise ValueError(f"y_mask contains unknown values: {unknown}")
    df["y_mask"] = converted.astype(int)
    logger.info("Converted y_mask to binary 0/1.")
    logger.info("Shape after y_mask conversion: %s", df.shape)
    return df


def columns_matching_keywords(columns: pd.Index, keywords: list[str]) -> list[str]:
    return [col for col in columns if any(keyword in col for keyword in keywords)]


def fill_max_clip_columns(
    df: pd.DataFrame, config: dict[str, Any], logger: logging.Logger
) -> tuple[pd.DataFrame, list[str], list[str]]:
    max_clip_col_columns = columns_matching_keywords(df.columns, config.get("max_clip_col", []))
    for col in max_clip_col_columns:
        max_value = pd.to_numeric(df[col], errors="coerce").max()
        if pd.isna(max_value):
            raise ValueError(f"Column {col} matched max_clip_col but has no numeric max value.")
        df[col] = df[col].fillna(np.ceil(max_value))

    max_clip_glo_columns = columns_matching_keywords(df.columns, config.get("max_clip_glo", []))
    if max_clip_glo_columns:
        numeric_values = df[max_clip_glo_columns].apply(pd.to_numeric, errors="coerce")
        global_max = numeric_values.max().max()
        if pd.isna(global_max):
            raise ValueError("Columns matched max_clip_glo but have no numeric global max value.")
        df[max_clip_glo_columns] = df[max_clip_glo_columns].fillna(np.ceil(global_max))

    logger.info("max_clip_col columns filled: %s", max_clip_col_columns)
    logger.info("max_clip_glo columns filled: %s", max_clip_glo_columns)
    logger.info("Shape after max clipping fills: %s", df.shape)
    return df, max_clip_col_columns, max_clip_glo_columns


def duplicate_feature_columns(
    df: pd.DataFrame, config: dict[str, Any], logger: logging.Logger
) -> pd.DataFrame:
    missing = [col for col in config.get("as_feat", []) if col not in df.columns]
    if missing:
        raise ValueError(f"Configured as_feat columns are missing: {missing}")
    for col in config.get("as_feat", []):
        df[f"{col}_feat"] = df[col]
    logger.info("Duplicated feature columns: %s", config.get("as_feat", []))
    logger.info("Shape after duplicating feature columns: %s", df.shape)
    return df


def drop_configured_columns(
    df: pd.DataFrame, config: dict[str, Any], logger: logging.Logger
) -> tuple[pd.DataFrame, list[str], list[str]]:
    to_drop = config.get("to_drop", [])
    dropped = [col for col in to_drop if col in df.columns]
    skipped = [col for col in to_drop if col not in df.columns]
    df = df.drop(columns=dropped)
    logger.info("Dropped columns: %s", dropped)
    logger.info("Skipped missing drop columns: %s", skipped)
    logger.info("Shape after dropping columns: %s", df.shape)
    return df, dropped, skipped


def encode_categorical_columns(
    df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
    warning_messages: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    configured = config.get("categorical", [])
    existing_configured = [col for col in configured if col in df.columns]
    missing_configured = [col for col in configured if col not in df.columns]
    if missing_configured:
        logger.info("Configured categorical columns not found and skipped: %s", missing_configured)
    if existing_configured:
        df = pd.get_dummies(df, columns=existing_configured, dummy_na=False, dtype=int)
    logger.info("One-hot encoded configured categorical columns: %s", existing_configured)
    logger.info("Shape after configured categorical encoding: %s", df.shape)

    non_numeric = df.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
    if non_numeric:
        message = f"Remaining non-numerical columns detected and one-hot encoded: {non_numeric}"
        record_warning(message, logger, warning_messages)
        df = pd.get_dummies(df, columns=non_numeric, dummy_na=False, dtype=int)
    logger.info("Shape after extra categorical encoding: %s", df.shape)
    return df, existing_configured, non_numeric


def reorder_columns(
    df: pd.DataFrame,
    config: dict[str, Any],
    logger: logging.Logger,
    warning_messages: list[str],
) -> pd.DataFrame:
    first_existing = [col for col in config.get("first_few", []) if col in df.columns]
    last_existing = [col for col in config.get("last_few", []) if col in df.columns]
    missing = [
        col
        for col in config.get("first_few", []) + config.get("last_few", [])
        if col not in df.columns
    ]
    if missing:
        record_warning(f"Configured reorder columns missing and skipped: {missing}", logger, warning_messages)

    edge_columns = set(first_existing + last_existing)
    middle = [col for col in df.columns if col not in edge_columns]
    df = df[first_existing + middle + last_existing]
    logger.info("Reordered columns with first=%s and last=%s", first_existing, last_existing)
    logger.info("Shape after reordering columns: %s", df.shape)
    return df


def sort_dataframe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    df = df.sort_values(["node_id", "timestep"]).reset_index(drop=True)
    logger.info("Sorted dataframe by node_id and timestep.")
    logger.info("Shape after sorting: %s", df.shape)
    return df


def run_correctness_checks(df: pd.DataFrame, output_folder: Path, logger: logging.Logger) -> None:
    for col in ["node_id", "timestep", "rent_per_sqft"]:
        if df[col].isna().any():
            raise ValueError(f"{col} has missing values.")
    if not set(df["y_mask"].unique()).issubset({0, 1}):
        raise ValueError("y_mask contains values outside 0 and 1.")

    duplicated = df.duplicated(["node_id", "timestep"], keep=False)
    if duplicated.any():
        duplicate_path = output_folder / "duplicated_node_timestep.csv"
        df.loc[duplicated].to_csv(duplicate_path, index=False)
        logger.info("Saved duplicated node-timestep rows to %s", duplicate_path)
        raise ValueError(f"Duplicated rows for node_id and timestep saved to {duplicate_path}")
    logger.info("Correctness checks passed.")


def save_full_dataframe(df: pd.DataFrame, output_folder: Path, logger: logging.Logger) -> Path:
    path = output_folder / "all_nodes.csv"
    df.to_csv(path, index=False)
    logger.info("Saved final full dataframe to %s", path)
    return path


def save_node_files(df: pd.DataFrame, output_folder: Path, logger: logging.Logger) -> list[Path]:
    nodes_folder = output_folder / "nodes"
    nodes_folder.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for node_id, node_df in df.groupby("node_id", sort=True):
        path = nodes_folder / f"{int(node_id):04d}.csv"
        node_df.sort_values("timestep").to_csv(path, index=False)
        saved_paths.append(path)
        logger.info("Saved node file: %s", path)
    logger.info("Saved %d node files under %s", len(saved_paths), nodes_folder)
    return saved_paths


def save_feature_columns(df: pd.DataFrame, output_folder: Path, logger: logging.Logger) -> Path:
    path = output_folder / "feature_columns.csv"
    pd.DataFrame({"feature_column": df.columns}).to_csv(path, index=False)
    logger.info("Saved feature columns to %s", path)
    return path


def save_summary(summary: dict[str, Any], output_folder: Path, logger: logging.Logger) -> Path:
    path = output_folder / "preprocessing_summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved preprocessing summary to %s", path)
    return path


def path_strings(paths: list[Path]) -> list[str]:
    return [str(path) for path in paths]


def preprocess(config: dict[str, Any], logger: logging.Logger) -> dict[str, Any]:
    output_folder = Path(config["output_folder"])
    output_folder.mkdir(parents=True, exist_ok=True)
    warning_messages: list[str] = []
    generated_files: list[Path] = [output_folder / "preprocessing.log"]

    input_path = Path(config["input_folder"])
    df = load_input_csv(input_path, logger)
    input_shape = list(df.shape)

    df, renamed_columns = rename_standard_columns(df, config, logger)
    df = ensure_integer_node_id(df, logger)
    df, timestep_mapping_path = ensure_integer_timestep(df, output_folder, logger)
    if timestep_mapping_path is not None:
        generated_files.append(timestep_mapping_path)
    df, project_mapping_path = ensure_integer_project_id(df, output_folder, logger)
    if project_mapping_path is not None:
        generated_files.append(project_mapping_path)
    df = convert_y_mask(df, logger)
    df, max_clip_col_columns, max_clip_glo_columns = fill_max_clip_columns(df, config, logger)
    df = duplicate_feature_columns(df, config, logger)
    df, dropped_columns, skipped_drop_columns = drop_configured_columns(df, config, logger)
    df, encoded_categorical, extra_categorical = encode_categorical_columns(
        df, config, logger, warning_messages
    )
    df = reorder_columns(df, config, logger, warning_messages)
    df = sort_dataframe(df, logger)
    run_correctness_checks(df, output_folder, logger)

    all_nodes_path = save_full_dataframe(df, output_folder, logger)
    generated_files.append(all_nodes_path)
    node_paths = save_node_files(df, output_folder, logger)
    generated_files.extend(node_paths)
    feature_columns_path = save_feature_columns(df, output_folder, logger)
    generated_files.append(feature_columns_path)

    summary = {
        "input_path": str(input_path),
        "output_folder": str(output_folder),
        "input_shape": input_shape,
        "final_shape": list(df.shape),
        "number_of_nodes": int(df["node_id"].nunique()),
        "number_of_timesteps": int(df["timestep"].nunique()),
        "number_of_final_columns": int(df.shape[1]),
        "renamed_columns": renamed_columns,
        "dropped_columns": dropped_columns,
        "skipped_drop_columns": skipped_drop_columns,
        "configured_categorical_columns": encoded_categorical,
        "extra_categorical_columns_detected": extra_categorical,
        "max_clip_col_columns": max_clip_col_columns,
        "max_clip_glo_columns": max_clip_glo_columns,
        "generated_files": path_strings(generated_files),
        "warnings": warning_messages,
    }
    summary_path = output_folder / "preprocessing_summary.json"
    summary["generated_files"].append(str(summary_path))
    save_summary(summary, output_folder, logger)
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_folder = Path(config["output_folder"])
    logger = setup_logging(output_folder)
    logger.info("Starting preprocessing with config %s", args.config)
    preprocess(config, logger)
    logger.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
