from pathlib import Path
import argparse

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
METADATA_PATH = BASE_DIR / "dataset" / "ccr" / "metadata.csv"
MRT_PATH = BASE_DIR / "dataset" / "Rail_Transport.csv"
DISTANCE_THRESHOLD_M = 300.0
PROJECT_COL = "project_name"
LAT_COL = "latitude"
LON_COL = "longitude"
MRT_NAME_COL = "station_name_english"
RULE_DISTANCE = "distance"
RULE_DISTANCE_OR_MRT_RADIUS = "distance_or_mrt_radius"


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metadata_path)
    required_columns = {PROJECT_COL, LAT_COL, LON_COL}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    df = (
        df[[PROJECT_COL, LAT_COL, LON_COL]]
        .dropna(subset=[PROJECT_COL, LAT_COL, LON_COL])
        .drop_duplicates(subset=[PROJECT_COL], keep="first")
        .sort_values(PROJECT_COL)
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError(f"No valid project rows found in {metadata_path}")
    return df


def load_mrt_data(mrt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(mrt_path)
    required_columns = {MRT_NAME_COL, LAT_COL, LON_COL}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required MRT columns: {sorted(missing_columns)}")

    df = (
        df[[MRT_NAME_COL, LAT_COL, LON_COL]]
        .dropna(subset=[MRT_NAME_COL, LAT_COL, LON_COL])
        .drop_duplicates(subset=[MRT_NAME_COL], keep="first")
        .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError(f"No valid MRT rows found in {mrt_path}")
    return df


def haversine_distance_matrix(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    earth_radius_m = 6371000.0
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)

    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return earth_radius_m * c


def haversine_cross_distance_matrix(
    source_lat_deg: np.ndarray,
    source_lon_deg: np.ndarray,
    target_lat_deg: np.ndarray,
    target_lon_deg: np.ndarray,
) -> np.ndarray:
    earth_radius_m = 6371000.0
    source_lat_rad = np.radians(source_lat_deg)
    source_lon_rad = np.radians(source_lon_deg)
    target_lat_rad = np.radians(target_lat_deg)
    target_lon_rad = np.radians(target_lon_deg)

    dlat = source_lat_rad[:, None] - target_lat_rad[None, :]
    dlon = source_lon_rad[:, None] - target_lon_rad[None, :]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(source_lat_rad[:, None])
        * np.cos(target_lat_rad[None, :])
        * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return earth_radius_m * c


def build_distance_edge_frame(df: pd.DataFrame, distance_threshold_m: float) -> pd.DataFrame:
    project_names = df[PROJECT_COL].astype(str).tolist()
    latitudes = df[LAT_COL].to_numpy(dtype=np.float64)
    longitudes = df[LON_COL].to_numpy(dtype=np.float64)
    distance_matrix = haversine_distance_matrix(latitudes, longitudes)
    node_count = len(project_names)
    edge_records = []

    for source_index in range(node_count):
        for target_index in range(source_index + 1, node_count):
            distance_m = distance_matrix[source_index, target_index]
            if distance_m <= distance_threshold_m:
                edge_records.append(
                    {
                        "source_index": source_index,
                        "target_index": target_index,
                        "source_project": project_names[source_index],
                        "target_project": project_names[target_index],
                        "distance_m": float(distance_m),
                        "rules": RULE_DISTANCE,
                        "shared_mrt": "",
                    }
                )

    return pd.DataFrame(
        edge_records,
        columns=[
            "source_index",
            "target_index",
            "source_project",
            "target_project",
            "distance_m",
            "rules",
            "shared_mrt",
        ],
    )


def build_mrt_radius_edge_frame(
    project_df: pd.DataFrame, mrt_df: pd.DataFrame, distance_threshold_m: float
) -> pd.DataFrame:
    project_names = project_df[PROJECT_COL].astype(str).tolist()
    project_latitudes = project_df[LAT_COL].to_numpy(dtype=np.float64)
    project_longitudes = project_df[LON_COL].to_numpy(dtype=np.float64)
    mrt_names = mrt_df[MRT_NAME_COL].astype(str).tolist()
    mrt_latitudes = mrt_df[LAT_COL].to_numpy(dtype=np.float64)
    mrt_longitudes = mrt_df[LON_COL].to_numpy(dtype=np.float64)
    distance_matrix = haversine_cross_distance_matrix(
        project_latitudes,
        project_longitudes,
        mrt_latitudes,
        mrt_longitudes,
    )
    mrt_to_projects: dict[str, list[int]] = {}

    for project_index in range(len(project_names)):
        for mrt_index, mrt_name in enumerate(mrt_names):
            distance_m = distance_matrix[project_index, mrt_index]
            if distance_m <= distance_threshold_m:
                mrt_to_projects.setdefault(mrt_name, []).append(project_index)

    edge_records = []
    seen_edges: set[tuple[int, int]] = set()

    for mrt_name, project_indices in mrt_to_projects.items():
        unique_project_indices = sorted(set(project_indices))
        for left_pos, source_index in enumerate(unique_project_indices):
            for target_index in unique_project_indices[left_pos + 1 :]:
                edge_key = (source_index, target_index)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                edge_records.append(
                    {
                        "source_index": source_index,
                        "target_index": target_index,
                        "source_project": project_names[source_index],
                        "target_project": project_names[target_index],
                        "distance_m": np.nan,
                        "rules": "mrt_radius",
                        "shared_mrt": mrt_name,
                    }
                )

    return pd.DataFrame(
        edge_records,
        columns=[
            "source_index",
            "target_index",
            "source_project",
            "target_project",
            "distance_m",
            "rules",
            "shared_mrt",
        ],
    )


def build_combined_edge_frame(
    project_df: pd.DataFrame, mrt_df: pd.DataFrame, distance_threshold_m: float
) -> tuple[pd.DataFrame, int, int]:
    distance_edge_frame = build_distance_edge_frame(project_df, distance_threshold_m)
    mrt_radius_edge_frame = build_mrt_radius_edge_frame(
        project_df, mrt_df, distance_threshold_m
    )

    combined_edge_frame = pd.concat(
        [distance_edge_frame, mrt_radius_edge_frame], ignore_index=True, sort=False
    )
    if combined_edge_frame.empty:
        return combined_edge_frame, 0, 0

    aggregated_records = []
    for _, group in combined_edge_frame.groupby(["source_index", "target_index"], sort=True):
        first_row = group.iloc[0].copy()
        unique_rules = sorted(set(group["rules"].dropna().astype(str)))
        shared_mrt_values = sorted(
            {value for value in group["shared_mrt"].dropna().astype(str) if value}
        )
        distance_values = group["distance_m"].dropna()

        first_row["rules"] = "+".join(unique_rules)
        first_row["shared_mrt"] = "|".join(shared_mrt_values)
        first_row["distance_m"] = (
            float(distance_values.min()) if not distance_values.empty else np.nan
        )
        aggregated_records.append(first_row.to_dict())

    return (
        pd.DataFrame(aggregated_records).reset_index(drop=True),
        len(distance_edge_frame),
        len(mrt_radius_edge_frame),
    )


def get_output_paths(rule: str, distance_threshold_m: float) -> tuple[Path, Path, Path]:
    threshold_label = f"{int(distance_threshold_m)}m"
    if rule == RULE_DISTANCE:
        output_dir = BASE_DIR / "dataset" / "ccr" / f"graph_link_{threshold_label}"
    else:
        output_dir = (
            BASE_DIR
            / "dataset"
            / "ccr"
            / f"graph_link_{threshold_label}_{RULE_DISTANCE_OR_MRT_RADIUS}"
        )
    return output_dir, output_dir / "links.csv", output_dir / "links.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build project graph edges from direct distance or MRT-radius rules."
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=METADATA_PATH,
        help="Path to the project metadata CSV.",
    )
    parser.add_argument(
        "--mrt-path",
        type=Path,
        default=MRT_PATH,
        help="Path to the MRT CSV used by the MRT-radius rule.",
    )
    parser.add_argument(
        "--distance-threshold-m",
        type=float,
        default=DISTANCE_THRESHOLD_M,
        help="Distance threshold in meters.",
    )
    parser.add_argument(
        "--rule",
        choices=[RULE_DISTANCE, RULE_DISTANCE_OR_MRT_RADIUS],
        default=RULE_DISTANCE,
        help="Edge rule to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_metadata(args.metadata_path)

    if args.rule == RULE_DISTANCE:
        edge_frame = build_distance_edge_frame(metadata, args.distance_threshold_m)
        distance_edge_count = len(edge_frame)
        mrt_radius_edge_count = 0
    else:
        mrt_data = load_mrt_data(args.mrt_path)
        edge_frame, distance_edge_count, mrt_radius_edge_count = build_combined_edge_frame(
            metadata, mrt_data, args.distance_threshold_m
        )

    output_dir, csv_output_path, txt_output_path = get_output_paths(
        args.rule, args.distance_threshold_m
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    edge_frame.to_csv(csv_output_path, index=False)

    txt_lines = [f"{row.source_index},{row.target_index}" for row in edge_frame.itertuples(index=False)]
    txt_output_path.write_text("\n".join(txt_lines) + ("\n" if txt_lines else ""), encoding="utf-8")

    print(f"Loaded metadata from: {args.metadata_path}")
    print(f"Projects: {len(metadata)}")
    print(f"Edge rule: {args.rule}")
    print(f"Distance threshold (m): {args.distance_threshold_m}")
    if args.rule == RULE_DISTANCE_OR_MRT_RADIUS:
        print(f"Loaded MRT data from: {args.mrt_path}")
        print(f"Direct distance edges: {distance_edge_count}")
        print(f"MRT-radius edges: {mrt_radius_edge_count}")
    print(f"Edges written: {len(edge_frame)}")
    print(f"Saved graph CSV to: {csv_output_path}")
    print(f"Saved graph TXT to: {txt_output_path}")


if __name__ == "__main__":
    main()
