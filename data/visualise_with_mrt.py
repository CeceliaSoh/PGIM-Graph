from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_CSV_PATH = (
    BASE_DIR / "dataset" / "URA_merged_ccr_project_coverage_check.csv"
)
DEFAULT_MRT_CSV_PATH = BASE_DIR / "dataset" / "mongodb_exports" / "Rail_Transport.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "dataset" / "singapore_project_mrt_map.png"
DEFAULT_DISTANCE_THRESHOLD_M = 1000.0


def load_project_file(project_csv_path: Path) -> pd.DataFrame:
    if not project_csv_path.exists():
        raise FileNotFoundError(f"Project CSV file was not found: {project_csv_path}")

    df = pd.read_csv(project_csv_path, dtype=str)
    if df.empty:
        raise ValueError(f"Project CSV file is empty: {project_csv_path}")

    expected_columns = {"project_name", "latitude", "longitude"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Project CSV is missing required columns: {sorted(missing_columns)}"
        )

    df["project_name"] = df["project_name"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    if df.empty:
        raise ValueError("No valid latitude/longitude rows were found in the CSV.")

    return df.drop_duplicates(subset=["project_name"]).reset_index(drop=True)


def load_mrt_file(mrt_csv_path: Path) -> pd.DataFrame:
    if not mrt_csv_path.exists():
        raise FileNotFoundError(f"MRT CSV file was not found: {mrt_csv_path}")

    df = pd.read_csv(mrt_csv_path, dtype=str)
    if df.empty:
        raise ValueError(f"MRT CSV file is empty: {mrt_csv_path}")

    expected_columns = {"station_name_english", "latitude", "longitude"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"MRT CSV is missing required columns: {sorted(missing_columns)}")

    df["station_name_english"] = df["station_name_english"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    if df.empty:
        raise ValueError("No valid MRT latitude/longitude rows were found in the CSV.")

    return df.drop_duplicates(subset=["station_name_english"]).reset_index(drop=True)


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


def build_project_mrt_edge_frame(
    project_df: pd.DataFrame, mrt_df: pd.DataFrame, distance_threshold_m: float
) -> pd.DataFrame:
    project_names = project_df["project_name"].astype(str).tolist()
    project_latitudes = project_df["latitude"].to_numpy(dtype=np.float64)
    project_longitudes = project_df["longitude"].to_numpy(dtype=np.float64)
    mrt_names = mrt_df["station_name_english"].astype(str).tolist()
    mrt_latitudes = mrt_df["latitude"].to_numpy(dtype=np.float64)
    mrt_longitudes = mrt_df["longitude"].to_numpy(dtype=np.float64)
    distance_matrix = haversine_cross_distance_matrix(
        project_latitudes,
        project_longitudes,
        mrt_latitudes,
        mrt_longitudes,
    )
    edge_records = []

    for project_index, project_name in enumerate(project_names):
        for mrt_index, mrt_name in enumerate(mrt_names):
            distance_m = distance_matrix[project_index, mrt_index]
            if distance_m <= distance_threshold_m:
                edge_records.append(
                    {
                        "project_name": project_name,
                        "mrt_name": mrt_name,
                        "project_latitude": float(project_latitudes[project_index]),
                        "project_longitude": float(project_longitudes[project_index]),
                        "mrt_latitude": float(mrt_latitudes[mrt_index]),
                        "mrt_longitude": float(mrt_longitudes[mrt_index]),
                        "distance_m": float(distance_m),
                    }
                )

    return pd.DataFrame(edge_records)


def build_map_figure(
    project_df: pd.DataFrame, mrt_df: pd.DataFrame, edge_df: pd.DataFrame, distance_threshold_m: float
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#eef6fb")

    if not edge_df.empty:
        segments = [
            [
                (row.project_longitude, row.project_latitude),
                (row.mrt_longitude, row.mrt_latitude),
            ]
            for row in edge_df.itertuples(index=False)
        ]
        line_collection = LineCollection(
            segments,
            linewidths=0.8,
            colors="#2f5d8a",
            alpha=0.30,
            zorder=1,
        )
        ax.add_collection(line_collection)

    ax.scatter(
        project_df["longitude"],
        project_df["latitude"],
        s=18,
        c="#d42b2b",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8,
        zorder=3,
    )
    ax.scatter(
        mrt_df["longitude"],
        mrt_df["latitude"],
        s=42,
        c="#1f77b4",
        marker="^",
        edgecolor="white",
        linewidth=0.6,
        alpha=0.95,
        zorder=4,
    )

    ax.set_title("Singapore Project to MRT Links")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    lon_mean = float(project_df["longitude"].mean())
    lat_mean = float(project_df["latitude"].mean())
    lon_std = float(project_df["longitude"].std())
    lat_std = float(project_df["latitude"].std())

    lon_half_range = max(0.002, 2.0 * lon_std)
    lat_half_range = max(0.002, 2.0 * lat_std)

    ax.set_xlim(lon_mean - lon_half_range, lon_mean + lon_half_range)
    ax.set_ylim(lat_mean - lat_half_range, lat_mean + lat_half_range)

    ax.text(
        0.01,
        0.98,
        (
            f"Projects: {len(project_df)} | MRT: {len(mrt_df)} | "
            f"Links: {len(edge_df)} | Threshold: {distance_threshold_m:.0f}m"
        ),
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot project-to-MRT links when projects are within a distance threshold of MRT stations."
    )
    parser.add_argument(
        "--project-file",
        type=Path,
        default=DEFAULT_PROJECT_CSV_PATH,
        help="Path to the CSV file containing project_name, latitude, and longitude.",
    )
    parser.add_argument(
        "--mrt-file",
        type=Path,
        default=DEFAULT_MRT_CSV_PATH,
        help="Path to the MRT CSV file containing station_name_english, latitude, and longitude.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the generated map image.",
    )
    parser.add_argument(
        "--distance-threshold-m",
        type=float,
        default=DEFAULT_DISTANCE_THRESHOLD_M,
        help="Edge distance threshold in meters, matching preprocess_to_graph.py logic.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the generated plot in a window after saving.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading project file: {args.project_file}")
    project_df = load_project_file(args.project_file)
    print(f"Found {len(project_df)} unique projects with valid coordinates.")

    print(f"Loading MRT file: {args.mrt_file}")
    mrt_df = load_mrt_file(args.mrt_file)
    print(f"Found {len(mrt_df)} unique MRT stations with valid coordinates.")

    edge_df = build_project_mrt_edge_frame(
        project_df, mrt_df, args.distance_threshold_m
    )
    print(
        f"Built {len(edge_df)} project-to-MRT edges using a distance threshold of "
        f"{args.distance_threshold_m:.0f} meters."
    )

    fig = build_map_figure(project_df, mrt_df, edge_df, args.distance_threshold_m)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_file, dpi=200, bbox_inches="tight")
    print(f"Saved point map to: {args.output_file}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
