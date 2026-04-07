from __future__ import annotations

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_POINT_CSV_PATH = (
    BASE_DIR / "dataset" / "URA_merged_ccr_project_coverage_check.csv"
)
DEFAULT_DISTANCE_THRESHOLD_M = 300.0
DEFAULT_OUTPUT_PATH = BASE_DIR / "dataset" / f"singapore_project_points_map_{DEFAULT_DISTANCE_THRESHOLD_M}.png"

def load_point_file(point_csv_path: Path) -> pd.DataFrame:
    if not point_csv_path.exists():
        raise FileNotFoundError(f"Point CSV file was not found: {point_csv_path}")

    df = pd.read_csv(point_csv_path, dtype=str)
    if df.empty:
        raise ValueError(f"Point CSV file is empty: {point_csv_path}")

    expected_columns = {"project_name", "latitude", "longitude"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Point CSV is missing required columns: {sorted(missing_columns)}"
        )

    df["project_name"] = df["project_name"].astype(str).str.strip()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    if df.empty:
        raise ValueError("No valid latitude/longitude rows were found in the CSV.")

    return df.drop_duplicates(subset=["project_name"]).reset_index(drop=True)


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


def build_edge_frame(point_df: pd.DataFrame, distance_threshold_m: float) -> pd.DataFrame:
    project_names = point_df["project_name"].astype(str).tolist()
    latitudes = point_df["latitude"].to_numpy(dtype=np.float64)
    longitudes = point_df["longitude"].to_numpy(dtype=np.float64)
    distance_matrix = haversine_distance_matrix(latitudes, longitudes)
    node_count = len(project_names)
    edge_records = []

    for source_index in range(node_count):
        for target_index in range(source_index + 1, node_count):
            distance_m = distance_matrix[source_index, target_index]
            if distance_m <= distance_threshold_m:
                edge_records.append(
                    {
                        "source_project": project_names[source_index],
                        "target_project": project_names[target_index],
                        "source_latitude": float(latitudes[source_index]),
                        "source_longitude": float(longitudes[source_index]),
                        "target_latitude": float(latitudes[target_index]),
                        "target_longitude": float(longitudes[target_index]),
                        "distance_m": float(distance_m),
                    }
                )

    return pd.DataFrame(edge_records)


def build_map_figure(
    point_df: pd.DataFrame, edge_df: pd.DataFrame, distance_threshold_m: float
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#eef6fb")

    if not edge_df.empty:
        segments = [
            [
                (row.source_longitude, row.source_latitude),
                (row.target_longitude, row.target_latitude),
            ]
            for row in edge_df.itertuples(index=False)
        ]
        line_collection = LineCollection(
            segments,
            linewidths=0.8,
            colors="#2f5d8a",
            alpha=0.35,
            zorder=1,
        )
        ax.add_collection(line_collection)

    ax.scatter(
        point_df["longitude"],
        point_df["latitude"],
        s=18,
        c="#d42b2b",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.8,
        zorder=3,
    )

    ax.set_title("Singapore Project Locations and Graph Edges")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    lon_mean = float(point_df["longitude"].mean())
    lat_mean = float(point_df["latitude"].mean())
    lon_std = float(point_df["longitude"].std())
    lat_std = float(point_df["latitude"].std())

    lon_half_range = max(0.002, 2.0 * lon_std)
    lat_half_range = max(0.002, 2.0 * lat_std)

    ax.set_xlim(lon_mean - lon_half_range, lon_mean + lon_half_range)
    ax.set_ylim(lat_mean - lat_half_range, lat_mean + lat_half_range)

    ax.text(
        0.01,
        0.98,
        f"Projects: {len(point_df)} | Edges: {len(edge_df)} | Threshold: {distance_threshold_m:.0f}m",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot project latitude/longitude points on a Singapore-focused map."
    )
    parser.add_argument(
        "--point-file",
        type=Path,
        default=DEFAULT_POINT_CSV_PATH,
        help="Path to the CSV file containing project_name, latitude, and longitude.",
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

    print(f"Loading point file: {args.point_file}")
    point_df = load_point_file(args.point_file)
    print(f"Found {len(point_df)} unique projects with valid coordinates.")
    edge_df = build_edge_frame(point_df, args.distance_threshold_m)
    print(
        f"Built {len(edge_df)} edges using a distance threshold of "
        f"{args.distance_threshold_m:.0f} meters."
    )

    fig = build_map_figure(point_df, edge_df, args.distance_threshold_m)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_file, dpi=200, bbox_inches="tight")
    print(f"Saved point map to: {args.output_file}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
