from __future__ import annotations

from pathlib import Path
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import LineCollection, PatchCollection

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_CSV_PATH = (
    BASE_DIR / "dataset" / "URA_merged_ccr_project_coverage_check.csv"
)
DEFAULT_PROJECT_METADATA_CSV_PATH = BASE_DIR / "dataset" / "mongodb_exports" / "Project.csv"
DEFAULT_MRT_CSV_PATH = BASE_DIR / "dataset" / "mongodb_exports" / "Rail_Transport.csv"
DEFAULT_DISTRICT_BOUNDARY_GEOJSON_PATH = (
    BASE_DIR / "dataset" / "district_and_planning_area.geojson"
)
DEFAULT_OUTPUT_PATH = BASE_DIR / "dataset" / "singapore_project_mrt_radius_map.png"
DEFAULT_DISTANCE_THRESHOLD_M = 250.0


def normalize_project_name(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )


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
    df["project_name_key"] = normalize_project_name(df["project_name"])
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    if df.empty:
        raise ValueError("No valid latitude/longitude rows were found in the CSV.")

    return df.drop_duplicates(subset=["project_name_key"]).reset_index(drop=True)


def load_project_metadata_file(project_metadata_csv_path: Path) -> pd.DataFrame:
    if not project_metadata_csv_path.exists():
        raise FileNotFoundError(
            f"Project metadata CSV file was not found: {project_metadata_csv_path}"
        )

    df = pd.read_csv(project_metadata_csv_path, dtype=str)
    if df.empty:
        raise ValueError(
            f"Project metadata CSV file is empty: {project_metadata_csv_path}"
        )

    expected_columns = {"project_name", "postal_district", "planning_area"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Project metadata CSV is missing required columns: {sorted(missing_columns)}"
        )

    df["project_name"] = df["project_name"].astype(str).str.strip()
    df["project_name_key"] = normalize_project_name(df["project_name"])
    df["postal_district"] = pd.to_numeric(df["postal_district"], errors="coerce")
    df["planning_area"] = df["planning_area"].astype(str).str.strip()
    df.loc[df["planning_area"].isin(["", "nan", "None"]), "planning_area"] = pd.NA
    df = df.dropna(subset=["postal_district"])
    df["postal_district"] = df["postal_district"].astype(int)
    df = df[df["postal_district"].between(1, 28)]

    if df.empty:
        raise ValueError("No valid postal_district rows were found in the metadata CSV.")

    return df.drop_duplicates(subset=["project_name_key"]).reset_index(drop=True)


def enrich_projects_with_postal_district(
    project_df: pd.DataFrame, project_metadata_df: pd.DataFrame
) -> pd.DataFrame:
    enriched_df = project_df.merge(
        project_metadata_df[["project_name_key", "postal_district", "planning_area"]],
        on="project_name_key",
        how="left",
    )
    return enriched_df


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


def build_project_radius_edge_frame(
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
    mrt_to_projects: dict[str, list[int]] = {}

    for project_index, project_name in enumerate(project_names):
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
                        "source_project": project_names[source_index],
                        "target_project": project_names[target_index],
                        "source_latitude": float(project_latitudes[source_index]),
                        "source_longitude": float(project_longitudes[source_index]),
                        "target_latitude": float(project_latitudes[target_index]),
                        "target_longitude": float(project_longitudes[target_index]),
                        "shared_mrt": mrt_name,
                    }
                )

    return pd.DataFrame(edge_records)


def compute_plot_bounds(
    project_df: pd.DataFrame,
) -> tuple[float, float, float, float, float, float]:
    lon_mean = float(project_df["longitude"].mean())
    lat_mean = float(project_df["latitude"].mean())
    lon_std = float(project_df["longitude"].std())
    lat_std = float(project_df["latitude"].std())

    lon_half_range = max(0.002, 2.0 * lon_std)
    lat_half_range = max(0.002, 2.0 * lat_std)

    return (
        lon_mean - lon_half_range,
        lon_mean + lon_half_range,
        lat_mean - lat_half_range,
        lat_mean + lat_half_range,
        lon_mean,
        lat_mean,
    )


def load_district_boundary_file(district_boundary_geojson_path: Path) -> dict:
    if not district_boundary_geojson_path.exists():
        raise FileNotFoundError(
            f"District boundary GeoJSON file was not found: {district_boundary_geojson_path}"
        )

    with district_boundary_geojson_path.open("r", encoding="utf-8") as file:
        geojson = json.load(file)

    features = geojson.get("features", [])
    if not features:
        raise ValueError(
            f"District boundary GeoJSON file does not contain any features: {district_boundary_geojson_path}"
        )

    return geojson


def iter_polygon_rings(geometry: dict) -> list[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])

    if geometry_type == "Polygon":
        return coordinates
    if geometry_type == "MultiPolygon":
        rings: list[list[list[float]]] = []
        for polygon in coordinates:
            rings.extend(polygon)
        return rings

    return []


def normalize_area_name(value: str) -> str:
    return " ".join(str(value).strip().upper().split())


def polygon_intersects_bounds(
    ring_array: np.ndarray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> bool:
    polygon_lon_min = float(ring_array[:, 0].min())
    polygon_lon_max = float(ring_array[:, 0].max())
    polygon_lat_min = float(ring_array[:, 1].min())
    polygon_lat_max = float(ring_array[:, 1].max())
    return not (
        polygon_lon_max < lon_min
        or polygon_lon_min > lon_max
        or polygon_lat_max < lat_min
        or polygon_lat_min > lat_max
    )


def add_district_boundary_layer(
    ax: plt.Axes,
    district_boundary_geojson: dict,
    selected_planning_areas: set[str],
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> list[str]:
    features = district_boundary_geojson.get("features", [])
    planning_area_names = sorted(
        {
            str(feature.get("properties", {}).get("planning_area", "")).strip()
            for feature in features
            if (
                str(feature.get("properties", {}).get("planning_area", "")).strip()
                and normalize_area_name(
                    str(feature.get("properties", {}).get("planning_area", "")).strip()
                )
                in selected_planning_areas
            )
        }
    )
    planning_area_to_color = {
        planning_area_name: plt.cm.Set3(color_index / max(len(planning_area_names), 1))
        for color_index, planning_area_name in enumerate(planning_area_names)
    }

    patches: list[Polygon] = []
    facecolors: list[tuple[float, float, float, float]] = []
    label_positions: dict[str, tuple[float, float]] = {}

    for feature in features:
        properties = feature.get("properties", {})
        planning_area_name = str(properties.get("planning_area", "")).strip()
        if not planning_area_name:
            continue
        if normalize_area_name(planning_area_name) not in selected_planning_areas:
            continue

        polygon_rings = iter_polygon_rings(feature.get("geometry", {}))
        if not polygon_rings:
            continue

        polygon_points: list[list[float]] = []
        for ring in polygon_rings:
            ring_array = np.asarray(ring, dtype=float)
            if ring_array.ndim != 2 or ring_array.shape[1] != 2 or len(ring_array) < 3:
                continue
            if not polygon_intersects_bounds(
                ring_array,
                lon_min=lon_min,
                lon_max=lon_max,
                lat_min=lat_min,
                lat_max=lat_max,
            ):
                continue
            patches.append(Polygon(ring_array, closed=True))
            facecolors.append(planning_area_to_color[planning_area_name])
            polygon_points.extend(ring)

        if planning_area_name not in label_positions and polygon_points:
            point_array = np.asarray(polygon_points, dtype=float)
            label_positions[planning_area_name] = (
                float(point_array[:, 0].mean()),
                float(point_array[:, 1].mean()),
            )

    if patches:
        patch_collection = PatchCollection(
            patches,
            facecolor=facecolors,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.22,
            zorder=-2,
        )
        ax.add_collection(patch_collection)

    for planning_area_name, (label_lon, label_lat) in label_positions.items():
        ax.text(
            label_lon,
            label_lat,
            planning_area_name,
            fontsize=8,
            color="#2b2b2b",
            ha="center",
            va="center",
            alpha=0.8,
            zorder=-1,
            bbox={"facecolor": "white", "alpha": 0.35, "edgecolor": "none", "pad": 1.5},
        )

    return list(label_positions.keys())


def build_map_figure(
    project_df: pd.DataFrame,
    mrt_df: pd.DataFrame,
    edge_df: pd.DataFrame,
    distance_threshold_m: float,
    district_boundary_geojson: dict | None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#eef6fb")

    lat_mean = float(project_df["latitude"].mean())
    meters_per_degree_lat = 111320.0
    meters_per_degree_lon = 111320.0 * np.cos(np.radians(lat_mean))
    radius_lon_deg = distance_threshold_m / meters_per_degree_lon
    radius_lat_deg = distance_threshold_m / meters_per_degree_lat
    lon_min, lon_max, lat_min, lat_max, lon_mean, lat_mean = compute_plot_bounds(project_df)
    planning_area_names: list[str] = []

    if district_boundary_geojson is not None:
        visible_project_df = project_df[
            project_df["longitude"].between(lon_min, lon_max)
            & project_df["latitude"].between(lat_min, lat_max)
        ].copy()
        selected_planning_areas = {
            normalize_area_name(value)
            for value in visible_project_df["planning_area"].dropna().tolist()
            if str(value).strip()
        }
        planning_area_names = add_district_boundary_layer(
            ax,
            district_boundary_geojson,
            selected_planning_areas=selected_planning_areas,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
        )

    for row in mrt_df.itertuples(index=False):
        radius_circle = Circle(
            (row.longitude, row.latitude),
            radius=radius_lon_deg,
            facecolor="#1f77b4",
            edgecolor="#1f77b4",
            linewidth=0.5,
            alpha=0.08,
            zorder=0,
        )
        radius_circle.set_transform(
            ax.transData
            + plt.matplotlib.transforms.Affine2D().scale(1.0, radius_lat_deg / radius_lon_deg)
        )
        ax.add_patch(radius_circle)

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

    for row in mrt_df.itertuples(index=False):
        ax.text(
            row.longitude,
            row.latitude,
            str(row.station_name_english),
            fontsize=7,
            color="#123b63",
            ha="left",
            va="bottom",
            alpha=0.9,
            zorder=5,
            bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none", "pad": 1.0},
        )

    ax.set_title("Project Edges Formed Within MRT Radius")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.text(
        0.01,
        0.98,
        (
            f"Projects: {len(project_df)} | MRT: {len(mrt_df)} | "
            f"Edges: {len(edge_df)} | MRT Radius: {distance_threshold_m:.0f}m | "
            f"Postal districts: {project_df['postal_district'].dropna().nunique()} | "
            f"Planning areas shown: {len(planning_area_names)}"
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
        description="Plot project-to-project edges when two projects fall within the same MRT radius."
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
        "--project-metadata-file",
        type=Path,
        default=DEFAULT_PROJECT_METADATA_CSV_PATH,
        help="Path to the project metadata CSV file containing project_name and postal_district.",
    )
    parser.add_argument(
        "--district-boundary-file",
        type=Path,
        default=DEFAULT_DISTRICT_BOUNDARY_GEOJSON_PATH,
        help="Path to the GeoJSON file containing district or planning-area boundaries.",
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
        help="Radius in meters for linking projects through the same MRT station.",
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

    print(f"Loading project metadata file: {args.project_metadata_file}")
    project_metadata_df = load_project_metadata_file(args.project_metadata_file)
    print(
        f"Found {len(project_metadata_df)} projects with valid postal district metadata."
    )

    project_df = enrich_projects_with_postal_district(project_df, project_metadata_df)
    matched_district_count = int(project_df["postal_district"].notna().sum())
    print(
        f"Matched postal districts for {matched_district_count} of {len(project_df)} plotted projects."
    )

    print(f"Loading MRT file: {args.mrt_file}")
    mrt_df = load_mrt_file(args.mrt_file)
    print(f"Found {len(mrt_df)} unique MRT stations with valid coordinates.")

    print(f"Loading district boundary file: {args.district_boundary_file}")
    district_boundary_geojson = load_district_boundary_file(args.district_boundary_file)
    print(
        f"Found {len(district_boundary_geojson.get('features', []))} district boundary features."
    )

    edge_df = build_project_radius_edge_frame(
        project_df, mrt_df, args.distance_threshold_m
    )
    print(
        f"Built {len(edge_df)} project-to-project edges using an MRT radius of "
        f"{args.distance_threshold_m:.0f} meters."
    )

    fig = build_map_figure(
        project_df,
        mrt_df,
        edge_df,
        args.distance_threshold_m,
        district_boundary_geojson,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_file, dpi=200, bbox_inches="tight")
    print(f"Saved point map to: {args.output_file}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
