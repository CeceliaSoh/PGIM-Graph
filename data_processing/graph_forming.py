from pathlib import Path

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0088
MAX_DISTANCE_KM = 1.0
CHUNK_SIZE = 500


def haversine_chunk(lat1_rad, lon1_rad, lat2_rad, lon2_rad):
    """Return pairwise distances in km between two coordinate batches."""
    dlat = lat2_rad[None, :] - lat1_rad[:, None]
    dlon = lon2_rad[None, :] - lon1_rad[:, None]

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad)[:, None] * np.cos(lat2_rad)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def build_project_graph(projects, max_distance_km=MAX_DISTANCE_KM, chunk_size=CHUNK_SIZE):
    """Build an undirected edge list for projects within the distance threshold."""
    lat_rad = np.radians(projects["latitude"].to_numpy())
    lon_rad = np.radians(projects["longitude"].to_numpy())
    project_names = projects["Project Name"].to_numpy()
    latitudes = projects["latitude"].to_numpy()
    longitudes = projects["longitude"].to_numpy()

    edge_frames = []
    node_count = len(projects)
    all_indices = np.arange(node_count)

    for start in range(0, node_count, chunk_size):
        end = min(start + chunk_size, node_count)
        distances = haversine_chunk(
            lat_rad[start:end],
            lon_rad[start:end],
            lat_rad,
            lon_rad,
        )

        source_indices = np.arange(start, end)
        valid_pairs = (distances < max_distance_km) & (all_indices[None, :] > source_indices[:, None])
        row_offsets, target_indices = np.nonzero(valid_pairs)

        if len(row_offsets) == 0:
            continue

        source_indices = source_indices[row_offsets]
        edge_frames.append(
            pd.DataFrame(
                {
                    "source_project": project_names[source_indices],
                    "target_project": project_names[target_indices],
                    "source_latitude": latitudes[source_indices],
                    "source_longitude": longitudes[source_indices],
                    "target_latitude": latitudes[target_indices],
                    "target_longitude": longitudes[target_indices],
                    "distance_km": distances[row_offsets, target_indices],
                }
            )
        )

    if not edge_frames:
        return pd.DataFrame(
            columns=[
                "source_project",
                "target_project",
                "source_latitude",
                "source_longitude",
                "target_latitude",
                "target_longitude",
                "distance_km",
            ]
        )

    return pd.concat(edge_frames, ignore_index=True)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    input_path = base_dir / "dataset" / "URA_enriched_with_99co_v3.csv"
    edge_output_path = base_dir / "dataset" / "project_graph_edges_within_5km.csv"
    degree_output_path = base_dir / "dataset" / "project_graph_node_degrees_within_5km.csv"

    df = pd.read_csv(input_path, usecols=["Project Name", "latitude", "longitude"])

    projects = (
        df[["Project Name", "latitude", "longitude"]]
        .dropna()
        .drop_duplicates(subset=["Project Name"])
        .sort_values("Project Name")
        .reset_index(drop=True)
    )

    edges = build_project_graph(projects)

    degree_counts = pd.concat(
        [
            edges["source_project"].value_counts(),
            edges["target_project"].value_counts(),
        ],
        axis=1,
    ).fillna(0)
    degree_counts.columns = ["source_degree", "target_degree"]
    degree_counts["edge_count"] = (
        degree_counts["source_degree"] + degree_counts["target_degree"]
    ).astype(int)

    node_degrees = (
        projects[["Project Name"]]
        .merge(
            degree_counts[["edge_count"]],
            how="left",
            left_on="Project Name",
            right_index=True,
        )
        .fillna({"edge_count": 0})
        .astype({"edge_count": int})
        .sort_values(["edge_count", "Project Name"], ascending=[False, True])
        .reset_index(drop=True)
    )

    edges.to_csv(edge_output_path, index=False)
    node_degrees.to_csv(degree_output_path, index=False)

    print(f"Unique projects: {len(projects):,}")
    print(f"Edges within {MAX_DISTANCE_KM:.1f} km: {len(edges):,}")
    print()
    print("Node edge-count statistics:")
    print(node_degrees["edge_count"].describe().to_string())
    print()
    print("Top 20 projects by edge count:")
    print(node_degrees.head(20).to_string(index=False))
    print()
    print(f"Saved edge list to: {edge_output_path}")
    print(f"Saved node degrees to: {degree_output_path}")


if __name__ == "__main__":
    main()
