from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
METADATA_PATH = BASE_DIR / "dataset" / "ccr" / "metadata.csv"
DISTANCE_THRESHOLD_M = 250.0
PROJECT_COL = "project_name"
LAT_COL = "latitude"
LON_COL = "longitude"
OUTPUT_DIR = BASE_DIR / "dataset" / "ccr" / f"graph_link_{int(DISTANCE_THRESHOLD_M)}m"
CSV_OUTPUT_PATH = OUTPUT_DIR / "links.csv"
TXT_OUTPUT_PATH = OUTPUT_DIR / "links.txt"


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


def build_edge_frame(df: pd.DataFrame, distance_threshold_m: float) -> pd.DataFrame:
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
                    }
                )

    return pd.DataFrame(
        edge_records,
        columns=["source_index", "target_index", "source_project", "target_project", "distance_m"],
    )


def main() -> None:
    metadata = load_metadata(METADATA_PATH)
    edge_frame = build_edge_frame(metadata, DISTANCE_THRESHOLD_M)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    edge_frame.to_csv(CSV_OUTPUT_PATH, index=False)

    txt_lines = [f"{row.source_index},{row.target_index}" for row in edge_frame.itertuples(index=False)]
    TXT_OUTPUT_PATH.write_text("\n".join(txt_lines) + ("\n" if txt_lines else ""), encoding="utf-8")

    print(f"Loaded metadata from: {METADATA_PATH}")
    print(f"Projects: {len(metadata)}")
    print(f"Distance threshold (m): {DISTANCE_THRESHOLD_M}")
    print(f"Edges written: {len(edge_frame)}")
    print(f"Saved graph CSV to: {CSV_OUTPUT_PATH}")
    print(f"Saved graph TXT to: {TXT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
