from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pymongo import MongoClient


TARGET_COLUMN = "Avg_rent_per_sqm"
TIMESTAMP_COLUMN = "Lease Commencement Date"
NODE_COLUMN = "Project Name"
LAT_COLUMN = "latitude"
LON_COLUMN = "longitude"
PROJECT_ID_COLUMN = "Project_ID"
INTERNAL_TARGET_COLUMN = "avg_rent_per_sqm"
INTERNAL_TIMESTAMP_COLUMN = "lease_commencement_date"
INTERNAL_NODE_COLUMN = "project_name"
INTERNAL_LAT_COLUMN = "latitude"
INTERNAL_LON_COLUMN = "longitude"
INTERNAL_PROJECT_ID_COLUMN = "project_id"
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "property_db"
EARTH_RADIUS_KM = 6371.0088
CHUNK_SIZE = 500
GRAPH_RULES = [
    {
        "name": "distance_le_1km",
        "description": "Connect projects within 1.0 km.",
        "max_distance_km": 1.0,
        "require_same_nearest_mrt": False,
    },
    {
        "name": "distance_le_0_5km",
        "description": "Connect projects within 0.5 km.",
        "max_distance_km": 0.5,
        "require_same_nearest_mrt": False,
    },
    {
        "name": "distance_le_1km_same_nearest_mrt",
        "description": "Connect projects within 1.0 km only if they share the same nearest MRT.",
        "max_distance_km": 1.0,
        "require_same_nearest_mrt": True,
    },
]
DEFAULT_GRAPH_NAME = GRAPH_RULES[0]["name"]

COLLECTION_CONFIG = {
    "Project": "Project",
    "Project_Location": "Project_Location",
    "Macro_Data": "Macro_Data",
    "Project_Top30_School": "Project_Top30_School",
    "Project_MRT_GgleMap": "Project_MRT_GgleMap",
    "Project_Facilities": "Project_Facilities",
    "Project_Rental_Aggregate": "Project_Rental_Aggregate",
}
FEATURE_GROUP_COLUMNS = {
    "macro": [],
    "school": [],
    "mrt": [],
    "facilities": [],
}

CANONICAL_ALIASES = {
    "project_id": ["project_id", "Project_ID", "projectid"],
    "project_name": ["project_name", "Project_Name", "Project Name"],
    "postal_district": ["postal_district", "Postal District"],
    "planning_region": ["planning_region", "Planning Region"],
    "planning_area": ["planning_area", "Planning Area"],
    "property_type_realis": ["property_type_realis", "Property_Type_REALIS", "Property Type"],
    "tenure": ["tenure", "Tenure"],
    "top_date": ["top_date", "TOP date", "topdate"],
    "project_size": ["project_size", "Project Size"],
    "latitude": ["latitude", "Latitude"],
    "longitude": ["longitude", "Longitude"],
    "lease_commencement_date": ["lease_commencement_date", "Lease Commencement Date"],
    "avg_rent_per_sqm": ["avg_rent_per_sqm", "Avg_rent_per_sqm"],
    "date": ["date", "Date"],
}


def log(message: str) -> None:
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def normalize_column_name(column_name: str) -> str:
    text = str(column_name).strip().lower()
    text = text.replace("&", "and")
    text = text.replace("%", "pct")
    text = text.replace("$", "")
    text = text.replace("/", "_")
    text = text.replace("-", "_")
    text = text.replace("'", "")
    text = text.replace(".", "_")
    text = text.replace("(", " ")
    text = text.replace(")", " ")
    text = text.replace(",", " ")
    text = text.replace(":", " ")
    text = " ".join(text.split())
    text = text.replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]
    return result


def standardize_string_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    object_columns = result.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        result[column] = result[column].map(
            lambda value: " ".join(str(value).split()) if isinstance(value, str) else value
        )
    return result


def find_column(df: pd.DataFrame, canonical_name: str) -> str | None:
    candidates = CANONICAL_ALIASES.get(canonical_name, [canonical_name])
    available = {column.casefold(): column for column in df.columns}
    for candidate in candidates:
        matched = available.get(candidate.casefold())
        if matched is not None:
            return matched
    return None


def ensure_canonical_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    result = df.copy()
    rename_map = {}
    for canonical_name in CANONICAL_ALIASES:
        matched = find_column(result, canonical_name)
        if matched is not None:
            rename_map[matched] = canonical_name
    result = result.rename(columns=rename_map)
    return result


def sanitize_identifier(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    text = "_".join(text.split())
    cleaned = "".join(character for character in text if character.isalnum() or character == "_")
    return cleaned or None


def ensure_project_keys(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    result = ensure_canonical_columns(normalize_column_names(standardize_string_whitespace(df)), dataset_name)

    if INTERNAL_NODE_COLUMN not in result.columns:
        candidate_columns = [column for column in result.columns if "project" in column and "name" in column]
        if candidate_columns:
            result[INTERNAL_NODE_COLUMN] = result[candidate_columns[0]]
            warn(
                f"{dataset_name}: '{INTERNAL_NODE_COLUMN}' missing, using '{candidate_columns[0]}' instead."
            )
        else:
            result[INTERNAL_NODE_COLUMN] = None
            warn(f"{dataset_name}: project name field missing; merge quality may be poor.")

    if INTERNAL_PROJECT_ID_COLUMN not in result.columns:
        result[INTERNAL_PROJECT_ID_COLUMN] = result[INTERNAL_NODE_COLUMN].map(sanitize_identifier)
        warn(
            f"{dataset_name}: '{INTERNAL_PROJECT_ID_COLUMN}' missing, generated from '{INTERNAL_NODE_COLUMN}'."
        )
    else:
        result[INTERNAL_PROJECT_ID_COLUMN] = result[INTERNAL_PROJECT_ID_COLUMN].map(sanitize_identifier)

    result["project_name_join"] = result[INTERNAL_NODE_COLUMN].map(sanitize_identifier)
    return result


def load_mongodb_collections():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB_NAME]
    collections = {}

    for logical_name, collection_name in COLLECTION_CONFIG.items():
        records = list(db[collection_name].find({}))
        frame = pd.DataFrame(records)
        if "_id" in frame.columns:
            frame = frame.drop(columns=["_id"])
        frame = ensure_project_keys(frame, logical_name)
        collections[logical_name] = frame
        log(f"Loaded MongoDB collection '{collection_name}': {len(frame):,} rows")

    return collections


def register_feature_group_columns(collections) -> None:
    FEATURE_GROUP_COLUMNS["macro"] = sorted(
        column for column in collections["Macro_Data"].columns if column not in {"date"}
    )
    FEATURE_GROUP_COLUMNS["school"] = sorted(
        column
        for column in collections["Project_Top30_School"].columns
        if column not in {INTERNAL_PROJECT_ID_COLUMN, INTERNAL_NODE_COLUMN, "project_name_join"}
    )
    FEATURE_GROUP_COLUMNS["mrt"] = sorted(
        column
        for column in collections["Project_MRT_GgleMap"].columns
        if column not in {INTERNAL_PROJECT_ID_COLUMN, INTERNAL_NODE_COLUMN, "project_name_join"}
    )
    FEATURE_GROUP_COLUMNS["facilities"] = sorted(
        column
        for column in collections["Project_Facilities"].columns
        if column not in {INTERNAL_PROJECT_ID_COLUMN, INTERNAL_NODE_COLUMN, "project_name_join"}
    )


def merge_with_fallback(left_df: pd.DataFrame, right_df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    left = ensure_project_keys(left_df.copy(), f"{dataset_name} left")
    right = ensure_project_keys(right_df.copy(), f"{dataset_name} right")

    if right.empty:
        warn(f"{dataset_name}: right-side dataframe is empty.")
        return left

    right_id = right[right[INTERNAL_PROJECT_ID_COLUMN].notna()].drop_duplicates(
        subset=[INTERNAL_PROJECT_ID_COLUMN]
    ).copy()
    merged = left.merge(
        right_id,
        on=INTERNAL_PROJECT_ID_COLUMN,
        how="left",
        suffixes=("", "_rhs"),
    )

    right_name = right.drop_duplicates(subset=["project_name_join"]).copy()
    fallback = left.merge(
        right_name,
        on="project_name_join",
        how="left",
        suffixes=("", "_fallback"),
    )

    right_columns = [
        column
        for column in right.columns
        if column not in {INTERNAL_PROJECT_ID_COLUMN, "project_name_join"}
    ]

    for column in right_columns:
        rhs_column = f"{column}_rhs"
        fallback_column = f"{column}_fallback"
        if rhs_column in merged.columns:
            merged[column] = merged[rhs_column]
        if column not in merged.columns:
            merged[column] = np.nan
        if fallback_column in fallback.columns:
            merged[column] = merged[column].where(merged[column].notna(), fallback[fallback_column])

    extra_columns = [column for column in merged.columns if column.endswith("_rhs")]
    fallback_columns = [column for column in fallback.columns if column.endswith("_fallback")]
    merged = merged.drop(columns=extra_columns, errors="ignore")
    fallback = fallback.drop(columns=fallback_columns, errors="ignore")

    if "project_name_join" not in merged.columns:
        merged["project_name_join"] = merged[INTERNAL_NODE_COLUMN].map(sanitize_identifier)

    return merged


def merge_project_temporal_data(collections):
    aggregate = collections["Project_Rental_Aggregate"].copy()
    project = collections["Project"].copy()
    location = collections["Project_Location"].copy()
    macro = collections["Macro_Data"].copy()
    schools = collections["Project_Top30_School"].copy()
    mrt = collections["Project_MRT_GgleMap"].copy()
    facilities = collections["Project_Facilities"].copy()

    for name, frame in [
        ("Project", project),
        ("Project_Location", location),
        ("Project_Top30_School", schools),
        ("Project_MRT_GgleMap", mrt),
        ("Project_Facilities", facilities),
    ]:
        duplicate_count = int(frame.duplicated(subset=[INTERNAL_PROJECT_ID_COLUMN]).sum())
        missing_keys = int(frame[INTERNAL_PROJECT_ID_COLUMN].isna().sum())
        log(f"{name}: duplicate project_id rows={duplicate_count:,}, missing keys={missing_keys:,}")

    aggregate[INTERNAL_TIMESTAMP_COLUMN] = pd.to_datetime(
        aggregate[INTERNAL_TIMESTAMP_COLUMN], errors="coerce"
    )
    macro["date"] = pd.to_datetime(macro["date"], errors="coerce")

    aggregate = aggregate.sort_values([INTERNAL_TIMESTAMP_COLUMN, INTERNAL_NODE_COLUMN]).reset_index(drop=True)
    log(f"Base aggregate rows for temporal modeling: {len(aggregate):,}")

    merged = aggregate.copy()
    merged = merge_with_fallback(merged, project, "Project merge")
    merged = merge_with_fallback(merged, location, "Project_Location merge")
    merged = merge_with_fallback(merged, schools, "Project_Top30_School merge")
    merged = merge_with_fallback(merged, mrt, "Project_MRT_GgleMap merge")
    merged = merge_with_fallback(merged, facilities, "Project_Facilities merge")
    merged = merged.merge(macro, left_on=INTERNAL_TIMESTAMP_COLUMN, right_on="date", how="left", suffixes=("", "_macro"))
    merged = merged.drop(columns=["date"], errors="ignore")

    merged = merged.rename(
        columns={
            INTERNAL_PROJECT_ID_COLUMN: PROJECT_ID_COLUMN,
            INTERNAL_NODE_COLUMN: NODE_COLUMN,
            INTERNAL_TIMESTAMP_COLUMN: TIMESTAMP_COLUMN,
            INTERNAL_TARGET_COLUMN: TARGET_COLUMN,
            INTERNAL_LAT_COLUMN: LAT_COLUMN,
            INTERNAL_LON_COLUMN: LON_COLUMN,
        }
    )

    log(f"Merged temporal dataframe rows: {len(merged):,}")
    return merged


def stable_one_hot_block(series: pd.Series, prefix: str):
    filled = series.fillna("UNKNOWN").astype(str)
    categories = sorted(filled.unique())
    block = pd.DataFrame(0.0, index=series.index, columns=[f"{prefix}__{category}" for category in categories])
    for category in categories:
        block.loc[filled == category, f"{prefix}__{category}"] = 1.0
    mapping = {category: index for index, category in enumerate(categories)}
    return block.astype(np.float32), mapping


def compute_age_of_property(df: pd.DataFrame) -> pd.Series:
    lease_year = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce").dt.year
    top_year = (
        df["top_date"]
        .astype("string")
        .str.extract(r"(\d{4})", expand=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    # Positive and intuitive: age at the lease observation date.
    age = lease_year - top_year
    return age.fillna(-1).astype(np.float32)


def build_ordered_feature_dataframe(df: pd.DataFrame):
    feature_blocks = []
    feature_columns = []
    report_sections = []

    postal_block, postal_mapping = stable_one_hot_block(df["postal_district"], "postal_district")
    feature_blocks.append(postal_block)
    feature_columns.extend(postal_block.columns.tolist())
    report_sections.append(("postal_district one-hot", postal_block.columns.tolist(), postal_mapping))

    region_block, region_mapping = stable_one_hot_block(df["planning_region"], "planning_region")
    feature_blocks.append(region_block)
    feature_columns.extend(region_block.columns.tolist())
    report_sections.append(("planning_region one-hot", region_block.columns.tolist(), region_mapping))

    area_block, area_mapping = stable_one_hot_block(df["planning_area"], "planning_area")
    feature_blocks.append(area_block)
    feature_columns.extend(area_block.columns.tolist())
    report_sections.append(("planning_area one-hot", area_block.columns.tolist(), area_mapping))

    property_type_block, property_type_mapping = stable_one_hot_block(
        df["property_type_realis"], "property_type_realis"
    )
    feature_blocks.append(property_type_block)
    feature_columns.extend(property_type_block.columns.tolist())
    report_sections.append(
        ("property_type_realis one-hot", property_type_block.columns.tolist(), property_type_mapping)
    )

    tenure_series = pd.to_numeric(df["tenure"], errors="coerce").fillna(9999).astype(np.float32)
    tenure_block = pd.DataFrame({"tenure": tenure_series}, index=df.index)
    feature_blocks.append(tenure_block)
    feature_columns.extend(tenure_block.columns.tolist())
    report_sections.append(("tenure scalar", tenure_block.columns.tolist(), None))

    project_size_block, project_size_mapping = stable_one_hot_block(df["project_size"], "project_size")
    feature_blocks.append(project_size_block)
    feature_columns.extend(project_size_block.columns.tolist())
    report_sections.append(("project_size one-hot", project_size_block.columns.tolist(), project_size_mapping))

    scalar_block = pd.DataFrame(
        {
            LAT_COLUMN: pd.to_numeric(df[LAT_COLUMN], errors="coerce").fillna(0).astype(np.float32),
            LON_COLUMN: pd.to_numeric(df[LON_COLUMN], errors="coerce").fillna(0).astype(np.float32),
            "age_of_property": compute_age_of_property(df),
        },
        index=df.index,
    )
    feature_blocks.append(scalar_block)
    feature_columns.extend(scalar_block.columns.tolist())
    report_sections.append(("latitude/longitude/age_of_property", scalar_block.columns.tolist(), None))

    macro_columns = [column for column in FEATURE_GROUP_COLUMNS["macro"] if column in df.columns]
    macro_block = (
        df[macro_columns].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        if macro_columns
        else pd.DataFrame(index=df.index)
    )
    feature_blocks.append(macro_block)
    feature_columns.extend(macro_block.columns.tolist())
    report_sections.append(("macro feature block", macro_block.columns.tolist(), None))

    school_columns = [column for column in FEATURE_GROUP_COLUMNS["school"] if column in df.columns]
    school_block = (
        df[school_columns].apply(pd.to_numeric, errors="coerce").fillna(99999).astype(np.float32)
        if school_columns
        else pd.DataFrame(index=df.index)
    )
    feature_blocks.append(school_block)
    feature_columns.extend(school_block.columns.tolist())
    report_sections.append(("Project_Top30_School block", school_block.columns.tolist(), None))

    mrt_columns = [column for column in FEATURE_GROUP_COLUMNS["mrt"] if column in df.columns]
    mrt_block = (
        df[mrt_columns].apply(pd.to_numeric, errors="coerce").fillna(99999).astype(np.float32)
        if mrt_columns
        else pd.DataFrame(index=df.index)
    )
    feature_blocks.append(mrt_block)
    feature_columns.extend(mrt_block.columns.tolist())
    report_sections.append(("Project_MRT_GgleMap block", mrt_block.columns.tolist(), None))

    facility_columns = [column for column in FEATURE_GROUP_COLUMNS["facilities"] if column in df.columns]
    facility_block = (
        df[facility_columns].apply(pd.to_numeric, errors="coerce").fillna(0).astype(np.float32)
        if facility_columns
        else pd.DataFrame(index=df.index)
    )
    feature_blocks.append(facility_block)
    feature_columns.extend(facility_block.columns.tolist())
    report_sections.append(("Project_Facilities block", facility_block.columns.tolist(), None))

    ordered_df = pd.concat(feature_blocks, axis=1).astype(np.float32)
    return ordered_df, report_sections


def preprocess_features(df):
    target = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").astype(np.float32)
    timestamps = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce").dt.to_period("M").astype(str)
    features_df, report_sections = build_ordered_feature_dataframe(df)
    return features_df, target, timestamps, report_sections


def build_temporal_tensors(df, node_frame):
    features_df, target_series, timestamps, report_sections = preprocess_features(df)

    node_names = node_frame[NODE_COLUMN].tolist()
    timestamp_values = sorted(timestamps.unique())
    feature_columns = features_df.columns.tolist()

    node_to_idx = {name: idx for idx, name in enumerate(node_names)}
    time_to_idx = {timestamp: idx for idx, timestamp in enumerate(timestamp_values)}

    num_timestamps = len(timestamp_values)
    num_nodes = len(node_names)
    num_features = len(feature_columns)

    features = np.zeros((num_timestamps, num_nodes, num_features), dtype=np.float32)
    targets = np.zeros((num_timestamps, num_nodes), dtype=np.float32)
    observation_mask = np.zeros((num_timestamps, num_nodes), dtype=bool)

    node_indices = df[NODE_COLUMN].map(node_to_idx)
    time_indices = timestamps.map(time_to_idx)
    valid_index_mask = node_indices.notna() & time_indices.notna()

    if not valid_index_mask.all():
        dropped_rows = int((~valid_index_mask).sum())
        warn(
            f"Dropping {dropped_rows:,} temporal rows with missing node/timestamp indices before tensor assembly."
        )

    valid_node_indices = node_indices[valid_index_mask].astype(int).to_numpy()
    valid_time_indices = time_indices[valid_index_mask].astype(int).to_numpy()
    valid_features = features_df.loc[valid_index_mask].to_numpy(dtype=np.float32)

    features[valid_time_indices, valid_node_indices, :] = valid_features
    valid_targets = (target_series.notna() & valid_index_mask).to_numpy()
    targets[time_indices[valid_targets].astype(int), node_indices[valid_targets].astype(int)] = target_series[
        valid_targets
    ].to_numpy(
        dtype=np.float32
    )
    observation_mask[valid_time_indices, valid_node_indices] = True

    # Fill missing target values by linear interpolation within each node's time series.
    # We only interpolate "inside" gaps bounded by earlier and later observed values;
    # no extrapolation is applied at the beginning or end of a series.
    targets_df = pd.DataFrame(targets, index=timestamp_values)
    interpolated_targets = (
        targets_df.mask(~observation_mask)
        .interpolate(method="linear", axis=0, limit_area="inside")
        .fillna(np.nan)
    )
    interpolated_count = int(
        np.count_nonzero(~observation_mask & interpolated_targets.notna().to_numpy())
    )
    if interpolated_count > 0:
        log(f"Interpolated {interpolated_count:,} missing target values using linear interpolation.")
    targets = np.nan_to_num(interpolated_targets.to_numpy(dtype=np.float32), nan=0.0)

    return features, targets, observation_mask, timestamp_values, feature_columns, report_sections


def haversine_chunk(lat1_rad, lon1_rad, lat2_rad, lon2_rad):
    dlat = lat2_rad[None, :] - lat1_rad[:, None]
    dlon = lon2_rad[None, :] - lon1_rad[:, None]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_rad)[:, None] * np.cos(lat2_rad)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def build_graph_variants(node_frame, graph_rules, chunk_size=CHUNK_SIZE):
    lat_rad = np.radians(node_frame[LAT_COLUMN].to_numpy())
    lon_rad = np.radians(node_frame[LON_COLUMN].to_numpy())
    nearest_mrt = node_frame["nearest_mrt"].to_numpy()
    node_count = len(node_frame)
    all_indices = np.arange(node_count)

    graph_parts = {
        rule["name"]: {"src_parts": [], "dst_parts": [], "dist_parts": []}
        for rule in graph_rules
    }

    for start in range(0, node_count, chunk_size):
        end = min(start + chunk_size, node_count)
        distances = haversine_chunk(
            lat_rad[start:end],
            lon_rad[start:end],
            lat_rad,
            lon_rad,
        )

        source_indices = np.arange(start, end)
        upper_triangle_mask = all_indices[None, :] > source_indices[:, None]

        for rule in graph_rules:
            valid_pairs = (distances < rule["max_distance_km"]) & upper_triangle_mask

            if rule["require_same_nearest_mrt"]:
                same_mrt = nearest_mrt[source_indices][:, None] == nearest_mrt[None, :]
                valid_pairs &= same_mrt

            row_offsets, target_indices = np.nonzero(valid_pairs)

            if len(row_offsets) == 0:
                continue

            selected_sources = source_indices[row_offsets]
            selected_distances = distances[row_offsets, target_indices].astype(np.float32)

            graph_parts[rule["name"]]["src_parts"].extend([selected_sources, target_indices])
            graph_parts[rule["name"]]["dst_parts"].extend([target_indices, selected_sources])
            graph_parts[rule["name"]]["dist_parts"].extend([selected_distances, selected_distances])

    graph_variants = {}
    for rule in graph_rules:
        parts = graph_parts[rule["name"]]
        if not parts["src_parts"]:
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, 1), dtype=np.float32)
        else:
            edge_index = np.vstack(
                [
                    np.concatenate(parts["src_parts"]).astype(np.int64),
                    np.concatenate(parts["dst_parts"]).astype(np.int64),
                ]
            )
            edge_attr = np.concatenate(parts["dist_parts"]).reshape(-1, 1).astype(np.float32)

        graph_variants[rule["name"]] = {
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "num_undirected_edges": int(edge_index.shape[1] // 2),
            "max_distance_km": float(rule["max_distance_km"]),
            "require_same_nearest_mrt": bool(rule["require_same_nearest_mrt"]),
            "description": rule["description"],
        }

    return graph_variants


def generate_feature_index_report(report_path: Path, report_sections, feature_columns) -> None:
    lines = []
    lines.append("Feature Index Report")
    lines.append("=" * 80)
    lines.append(f"Total feature dimension: {len(feature_columns)}")
    lines.append("")

    cursor = 0
    scalar_features = []

    for section_name, columns, mapping in report_sections:
        if not columns:
            continue
        start_idx = cursor
        end_idx = cursor + len(columns)
        if len(columns) == 1:
            lines.append(f"feature[{start_idx}]: {section_name} -> {columns[0]}")
        else:
            lines.append(f"feature[{start_idx}:{end_idx}]: {section_name}")
        if mapping:
            lines.append("  category_to_index:")
            for category, index in mapping.items():
                lines.append(f"    {category}: {start_idx + index}")
        if not mapping and len(columns) > 1:
            lines.append("  columns:")
            for index, column in enumerate(columns, start=start_idx):
                lines.append(f"    {index}: {column}")
        if not mapping:
            scalar_features.extend(columns)
        cursor = end_idx
        lines.append("")

    lines.append("Ordered numeric scalar features:")
    if scalar_features:
        for feature_name in scalar_features:
            lines.append(f"- {feature_name}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("Final ordered feature_columns:")
    for index, column in enumerate(feature_columns):
        lines.append(f"{index}: {column}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Saved feature index report to: {report_path}")


def main():
    base_dir = Path(__file__).resolve().parent
    feature_output_path = base_dir / "dataset" / "feature.npy"
    graph_output_path = base_dir / "dataset" / "graph.pt"
    report_output_path = base_dir / "dataset" / "feature_index_report.txt"

    collections = load_mongodb_collections()
    register_feature_group_columns(collections)
    df = merge_project_temporal_data(collections)
    df[TIMESTAMP_COLUMN] = pd.to_datetime(df[TIMESTAMP_COLUMN], errors="coerce")
    df = df.dropna(subset=[TIMESTAMP_COLUMN, NODE_COLUMN]).sort_values([TIMESTAMP_COLUMN, NODE_COLUMN]).reset_index(
        drop=True
    )

    mrt_source = collections["Project_MRT_GgleMap"].copy()
    mrt_source = ensure_canonical_columns(normalize_column_names(mrt_source), "Project_MRT_GgleMap")
    mrt_distance_columns = [
        column
        for column in mrt_source.columns
        if column not in {INTERNAL_PROJECT_ID_COLUMN, INTERNAL_NODE_COLUMN, "project_name_join"}
    ]
    mrt_distance_columns = [column for column in mrt_distance_columns if pd.api.types.is_numeric_dtype(mrt_source[column])]

    node_frame = (
        df[[NODE_COLUMN, LAT_COLUMN, LON_COLUMN, PROJECT_ID_COLUMN]]
        .drop_duplicates(subset=[NODE_COLUMN])
        .sort_values(NODE_COLUMN)
        .reset_index(drop=True)
    )
    node_frame_input = ensure_project_keys(
        node_frame.rename(columns={PROJECT_ID_COLUMN: INTERNAL_PROJECT_ID_COLUMN, NODE_COLUMN: INTERNAL_NODE_COLUMN}),
        "Node_Frame",
    )
    node_frame = merge_with_fallback(
        node_frame_input,
        mrt_source[[INTERNAL_PROJECT_ID_COLUMN, INTERNAL_NODE_COLUMN, "project_name_join"] + mrt_distance_columns],
        "Node MRT merge",
    ).rename(columns={INTERNAL_PROJECT_ID_COLUMN: PROJECT_ID_COLUMN, INTERNAL_NODE_COLUMN: NODE_COLUMN})

    if mrt_distance_columns:
        mrt_distances = node_frame[mrt_distance_columns].apply(pd.to_numeric, errors="coerce")
        nearest_mrt = pd.Series("UNKNOWN_MRT", index=node_frame.index, dtype=object)
        valid_mrt_rows = mrt_distances.notna().any(axis=1)
        nearest_mrt.loc[valid_mrt_rows] = mrt_distances.loc[valid_mrt_rows].idxmin(axis=1)
        node_frame["nearest_mrt"] = nearest_mrt
    else:
        node_frame["nearest_mrt"] = "UNKNOWN_MRT"
        warn("No MRT distance columns found for nearest MRT derivation.")

    duplicate_nodes = int(node_frame.duplicated(subset=[NODE_COLUMN]).sum())
    missing_node_keys = int(node_frame[PROJECT_ID_COLUMN].isna().sum())
    log(f"Node frame duplicates={duplicate_nodes:,}, missing project_id={missing_node_keys:,}")

    features, targets, observation_mask, timestamp_values, feature_columns, report_sections = build_temporal_tensors(
        df, node_frame
    )
    graph_variants = build_graph_variants(node_frame, GRAPH_RULES)
    default_graph = graph_variants[DEFAULT_GRAPH_NAME]

    np.save(feature_output_path, features)
    generate_feature_index_report(report_output_path, report_sections, feature_columns)

    graph_data = {
        "edge_index": torch.from_numpy(default_graph["edge_index"]).long(),
        "edge_attr": torch.from_numpy(default_graph["edge_attr"]).float(),
        "y": torch.from_numpy(targets).float(),
        "y_mask": torch.from_numpy(observation_mask),
        "num_nodes": int(len(node_frame)),
        "num_timestamps": int(len(timestamp_values)),
        "node_names": node_frame[NODE_COLUMN].tolist(),
        "node_coordinates": torch.from_numpy(node_frame[[LAT_COLUMN, LON_COLUMN]].to_numpy(np.float32)),
        "nearest_mrt": node_frame["nearest_mrt"].tolist(),
        "timestamps": timestamp_values,
        "feature_columns": feature_columns,
        "feature_path": str(feature_output_path),
        "distance_threshold_km": default_graph["max_distance_km"],
        "target_column": TARGET_COLUMN,
        "default_graph_name": DEFAULT_GRAPH_NAME,
        "graphs": {
            name: {
                "edge_index": torch.from_numpy(graph["edge_index"]).long(),
                "edge_attr": torch.from_numpy(graph["edge_attr"]).float(),
                "num_undirected_edges": graph["num_undirected_edges"],
                "max_distance_km": graph["max_distance_km"],
                "require_same_nearest_mrt": graph["require_same_nearest_mrt"],
                "description": graph["description"],
            }
            for name, graph in graph_variants.items()
        },
    }
    torch.save(graph_data, graph_output_path)

    print(f"Saved features to: {feature_output_path}")
    print(f"Saved graph to: {graph_output_path}")
    print(f"Saved feature report to: {report_output_path}")
    print(f"Feature tensor shape [T, N, F]: {features.shape}")
    print(f"Target tensor shape [T, N]: {targets.shape}")
    print(f"Observation mask shape [T, N]: {observation_mask.shape}")
    print(f"Unique nodes: {len(node_frame):,}")
    print(f"Unique timestamps: {len(timestamp_values):,}")
    print(f"Feature dimension: {len(feature_columns):,}")
    print("Graph variants:")
    for name, graph in graph_variants.items():
        print(
            f"- {name}: {graph['num_undirected_edges']:,} undirected edges | "
            f"max_distance_km={graph['max_distance_km']:.1f} | "
            f"same_nearest_mrt={graph['require_same_nearest_mrt']}"
        )


if __name__ == "__main__":
    main()
