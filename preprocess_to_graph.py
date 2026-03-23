import os
import numpy as np
import pandas as pd
import random
import torch

DF_PATH = r"dataset\URA_merged_ccr_v3.0.csv"
DF = pd.read_csv(DF_PATH)
DISTANCE_THRESHOLD_M = 200
SPLIT_YEAR = 2023
POSTFIX_STR = f"ccr_v3.0_within_{int(DISTANCE_THRESHOLD_M)}m"
POSTFIX_STR_TRAIN = f"{POSTFIX_STR}_train"
POSTFIX_STR_TEST = f"{POSTFIX_STR}_test"
OUTPUT_DIR = r"dataset"

print("Data loaded: ", DF_PATH)

df_train = DF[DF["year"] <= SPLIT_YEAR].copy()
df_test = DF[DF["year"] > SPLIT_YEAR].copy()
project_list = sorted(DF["Project Name"].dropna().unique().tolist())

def save_feature_npy(
    df: pd.DataFrame,
    postfix_str: str,
    output_dir: str = ".",
    project_col: str = "Project Name",
    time_col: str = "Lease Commencement Date",
    project_list: list | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[project_col, time_col]).copy()

    # ensure unique
    df = (
        df.sort_values([project_col, time_col])
          .drop_duplicates(subset=[project_col, time_col], keep="first")
          .copy()
    )

    # --- define columns ---
    base_exclude = {
        project_col,
        time_col,
        "year",
        "rent_per_sqft",
        "y_mask",
    }

    # base features (numeric only)
    base_feature_cols = [
        col for col in df.columns
        if col not in base_exclude and pd.api.types.is_numeric_dtype(df[col])
    ]

    # ensure target + mask exist
    assert "rent_per_sqft" in df.columns, "rent_per_sqft missing"
    assert "y_mask" in df.columns, "y_mask missing"

    # final feature order
    feature_columns = base_feature_cols + ["rent_per_sqft", "y_mask"]

    # --- indexing ---
    if project_list is None:
        project_list = sorted(df[project_col].unique().tolist())
    else:
        project_list = list(project_list)
    time_list = sorted(df[time_col].unique().tolist())

    project_to_idx = {p: i for i, p in enumerate(project_list)}
    time_to_idx = {t: i for i, t in enumerate(time_list)}

    N = len(project_list)
    T = len(time_list)
    D = len(feature_columns)

    feature_array = np.zeros((N, T, D), dtype=np.float32)

    # --- fill tensor ---
    for _, row in df.iterrows():
        i = project_to_idx[row[project_col]]
        t = time_to_idx[row[time_col]]

        values = row[feature_columns].to_numpy(dtype=np.float32)
        feature_array[i, t, :] = values

    # --- save ---
    save_path = os.path.join(output_dir, f"feature_{postfix_str}.npy")
    np.save(save_path, feature_array)

    print(f"Saved to: {save_path}")
    print(f"Feature shape: {feature_array.shape}")
    print(f"N={N}, T={T}, D={D}")
    print(f"Last columns: {feature_columns[-2:]}")  # sanity check

    return feature_array, feature_columns, project_list, time_list

def check_feature_npy_consistency(
    df: pd.DataFrame,
    feature_array: np.ndarray,
    feature_columns: list,
    project_list: list,
    time_list: list,
    num_samples: int = 10,
    project_col: str = "Project Name",
    time_col: str = "Lease Commencement Date",
    atol: float = 1e-5
):
    """
    Randomly check whether feature_array matches df values
    """

    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # build lookup maps
    project_to_idx = {p: i for i, p in enumerate(project_list)}
    time_to_idx = {t: i for i, t in enumerate(time_list)}

    # ensure df is unique
    df_unique = (
        df.sort_values([project_col, time_col])
          .drop_duplicates(subset=[project_col, time_col])
    )

    total_checks = 0
    total_errors = 0

    for _ in range(num_samples):
        row = df_unique.sample(1).iloc[0]

        proj = row[project_col]
        time = row[time_col]

        if proj not in project_to_idx or time not in time_to_idx:
            continue

        i = project_to_idx[proj]
        t = time_to_idx[time]

        df_values = row[feature_columns].to_numpy(dtype=np.float32)
        npy_values = feature_array[i, t, :]

        if not np.allclose(df_values, npy_values, atol=atol, equal_nan=True):
            print("❌ Mismatch found:")
            print(f"Project: {proj}")
            print(f"Time: {time}")

            diff_idx = np.where(~np.isclose(df_values, npy_values, atol=atol, equal_nan=True))[0]

            for idx in diff_idx[:10]:  # limit output
                print(
                    f"  Column: {feature_columns[idx]} | "
                    f"DF: {df_values[idx]} | NPY: {npy_values[idx]}"
                )

            total_errors += 1
        else:
            print(f"✅ OK: {proj} | {time}")

        total_checks += 1

    print("\n==== Summary ====")
    print(f"Checked: {total_checks}")
    print(f"Errors: {total_errors}")

def save_graph_pt(
    df: pd.DataFrame,
    postfix_str: str,
    output_dir: str = ".",
    project_col: str = "Project Name",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    distance_threshold_m: float = 200.0,
    add_self_loops: bool = False,
    project_list: list | None = None,
):
    """
    Build a static graph among projects based on geographic distance and save to graph_{postfix}.pt

    Parameters
    ----------
    df : pd.DataFrame
    postfix_str : str
    output_dir : str
    project_col : str
    lat_col : str
    lon_col : str
    distance_threshold_m : float
        Connect two projects if distance <= this threshold (in meters)
    add_self_loops : bool
        Whether to add self-loop edges

    Returns
    -------
    graph_dict : dict
        {
            "edge_index": LongTensor [2, E],
            "node_names": list[str],
            "node_pos": FloatTensor [N, 2],   # [latitude, longitude]
            "distance_threshold_m": float,
            "project_to_idx": dict,
        }
    """

    os.makedirs(output_dir, exist_ok=True)

    df = df.copy()

    # keep one row per project with valid lat/lon
    node_df = (
        df[[project_col, lat_col, lon_col]]
        .dropna(subset=[project_col, lat_col, lon_col])
        .drop_duplicates(subset=[project_col], keep="first")
    )

    if project_list is None:
        node_df = node_df.sort_values(project_col).reset_index(drop=True)
    else:
        project_list = list(project_list)
        node_df = (
            node_df.set_index(project_col)
            .reindex(project_list)
            .reset_index()
        )
        missing_projects = node_df[[lat_col, lon_col]].isna().any(axis=1)
        if missing_projects.any():
            missing_names = node_df.loc[missing_projects, project_col].tolist()
            raise ValueError(
                "Missing latitude/longitude for projects in project_list: "
                f"{missing_names[:10]}"
            )

    if node_df.empty:
        raise ValueError("No valid project/latitude/longitude rows found.")

    node_names = node_df[project_col].tolist()
    latitudes = node_df[lat_col].to_numpy(dtype=np.float64)
    longitudes = node_df[lon_col].to_numpy(dtype=np.float64)

    N = len(node_names)
    project_to_idx = {p: i for i, p in enumerate(node_names)}

    # ------------------------------------------------------------------
    # Haversine distance matrix in meters
    # ------------------------------------------------------------------
    def haversine_distance_matrix(lat_deg, lon_deg):
        R = 6371000.0  # meters

        lat = np.radians(lat_deg)
        lon = np.radians(lon_deg)

        dlat = lat[:, None] - lat[None, :]
        dlon = lon[:, None] - lon[None, :]

        a = (
            np.sin(dlat / 2.0) ** 2
            + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    dist_mat = haversine_distance_matrix(latitudes, longitudes)

    # ------------------------------------------------------------------
    # Build edges
    # ------------------------------------------------------------------
    edge_list = []

    for i in range(N):
        for j in range(N):
            if i == j and not add_self_loops:
                continue
            if dist_mat[i, j] <= distance_threshold_m:
                edge_list.append((i, j))

    if len(edge_list) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    graph_dict = {
        "edge_index": edge_index,  # [2, E]
        "node_names": node_names,
        "node_pos": torch.tensor(
            np.stack([latitudes, longitudes], axis=1),
            dtype=torch.float32
        ),  # [N, 2]
        "distance_threshold_m": float(distance_threshold_m),
        "project_to_idx": project_to_idx,
    }

    save_path = os.path.join(output_dir, f"graph_{postfix_str}.pt")
    torch.save(graph_dict, save_path)

    print(f"Saved to: {save_path}")
    print(f"Number of nodes: {N}")
    print(f"Number of edges: {edge_index.shape[1]}")
    print(f"Distance threshold (m): {distance_threshold_m}")

    return graph_dict

print("\n\n=== TRAIN SET ===\n\n")
feature_array, feature_columns, project_list, time_list = save_feature_npy(
    df=df_train,
    postfix_str=POSTFIX_STR_TRAIN,
    output_dir=OUTPUT_DIR,
    project_list=project_list,
)

  
check_feature_npy_consistency(
    df=df_train,
    feature_array=feature_array,
    feature_columns=feature_columns,
    project_list=project_list,
    time_list=time_list,
    num_samples=20
)

print("\n\n=== TEST SET ===\n\n")
feature_array, feature_columns, project_list, time_list = save_feature_npy(
    df=df_test,
    postfix_str=POSTFIX_STR_TEST,
    output_dir=OUTPUT_DIR,
    project_list=project_list,
)

  
check_feature_npy_consistency(
    df=df_test,
    feature_array=feature_array,
    feature_columns=feature_columns,
    project_list=project_list,
    time_list=time_list,
    num_samples=20
)

graph_dict = save_graph_pt(
    df=DF,
    postfix_str=POSTFIX_STR,
    output_dir=OUTPUT_DIR,
    distance_threshold_m=DISTANCE_THRESHOLD_M,
    project_list=project_list,
)
