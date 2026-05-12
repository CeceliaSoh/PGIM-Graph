"""Preprocessing for URA LightGBM rental forecasting.

This module refactors the preprocessing and feature engineering logic from
`URA_imputed_rental_lgbm_v3.2_optuna_h24obs.ipynb` into reusable functions.
It intentionally stops before model training.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


CSV_PATH = Path("dataset/URA_renting_imputed_saits_no_cutoff_macro.csv")
FALLBACK_PROJECT_PATH = Path("URA_merged_full_v3.csv")

TRAIN_START = 200001
TEST_CUTOFF = 202001
N_SPLITS = 3
VAL_MONTHS = 12

TARGET_BASE_COL = "rent_psf_imp"
OBS_TARGET_BASE_COL = "rent_psf_obs"
PROJECT_COL = "project_idx"
DATE_COL = "date"
DISTRICT_COL = "district"
OBS_FLAG_COL = "was_observed"

REQUIRED_INPUT_COLS = [
    DATE_COL,
    PROJECT_COL,
    DISTRICT_COL,
    TARGET_BASE_COL,
    OBS_TARGET_BASE_COL,
    OBS_FLAG_COL,
    "condo_name",
    "Condo_Age_2026",
    "tenure_remaining_years",
    "Large_Dev_200plus",
    "nearest_mrt_dist",
]

MACRO_COLS = [
    "cpi_all_items_infl_yoy_pct",
    "unemployment_rate_sa_pct",
    "sora_compounded_3m_pct",
    "sg_govt_bond_yield_10y_pct",
    "sgd_per_usd_logret_12m_pct",
    "ura_private_rental_index_yoy_pct",
    "ura_private_price_index_avg_3seg_yoy_pct",
    "hdb_resale_price_index_yoy_pct",
    "gva_yoy_growth_pct",
    "rental_index_nonlanded_yoy_log_growth_lag1q",
    "rental_index_nonlanded_in_ccr_yoy_log_growth_lag1q",
    "rental_index_nonlanded_in_ocr_yoy_log_growth_lag1q",
    "rental_index_nonlanded_in_rcr_yoy_log_growth_lag1q",
    "ppi_nonlanded_yoy_log_growth_lag1q",
    "ppi_nonlanded_in_ccr_yoy_log_growth_lag1q",
    "ppi_nonlanded_in_ocr_yoy_log_growth_lag1q",
    "ppi_nonlanded_in_rcr_yoy_log_growth_lag1q",
    "log1p_vacant_nonlanded_units_lag1q",
    "vacancy_rate_nonlanded_units_lag1q",
    "log1p_nonlanded_private_units_in_the_pipeline_lag1q",
    "nonlanded_private_units_in_the_pipeline_under_construction_share_lag1q",
    "log1p_non_landed_units_launched_in_private_res_projects_with_prereq_for_sale_lag1q",
    "non_landed_units_in_private_res_projects_with_prereq_not_launched_share_lag1q",
    "log1p_unsold_completed_nonlanded_private_units_lag1q",
    "log1p_unsold_non_landed_units_in_launched_private_res_projects_lag1q",
    "assessed_non_landed_private_residential_owned_by_companies_share_lag1q",
    "assessed_non_landed_private_residential_owned_by_pr_foreigners_share_lag1q",
    "uncompleted_non_landed_private_residential_purchased_by_companies_share_lag1q",
    "uncompleted_non_landed_private_residential_purchased_by_foreigners_share_lag1q",
    "uncompleted_non_landed_private_residential_purchased_by_spr_share_lag1q",
]

BASE_FEATURES = [
    "area_sqft_med",
    "Condo_Age_2026",
    "tenure_freehold_like",
    "tenure_medium_lease_60_80",
    "tenure_more_than_80",
    "tenure_short_lease_lt60",
    "tenure_unknown",
    "tenure_remaining_years",
    "project_size_small",
    "project_size_medium",
    "project_size_large",
    "Large_Dev_200plus",
    "Postal District",
    "Planning Area",
    "Neighbourhood",
    "nearest_mrt_dist",
    "size_tier",
    "nearest_bus_stops_dist",
    "n_bus_stops_top3",
    "nearest_supermarkets_dist",
    "n_supermarkets_top3",
    "nearest_parks_dist",
    "n_parks_top3",
    "nearest_clinics_dist",
    "n_clinics_top3",
    "nearest_bank_dist",
    "n_bank_top3",
    "nearest_atms_dist",
    "n_atms_top3",
    "nearest_school_dist",
    "latitude",
    "longitude",
    "month_of_year",
    "glob_med_1m",
    "glob_med_3m",
    "glob_med_12m",
    "glob_trend",
    "dist_med_1m",
    "dist_med_3m",
    "dist_med_12m",
    "dist_trend",
    "condo_level_1m",
    "condo_level_12m",
    "obs_frac_12m",
    "ura_private_rental_index_yoy_pct",
    "sora_compounded_3m_pct",
    "gva_yoy_growth_pct",
    "cpi_all_items_infl_yoy_pct",
    "unemployment_rate_sa_pct",
    "sg_govt_bond_yield_10y_pct",
    "sgd_per_usd_logret_12m_pct",
    "ura_private_price_index_avg_3seg_yoy_pct",
    "hdb_resale_price_index_yoy_pct",
    "rental_index_nonlanded_yoy_log_growth_lag1q",
    "rental_index_nonlanded_in_ccr_yoy_log_growth_lag1q",
    "rental_index_nonlanded_in_ocr_yoy_log_growth_lag1q",
    "rental_index_nonlanded_in_rcr_yoy_log_growth_lag1q",
    "ppi_nonlanded_yoy_log_growth_lag1q",
    "ppi_nonlanded_in_ccr_yoy_log_growth_lag1q",
    "ppi_nonlanded_in_ocr_yoy_log_growth_lag1q",
    "ppi_nonlanded_in_rcr_yoy_log_growth_lag1q",
    "log1p_vacant_nonlanded_units_lag1q",
    "vacancy_rate_nonlanded_units_lag1q",
    "log1p_nonlanded_private_units_in_the_pipeline_lag1q",
    "nonlanded_private_units_in_the_pipeline_under_construction_share_lag1q",
    "log1p_non_landed_units_launched_in_private_res_projects_with_prereq_for_sale_lag1q",
    "non_landed_units_in_private_res_projects_with_prereq_not_launched_share_lag1q",
    "log1p_unsold_completed_nonlanded_private_units_lag1q",
    "log1p_unsold_non_landed_units_in_launched_private_res_projects_lag1q",
    "assessed_non_landed_private_residential_owned_by_companies_share_lag1q",
    "assessed_non_landed_private_residential_owned_by_pr_foreigners_share_lag1q",
    "uncompleted_non_landed_private_residential_purchased_by_companies_share_lag1q",
    "uncompleted_non_landed_private_residential_purchased_by_foreigners_share_lag1q",
    "uncompleted_non_landed_private_residential_purchased_by_spr_share_lag1q",
]


@dataclass
class LGBMPreprocessResult:
    processed_df: pd.DataFrame
    model_df: pd.DataFrame
    feature_cols: list[str]
    categorical_cols: list[str]
    target_col: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame | None
    y_valid: pd.Series | None
    X_test: pd.DataFrame
    y_test: pd.Series
    cv_splits: list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]
    metadata: dict


def log_shape(label: str, df: pd.DataFrame) -> None:
    print(f"{label}: {len(df):,} rows x {len(df.columns):,} columns")


def validate_required_columns(df: pd.DataFrame, required_cols: Iterable[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_input_data(csv_path: Path | str = CSV_PATH) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])
    validate_required_columns(df, REQUIRED_INPUT_COLS)
    print(f"Loaded source file: {csv_path}")
    log_shape("Raw data", df)
    return df


def prepare_date_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df, [DATE_COL, PROJECT_COL])
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="raise")
    df = df.sort_values([PROJECT_COL, DATE_COL]).reset_index(drop=True)
    df["period"] = df[DATE_COL].dt.year * 100 + df[DATE_COL].dt.month
    df["month_of_year"] = df[DATE_COL].dt.month
    df["lease_year"] = df[DATE_COL].dt.year
    df["lease_quarter"] = df[DATE_COL].dt.quarter
    print(
        "Date range: "
        f"{df[DATE_COL].min().strftime('%Y-%m')} -> {df[DATE_COL].max().strftime('%Y-%m')}"
    )
    log_shape("After date features", df)
    return df


def convert_range_to_number(value: object) -> float:
    if pd.isna(value) or not isinstance(value, str):
        return np.nan
    parts = value.replace(",", "").split(" to ")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return np.nan
    return float(sum(nums) / len(nums))


def make_size_bucket(area_sqft: pd.Series, q: int = 5) -> pd.Series:
    s_valid = area_sqft.dropna()
    out = pd.Series(index=area_sqft.index, dtype="object")
    if s_valid.nunique() < 2:
        out[:] = "SZ1"
        return out
    q_use = min(q, int(s_valid.nunique()))
    labels = [f"SZ{i + 1}" for i in range(q_use)]
    try:
        bucket = pd.qcut(s_valid.rank(method="first"), q=q_use, labels=labels, duplicates="drop")
    except ValueError:
        bucket = pd.cut(s_valid, bins=q_use, labels=labels, include_lowest=True, duplicates="drop")
    out.loc[s_valid.index] = bucket.astype(str)
    return out.fillna("SZ1")


def fix_missing_condo_name(
    df: pd.DataFrame,
    fallback_project_path: Path | str = FALLBACK_PROJECT_PATH,
) -> pd.DataFrame:
    validate_required_columns(df, ["condo_name", PROJECT_COL])
    df = df.copy()
    mask = df["condo_name"].isna() | (df["condo_name"] == "Unknown")
    if not mask.any():
        print("condo_name OK")
        return df

    fallback_project_path = Path(fallback_project_path)
    if not fallback_project_path.exists():
        raise FileNotFoundError(
            "condo_name contains missing/Unknown values, but fallback file is missing: "
            f"{fallback_project_path}"
        )

    required = ["Project Name", "Postal District", "Floor Area (SQFT)"]
    df_tmp = pd.read_csv(fallback_project_path, usecols=required)
    df_tmp = df_tmp[pd.to_numeric(df_tmp["Postal District"], errors="coerce").isin([1, 2, 9, 10, 11])]
    df_tmp["Area"] = df_tmp["Floor Area (SQFT)"].apply(convert_range_to_number)
    df_tmp["sz"] = make_size_bucket(df_tmp["Area"], q=5)
    df_tmp["ps"] = df_tmp["Project Name"].astype(str).str.strip() + " | " + df_tmp["sz"]
    idx_to_series = {i: name for i, name in enumerate(sorted(df_tmp["ps"].unique()))}
    size_label_map = {
        "SZ1": "very_small",
        "SZ2": "small",
        "SZ3": "medium",
        "SZ4": "large",
        "SZ5": "very_large",
    }

    mapped = df.loc[mask, PROJECT_COL].map(
        lambda i: idx_to_series.get(i, "Unknown | Unknown").rsplit(" | ", 1)
    )
    df.loc[mask, "condo_name"] = mapped.map(lambda x: x[0])
    df.loc[mask, "size_tier"] = mapped.map(lambda x: x[-1])
    df.loc[mask, "size_label"] = df.loc[mask, "size_tier"].map(size_label_map)
    print(f"Fixed condo_name. Still missing: {df['condo_name'].isna().sum():,}")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    validate_required_columns(df, ["Condo_Age_2026", "tenure_remaining_years", "Large_Dev_200plus"])

    amenity_dist_cols = [
        c
        for c in df.columns
        if c.startswith("nearest_")
        and c
        not in ["nearest_mrt_dist", "nearest_mrt_name", "nearest_school_dist", "nearest_school_name"]
    ]
    for col in amenity_dist_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in [c for c in df.columns if c.endswith("_top3")]:
        df[col] = df[col].fillna(0)

    school_dist_cols = [c for c in df.columns if "School" in c or "Junior" in c or "Institution" in c]
    for col in school_dist_cols + ["nearest_school_dist"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    if "nearest_school_name" in df.columns:
        df["nearest_school_name"] = df["nearest_school_name"].fillna("Unknown")

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df["Condo_Age_2026"] = df.groupby(DISTRICT_COL)["Condo_Age_2026"].transform(
        lambda x: x.fillna(x.median())
    )
    df["Condo_Age_2026"] = df["Condo_Age_2026"].fillna(df["Condo_Age_2026"].median())
    df["tenure_remaining_years"] = df["tenure_remaining_years"].fillna(
        df["tenure_remaining_years"].median()
    )

    macro_cols = [c for c in MACRO_COLS if c in df.columns]
    if macro_cols:
        df[macro_cols] = df.sort_values(DATE_COL).groupby(DATE_COL)[macro_cols].transform("first")
        df[macro_cols] = df[macro_cols].ffill().bfill()

    df["Large_Dev_200plus"] = df["Large_Dev_200plus"].fillna(0)
    for col in ["project_size_small", "project_size_medium", "project_size_large"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    df["nearest_mrt_dist"] = df["nearest_mrt_dist"].fillna(df["nearest_mrt_dist"].median())
    for col in [
        "Planning Area",
        "Neighbourhood",
        "condo_name",
        "size_tier",
        "size_label",
        "segment",
        "nearest_mrt_name",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    missing = df.isna().sum()
    remaining = missing[missing > 0].to_dict()
    print(f"Missing after fill: {remaining if remaining else 'None'}")
    log_shape("After missing-value fill", df)
    return df


def build_month_grid(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df, [DATE_COL, "period"])
    month_grid = pd.DataFrame({"date": pd.date_range(df[DATE_COL].min(), df[DATE_COL].max(), freq="MS")})
    month_grid["period"] = month_grid["date"].dt.year * 100 + month_grid["date"].dt.month
    return month_grid


def _suffix_for_shift(shift_months: int, observed_only: bool) -> str:
    suffix = "_obs" if observed_only else ""
    if shift_months != 1:
        suffix += f"_t{shift_months}"
    return suffix


def build_global_lag_features(
    df: pd.DataFrame,
    month_grid: pd.DataFrame,
    raw_col: str,
    shift_months: int = 1,
    observed_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = _suffix_for_shift(shift_months, observed_only)
    raw = df.groupby("period")[raw_col].median().reset_index().rename(columns={raw_col: "_raw"})
    glob = month_grid.merge(raw, on="period", how="left").sort_values("period").reset_index(drop=True)
    # Leakage-sensitive: shift before rolling so the current target month is never used.
    shifted = glob["_raw"].shift(shift_months)
    glob[f"glob_med_1m{suffix}"] = shifted
    glob[f"glob_med_3m{suffix}"] = shifted.rolling(3, min_periods=1).mean()
    glob[f"glob_med_12m{suffix}"] = shifted.rolling(12, min_periods=3).mean()
    glob[f"glob_trend{suffix}"] = glob[f"glob_med_1m{suffix}"] - glob[f"glob_med_12m{suffix}"]
    cols = ["period", f"glob_med_1m{suffix}", f"glob_med_3m{suffix}", f"glob_med_12m{suffix}", f"glob_trend{suffix}"]
    out = df.merge(glob[cols], on="period", how="left")
    return out, raw


def build_district_lag_features(
    df: pd.DataFrame,
    month_grid: pd.DataFrame,
    raw_by_district: pd.DataFrame,
    shift_months: int = 1,
    observed_only: bool = False,
) -> pd.DataFrame:
    suffix = _suffix_for_shift(shift_months, observed_only)
    parts = []
    for district in df[DISTRICT_COL].dropna().unique():
        g = month_grid[["period"]].copy()
        g[DISTRICT_COL] = district
        district_raw = raw_by_district[raw_by_district[DISTRICT_COL] == district][["period", "_raw"]]
        g = g.merge(district_raw, on="period", how="left").sort_values("period")
        # Leakage-sensitive: shift before rolling so validation/test months do not leak into features.
        shifted = g["_raw"].shift(shift_months)
        g[f"dist_med_1m{suffix}"] = shifted
        g[f"dist_med_3m{suffix}"] = shifted.rolling(3, min_periods=1).mean()
        g[f"dist_med_12m{suffix}"] = shifted.rolling(12, min_periods=3).mean()
        g[f"dist_trend{suffix}"] = g[f"dist_med_1m{suffix}"] - g[f"dist_med_12m{suffix}"]
        parts.append(g)
    if not parts:
        raise ValueError("No district values available to build district lag features.")
    lag = pd.concat(parts, ignore_index=True)
    cols = [
        DISTRICT_COL,
        "period",
        f"dist_med_1m{suffix}",
        f"dist_med_3m{suffix}",
        f"dist_med_12m{suffix}",
        f"dist_trend{suffix}",
    ]
    return df.merge(lag[cols], on=[DISTRICT_COL, "period"], how="left")


def build_condo_lag_features(
    df: pd.DataFrame,
    month_grid: pd.DataFrame,
    raw_by_project: pd.DataFrame,
    shift_months: int = 1,
    observed_only: bool = False,
) -> pd.DataFrame:
    suffix = _suffix_for_shift(shift_months, observed_only)
    parts = []
    for project_idx in df[PROJECT_COL].unique():
        g = month_grid[["period"]].copy()
        g[PROJECT_COL] = project_idx
        project_raw = raw_by_project[raw_by_project[PROJECT_COL] == project_idx][["period", "_raw"]]
        g = g.merge(project_raw, on="period", how="left").sort_values("period")
        # Leakage-sensitive: shift before rolling so only historical project rents are used.
        shifted = g["_raw"].shift(shift_months)
        g[f"condo_level_1m{suffix}"] = shifted
        g[f"condo_level_12m{suffix}"] = shifted.rolling(12, min_periods=3).mean()
        parts.append(g)
    if not parts:
        raise ValueError("No project_idx values available to build condo lag features.")
    lag = pd.concat(parts, ignore_index=True)
    cols = [PROJECT_COL, "period", f"condo_level_1m{suffix}", f"condo_level_12m{suffix}"]
    out = df.merge(lag[cols], on=[PROJECT_COL, "period"], how="left")
    out[f"condo_level_1m{suffix}"] = (
        out[f"condo_level_1m{suffix}"]
        .fillna(out[f"dist_med_1m{suffix}"])
        .fillna(out[f"glob_med_1m{suffix}"])
    )
    out[f"condo_level_12m{suffix}"] = (
        out[f"condo_level_12m{suffix}"]
        .fillna(out[f"dist_med_12m{suffix}"])
        .fillna(out[f"glob_med_12m{suffix}"])
    )
    return out


def build_lag_feature_family(
    df: pd.DataFrame,
    month_grid: pd.DataFrame,
    raw_col: str,
    shift_months: int = 1,
    observed_only: bool = False,
) -> pd.DataFrame:
    df, _ = build_global_lag_features(df, month_grid, raw_col, shift_months, observed_only)
    raw_by_district = (
        df.groupby([DISTRICT_COL, "period"])[raw_col].median().reset_index().rename(columns={raw_col: "_raw"})
    )
    df = build_district_lag_features(df, month_grid, raw_by_district, shift_months, observed_only)
    raw_by_project = (
        df.groupby([PROJECT_COL, "period"])[raw_col].median().reset_index().rename(columns={raw_col: "_raw"})
    )
    df = build_condo_lag_features(df, month_grid, raw_by_project, shift_months, observed_only)
    label = f"lag-{shift_months}{' observed-only' if observed_only else ''}"
    print(f"Built {label} features")
    return df


def build_all_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    month_grid = build_month_grid(df)

    df = build_lag_feature_family(df, month_grid, TARGET_BASE_COL, shift_months=1, observed_only=False)
    df = build_lag_feature_family(df, month_grid, TARGET_BASE_COL, shift_months=12, observed_only=False)
    df = build_lag_feature_family(df, month_grid, TARGET_BASE_COL, shift_months=24, observed_only=False)

    df["_obs_raw"] = df[OBS_TARGET_BASE_COL]
    df = build_lag_feature_family(df, month_grid, "_obs_raw", shift_months=1, observed_only=True)
    df = build_lag_feature_family(df, month_grid, "_obs_raw", shift_months=12, observed_only=True)
    df = build_lag_feature_family(df, month_grid, "_obs_raw", shift_months=24, observed_only=True)
    df = df.drop(columns=["_obs_raw"])
    log_shape("After lag features", df)
    return df


def prepare_categorical_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    df["obs_frac_12m"] = df.groupby(PROJECT_COL)[OBS_FLAG_COL].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).mean()
    )
    df["obs_frac_12m_t12"] = df.groupby(PROJECT_COL)[OBS_FLAG_COL].transform(
        lambda x: x.shift(12).rolling(12, min_periods=1).mean()
    )
    df["obs_frac_12m_t24"] = df.groupby(PROJECT_COL)[OBS_FLAG_COL].transform(
        lambda x: x.shift(24).rolling(12, min_periods=1).mean()
    )

    df["Postal District"] = df[DISTRICT_COL].astype(str)
    categorical_cols = ["Postal District", "Planning Area", "Neighbourhood", "size_tier"]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    print(f"Categorical features: {categorical_cols}")
    return df, categorical_cols


def build_feature_columns(df: pd.DataFrame, categorical_cols: list[str]) -> dict[str, list[str]]:
    has_cols = sorted([c for c in df.columns if c.startswith("Has_")])
    features_a = list(dict.fromkeys(BASE_FEATURES + has_cols))
    features_a = [c for c in features_a if c in df.columns]
    cat_features_a = [c for c in categorical_cols if c in features_a]

    lag_imp_cols = [
        "glob_med_1m",
        "glob_med_3m",
        "glob_med_12m",
        "glob_trend",
        "dist_med_1m",
        "dist_med_3m",
        "dist_med_12m",
        "dist_trend",
        "condo_level_1m",
        "condo_level_12m",
    ]
    features_obs = [
        f for f in features_a if f not in lag_imp_cols
    ] + [
        "glob_med_1m_obs",
        "glob_med_3m_obs",
        "glob_med_12m_obs",
        "glob_trend_obs",
        "dist_med_1m_obs",
        "dist_med_3m_obs",
        "dist_med_12m_obs",
        "dist_trend_obs",
        "condo_level_1m_obs",
        "condo_level_12m_obs",
    ]
    features_obs = [f for f in features_obs if f in df.columns]

    lag1_cols = lag_imp_cols + ["obs_frac_12m"]
    lag12_cols = [
        "glob_med_1m_t12",
        "glob_med_3m_t12",
        "glob_med_12m_t12",
        "glob_trend_t12",
        "dist_med_1m_t12",
        "dist_med_3m_t12",
        "dist_med_12m_t12",
        "dist_trend_t12",
        "condo_level_1m_t12",
        "condo_level_12m_t12",
        "obs_frac_12m_t12",
    ]
    features_t12_base = list(dict.fromkeys([f for f in features_a if f not in lag1_cols] + lag12_cols))

    obs_lag1_cols = [
        "glob_med_1m_obs",
        "glob_med_3m_obs",
        "glob_med_12m_obs",
        "glob_trend_obs",
        "dist_med_1m_obs",
        "dist_med_3m_obs",
        "dist_med_12m_obs",
        "dist_trend_obs",
        "condo_level_1m_obs",
        "condo_level_12m_obs",
        "obs_frac_12m",
    ]
    obs_lag12_cols = [
        "glob_med_1m_obs_t12",
        "glob_med_3m_obs_t12",
        "glob_med_12m_obs_t12",
        "glob_trend_obs_t12",
        "dist_med_1m_obs_t12",
        "dist_med_3m_obs_t12",
        "dist_med_12m_obs_t12",
        "dist_trend_obs_t12",
        "condo_level_1m_obs_t12",
        "condo_level_12m_obs_t12",
        "obs_frac_12m_t12",
    ]
    features_t12_obs_base = list(
        dict.fromkeys([f for f in features_obs if f not in obs_lag1_cols] + obs_lag12_cols)
    )

    lag24_cols = [
        "glob_med_1m_t24",
        "glob_med_3m_t24",
        "glob_med_12m_t24",
        "glob_trend_t24",
        "dist_med_1m_t24",
        "dist_med_3m_t24",
        "dist_med_12m_t24",
        "dist_trend_t24",
        "condo_level_1m_t24",
        "condo_level_12m_t24",
        "obs_frac_12m_t24",
    ]
    features_t24_base = list(dict.fromkeys([f for f in features_a if f not in lag1_cols] + lag24_cols))

    lag24_obs_cols = [
        "glob_med_1m_obs_t24",
        "glob_med_3m_obs_t24",
        "glob_med_12m_obs_t24",
        "glob_trend_obs_t24",
        "dist_med_1m_obs_t24",
        "dist_med_3m_obs_t24",
        "dist_med_12m_obs_t24",
        "dist_trend_obs_t24",
        "condo_level_1m_obs_t24",
        "condo_level_12m_obs_t24",
        "obs_frac_12m_t24",
    ]
    features_t24_obs_base = list(
        dict.fromkeys([f for f in features_obs if f not in obs_lag1_cols] + lag24_obs_cols)
    )

    features = {
        "h1": features_a,
        "h1_obs": features_obs,
        "h12": [f for f in features_t12_base if f in df.columns],
        "h12_obs": [f for f in features_t12_obs_base if f in df.columns],
        "h24": [f for f in features_t24_base if f in df.columns],
        "h24_obs": [f for f in features_t24_obs_base if f in df.columns],
        "cat_h1": cat_features_a,
    }
    print(f"Task A features          : {len(features['h1'])} ({len(cat_features_a)} categorical)")
    print(f"Observed-only features   : {len(features['h1_obs'])} ({len(cat_features_a)} categorical)")
    print(f"Task B features (h=12)   : {len(features['h12'])}")
    print(f"Task C features (h=24)   : {len(features['h24'])}")
    print(f"Task C Obs-only (h=24)   : {len(features['h24_obs'])}")
    return features


def add_months(period: int, months: int) -> int:
    year, month = divmod(int(period), 100)
    month += months
    year += (month - 1) // 12
    month = (month - 1) % 12 + 1
    return year * 100 + month


def make_forward_target(
    df: pd.DataFrame,
    horizon: int,
    source_target_col: str = TARGET_BASE_COL,
    observed_only: bool = False,
) -> pd.DataFrame:
    validate_required_columns(df, [PROJECT_COL, "period", source_target_col])
    out = df.copy()
    target_col = f"target_t{horizon}"
    join_col = f"_target_period_t{horizon}"
    out = out.drop(columns=[c for c in [join_col, target_col] if c in out.columns])
    out[join_col] = out["period"].apply(lambda p: add_months(p, horizon))
    ref = df[[PROJECT_COL, "period", source_target_col]].rename(
        columns={"period": join_col, source_target_col: target_col}
    )
    if observed_only:
        ref = df.loc[df[OBS_FLAG_COL], [PROJECT_COL, "period", source_target_col]].rename(
            columns={"period": join_col, source_target_col: target_col}
        )
    out = out.merge(ref, on=[PROJECT_COL, join_col], how="left").drop(columns=join_col)
    return out


def month_cv_split(
    X_df: pd.DataFrame,
    y_ser: pd.Series,
    period_series: pd.Series,
    periods_arr: np.ndarray,
    tscv: TimeSeriesSplit,
) -> list[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
    period_col = period_series.loc[X_df.index]
    splits = []
    for tr_idx, va_idx in tscv.split(periods_arr):
        tr_mask = period_col.isin(set(periods_arr[tr_idx]))
        va_mask = period_col.isin(set(periods_arr[va_idx]))
        splits.append((X_df.loc[tr_mask], y_ser.loc[tr_mask], X_df.loc[va_mask], y_ser.loc[va_mask]))
    return splits


def make_time_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    horizon: int,
    train_start: int = TRAIN_START,
    test_cutoff: int = TEST_CUTOFF,
    n_splits: int = N_SPLITS,
    val_months: int = VAL_MONTHS,
) -> LGBMPreprocessResult:
    validate_required_columns(df, ["period", target_col, PROJECT_COL, OBS_FLAG_COL])
    model_df = df.dropna(subset=[target_col]).copy()
    all_periods = np.sort(model_df["period"].unique())
    train_periods = all_periods[(all_periods >= train_start) & (all_periods < test_cutoff)]
    test_periods = all_periods[all_periods >= test_cutoff]
    if len(train_periods) == 0 or len(test_periods) == 0:
        raise ValueError(
            f"Time split produced empty train/test periods. "
            f"train={len(train_periods)}, test={len(test_periods)}"
        )

    mask_train = (model_df["period"] >= train_start) & (model_df["period"] < test_cutoff)
    mask_test = model_df["period"] >= test_cutoff
    X_train = model_df.loc[mask_train, feature_cols].copy()
    y_train = model_df.loc[mask_train, target_col].copy()
    X_test = model_df.loc[mask_test, feature_cols].copy()
    y_test = model_df.loc[mask_test, target_col].copy()
    train_period_series = model_df.loc[mask_train, "period"].copy()

    gap_months = max(horizon - 1, 0)
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=val_months, gap=gap_months)
    cv_splits = month_cv_split(X_train, y_train, train_period_series, train_periods, tscv)
    X_valid = cv_splits[-1][2].copy() if cv_splits else None
    y_valid = cv_splits[-1][3].copy() if cv_splits else None

    obs_lookup = df.set_index([PROJECT_COL, "period"])[OBS_FLAG_COL]
    target_periods = np.array([add_months(p, horizon) for p in model_df.loc[mask_test, "period"].values])
    proj_ids = model_df.loc[mask_test, PROJECT_COL].values
    obs_mask_test = np.array([obs_lookup.get((pid, tp), False) for pid, tp in zip(proj_ids, target_periods)])

    print("=" * 60)
    print(f"Train range                 : {train_periods[0]} - {train_periods[-1]}")
    print(f"Test range                  : {test_periods[0]} - {test_periods[-1]}")
    print(f"Train rows                  : {len(y_train):,}")
    print(f"Test rows (total)           : {len(y_test):,}")
    print(f"Test rows (target observed) : {obs_mask_test.sum():,} ({100 * obs_mask_test.mean():.1f}%)")
    print(f"Walk-forward CV             : n_splits={n_splits}, val_months={val_months}, gap={gap_months}")
    print("=" * 60)

    metadata = {
        "horizon": horizon,
        "train_start": train_start,
        "test_cutoff": test_cutoff,
        "train_periods": train_periods,
        "test_periods": test_periods,
        "obs_mask_test": obs_mask_test,
        "gap_months": gap_months,
        "n_splits": n_splits,
        "val_months": val_months,
    }

    return LGBMPreprocessResult(
        processed_df=df,
        model_df=model_df,
        feature_cols=feature_cols,
        categorical_cols=[],
        target_col=target_col,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        cv_splits=cv_splits,
        metadata=metadata,
    )


def prepare_lgbm_inputs(
    df: pd.DataFrame | None = None,
    csv_path: Path | str = CSV_PATH,
    fallback_project_path: Path | str = FALLBACK_PROJECT_PATH,
    horizon: int = 24,
    observed_only: bool = False,
) -> LGBMPreprocessResult:
    if horizon not in {1, 12, 24}:
        raise ValueError("horizon must be one of {1, 12, 24} to match the notebook.")

    if df is None:
        df = load_input_data(csv_path)
    else:
        df = df.copy()
        validate_required_columns(df, REQUIRED_INPUT_COLS)

    df = prepare_date_features(df)
    df = fix_missing_condo_name(df, fallback_project_path=fallback_project_path)
    df = fill_missing_values(df)
    df = build_all_lag_features(df)
    df, categorical_cols = prepare_categorical_features(df)
    feature_sets = build_feature_columns(df, categorical_cols)

    if observed_only:
        source_target_col = OBS_TARGET_BASE_COL
        base_df = df[df[OBS_FLAG_COL]].copy()
        feature_key = f"h{horizon}_obs"
    else:
        source_target_col = TARGET_BASE_COL
        base_df = df.copy()
        feature_key = f"h{horizon}"

    model_df = make_forward_target(
        base_df,
        horizon=horizon,
        source_target_col=source_target_col,
        observed_only=observed_only,
    )
    target_col = f"target_t{horizon}"
    if horizon == 1 and not observed_only:
        feature_key = "h1"
    elif horizon == 1 and observed_only:
        feature_key = "h1_obs"

    feature_cols = [c for c in feature_sets[feature_key] if c in model_df.columns]
    categorical_feature_cols = [c for c in categorical_cols if c in feature_cols]
    result = make_time_split(model_df, feature_cols, target_col, horizon=horizon)
    result.processed_df = df
    result.categorical_cols = categorical_feature_cols
    result.metadata.update(
        {
            "csv_path": str(csv_path),
            "fallback_project_path": str(fallback_project_path),
            "observed_only": observed_only,
            "feature_key": feature_key,
            "feature_sets": feature_sets,
        }
    )
    print(f"Ready for LightGBM: X_train={result.X_train.shape}, X_test={result.X_test.shape}")
    return result


def save_preprocess_result(result: LGBMPreprocessResult, output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(result, f)
    print(f"Saved preprocessing result: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare URA LightGBM inputs without training a model.")
    parser.add_argument("--csv-path", default=str(CSV_PATH))
    parser.add_argument("--fallback-project-path", default=str(FALLBACK_PROJECT_PATH))
    parser.add_argument("--horizon", type=int, default=24, choices=[1, 12, 24])
    parser.add_argument("--observed-only", action="store_true")
    parser.add_argument("--output", default="temp.csv")
    args = parser.parse_args()

    result = prepare_lgbm_inputs(
        csv_path=args.csv_path,
        fallback_project_path=args.fallback_project_path,
        horizon=args.horizon,
        observed_only=args.observed_only,
    )
    if args.output:
        save_preprocess_result(result, args.output)


if __name__ == "__main__":
    main()
