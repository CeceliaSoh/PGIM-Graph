"""Create the horizontally shifted target dataframe from the URA notebook.

This script intentionally reproduces only the minimum target-shift logic from
`URA_imputed_rental_lgbm_v3.2_optuna_h24obs.ipynb`.

It does not build lag features, run LightGBM, or run Optuna.
"""

from pathlib import Path

import pandas as pd


# Same input file used in the notebook.
CSV_PATH = Path("dataset/URA_renting_imputed_saits_no_cutoff_macro.csv")

# Change this to 1 or 12 if you want to inspect the h=1 or h=12 notebook target.
# The notebook's Task C / h24obs block uses HORIZON = 24.
HORIZON = 24

TARGET_SOURCE_COL = "rent_psf_imp"
TARGET_COL = f"target_t{HORIZON}"
TARGET_PERIOD_COL = f"_target_period_t{HORIZON}"
OUT_CSV = Path("temp.csv")

REQUIRED_COLS = [
    "date",
    "project_idx",
    TARGET_SOURCE_COL,
]


def validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """Fail early if the notebook's shift keys or target source are missing."""
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def add_months(p: int, m: int) -> int:
    """Same calendar-month helper used in the notebook."""
    y, mo = divmod(int(p), 100)
    mo += m
    y += (mo - 1) // 12
    mo = (mo - 1) % 12 + 1
    return y * 100 + mo


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {CSV_PATH}")

    # Load only the raw file and parse date, exactly as needed to create period.
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])
    validate_required_columns(df, REQUIRED_COLS)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    # The notebook sorts by project/date before creating period.
    df = df.sort_values(["project_idx", "date"]).reset_index(drop=True)

    # Minimum date preprocessing required for the horizontal shift join.
    df["period"] = df["date"].dt.year * 100 + df["date"].dt.month
    print(f"Before shift: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
    print(f"Period range: {df['period'].min()} -> {df['period'].max()}")

    # Keep a diagnostic copy of the future period. The notebook drops this
    # helper column after merging, but keeping it here makes alignment auditable.
    df[TARGET_PERIOD_COL] = df["period"].apply(lambda p: add_months(p, HORIZON))

    # Same horizontal shift as the notebook:
    # current row (project_idx, period=t) joins to the same project's row at
    # period=t+HORIZON, and pulls that future rent into target_t{HORIZON}.
    future_target = df[["project_idx", "period", TARGET_SOURCE_COL]].rename(
        columns={
            "period": TARGET_PERIOD_COL,
            TARGET_SOURCE_COL: TARGET_COL,
        }
    )
    shifted = df.merge(future_target, on=["project_idx", TARGET_PERIOD_COL], how="left")
    print(f"After shift merge: {shifted.shape[0]:,} rows x {shifted.shape[1]:,} columns")

    # Optional diagnostic: whether the future target row was observed or imputed.
    # This mirrors the notebook's later obs_lookup idea, but is not a feature.
    if "was_observed" in df.columns:
        future_observed = df[["project_idx", "period", "was_observed"]].rename(
            columns={
                "period": TARGET_PERIOD_COL,
                "was_observed": f"target_was_observed_t{HORIZON}",
            }
        )
        shifted = shifted.merge(future_observed, on=["project_idx", TARGET_PERIOD_COL], how="left")

    # The notebook drops rows whose future target is unavailable.
    shifted_non_null = shifted.dropna(subset=[TARGET_COL]).copy()
    print(
        f"After dropping missing {TARGET_COL}: "
        f"{shifted_non_null.shape[0]:,} rows x {shifted_non_null.shape[1]:,} columns"
    )

    # Keep only columns useful for verifying the horizontal alignment.
    useful_cols = [
        "project_idx",
        "condo_name",
        "size_tier",
        "district",
        "segment",
        "date",
        "period",
        TARGET_PERIOD_COL,
        TARGET_SOURCE_COL,
        "rent_psf_obs",
        "was_observed",
        TARGET_COL,
        f"target_was_observed_t{HORIZON}",
    ]
    useful_cols = [c for c in useful_cols if c in shifted_non_null.columns]
    shifted_check = shifted_non_null[useful_cols].copy()

    # Show a few rows per project so you can verify:
    # period + HORIZON == _target_period_tH, and target_tH is future rent.
    print("\nSample shifted rows by project:")
    sample_project_ids = shifted_check["project_idx"].drop_duplicates().head(3).tolist()
    for pid in sample_project_ids:
        print(f"\nproject_idx={pid}")
        print(shifted_check.loc[shifted_check["project_idx"] == pid].head(5).to_string(index=False))

    shifted_check.to_csv(OUT_CSV, index=False)
    print(f"\nSaved shifted dataframe to {OUT_CSV} ({shifted_check.shape[0]:,} rows x {shifted_check.shape[1]:,} columns)")


if __name__ == "__main__":
    main()
