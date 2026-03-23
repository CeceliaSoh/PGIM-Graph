from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from pymongo import ASCENDING, MongoClient


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "dataset" / "mongodb_exports"

URA_V2_PATH = DATASET_DIR / "URA_enriched_with_99co_v2.csv"
URA_V3_PATH = DATASET_DIR / "URA_enriched_with_99co_v3.csv"
PROPERTY_DETAILS_PATH = DATASET_DIR / "property_details_facilities.csv"
MRT_LRT_PATH = DATASET_DIR / "mrt_lrt_stations.csv"
AMENITIES_PATH = DATASET_DIR / "nearby_5km_amenities.csv"

MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "property_db"
INSERT_TO_MONGODB = True
CLEAR_EXISTING_COLLECTIONS = True


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
WARNINGS: list[str] = []


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def log(message: str) -> None:
    """Print a clear progress message."""
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    """Print and collect a warning."""
    WARNINGS.append(message)
    print(f"[WARN] {message}")


def sanitize_identifier(value: str | None) -> str | None:
    """Return a normalized identifier-safe string."""
    if value is None:
        return None
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text or None


def normalize_column_name(column_name: str) -> str:
    """Convert a column name to lowercase snake_case without symbols/spaces."""
    text = str(column_name).strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"\(\s*\$\s*\)", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def normalize_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all output columns to lowercase snake_case."""
    result = df.copy()
    result.columns = [normalize_column_name(column) for column in result.columns]
    return result


def standardize_string_value(value):
    """Trim and normalize repeated whitespace for scalar string values."""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return value


def standardize_string_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize whitespace across all object columns."""
    result = df.copy()
    object_columns = result.select_dtypes(include=["object"]).columns
    for column in object_columns:
        result[column] = result[column].map(standardize_string_value)
    return result


def load_csv(path: Path, usecols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load a CSV file with logging and optional column selection."""
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")

    usecols_list = list(usecols) if usecols is not None else None
    selected_columns = None

    if usecols_list is not None:
        actual_columns = load_header(path)
        selected_columns = [column for column in usecols_list if column in actual_columns]
        missing_columns = [column for column in usecols_list if column not in actual_columns]
        if missing_columns:
            for column in missing_columns:
                warn(
                    f"{path.name}: requested column '{column}' was not found during load. "
                    f"Closest matches: {build_close_match_message(column, actual_columns)}"
                )

    log(
        f"Loading {path.name}"
        + (f" with {len(selected_columns)} selected columns" if selected_columns is not None else "")
    )
    df = pd.read_csv(path, usecols=selected_columns, low_memory=False)
    df = standardize_string_whitespace(df)
    log(f"Loaded {path.name}: {len(df):,} rows x {len(df.columns):,} columns")
    return df


def load_header(path: Path) -> list[str]:
    """Read only the header row from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")
    return pd.read_csv(path, nrows=0).columns.tolist()


def build_close_match_message(expected: str, actual_columns: list[str]) -> str:
    """Return a small list of close matches for a missing column."""
    matches = difflib.get_close_matches(expected, actual_columns, n=5, cutoff=0.4)
    return ", ".join(matches) if matches else "No close matches found"


def safe_select_columns(df: pd.DataFrame, columns: list[str], dataset_name: str) -> pd.DataFrame:
    """Select columns safely and warn for any that are missing."""
    available = [column for column in columns if column in df.columns]
    missing = [column for column in columns if column not in df.columns]

    if missing:
        actual_columns = df.columns.tolist()
        for column in missing:
            warn(
                f"{dataset_name}: missing expected column '{column}'. "
                f"Closest matches: {build_close_match_message(column, actual_columns)}"
            )

    return df[available].copy()


def find_first_existing_column(actual_columns: list[str], candidates: list[str]) -> str | None:
    """Return the first matching column name from a list of aliases."""
    normalized_lookup = {column.casefold(): column for column in actual_columns}
    for candidate in candidates:
        direct = normalized_lookup.get(candidate.casefold())
        if direct:
            return direct
    return None


def rename_using_aliases(
    df: pd.DataFrame,
    alias_map: dict[str, list[str]],
    dataset_name: str,
) -> pd.DataFrame:
    """Rename raw columns to canonical names using alias candidates."""
    result = df.copy()
    actual_columns = result.columns.tolist()
    rename_map: dict[str, str] = {}

    for canonical_name, candidates in alias_map.items():
        matched = find_first_existing_column(actual_columns, candidates)
        if matched is None:
            warn(
                f"{dataset_name}: expected one of {candidates} for '{canonical_name}', "
                f"but none were found."
            )
            continue
        rename_map[matched] = canonical_name

    return result.rename(columns=rename_map)


def add_project_id(df: pd.DataFrame, dataset_name: str, name_candidates: list[str]) -> pd.DataFrame:
    """Ensure a Project_ID column exists, deriving it from a project-name field if needed."""
    result = df.copy()

    if "Project_ID" in result.columns:
        result["Project_ID"] = result["Project_ID"].map(sanitize_identifier)
        return result

    source_column = find_first_existing_column(result.columns.tolist(), name_candidates)
    if source_column is None:
        warn(
            f"{dataset_name}: could not create Project_ID because none of these columns exist: {name_candidates}"
        )
        result["Project_ID"] = None
        return result

    result["Project_ID"] = result[source_column].map(sanitize_identifier)
    warn(
        f"{dataset_name}: Project_ID not found in source; generated from '{source_column}'. "
        f"Manual verification is recommended."
    )
    return result


def add_project_name(df: pd.DataFrame, dataset_name: str, name_candidates: list[str]) -> pd.DataFrame:
    """Ensure a Project_Name column exists using the best available source column."""
    result = df.copy()

    if "Project_Name" in result.columns:
        return result

    source_column = find_first_existing_column(result.columns.tolist(), name_candidates)
    if source_column is None:
        warn(
            f"{dataset_name}: could not create Project_Name because none of these columns exist: {name_candidates}"
        )
        result["Project_Name"] = None
        return result

    result["Project_Name"] = result[source_column]
    warn(
        f"{dataset_name}: Project_Name not found in source; generated from '{source_column}'. "
        f"Manual verification is recommended."
    )
    return result


def add_join_name(df: pd.DataFrame) -> pd.DataFrame:
    """Add a normalized join key derived from Project_Name."""
    result = df.copy()
    if "Project_Name" not in result.columns:
        result["Project_Name"] = None
    result["Project_Name_Join"] = result["Project_Name"].map(sanitize_identifier)
    return result


def normalize_project_size(value):
    """Normalize strings like 'Small (265 units)' to 'Small'."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    if " (" in text:
        return text.split(" (", 1)[0].strip()
    return text


def merge_project_data(left_df: pd.DataFrame, right_df: pd.DataFrame) -> pd.DataFrame:
    """Merge project-level data primarily by Project_ID and fallback by Project_Name."""
    left = add_join_name(left_df)
    right = add_join_name(right_df)

    if "Project_ID" in right.columns:
        right_non_null_ids = right[right["Project_ID"].notna()].copy()
    else:
        right_non_null_ids = right.iloc[0:0].copy()

    merged = left.merge(
        right_non_null_ids,
        on="Project_ID",
        how="left",
        suffixes=("", "_right"),
    )

    right_name_lookup = right.drop_duplicates(subset=["Project_Name_Join"]).copy()
    fallback = left.merge(
        right_name_lookup,
        on="Project_Name_Join",
        how="left",
        suffixes=("", "_fallback"),
    )

    left_columns = set(left.columns)
    right_columns = [column for column in right.columns if column not in {"Project_ID", "Project_Name_Join"}]

    for column in right_columns:
        merged_column = column
        fallback_column = f"{column}_fallback"
        if merged_column not in merged.columns and fallback_column not in fallback.columns:
            continue

        if merged_column not in merged.columns:
            merged[merged_column] = None

        fallback_values = fallback[fallback_column] if fallback_column in fallback.columns else None
        if fallback_values is not None:
            merged[merged_column] = merged[merged_column].where(merged[merged_column].notna(), fallback_values)

    extra_right_columns = [column for column in merged.columns if column.endswith("_right")]
    extra_fallback_columns = [column for column in merged.columns if column.endswith("_fallback")]
    merged = merged.drop(columns=extra_right_columns + extra_fallback_columns + ["Project_Name_Join"], errors="ignore")

    overlapping = [column for column in right_columns if column in left_columns]
    if overlapping:
        warn(
            "Project merge: overlapping columns required fallback reconciliation. "
            f"Please manually verify: {overlapping}"
        )

    return merged


def coerce_numeric_if_possible(series: pd.Series) -> pd.Series:
    """Convert a series to numeric where that is safe and helpful."""
    converted = pd.to_numeric(series, errors="coerce")
    if converted.notna().sum() > 0:
        return converted
    return series


def parse_rank_prefix(rank_text: str) -> int | None:
    """Convert prefixes like '1st' or '10th' into an integer rank."""
    match = re.match(r"(\d+)(?:st|nd|rd|th)$", rank_text.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def slug_to_title(slug: str) -> str:
    """Convert an identifier slug to title case."""
    return " ".join(part.capitalize() for part in slug.split("_") if part)


def infer_amenity_groups(columns: list[str]) -> dict[str, dict[int, dict[str, str]]]:
    """Infer ranked amenity columns like '1st MRT' and '1st MRT dist'."""
    groups: dict[str, dict[int, dict[str, str]]] = {}
    pattern = re.compile(r"^(?P<rank>\d+(?:st|nd|rd|th))\s+(?P<group>.+?)(?P<dist>\s+dist)?$", re.IGNORECASE)

    for column in columns:
        match = pattern.match(column.strip())
        if not match:
            continue

        rank = parse_rank_prefix(match.group("rank"))
        if rank is None:
            continue

        group_name = match.group("group").strip()
        group_key = sanitize_identifier(group_name) or group_name.lower()
        kind = "distance" if match.group("dist") else "name"

        groups.setdefault(group_key, {}).setdefault(rank, {})[kind] = column

    return groups


def make_amenity_labels(group_key: str) -> tuple[str, str]:
    """Return dataset name and singular entity label for an amenity group."""
    explicit = {
        "mrt": ("Project_MRT", "MRT"),
        "schools": ("Project_School", "School"),
        "supermarkets": ("Project_Supermarket", "Supermarket"),
        "parks": ("Project_Park", "Park"),
        "bus_stops": ("Project_Bus_Stop", "Bus_Stop"),
        "clinics": ("Project_Clinic", "Clinic"),
        "bank": ("Project_Bank", "Bank"),
        "atms": ("Project_ATM", "ATM"),
        "post_boxes": ("Project_Post_Box", "Post_Box"),
        "post_offices": ("Project_Post_Office", "Post_Office"),
    }
    if group_key in explicit:
        return explicit[group_key]

    singular = group_key
    if singular.endswith("ies"):
        singular = singular[:-3] + "y"
    elif singular.endswith("s") and not singular.endswith("ss"):
        singular = singular[:-1]

    dataset_name = f"Project_{slug_to_title(singular).replace(' ', '_')}"
    label = slug_to_title(singular).replace(" ", "_")
    warn(
        f"Amenity group '{group_key}' was inferred automatically as dataset '{dataset_name}'. "
        "Please manually verify the naming."
    )
    return dataset_name, label


def reshape_ranked_amenity_wide_to_long(
    df: pd.DataFrame,
    group_key: str,
    rank_columns: dict[int, dict[str, str]],
) -> tuple[str, pd.DataFrame]:
    """Convert a ranked wide amenity group into a long normalized dataset."""
    dataset_name, label = make_amenity_labels(group_key)
    required_base = ["Project_ID", "Project_Name"]
    missing_base = [column for column in required_base if column not in df.columns]
    if missing_base:
        warn(
            f"{dataset_name}: missing base columns {missing_base}. "
            "This dataset may need manual verification."
        )

    rows: list[dict] = []
    rank_column_name = f"{label}_Rank"
    name_column_name = f"{label}_Name"
    distance_column_name = f"{label}_Distance"

    for _, record in df.iterrows():
        for rank in sorted(rank_columns):
            spec = rank_columns[rank]
            name_column = spec.get("name")
            distance_column = spec.get("distance")

            amenity_name = record.get(name_column) if name_column else None
            amenity_distance = record.get(distance_column) if distance_column else None

            if pd.isna(amenity_name) and pd.isna(amenity_distance):
                continue

            rows.append(
                {
                    "Project_ID": record.get("Project_ID"),
                    "Project_Name": record.get("Project_Name"),
                    rank_column_name: rank,
                    name_column_name: amenity_name,
                    distance_column_name: amenity_distance,
                }
            )

    long_df = pd.DataFrame(rows)
    if not long_df.empty and distance_column_name in long_df.columns:
        long_df[distance_column_name] = coerce_numeric_if_possible(long_df[distance_column_name])

    long_df = standardize_string_whitespace(long_df)
    return dataset_name, long_df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame to CSV, ensuring the parent directory exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log(f"Saved {path.name} with {len(df):,} rows")


def insert_into_mongodb(
    df: pd.DataFrame,
    collection_name: str,
    db,
    clear_existing: bool = False,
    index_fields: list[tuple[str, bool]] | None = None,
) -> None:
    """Insert a DataFrame into MongoDB with optional indexes."""
    if df.empty:
        log(f"Skipping MongoDB insert for {collection_name}: dataset is empty")
        return

    collection = db[collection_name]
    if clear_existing:
        collection.delete_many({})

    records = df.where(pd.notnull(df), None).to_dict("records")
    collection.insert_many(records)
    log(f"Inserted {len(records):,} records into MongoDB collection '{collection_name}'")

    if index_fields:
        for field_name, unique in index_fields:
            collection.create_index([(field_name, ASCENDING)], unique=unique)
            log(
                f"Created {'unique ' if unique else ''}index on "
                f"{collection_name}.{field_name}"
            )


def print_dataset_summary(dataset_name: str, df: pd.DataFrame) -> None:
    """Print row count and final column names for a dataset."""
    print()
    print(f"{dataset_name}:")
    print(f"  rows = {len(df):,}")
    print(f"  columns = {df.columns.tolist()}")


def build_project_dataset() -> pd.DataFrame:
    """Build the Project dataset."""
    ura_aliases = {
        "Project_Name": ["Project_Name", "Project Name"],
        "Postal District": ["Postal District"],
        "Planning Region": ["Planning Region"],
        "Planning Area": ["Planning Area"],
        "Property Type": ["Property Type"],
        "tenure_remaining_years": ["tenure_remaining_years"],
    }
    details_aliases = {
        "Project_Name": ["Project_Name", "Project Name"],
        "Project_Name_in_Realis": ["Project Name in Realis"],
        "TOP date": ["TOP date"],
        "Project Size": ["Project Size"],
        "Number of Units": ["Number of Units"],
        "Blocks": ["Blocks"],
        "Property Type": ["Property Type", "Property type"],
    }

    ura_columns = [
        "Project Name",
        "Postal District",
        "Planning Region",
        "Planning Area",
        "Property Type",
        "tenure_remaining_years",
    ]
    details_columns = [
        "Project Name",
        "Project Name in Realis",
        "TOP date",
        "Project Size",
        "Number of Units",
        "Blocks",
        "Property type",
    ]

    ura = rename_using_aliases(load_csv(URA_V2_PATH, ura_columns), ura_aliases, "Project")
    ura = add_project_name(ura, "Project", ["Project_Name", "Project Name"])
    ura = add_project_id(ura, "Project", ["Project_Name", "Project Name"])
    ura = safe_select_columns(
        ura,
        [
            "Project_ID",
            "Project_Name",
            "Postal District",
            "Planning Region",
            "Planning Area",
            "Property Type",
            "tenure_remaining_years",
        ],
        "Project",
    ).rename(
        columns={
            "Property Type": "Property_Type_REALIS",
            "tenure_remaining_years": "Tenure",
        }
    )

    details = rename_using_aliases(
        load_csv(PROPERTY_DETAILS_PATH, details_columns),
        details_aliases,
        "Project",
    )
    details = add_project_name(
        details,
        "Project",
        ["Project_Name", "Project Name", "Project_Name_in_Realis", "Project Name in Realis"],
    )
    details = add_project_id(
        details,
        "Project",
        ["Project_Name_in_Realis", "Project Name in Realis", "Project_Name", "Project Name"],
    )
    details["Project Size"] = details["Project Size"].map(normalize_project_size)
    details = safe_select_columns(
        details,
        [
            "Project_ID",
            "Project_Name",
            "TOP date",
            "Project Size",
            "Number of Units",
            "Blocks",
            "Property Type",
        ],
        "Project",
    ).rename(columns={"Property Type": "Property_Type_99co"})

    project_df = merge_project_data(ura, details)
    project_df = standardize_string_whitespace(project_df)
    project_df = project_df.drop_duplicates(subset=["Project_ID"]).copy()
    return project_df


def build_project_location_dataset() -> pd.DataFrame:
    """Build the Project_Location dataset."""
    alias_map = {
        "Project_Name": ["Project_Name", "Project Name"],
        "onemap_address": ["onemap_address"],
        "latitude": ["latitude"],
        "longitude": ["longitude"],
    }
    columns = ["Project Name", "onemap_address", "latitude", "longitude"]
    df = rename_using_aliases(load_csv(URA_V2_PATH, columns), alias_map, "Project_Location")
    df = add_project_name(df, "Project_Location", ["Project_Name", "Project Name"])
    df = add_project_id(df, "Project_Location", ["Project_Name", "Project Name"])
    df = safe_select_columns(
        df,
        ["Project_ID", "Project_Name", "onemap_address", "latitude", "longitude"],
        "Project_Location",
    )
    df = df.drop_duplicates(subset=["Project_ID"]).copy()
    return df


def build_project_rental_dataset() -> pd.DataFrame:
    """Build the row-level Project_Rental dataset from URA v2."""
    v2_aliases = {
        "Project_Name": ["Project_Name", "Project Name"],
        "Lease Commencement Date": ["Lease Commencement Date"],
        "Monthly Rent ($)": ["Monthly Rent ($)"],
        "Floor Area (SQM)": ["Floor Area (SQM)"],
    }

    v2_columns = [
        "Project Name",
        "Lease Commencement Date",
        "Monthly Rent ($)",
        "Floor Area (SQM)",
    ]

    v2 = rename_using_aliases(load_csv(URA_V2_PATH, v2_columns), v2_aliases, "Project_Rental")
    v2 = add_project_name(v2, "Project_Rental", ["Project_Name", "Project Name"])
    v2 = add_project_id(v2, "Project_Rental", ["Project_Name", "Project Name"])
    v2 = safe_select_columns(
        v2,
        [
            "Project_ID",
            "Project_Name",
            "Lease Commencement Date",
            "Monthly Rent ($)",
            "Floor Area (SQM)",
        ],
        "Project_Rental",
    )
    v2 = standardize_string_whitespace(v2)
    return v2


def build_aggregate_dataset() -> pd.DataFrame:
    """Build the aggregate rental dataset from URA v3."""
    v3_aliases = {
        "Project_Name": ["Project_Name", "Project Name"],
        "Lease Commencement Date": ["Lease Commencement Date"],
        "FloorArea_avg_sqm": ["FloorArea_avg_sqm"],
        "Avg_rent_per_sqm": ["Avg_rent_per_sqm"],
        "ave_rent": ["ave_rent"],
    }

    v3_columns = [
        "Project Name",
        "Lease Commencement Date",
        "FloorArea_avg_sqm",
        "Avg_rent_per_sqm",
        "ave_rent",
    ]

    v3 = rename_using_aliases(
        load_csv(URA_V3_PATH, v3_columns),
        v3_aliases,
        "Project_Rental_Aggregate",
    )
    v3 = add_project_name(v3, "Project_Rental_Aggregate", ["Project_Name", "Project Name"])
    v3 = add_project_id(v3, "Project_Rental_Aggregate", ["Project_Name", "Project Name"])
    v3 = safe_select_columns(
        v3,
        [
            "Project_ID",
            "Project_Name",
            "Lease Commencement Date",
            "FloorArea_avg_sqm",
            "Avg_rent_per_sqm",
            "ave_rent",
        ],
        "Project_Rental_Aggregate",
    )
    v3 = v3.drop_duplicates(subset=["Project_ID", "Lease Commencement Date"]).copy()
    v3 = standardize_string_whitespace(v3)
    return v3


def build_macro_dataset() -> pd.DataFrame:
    """Build the Macro_Data dataset."""
    alias_map = {
        "Lease Commencement Date": ["Lease Commencement Date"],
        "cpi_all_items_infl_yoy_pct": ["cpi_all_items_infl_yoy_pct"],
        "unemployment_rate_sa_pct": ["unemployment_rate_sa_pct"],
        "sora_overnight_pct": ["sora_overnight_pct"],
        "sora_compounded_3m_pct": ["sora_compounded_3m_pct"],
        "sg_govt_bond_yield_10y_pct": ["sg_govt_bond_yield_10y_pct"],
        "sgd_per_usd_logret_12m_pct": ["sgd_per_usd_logret_12m_pct"],
        "ura_private_rental_index_yoy_pct": ["ura_private_rental_index_yoy_pct"],
        "ura_private_price_index_avg_3seg_yoy_pct": ["ura_private_price_index_avg_3seg_yoy_pct"],
        "hdb_resale_price_index_yoy_pct": ["hdb_resale_price_index_yoy_pct"],
        "gva_yoy_growth_pct": ["gva_yoy_growth_pct"],
    }
    columns = list(alias_map.keys())
    df = rename_using_aliases(load_csv(URA_V2_PATH, columns), alias_map, "Macro_Data")
    df = safe_select_columns(df, columns, "Macro_Data")
    df = df.rename(columns={"Lease Commencement Date": "Date"})
    df = df.drop_duplicates(subset=["Date"]).copy()
    return df


def build_project_facilities_dataset() -> pd.DataFrame:
    """Build the Project_Facilities dataset."""
    facility_columns = [
        "Adventure Park",
        "Aerobic Pool",
        "Amphitheatre",
        "BBQ",
        "Badminton Hall",
        "Basketball Court",
        "Billiards Room",
        "Bowling Alley",
        "Bridge",
        "Clubhouse",
        "Concierge",
        "Driving Range",
        "Fitness Corner",
        "Fountain",
        "Fun Pool",
        "Function Room",
        "Games Room",
        "Gym",
        "Hammocks",
        "Hydrotherapy Pool",
        "Infinity Pool",
        "Jacuzzi",
        "Jet Pool",
        "Jogging Track",
        "Karaoke",
        "Lap Pool",
        "Library",
        "Lounge",
        "Meeting Room",
        "Mini Golf Range",
        "Mini Mart",
        "Multi Purpose Hall",
        "Open Terrace",
        "Outdoor Dining",
        "Parking",
        "Pavilion",
        "Playground",
        "Pond",
        "Pool Deck",
        "Reflective Pool",
        "Reflexology Path",
        "Retail Shops",
        "Rooftop Pool",
        "Sauna",
        "Sculpture",
        "Security",
        "Sky Lounge",
        "Sky Terrace",
        "Spa Pavillion",
        "Spa Pool",
        "Squash Court",
        "Steam Room",
        "Study Room",
        "Swimming Pool",
        "Tennis Court",
        "Timber Deck",
        "Underwater Fitness Station",
        "Viewing Deck",
        "Wading Pool",
        "Water Channel",
        "Water Feature",
        "Waterfall",
        "Wine And Cigar Room",
        "Yoga Corner",
    ]
    columns = ["Project Name", "Project Name in Realis"] + facility_columns
    alias_map = {"Project_Name": ["Project Name"], "Project_Name_in_Realis": ["Project Name in Realis"]}
    for facility in facility_columns:
        alias_map[facility] = [facility]

    df = rename_using_aliases(load_csv(PROPERTY_DETAILS_PATH, columns), alias_map, "Project_Facilities")
    df = add_project_name(
        df,
        "Project_Facilities",
        ["Project_Name_in_Realis", "Project Name in Realis", "Project_Name", "Project Name"],
    )
    df = add_project_id(
        df,
        "Project_Facilities",
        ["Project_Name_in_Realis", "Project Name in Realis", "Project_Name", "Project Name"],
    )
    df = safe_select_columns(df, ["Project_ID", "Project_Name"] + facility_columns, "Project_Facilities")
    df = df.drop_duplicates(subset=["Project_ID"]).copy()
    return df


def build_project_top30_school_dataset() -> pd.DataFrame:
    """Build the Project_Top30_School dataset from URA v3."""
    school_columns = [
        "Nanyang Primary School",
        "Rosyth School",
        "Henry Park Primary School",
        "Tao Nan School",
        "Raffles Girls' Primary School",
        "St. Hilda's Primary School",
        "Pei Hwa Presbyterian Primary School",
        "Methodist Girls' School (Primary)",
        "Nan Hua Primary School",
        "Chij St. Nicholas Girls' School",
        "Anglo-Chinese School (Primary)",
        "Catholic High School (Primary)",
        "Rulanh Primary School",
        "Red Swastika School",
        "Ai Tong School",
        "St. Joseph's Institution Junior",
        "Kong Hwa School",
        "South View Primary School",
        "Chongfu School",
        "Pei Chun Public School",
        "Holy Innocents' Primary School",
        "Maris Stella High School (Primary)",
        "Singapore Chinese Girls' Primary School",
        "Canberra Primary School",
        "Radin Mas Primary School",
        "River Valley Primary School",
        "Gongshang Primary School",
        "Temasek Primary School",
        "Anderson Primary School",
        "Princess Elizabeth Primary School",
    ]
    columns = ["Project Name"] + school_columns
    alias_map = {"Project_Name": ["Project_Name", "Project Name"]}
    for school in school_columns:
        alias_map[school] = [school]

    df = rename_using_aliases(
        load_csv(URA_V3_PATH, columns),
        alias_map,
        "Project_Top30_School",
    )
    df = add_project_name(df, "Project_Top30_School", ["Project_Name", "Project Name"])
    df = add_project_id(df, "Project_Top30_School", ["Project_Name", "Project Name"])
    df = safe_select_columns(
        df,
        ["Project_ID", "Project_Name"] + school_columns,
        "Project_Top30_School",
    )
    df = df.drop_duplicates(subset=["Project_ID"]).copy()
    return df


def build_project_mrt_gglemap_dataset() -> pd.DataFrame:
    """Build the Project_MRT_GgleMap dataset from URA v2 MRT-distance columns."""
    mrt_columns = [
        "Admiralty_MRT",
        "Aljunied_MRT",
        "Ang Mo Kio_MRT",
        "Bangkit_MRT",
        "Bartley_MRT",
        "Bayfront_MRT",
        "Bayshore_MRT",
        "Beauty World_MRT",
        "Bedok_MRT",
        "Bedok North_MRT",
        "Bedok Reservoir_MRT",
        "Bencoolen_MRT",
        "Bendemeer_MRT",
        "Bishan_MRT",
        "Boon Keng_MRT",
        "Boon Lay_MRT",
        "Botanic Gardens_MRT",
        "Braddell_MRT",
        "Bras Basah_MRT",
        "Bright Hill_MRT",
        "Buangkok_MRT",
        "Bugis_MRT",
        "Bukit Batok_MRT",
        "Bukit Gombak_MRT",
        "Bukit Panjang_MRT",
        "Buona Vista_MRT",
        "Caldecott_MRT",
        "Canberra_MRT",
        "Canberra Station (NS12)_MRT",
        "Cashew_MRT",
        "Cashew MRT Station (Downtown Line)_MRT",
        "Changi Airport_MRT",
        "Chinatown_MRT",
        "Chinese Garden_MRT",
        "Choa Chu Kang_MRT",
        "City Hall_MRT",
        "Clarke Quay_MRT",
        "Clementi_MRT",
        "Commonwealth_MRT",
        "Dakota_MRT",
        "Damai_MRT",
        "Dhoby Ghaut_MRT",
        "Dover_MRT",
        "Downtown_MRT",
        "Esplanade_MRT",
        "Eunos_MRT",
        "Expo_MRT",
        "Fajar_MRT",
        "Farrer Park_MRT",
        "Farrer Road_MRT",
        "Fort Canning_MRT",
        "Gardens by the Bay_MRT",
        "Geylang Bahru_MRT",
        "Great World_MRT",
        "Harbourfront_MRT",
        "Havelock_MRT",
        "Haw Par Villa_MRT",
        "Hillview_MRT",
        "Holland Village_MRT",
        "Hougang_MRT",
        "Hume_MRT",
        "Jalan Besar_MRT",
        "Jelapang_MRT",
        "Joo Koon_MRT",
        "Jurong East_MRT",
        "Kaki Bukit_MRT",
        "Kallang_MRT",
        "Katong Park_MRT",
        "Keat Hong_MRT",
        "Kembangan_MRT",
        "Kent Ridge_MRT",
        "Khatib_MRT",
        "King Albert Park_MRT",
        "Kovan_MRT",
        "Kranji_MRT",
        "Labrador Park_MRT",
        "Lakeside_MRT",
        "Lavender_MRT",
        "Lentor_MRT",
        "Little India_MRT",
        "Lorong Chuan_MRT",
        "Macpherson_MRT",
        "Marina Bay_MRT",
        "Marina South Pier_MRT",
        "Marine Parade_MRT",
        "Marine Terrace_MRT",
        "Marsiling_MRT",
        "Marymount_MRT",
        "Mattar_MRT",
        "Maxwell_MRT",
        "Mayflower_MRT",
        "Mountbatten_MRT",
        "Napier_MRT",
        "Newton_MRT",
        "Nicoll Highway_MRT",
        "Novena_MRT",
        "One-North_MRT",
        "Orchard_MRT",
        "Orchard Boulevard_MRT",
        "Outram Park_MRT",
        "Pasir Panjang_MRT",
        "Pasir Ris_MRT",
        "Paya Lebar_MRT",
        "Pending_MRT",
        "Petir_MRT",
        "Phoenix_MRT",
        "Pioneer_MRT",
        "Potong Pasir_MRT",
        "Promenade_MRT",
        "Punggol_MRT",
        "Punggol Coast_MRT",
        "Queenstown_MRT",
        "Raffles Place_MRT",
        "Redhill_MRT",
        "Rochor_MRT",
        "Segar_MRT",
        "Sembawang_MRT",
        "Sengkang_MRT",
        "Senja_MRT",
        "Serangoon_MRT",
        "Shenton Way_MRT",
        "Siglap_MRT",
        "Simei_MRT",
        "Sixth Avenue_MRT",
        "Somerset_MRT",
        "South View_MRT",
        "Springleaf_MRT",
        "Stadium_MRT",
        "Stevens_MRT",
        "Tai Seng_MRT",
        "Tampines_MRT",
        "Tampines East_MRT",
        "Tampines West_MRT",
        "Tan Kah Kee_MRT",
        "Tanah Merah_MRT",
        "Tanjong Katong_MRT",
        "Tanjong Pagar_MRT",
        "Tanjong Rhu_MRT",
        "Teck Whye_MRT",
        "Telok Ayer_MRT",
        "Telok Blangah_MRT",
        "The 'Eylim_MRT",
        "Tiong Bahru_MRT",
        "Toa Payoh_MRT",
        "Ubi_MRT",
        "Upper Changi_MRT",
        "Upper Thomson_MRT",
        "Woodlands_MRT",
        "Woodlands North_MRT",
        "Woodlands South_MRT",
        "Woodleigh_MRT",
        "Yew Tee_MRT",
        "Yio Chu Kang_MRT",
        "Yishun_MRT",
    ]
    columns = ["Project Name"] + mrt_columns
    alias_map = {"Project_Name": ["Project_Name", "Project Name"]}
    for mrt_column in mrt_columns:
        alias_map[mrt_column] = [mrt_column]

    df = rename_using_aliases(
        load_csv(URA_V2_PATH, columns),
        alias_map,
        "Project_MRT_GgleMap",
    )
    df = add_project_name(df, "Project_MRT_GgleMap", ["Project_Name", "Project Name"])
    df = add_project_id(df, "Project_MRT_GgleMap", ["Project_Name", "Project Name"])
    df = safe_select_columns(
        df,
        ["Project_ID", "Project_Name"] + mrt_columns,
        "Project_MRT_GgleMap",
    )
    rename_map = {column: column.removesuffix("_MRT") for column in mrt_columns if column in df.columns}
    df = df.rename(columns=rename_map)
    df = df.drop_duplicates(subset=["Project_ID"]).copy()
    return df


def build_rail_transport_dataset() -> pd.DataFrame:
    """Build the Rail_Transport dataset."""
    columns = [
        "ALPHANUMERIC_CODE",
        "STATION_NAME_ENGLISH",
        "LINE_ENGLISH",
        "LINE_COLOR",
        "OPENING_DATE",
        "TRANSPORT_TYPE",
        "SEARCHVAL",
        "BLK_NO",
        "ROAD_NAME",
        "BUILDING",
        "ADDRESS",
        "POSTAL",
        "X",
        "Y",
        "LATITUDE",
        "LONGITUDE",
    ]
    df = load_csv(MRT_LRT_PATH)
    if df.columns.tolist() and (df.columns[0] == "" or str(df.columns[0]).startswith("Unnamed:")):
        df = df.drop(columns=[df.columns[0]])
    df = safe_select_columns(df, columns, "Rail_Transport")
    df = df.drop_duplicates(subset=["ALPHANUMERIC_CODE"]).copy()
    return df


def build_project_amenity_datasets() -> dict[str, pd.DataFrame]:
    """Build all Project_* amenity datasets from nearby_5km_amenities.csv."""
    nearby = load_csv(AMENITIES_PATH)
    nearby = rename_using_aliases(
        nearby,
        {
            "Project_Name": ["Project_Name", "Project Name"],
            "Project_Name_in_Realis": ["Project Name in Realis"],
        },
        "Nearby_Amenities",
    )
    nearby = add_project_name(
        nearby,
        "Nearby_Amenities",
        ["Project_Name_in_Realis", "Project Name in Realis", "Project_Name", "Project Name"],
    )
    nearby = add_project_id(
        nearby,
        "Nearby_Amenities",
        ["Project_Name_in_Realis", "Project Name in Realis", "Project_Name", "Project Name"],
    )

    groups = infer_amenity_groups(nearby.columns.tolist())
    if not groups:
        warn("No ranked amenity column groups were inferred from nearby_5km_amenities.csv")
        return {}

    datasets: dict[str, pd.DataFrame] = {}
    for group_key, rank_columns in sorted(groups.items()):
        dataset_name, long_df = reshape_ranked_amenity_wide_to_long(nearby, group_key, rank_columns)
        datasets[dataset_name] = long_df
    return datasets


def maybe_insert_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    db,
    index_fields: list[tuple[str, bool]] | None,
) -> None:
    """Insert a dataset into MongoDB when enabled."""
    if not INSERT_TO_MONGODB:
        return
    insert_into_mongodb(
        df=df,
        collection_name=dataset_name,
        db=db,
        clear_existing=CLEAR_EXISTING_COLLECTIONS,
        index_fields=index_fields,
    )


def main() -> None:
    """Run the full ETL pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mongo_db = None

    if INSERT_TO_MONGODB:
        client = MongoClient(MONGO_URI)
        mongo_db = client[MONGO_DB_NAME]
        log(f"Connected to MongoDB database '{MONGO_DB_NAME}'")
    else:
        log("MongoDB insertion is disabled")

    datasets: dict[str, pd.DataFrame] = {}

    datasets["Project"] = build_project_dataset()
    datasets["Project_Location"] = build_project_location_dataset()
    datasets["Project_Rental"] = build_project_rental_dataset()
    datasets["Project_Rental_Aggregate"] = build_aggregate_dataset()
    datasets["Macro_Data"] = build_macro_dataset()
    datasets["Project_Facilities"] = build_project_facilities_dataset()
    datasets["Project_MRT_GgleMap"] = build_project_mrt_gglemap_dataset()
    datasets["Project_Top30_School"] = build_project_top30_school_dataset()
    datasets["Rail_Transport"] = build_rail_transport_dataset()
    datasets.update(build_project_amenity_datasets())

    mongo_indexes: dict[str, list[tuple[str, bool]]] = {
        "Project": [("project_id", True)],
        "Project_Location": [("project_id", False)],
        "Project_Rental": [("project_id", False)],
        "Project_Rental_Aggregate": [("project_id", False)],
        "Macro_Data": [("date", False)],
        "Project_Facilities": [("project_id", False)],
        "Project_MRT_GgleMap": [("project_id", False)],
        "Project_Top30_School": [("project_id", False)],
        "Rail_Transport": [("alphanumeric_code", False)],
    }
    for dataset_name in datasets:
        if dataset_name.startswith("Project_") and dataset_name not in mongo_indexes:
            mongo_indexes[dataset_name] = [("project_id", False)]

    print()
    print("=" * 60)
    print("ETL OUTPUT SUMMARY")
    print("=" * 60)

    for dataset_name, df in datasets.items():
        normalized_df = normalize_output_columns(df)
        output_path = OUTPUT_DIR / f"{dataset_name}.csv"
        save_csv(normalized_df, output_path)
        print_dataset_summary(dataset_name, normalized_df)
        maybe_insert_dataset(dataset_name, normalized_df, mongo_db, mongo_indexes.get(dataset_name))

    print()
    print("=" * 60)
    print("WARNINGS")
    print("=" * 60)
    if WARNINGS:
        for message in WARNINGS:
            print(f"- {message}")
    else:
        print("No warnings.")


if __name__ == "__main__":
    main()
