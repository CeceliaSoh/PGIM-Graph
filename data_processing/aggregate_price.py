import pandas as pd

# Load data
df = pd.read_csv(r"dataset/URA_enriched_with_99co_v2.csv")

# ----------------------------
# 1. Clean and convert Floor Area (SQM)
# ----------------------------
area = (
    df["Floor Area (SQM)"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

# Split into lower / upper bounds if value is a range like "40 to 50"
area_split = area.str.split(" to ", expand=True)

area_low = pd.to_numeric(area_split[0], errors="coerce")
area_high = pd.to_numeric(area_split[1], errors="coerce")

# If there is an upper bound, take midpoint; otherwise use the single value
df["FloorArea_avg_sqm"] = area_low.where(area_high.isna(), (area_low + area_high) / 2)

# ----------------------------
# 2. Compute row-level rent per sqm
# ----------------------------
df["Monthly Rent ($)"] = pd.to_numeric(df["Monthly Rent ($)"], errors="coerce")
df["rent_per_sqm"] = df["Monthly Rent ($)"] / df["FloorArea_avg_sqm"]

# ----------------------------
# 3. Define grouping and columns
# ----------------------------
group_cols = ["Project Name", "Lease Commencement Date"]
drop_cols = ["No of Bedroom", "Monthly Rent ($)", "Floor Area (SQM)", "Floor Area (SQFT)"]

# Keep all other columns using first(), since they should be identical within each group
other_cols = [
    col for col in df.columns
    if col not in group_cols + drop_cols + ["rent_per_sqm", "FloorArea_avg_sqm"]
]

# ----------------------------
# 4. Build aggregation dictionary
# ----------------------------
agg_dict = {col: "first" for col in other_cols}
agg_dict.update({
    "FloorArea_avg_sqm": "mean",
    "rent_per_sqm": "mean",
    "Monthly Rent ($)": "mean",
})

# ----------------------------
# 5. Group and aggregate
# ----------------------------
small_df = (
    df.groupby(group_cols, as_index=False)
      .agg(agg_dict)
      .rename(columns={
          "rent_per_sqm": "Avg_rent_per_sqm",
          "Monthly Rent ($)": "ave_rent"
      })
)

# ----------------------------
# 6. Save and print
# ----------------------------
small_df.to_csv("dataset/URA_enriched_with_99co_v3.csv", index=False)
print(small_df)