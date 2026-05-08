import pandas as pd

# Path
file_path = "/home/cecelia/project/PGIM-Graph/database_v3/macro_data_processed_v0.csv"

# Load data
df = pd.read_csv(file_path)

# (Optional) ensure date column is parsed correctly
df["Lease Commencement Date"] = pd.to_datetime(df["Lease Commencement Date"], errors="coerce")

# Group by timestep + Lease Commencement Date
df_avg = (
    df
    .groupby(["timestep", "Lease Commencement Date"], as_index=False)
    .mean(numeric_only=True)
)

# Sort for readability
df_avg = df_avg.sort_values(["timestep", "Lease Commencement Date"])

# Save (optional)
output_path = "/home/cecelia/project/PGIM-Graph/database_v3/macro_data_processed.csv"
df_avg.to_csv(output_path, index=False)

# Preview
print(df_avg.head())