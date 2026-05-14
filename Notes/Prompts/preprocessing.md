Please create two files under the data/ folder:

1. data/config.yaml
2. data/preprocessing.py

The goal is to build a configurable preprocessing pipeline for rental forecasting data, with proper logging, validation, and saved metadata.

config.yaml should contain:

input_folder: dataset/V250506.csv
output_folder: dataset/database_260512

timestep: month_idx
node_id: project_idx
project_id: condo_name
rent_per_sqft: rent_psf_imp
y_mask: was_observed

max_clip_col: ["dist"]
max_clip_glo: ["School", "Institution"]

as_feat:
  - timestep
  - node_id
  - rent_per_sqft

to_drop:
  - is_post_cutoff
  - rent_psf_obs
  - date

categorical:
  - district
  - size_tier
  - size_label
  - segment
  - Planning Area
  - Neighbourhood
  - nearest_mrt_name
  - nearest_school_name

first_few:
  - timestep
  - node_id
  - project_id

last_few:
  - rent_per_sqft_feat
  - rent_per_sqft
  - y_mask

preprocessing.py requirements:

1. Use pathlib, pandas, numpy, yaml, json, logging, and warnings.
2. Implement the pipeline in clean modular functions, not one long script.
3. Add a main() function and allow running with:

   python data/preprocessing.py --config data/config.yaml

4. Set up logging:
   - log to console
   - also save logs to {output_folder}/preprocessing.log
   - log dataframe shape after each major step
   - log renamed columns, dropped columns, encoded categorical columns, generated mapping files, and saved node files

5. Load input CSV from config["input_folder"].

6. Rename configured columns to standard names:
   - timestep
   - node_id
   - project_id
   - rent_per_sqft
   - y_mask

7. Validate required columns exist before renaming. If missing, raise a clear ValueError.

8. Ensure node_id is integer.

9. Ensure timestep is integer.
   - If timestep is date-like, convert it to integer timestep IDs based on sorted unique dates.
   - If a date column exists, save mapping between timestep and date to:
     {output_folder}/timestep.csv

10. Ensure project_id is integer.
    - If project_id is not integer, sort unique project IDs, assign integer IDs, and save mapping to:
      {output_folder}/project_id.csv

11. Convert y_mask to binary 0/1.
    - True, true, yes, 1 -> 1
    - False, false, no, 0 -> 0
    - If there are unknown values, raise ValueError.

12. For columns whose names contain keywords in max_clip_col:
    - fill missing values with ceil(max value of that individual column)

13. For columns whose names contain keywords in max_clip_glo:
    - find all matched columns
    - compute global max across all matched columns
    - fill missing values in all matched columns with ceil(global max)

14. Duplicate columns listed in as_feat:
    - create {column}_feat for each configured column
    - if a column is missing, raise ValueError

15. Drop columns listed in to_drop:
    - skip missing columns
    - log which columns are dropped and which are skipped

16. One-hot encode configured categorical columns.
    - Only encode columns that exist.
    - Then detect any remaining non-numerical columns.
    - If remaining non-numerical columns exist, raise a warning and one-hot encode them as well.
    - Use dummy_na=False.

17. Reorder columns:
    - columns in first_few should appear first if they exist
    - columns in last_few should appear last if they exist
    - all other columns stay in the middle
    - warn if configured first_few or last_few columns are missing

18. Sort final dataframe by:
    node_id, timestep

19. Run correctness checks before saving:
    - node_id has no missing values
    - timestep has no missing values
    - rent_per_sqft has no missing values
    - y_mask only contains 0 and 1
    - no duplicated rows for the same node_id and timestep
    If duplicated node_id-timestep rows exist, raise ValueError and save the duplicated rows to:
      {output_folder}/duplicated_node_timestep.csv

20. Save final full dataframe to:
    {output_folder}/all_nodes.csv

21. Split dataframe by node_id and save each node as a CSV under:
    {output_folder}/nodes/

    Filename format:
    node_id = 0  -> 0000.csv
    node_id = 12 -> 0012.csv

    Each node file should contain all columns and be sorted by timestep.

22. Save metadata files:
    - {output_folder}/feature_columns.csv
    - {output_folder}/preprocessing_summary.json

The preprocessing_summary.json should include:
- input_path
- output_folder
- input_shape
- final_shape
- number_of_nodes
- number_of_timesteps
- number_of_final_columns
- renamed_columns
- dropped_columns
- skipped_drop_columns
- configured_categorical_columns
- extra_categorical_columns_detected
- max_clip_col_columns
- max_clip_glo_columns
- generated_files
- warnings

Important coding style:
- Keep functions small and readable.
- Avoid hidden defaults that silently change experiment behavior.
- Prefer clear ValueError for serious correctness issues.
- Use logging.info for normal progress.
- Use logging.warning for suspicious but recoverable issues.
- Use comments only where they explain non-obvious logic.
- The code should be easy to modify later for research experiments.