# Preprocessing Reverse Engineering Note

Target files reviewed:

- `data/config.yaml`
- `data/preprocessing.py`

Current note is based on the on-disk files at review time. Important: `data/config.yaml` currently differs from the first requested version: it uses `dataset/V260506.csv`, has `max_clip_col: []`, and has `max_clip_glo: ["dist", "School", "Institution"]`.

## Clear Bugs or Possible Bugs Found First

| Issue | Location | Why It Matters | Suggested Minimal Patch |
|---|---:|---|---|
| `project_id.csv` column naming is ambiguous after mapping string condo names to integers. | `ensure_integer_project_id`, lines 165-170 | The output uses `project_id` for the new integer ID and `original_project_id` for the condo name. This is valid, but easy to confuse with the standardized `project_id` concept. | Rename mapping file columns to `project_id` and `project_name`, or `project_id_int` and `project_id_original`. |
| `feature_columns.csv` currently records every final column, not only model feature columns. | `save_feature_columns`, lines 358-361 | The file includes target-like columns such as `rent_per_sqft` and `y_mask`, plus ID columns, unless downstream code filters them later. | Rename to `final_columns.csv`, or explicitly exclude labels/masks/IDs when producing true feature columns. |
| Keyword matching is broad substring matching. | `columns_matching_keywords`, lines 212-213 | A keyword such as `dist` matches `district` and any other substring occurrence. | Use explicit column lists or regex anchors/configured match modes. |
| One-hot encoding is performed on the full dataset. | `encode_categorical_columns`, lines 266-288 | It can leak category availability from future/test periods into train-time feature schema. | Fit categorical vocabulary on train rows, then transform validation/test using the same schema. |

No preprocessing code was rewritten during this review.

## 1. High-Level Purpose

The pipeline converts a raw rental forecasting CSV into a model-ready tabular time-series dataset. It standardizes important column names, validates core identifiers and target/mask columns, fills selected missing distance or POI-related columns, duplicates configured columns as feature columns, drops configured raw columns, one-hot encodes categorical fields, sorts rows by node and time, checks for invalid node-time duplicates, and saves both a full CSV and one CSV per node.

The intended downstream shape is one row per `(node_id, timestep)`, with all nodes sharing the same final columns. This is consistent with graph/time-series preprocessing where each node is a project and each timestep is a month.

## 2. Input and Output

| Item | Current Value / Behavior |
|---|---|
| Input CSV path | `dataset/V260506.csv` from `input_folder` in `data/config.yaml`, line 1. |
| Output folder | `dataset/database_260512` from `output_folder`, line 2. |
| CLI entry point | `python data/preprocessing.py`; optional override with `--config`, default is `data/config.yaml` at `parse_args`, lines 18-26. |
| Full output CSV | `{output_folder}/all_nodes.csv`, created by `save_full_dataframe`, lines 338-342. |
| Node-level CSV outputs | `{output_folder}/nodes/{node_id:04d}.csv`, created by `save_node_files`, lines 345-355. |
| Metadata files | `timestep.csv`, `project_id.csv`, `feature_columns.csv`, `preprocessing_summary.json`, and `preprocessing.log`. |
| Failure artifact | `duplicated_node_timestep.csv` is created only if duplicate `(node_id, timestep)` rows are detected, lines 329-334. |

## 3. Config-to-Code Mapping

| Config Key | Expected Meaning | Where Used | Matches Intention? |
|---|---|---:|---|
| `input_folder` | Path to raw input CSV. Despite the name, it is a file path. | `preprocess`, line 383; `load_input_csv`, lines 62-65. | Mostly yes, but name is misleading. `input_path` would be clearer. |
| `output_folder` | Folder for all generated files. | `main`, line 440; `setup_logging`, lines 37-53; `preprocess`, lines 378-381. | Yes. |
| `timestep` | Source column to rename to standard `timestep`. | `validate_required_columns`, lines 68-77; `rename_standard_columns`, lines 80-92. | Yes. |
| `node_id` | Source node/project numeric ID column. | Same rename functions; then `ensure_integer_node_id`, lines 100-108. | Yes, but it must already be integer-like. |
| `project_id` | Source project identity column, currently `condo_name`. | Rename; then `ensure_integer_project_id`, lines 154-174. | Partially. It maps string names to integer IDs, but this may duplicate or conflict conceptually with `node_id`. |
| `rent_per_sqft` | Source target/imputed rent column. | Rename; duplicated if in `as_feat`; checked in `run_correctness_checks`, lines 322-325. | Yes, with a leakage risk if duplicated as input. |
| `y_mask` | Source observation mask column. | Rename; `convert_y_mask`, lines 177-209; checked at lines 326-327. | Yes. |
| `max_clip_col` | Keywords for per-column missing-value fill using each column max. | `fill_max_clip_columns`, lines 219-224. Current config is empty. | Yes mechanically, but broad keyword matching can be risky. |
| `max_clip_glo` | Keywords for global missing-value fill across all matched columns. | `fill_max_clip_columns`, lines 226-232. Current config includes `dist`, `School`, `Institution`. | Partially. It may match unintended columns such as `district`. |
| `as_feat` | Columns to duplicate as `{column}_feat`. | `duplicate_feature_columns`, lines 240-250. | Yes. Experimentally risky for `rent_per_sqft`. |
| `to_drop` | Columns to remove after feature duplication and before encoding. | `drop_configured_columns`, lines 253-263. | Yes. Missing columns are silently skipped with log info. |
| `categorical` | Known categorical columns to one-hot encode. | `encode_categorical_columns`, lines 272-280. | Yes for existing columns. Missing categorical columns are skipped. |
| `first_few` | Columns to place first in final CSV. | `reorder_columns`, lines 297-309. | Yes, missing entries only warn. |
| `last_few` | Columns to place last in final CSV. | `reorder_columns`, lines 297-309. | Yes, missing entries only warn. |

## 4. Step-by-Step Pipeline Trace

| Function | Input | Output | Side Effects | Assumptions / Failure Cases |
|---|---|---|---|---|
| `parse_args` lines 18-26 | CLI args | Namespace with `config` path | None | Defaults to `data/config.yaml`. Fails if later file load fails. |
| `load_config` lines 29-34 | YAML path | Config dict | Reads file | Raises `ValueError` if YAML is not a mapping. |
| `setup_logging` lines 37-53 | Output folder | Logger | Creates output folder; writes `preprocessing.log` | Overwrites previous log each run with mode `w`. |
| `record_warning` lines 56-59 | Message, logger, warning list | None | Emits Python warning and log warning | Used for recoverable suspicious states. |
| `load_input_csv` lines 62-65 | Input path | DataFrame | Reads CSV | Fails if path is wrong or CSV cannot parse. |
| `validate_required_columns` lines 68-77 | DataFrame, config | None | None | Requires configured source columns for standard names before renaming. |
| `rename_standard_columns` lines 80-92 | DataFrame, config | Renamed DataFrame and rename map | Logs rename map and shape | Does not check for duplicate column names after renaming. |
| `is_integer_series` lines 95-97 | Series | Boolean | None | Treats numeric-looking strings as integer-like if all values coerce and modulo 1 is zero. |
| `ensure_integer_node_id` lines 100-108 | DataFrame | DataFrame with integer `node_id` | Logs shape | Raises if missing or non-integer-like. Does not create mapping. |
| `ensure_integer_timestep` lines 111-151 | DataFrame, output folder | DataFrame and optional mapping path | May save `timestep.csv` | Integer-like timesteps are kept. Date-like timesteps are mapped by sorted unique dates. If `date` exists, an integer timestep/date mapping is saved. |
| `ensure_integer_project_id` lines 154-174 | DataFrame, output folder | DataFrame and optional mapping path | May save `project_id.csv` | If not integer-like, converts sorted unique string IDs to integers. Raises if missing. |
| `convert_y_mask` lines 177-209 | DataFrame | DataFrame with integer `y_mask` | Logs shape | Accepts true/false/yes/no/1/0 variants. Raises on unknown or missing values. |
| `columns_matching_keywords` lines 212-213 | Columns and keywords | Matched column names | None | Uses substring matching and is case-sensitive. |
| `fill_max_clip_columns` lines 216-237 | DataFrame and config | DataFrame plus matched column lists | Logs matched columns and shape | Per-column mode raises if a matched column has no numeric max. Global mode raises only if all matched values have no numeric max. |
| `duplicate_feature_columns` lines 240-250 | DataFrame and config | DataFrame with `{col}_feat` copies | Logs duplicated columns | Raises if any configured source column is absent. |
| `drop_configured_columns` lines 253-263 | DataFrame and config | DataFrame, dropped list, skipped list | Logs dropped/skipped | Missing drop columns are skipped, not warned or errored. |
| `encode_categorical_columns` lines 266-288 | DataFrame and config | Encoded DataFrame and encoded-column lists | Logs, may warn | Encodes configured existing categorical columns with `dtype=int`; then encodes any remaining non-numeric columns. |
| `reorder_columns` lines 291-312 | DataFrame and config | Reordered DataFrame | Logs, may warn | Missing configured edge columns warn but do not fail. |
| `sort_dataframe` lines 315-319 | DataFrame | Sorted DataFrame | Logs shape | Assumes `node_id` and `timestep` exist. |
| `run_correctness_checks` lines 322-335 | DataFrame, output folder | None | May save duplicate rows file | Raises if key columns are missing values, `y_mask` has values outside `{0,1}`, or duplicates exist. |
| `save_full_dataframe` lines 338-342 | DataFrame, output folder | Path | Saves `all_nodes.csv` | Overwrites existing file. |
| `save_node_files` lines 345-355 | DataFrame, output folder | List of paths | Creates `nodes/`; saves one CSV per node | Assumes integer `node_id`; overwrites existing node CSVs. |
| `save_feature_columns` lines 358-362 | DataFrame, output folder | Path | Saves final column list | Name suggests features but includes all final columns. |
| `save_summary` lines 365-370 | Summary dict | Path | Saves JSON | Overwrites summary. |
| `preprocess` lines 377-434 | Config and logger | Summary dict | Coordinates all transformations and writes outputs | Main sequencing logic. |
| `main` lines 437-444 | CLI args | None | Initializes logging and runs pipeline | Reads output folder before pipeline begins. |

## 5. Column Transformation Flow

| Column | Flow |
|---|---|
| `timestep` | Source `month_idx` is validated and renamed to `timestep` at lines 68-92. It is converted to integer in `ensure_integer_timestep`, lines 111-151. If it is date-like, sorted unique dates are assigned integer IDs. If it is already integer and `date` exists, `timestep.csv` maps integer timestep to date. It is duplicated to `timestep_feat` because it appears in `as_feat`, lines 240-250. It is placed first if present, lines 291-312. |
| `node_id` | Source `project_idx` is renamed to `node_id`, then required to be integer-like at lines 100-108. It is duplicated to `node_id_feat` because it appears in `as_feat`. It is used for sorting and splitting node files. |
| `project_id` | Source `condo_name` is renamed to `project_id`. If string-like, it is sorted and mapped to integer IDs at lines 164-170, with mapping saved to `project_id.csv`. It is not duplicated by the current config. |
| `rent_per_sqft` | Source `rent_psf_imp` is renamed to `rent_per_sqft`. It is duplicated to `rent_per_sqft_feat`, then remains in the final dataframe as a label-like column near the end. Missing values are checked before saving. |
| `y_mask` | Source `was_observed` is renamed to `y_mask`, converted to integer 0/1, and kept as a final column near the end. It is not duplicated, but it is present in final CSVs unless downstream code removes it from model inputs. |
| `date` | If present, it is used for `timestep.csv` when timestep is already integer. It is later dropped by `to_drop`, lines 253-263. It is not preserved in final node files. |
| Categorical columns | Existing configured categorical columns are one-hot encoded with integer dummy columns at lines 272-280. Missing configured categorical columns are logged and skipped. Any remaining non-numeric columns are warned about and also one-hot encoded at lines 282-286. |
| `{column}_feat` columns | Created before dropping and encoding at lines 240-250. Current config creates `timestep_feat`, `node_id_feat`, and `rent_per_sqft_feat`. These are simple copies, not normalized or lagged. |

## 6. Correctness Checks

| Check | Location | Prevents | Strict Enough? | Possibly Too Strict? |
|---|---:|---|---|---|
| Config must be YAML mapping. | Lines 29-34 | Invalid config shape. | Yes. | No. |
| Required configured source columns must exist before rename. | Lines 68-77 | Silent missing core columns. | Yes. | No. |
| `node_id` has no missing values and is integer-like. | Lines 100-108 | Invalid node grouping and filenames. | Yes. | Could be too strict if project IDs need mapping like `project_id`. |
| `timestep` has no missing values and is integer-like or date-like. | Lines 111-151 | Invalid temporal order. | Yes. | Could be too strict if partial unknown dates need filtering. |
| `project_id` has no missing values and is mapped if non-integer. | Lines 154-174 | Missing project identity. | Yes. | Could be too strict if some projects legitimately lack names. |
| `y_mask` accepts only known binary encodings. | Lines 177-209 | Invalid loss/observation masks. | Yes. | It rejects missing masks rather than defaulting to 0. That is probably good. |
| Per-column max clip requires numeric max. | Lines 219-224 | Filling with nonsense for all-missing/non-numeric columns. | Yes. | Can be too strict when all-missing columns should be filled by a default or dropped. |
| Global max clip requires at least one numeric max across matched columns. | Lines 226-232 | All-missing global group fill. | Somewhat. | It may still fill unrelated matched columns with a global max. |
| `as_feat` columns must exist. | Lines 240-245 | Missing configured features. | Yes. | No, because missing feature copies should be explicit. |
| Missing `to_drop` columns are skipped. | Lines 253-263 | Allows flexible configs. | Weak. | May hide typo in config. |
| Remaining non-numeric columns are warned and encoded. | Lines 282-286 | Avoids saving object/string columns accidentally. | Medium. | It may silently turn identifiers into high-cardinality features. |
| Missing `first_few`/`last_few` columns warn. | Lines 297-305 | Alerts layout mismatch. | Medium. | For experiment reproducibility, errors may be better. |
| `node_id`, `timestep`, `rent_per_sqft` have no missing values before save. | Lines 322-325 | Broken core panel data. | Yes. | Could be too strict if missing targets are expected and masked by `y_mask`. |
| `y_mask` only contains 0 and 1. | Lines 326-327 | Invalid mask values. | Yes. | No. |
| No duplicated `(node_id, timestep)` rows. | Lines 329-334 | Ambiguous node-time rows. | Yes. | Could be too strict if duplicates should be aggregated. |

## 7. Logging Behavior

Logs are written to console and to `{output_folder}/preprocessing.log`, configured in `setup_logging`, lines 37-53. The file log is overwritten each run.

Important logged events include input load shape, renamed columns, shape after most major steps, ID conversions, mapping file saves, max clipping matched columns, duplicated feature columns, dropped and skipped drop columns, categorical columns encoded, extra categorical warnings, sorting, correctness checks, every node file saved, feature columns saved, summary saved, and completion.

Warnings are raised and recorded for:

- Remaining non-numeric columns that are automatically one-hot encoded, lines 282-286.
- Missing configured `first_few` or `last_few` columns, lines 299-305.

The logging is useful for debugging pipeline flow and output inventory. For experiment correctness, it could be stronger by logging final column counts by type, exact dummy-column counts per categorical column, missing-value summaries before and after clipping, and whether target-like columns are present in the final feature list.

## 8. Generated File Inventory

| File / Folder | Created When | Contains | How To Verify Correctness |
|---|---|---|---|
| `{output_folder}/preprocessing.log` | At startup in `setup_logging`. | Console-equivalent progress logs. | Check input path, shapes after each step, matched clipping columns, categorical columns, warnings, and final save messages. |
| `{output_folder}/timestep.csv` | In `ensure_integer_timestep` if timestep is date-like, or if timestep is integer and `date` exists. | Mapping between integer `timestep` and `date`. | Confirm month order is correct and one timestep maps to intended date. |
| `{output_folder}/project_id.csv` | In `ensure_integer_project_id` if `project_id` is non-integer. | Integer `project_id` to original string project identity. | Spot-check condo names and verify stable sorted mapping. |
| `{output_folder}/duplicated_node_timestep.csv` | Only on duplicate failure in `run_correctness_checks`. | All duplicated `(node_id, timestep)` rows. | Inspect whether duplicates are true data errors or need aggregation. |
| `{output_folder}/all_nodes.csv` | After all checks pass. | Final full dataframe with every node and timestep row. | Check shape, no duplicates, final columns, and sorted `(node_id, timestep)`. |
| `{output_folder}/nodes/` | In `save_node_files`. | Folder of per-node CSVs. | Count files equals unique `node_id` count. |
| `{output_folder}/nodes/*.csv` | One per node in `save_node_files`. | All final columns for one node, sorted by timestep. | Open a few files and confirm monotonic timestep and same columns as `all_nodes.csv`. |
| `{output_folder}/feature_columns.csv` | In `save_feature_columns`. | Current final column names. | Verify this is actually what downstream code expects; it includes labels and masks unless filtered later. |
| `{output_folder}/preprocessing_summary.json` | At end of `preprocess`. | Input/output paths, shapes, node/timestep counts, renamed/dropped/skipped columns, categorical info, clipping columns, generated files, warnings. | Compare with log and expected experiment config. |

## 9. Potential Correctness Risks

| Risk | Why It Matters | Code Location | Severity | Suggested Fix or Check |
|---|---|---:|---|---|
| Accidental leakage from one-hot encoding before train/test split. | Future/test categories affect train-time feature schema. This may be acceptable for transductive experiments but not strict forecasting. | `encode_categorical_columns`, lines 266-288. | High | Fit categorical vocabulary on train period only, then apply fixed columns to all splits. |
| Dropping `date` while still needing it for analysis. | Final node files lose human-readable dates. Debugging forecasts by calendar month becomes harder. | `to_drop` in config line 21; `drop_configured_columns`, lines 253-263. | Medium | Preserve `date` in a metadata file or keep it out of model features but in inspection outputs. |
| Project ID mapping can change between runs if source names change. | Integer IDs are assigned by sorted unique strings. New/changed names shift or expand mapping. | `ensure_integer_project_id`, lines 164-170. | Medium | Save and reuse a frozen mapping for comparable experiments. |
| Categorical columns not listed in config are still encoded. | This avoids object columns, but can introduce high-cardinality accidental features. | `encode_categorical_columns`, lines 282-286. | Medium | Make extra encoding optional or fail unless explicitly allowed. |
| Keyword-based max clipping matches unintended columns. | Current `max_clip_glo` includes `dist`, which matches `district`; broad keywords can fill unrelated columns. | `columns_matching_keywords`, lines 212-213; config lines 10-11. | High | Use explicit column names or regex/match modes. Audit `max_clip_*_columns` in summary. |
| Duplicated node/timestep rows stop the pipeline. | Good if one row per node-month is required; bad if raw transactions need aggregation first. | `run_correctness_checks`, lines 329-334. | High | Decide whether duplicates should be aggregated, filtered, or treated as data corruption. |
| `rent_per_sqft_feat` duplicates target-like information. | If predicting current rent, this is direct target leakage. If predicting shifted future rent, it may be intended as history/current covariate. | Config lines 13-17; `duplicate_feature_columns`, lines 240-250. | High | Confirm target definition and shift logic downstream. Consider lagging `rent_per_sqft_feat`. |
| `y_mask` may be used as a feature unintentionally. | Observation masks can leak missingness/availability information and should usually be used for loss masking, not model input, unless intentional. | `convert_y_mask`, lines 177-209; final retention via `last_few`, config lines 38-41. | Medium | Ensure downstream feature selection excludes `y_mask` from input features. |
| Node files could inherit unintended final columns. | All node CSVs include every final column, including labels, masks, IDs, and dummy columns. | `save_node_files`, lines 345-355. | Medium | Verify downstream loader selects the intended feature columns only. |
| Silent skipping of missing `to_drop` columns. | A typo in config can leave unwanted columns in the dataset. | `drop_configured_columns`, lines 256-262. | Medium | Convert skipped drops to warnings or errors in strict mode. |
| Missing configured categorical columns are only logged. | Schema drift may go unnoticed in experiment runs. | `encode_categorical_columns`, lines 272-277. | Medium | Record skipped categoricals in summary, or error in strict mode. |
| `feature_columns.csv` name may overpromise. | It lists all final columns, not necessarily train features. | `save_feature_columns`, lines 358-361. | Medium | Rename file or create `model_feature_columns.csv` with exclusions. |
| Output folder is overwritten incrementally. | Old files in `nodes/` can remain if a later run has fewer nodes. | `save_node_files`, lines 345-355. | Medium | Clean or version output directories before saving, with explicit user approval. |

## 10. Readability and Maintainability Review

The code is generally modular and readable. Most functions have a single clear job: loading config, logging setup, standard renaming, ID conversion, mask conversion, clipping, feature duplication, dropping, encoding, ordering, sorting, checking, and saving are separated.

Names are mostly clear. The main naming concern is the config key `input_folder`, which is actually a CSV file path. Another concern is `feature_columns.csv`, because it currently records final columns rather than confirmed model input features.

Error messages are usually actionable. The max clipping error that found an all-NaN distance column is especially useful because it names the matched column. The duplicate error also saves a diagnostic CSV.

Config behavior is explicit for major transformations, but some fallback behavior is implicit: missing drop columns are skipped, missing categorical columns are only logged, and extra object columns are automatically encoded. Those are convenient for exploration but can hide schema drift in serious experiment runs.

The most fragile part is keyword-based clipping. It is compact, but substring matching is too blunt for research pipelines where feature names carry many meanings.

## 11. Questions I Should Answer Before Trusting This Pipeline

1. Should `rent_per_sqft_feat` be included as an input feature, or should it be lagged/shifted to avoid target leakage?
2. Should `y_mask` be included as a feature, or only used for loss masking and evaluation?
3. Should one-hot encoding be fitted only on training data, especially if the experiment claims strict forecasting?
4. Should duplicate `(node_id, timestep)` rows be aggregated, deduplicated by rule, or treated as a hard data error?
5. Should `date` be preserved in final output for debugging and temporal audits?
6. Should missing `to_drop`, `first_few`, or `last_few` columns be warnings or errors?
7. Should missing configured categorical columns be warnings or errors?
8. Should `node_id` and `project_id` both remain in final data, and what is the difference between them for modeling?
9. Should max clipping use explicit column names instead of keyword matching?
10. Should all-missing distance columns be dropped, filled from global distance max, or filled with a domain-specific sentinel?
11. Should output directories be versioned or cleaned to prevent stale node files?
12. Does downstream training use `feature_columns.csv` as input features, or does it know to remove target/mask/ID columns?

## 12. Final Verdict

Verdict: partially matches the intended preprocessing logic.

Main strengths:

- The pipeline is modular and easy to inspect.
- Core columns are validated before renaming.
- IDs, timesteps, and masks get explicit conversions.
- Duplicate node-timestep rows are treated seriously and produce a diagnostic file.
- Logs and summary metadata make runs auditable.
- One-hot outputs are now integer `0/1`, matching the requested format.

Main risks:

- `rent_per_sqft_feat` may leak the target depending on the downstream prediction horizon.
- One-hot encoding is fitted on the full dataset.
- Broad keyword clipping can match unintended columns such as `district`.
- `feature_columns.csv` may be mistaken for a clean model feature list.
- Several config mismatches are skipped or warned rather than failed.

Highest-priority changes before using for experiments:

1. Decide whether `rent_per_sqft_feat` and `y_mask` are model inputs or only training/evaluation support columns.
2. Replace keyword-based clipping with explicit column lists or a stricter matching mode.
3. Decide whether categorical encoding should be fit on train only.
4. Clarify whether duplicates should be aggregated or should remain hard errors.
5. Rename or redefine `feature_columns.csv` so it cannot be confused with the actual model input feature set.
