from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

class FormNodeProcessor:
    def __init__(self, config):
        self.config = config
        self.in_paths, self.out_paths = self.set_io_paths(config["io_paths"], config["io_node_paths"])
        
    def set_io_paths(self, io_paths, io_node_paths):
        input_path = Path(io_paths["input_path"])
        output_folder = Path(io_paths["output_folder"])
        output_folder.mkdir(parents=True, exist_ok=True)
        full_project_location_path = io_paths.get("full_project_location_path")

        data_file_ext = io_node_paths.get("data_file_ext", ".csv")
        if data_file_ext and not data_file_ext.startswith("."):
            data_file_ext = f".{data_file_ext}"

        def output_data_path(filename_key):
            filename = Path(io_node_paths[filename_key])
            if filename.suffix:
                return output_folder / filename
            return output_folder / f"{filename}{data_file_ext}"

        def output_metadata_folder(filename_key):
            filename = Path(io_node_paths[filename_key])
            return output_folder / filename.with_suffix("")

        project_node_folder = output_metadata_folder("project_node_filename")
        size_node_folder = output_metadata_folder("size_node_filename")
        project_node_folder.mkdir(parents=True, exist_ok=True)
        size_node_folder.mkdir(parents=True, exist_ok=True)

        in_paths = {
            "input_path": input_path,
            "full_project_location_path": (
                Path(full_project_location_path)
                if full_project_location_path is not None
                else None
            ),
        }
        out_paths = {
            "output_folder": output_folder,
            "project_info_path": output_data_path("project_info_filename"),
            "timestep_info_path": output_data_path("timestep_info_filename"),
            "node_id_path": output_data_path("node_id_filename"),
            "project_node_path": output_data_path("project_node_filename"),
            "size_node_path": output_data_path("size_node_filename"),
            "project_node_folder": project_node_folder,
            "size_node_folder": size_node_folder,
            "project_node_standardisation_path": project_node_folder / io_node_paths["standardisation_file"],
            "project_node_normalisation01_path": project_node_folder / io_node_paths["normalisation01_file"],
            "project_node_normalisation11_path": project_node_folder / io_node_paths["normalisation11_file"],
            "size_node_standardisation_path": size_node_folder / io_node_paths["standardisation_file"],
            "size_node_normalisation01_path": size_node_folder / io_node_paths["normalisation01_file"],
            "size_node_normalisation11_path": size_node_folder / io_node_paths["normalisation11_file"],
        }

        self.data_file_ext = data_file_ext
        for name, path in {**in_paths, **out_paths}.items():
            setattr(self, name, path)
        return in_paths, out_paths

    def fill_node_id(self, df):
        if "node_id" not in df.columns:
            df["node_id"] = np.arange(len(df))
        return df
    
    def fill_timestep(self, df):
        if "timestep" not in df.columns:
            if "date" not in df.columns:
                raise KeyError("Missing date column before filling timestep")

            parsed_dates = pd.to_datetime(df["date"], errors="raise")
            if parsed_dates.isna().any():
                raise ValueError("date contains missing values")

            timestep_map = {
                date: timestep
                for timestep, date in enumerate(sorted(parsed_dates.unique()))
            }
            df["timestep"] = parsed_dates.map(timestep_map).astype(int)
        return df
    
    def fill_project_id(self, df):
        if "project_id" not in df.columns:
            if "project_name" not in df.columns:
                raise KeyError("Missing project_name column before filling project_id")
            if df["project_name"].isna().any():
                raise ValueError("project_name contains missing values")

            project_id_map = {
                project_name: project_id
                for project_id, project_name in enumerate(sorted(df["project_name"].unique()))
            }
            df["project_id"] = df["project_name"].map(project_id_map).astype(int)
        return df

    def fill_project_location_col(self, df, location_col):
        if "project_name" not in df.columns:
            raise KeyError(f"Missing project_name column before filling {location_col}")
        if df["project_name"].isna().any():
            raise ValueError("project_name contains missing values")

        full_project_location_path = getattr(self, "full_project_location_path", None)
        if full_project_location_path is None:
            raise ValueError(f"full_project_location_path is required to fill {location_col}")

        location_df = pd.read_csv(full_project_location_path)
        required_cols = ["project_name", location_col]
        missing_cols = [
            col for col in required_cols if col not in location_df.columns
        ]
        if missing_cols:
            raise KeyError(
                f"Missing columns in full project location file: {missing_cols}"
            )
        if location_df["project_name"].isna().any():
            raise ValueError("project_name contains missing values in full project location file")

        location_counts = location_df.groupby("project_name")[location_col].nunique(dropna=False)
        if (location_counts > 1).any():
            raise ValueError(
                f"project_name must map to exactly one {location_col} in full project location file"
            )

        location_map = (
            location_df
            .drop_duplicates("project_name")
            .set_index("project_name")[location_col]
        )
        location_values = df["project_name"].map(location_map)
        if location_values.isna().any():
            missing_project_names = sorted(
                df.loc[location_values.isna(), "project_name"].unique()
            )
            raise KeyError(
                f"Missing {location_col} for project_name values: {missing_project_names}"
            )

        df[location_col] = pd.to_numeric(location_values, errors="raise").astype(float)
        return df

    def fill_longitude(self, df):
        if "longitude" not in df.columns:
            df = self.fill_project_location_col(df, "longitude")
        return df
    
    def fill_latitude(self, df):
        if "latitude" not in df.columns:
            df = self.fill_project_location_col(df, "latitude")
        return df

    def rename(self, df, config):
        fill_methods = {
            "node_id": self.fill_node_id,
            "project_id": self.fill_project_id,
            "timestep": self.fill_timestep,
            "longitude": self.fill_longitude,
            "latitude": self.fill_latitude,
        }

        missing_cols = [
            source_col
            for source_col in config.values()
            if source_col is not None
            if source_col not in df.columns
        ]

        if missing_cols:
            raise KeyError(f"Missing dataframe columns before rename: {missing_cols}")

        renamed_df = df.copy()
        for target_col, source_col in config.items():
            if source_col is None:
                if target_col not in fill_methods:
                    raise ValueError(
                        f"Null rename source is only supported for {list(fill_methods)}"
                    )
                renamed_df = fill_methods[target_col](renamed_df)
                continue

            renamed_df[target_col] = renamed_df[source_col]

        # remove any columns that have been renamed but are not in the target columns
        source_cols = {source_col for source_col in config.values() if source_col is not None}
        target_cols = set(config.keys())
        cols_to_drop = [
            col
            for col in source_cols - target_cols
            if col in renamed_df.columns
        ]
        renamed_df = renamed_df.drop(columns=cols_to_drop)

        return renamed_df
    
    def check_std(self, df):
        standard_cols = self.config["standard_cols"]
        missing_cols = [col for col in standard_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Missing standard columns after rename: {missing_cols}")

        checked_df = df.copy()

        def convert_int_col(col):
            values = pd.to_numeric(checked_df[col], errors="raise")
            if values.isna().any():
                raise ValueError(f"{col} contains missing values")
            if not np.isclose(values % 1, 0).all():
                raise ValueError(f"{col} must contain integer-like values")
            checked_df[col] = values.astype(int)

        convert_int_col("node_id")
        convert_int_col("timestep")
        if checked_df["project_name"].isna().any():
            raise ValueError("project_name contains missing values")

        name_counts = checked_df.groupby("project_id")["project_name"].nunique(dropna=False)
        if (name_counts > 1).any():
            raise ValueError("project_id must map to exactly one project_name")
        id_counts = checked_df.groupby("project_name")["project_id"].nunique(dropna=False)
        if (id_counts > 1).any():
            raise ValueError("project_name must map to exactly one project_id")

        project_names = sorted(checked_df["project_name"].dropna().unique())
        project_id_map = {
            project_name: project_id
            for project_id, project_name in enumerate(project_names)
        }
        checked_df["project_id"] = checked_df["project_name"].map(project_id_map).astype(int)

        parsed_dates = pd.to_datetime(checked_df["date"], errors="raise")
        checked_df["date"] = parsed_dates
        timestep_date_counts = checked_df.groupby("timestep")["date"].nunique(dropna=False)
        if (timestep_date_counts > 1).any():
            raise ValueError("timestep must map to exactly one date")
        date_timestep_counts = checked_df.groupby("date")["timestep"].nunique(dropna=False)
        if (date_timestep_counts > 1).any():
            raise ValueError("date must map to exactly one timestep")

        timestep_dates = (
            checked_df[["timestep", "date"]]
            .drop_duplicates()
            .sort_values("timestep")
        )
        if not timestep_dates["date"].is_monotonic_increasing:
            raise ValueError("date order must increase with timestep")
        timestep_id_map = {
            timestep: timestep_id
            for timestep_id, timestep in enumerate(timestep_dates["timestep"])
        }
        checked_df["timestep"] = checked_df["timestep"].map(timestep_id_map).astype(int)

        size_tier_map = {f"SZ{idx}": idx for idx in range(1, 6)}
        size_tier = checked_df["size_tier"]
        if pd.api.types.is_numeric_dtype(size_tier):
            size_tier_values = pd.to_numeric(size_tier, errors="raise")
        else:
            size_tier_values = size_tier.astype(str).str.upper().map(size_tier_map)
        if size_tier_values.isna().any():
            raise ValueError("size_tier must be one of SZ1, SZ2, SZ3, SZ4, SZ5 or 1, 2, 3, 4, 5")
        if not size_tier_values.isin(range(1, 6)).all():
            raise ValueError("size_tier must be in the range 1 to 5")
        checked_df["size_tier"] = size_tier_values.astype(int)

        checked_df["rent_per_sqft"] = pd.to_numeric(
            checked_df["rent_per_sqft"],
            errors="raise",
        ).astype(float)

        y_mask_map = {
            True: 1,
            False: 0,
            "true": 1,
            "false": 0,
            "True": 1,
            "False": 0,
            "TRUE": 1,
            "FALSE": 0,
        }
        if checked_df["y_mask"].dtype == bool:
            y_mask = checked_df["y_mask"].astype(int)
        else:
            y_mask = checked_df["y_mask"].replace(y_mask_map)
            y_mask = pd.to_numeric(y_mask, errors="raise")
        if y_mask.isna().any() or not y_mask.isin([0, 1]).all():
            raise ValueError("y_mask must contain only 0/1 or true/false values")
        checked_df["y_mask"] = y_mask.astype(int)

        return checked_df

    def bfr_save_preprocessing(self, df, config):
        processed_df = df.copy()

        if not config["other_feat"]:
            missing_cols = [col for col in config["cols"] if col not in processed_df.columns]
            if missing_cols:
                raise KeyError(f"Missing configured columns before save: {missing_cols}")
            processed_df = processed_df[config["cols"]]

        to_drop = config.get("to_drop", [])
        if to_drop:
            missing_drop_cols = [col for col in to_drop if col not in processed_df.columns]
            if missing_drop_cols:
                raise KeyError(f"Missing columns configured to drop: {missing_drop_cols}")
            processed_df = processed_df.drop(columns=to_drop)

        first_few = config.get("first_few", [])
        last_few = config.get("last_few", [])
        sequence_cols = first_few + last_few
        missing_sequence_cols = [
            col
            for col in sequence_cols
            if col not in processed_df.columns
        ]
        if missing_sequence_cols:
            raise KeyError(f"Missing columns configured for ordering: {missing_sequence_cols}")

        duplicated_sequence_cols = [
            col
            for col in first_few
            if col in last_few
        ]
        if duplicated_sequence_cols:
            raise ValueError(
                f"Columns cannot appear in both first_few and last_few: {duplicated_sequence_cols}"
            )

        sort_by = config["sort_by"]
        missing_sort_cols = [col for col in sort_by if col not in processed_df.columns]
        if missing_sort_cols:
            raise KeyError(f"Missing columns configured for sorting: {missing_sort_cols}")
        if sort_by:
            processed_df = processed_df.sort_values(sort_by).reset_index(drop=True)

        edge_cols = set(sequence_cols)
        middle_cols = [col for col in processed_df.columns if col not in edge_cols]
        return processed_df[first_few + middle_cols + last_few]

    def save_meta_info(self, df, project_info_path, timestep_info_path):

        project_info = (
            df[self.config["project_info"]["cols"]]
            .drop_duplicates()
            .sort_values("project_id")
        )
        timestep_info = (
            df[self.config["timestep_info"]["cols"]]
            .drop_duplicates()
            .sort_values("timestep")
        )
        df_proj = self.bfr_save_preprocessing(project_info, self.config["project_info"])
        df_timestep = self.bfr_save_preprocessing(timestep_info, self.config["timestep_info"])
        df_proj.to_csv(project_info_path, index=False)
        df_timestep.to_csv(timestep_info_path, index=False)

    def prepare_projlv_df(self, df, config):
        aggregate_df = df.copy()

        required_keys = ["to_drop", "group_by"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise KeyError(f"Missing project-level aggregation config keys: {missing_keys}")

        group_by = config["group_by"]
        missing_group_cols = [col for col in group_by if col not in aggregate_df.columns]
        if missing_group_cols:
            raise KeyError(f"Missing columns configured for grouping: {missing_group_cols}")

        to_drop = config["to_drop"]
        missing_drop_cols = [col for col in to_drop if col not in aggregate_df.columns]
        if missing_drop_cols:
            raise KeyError(f"Missing columns configured to drop: {missing_drop_cols}")
        if to_drop:
            aggregate_df = aggregate_df.drop(columns=to_drop)

        if "rent_per_sqft" not in aggregate_df.columns:
            raise KeyError("Missing rent_per_sqft column for project-level aggregation")

        missing_group_cols = [col for col in group_by if col not in aggregate_df.columns]
        if missing_group_cols:
            raise KeyError(f"Grouping columns were dropped before aggregation: {missing_group_cols}")

        agg_funcs = {
            col: "first"
            for col in aggregate_df.columns
            if col not in group_by
        }
        agg_funcs["rent_per_sqft"] = "mean"

        aggregate_df = (
            aggregate_df
            .groupby(group_by, as_index=False, dropna=False)
            .agg(agg_funcs)
        )
        aggregate_df["y_mask"] = aggregate_df["rent_per_sqft"].notna().astype(int)
        aggregate_df["size_tier"] = 0
        return aggregate_df

    def reindex_nodes_id(self, project_node_df, original_df, config):
        if "sort_by" not in config:
            raise KeyError("Missing reindex node id config key: sort_by")

        logger.info(
            "Reindexing node ids: project_node_shape=%s original_shape=%s",
            project_node_df.shape,
            original_df.shape,
        )

        df_concat = pd.concat(
            [project_node_df.copy(), original_df.copy()],
            ignore_index=True,
            sort=False,
        )

        sort_by = config["sort_by"]
        group_by = config["group_by"]
        missing_group_cols = [col for col in group_by if col not in df_concat.columns]
        if missing_group_cols:
            raise KeyError(f"Missing columns configured for node-id grouping: {missing_group_cols}")
        missing_sort_cols = [col for col in sort_by if col not in df_concat.columns]
        if missing_sort_cols:
            raise KeyError(f"Missing columns configured for node-id sorting: {missing_sort_cols}")

        df_concat_index = (
            df_concat
            .groupby(group_by, dropna=False)
            .size()
            .reset_index(name="_group_size")
            .drop(columns="_group_size")
            .sort_values(sort_by)
            .reset_index(drop=True)
        )
        df_concat_index["node_id"] = np.arange(len(df_concat_index))

        df_concat_index.to_csv(self.out_paths["node_id_path"], index=False)

        df_concat = df_concat.drop(columns=["node_id"], errors="ignore")
        df_concat = df_concat.merge(df_concat_index, on=group_by, how="left")
        if df_concat["node_id"].isna().any():
            raise ValueError("Failed to map node_id for some rows during reindexing")
        df_concat["node_id"] = df_concat["node_id"].astype(int)

        project_node_df = df_concat[df_concat["size_tier"] == 0].reset_index(drop=True)
        size_node_df = df_concat[df_concat["size_tier"] != 0].reset_index(drop=True)

        logger.info(
            "Reindexed node ids: combined_shape=%s project_node_shape=%s size_node_shape=%s",
            df_concat.shape,
            project_node_df.shape,
            size_node_df.shape,
        )

        return project_node_df, size_node_df

    def feature_preprocessing(self, df, config):
        """Categorical encoding and duplicate columns as features."""
        processed_df = df.copy()
        new_cols = []

        categorical_cols = config["categorical"]
        missing_categorical_cols = [
            col for col in categorical_cols if col not in processed_df.columns
        ]
        if missing_categorical_cols:
            raise KeyError(
                f"Missing configured categorical columns before feature preprocessing: {missing_categorical_cols}"
            )
        if categorical_cols:
            encoded_df = pd.get_dummies(
                processed_df[categorical_cols],
                columns=categorical_cols,
                dummy_na=False,
                dtype=int,
            )
            new_cols.extend(encoded_df.columns.tolist())
            processed_df = pd.concat([processed_df, encoded_df], axis=1)

        as_feat_cols = config["as_feat"]
        missing_as_feat_cols = [
            col for col in as_feat_cols if col not in processed_df.columns
        ]
        if missing_as_feat_cols:
            raise KeyError(
                f"Missing configured as_feat columns before feature preprocessing: {missing_as_feat_cols}"
            )
        for col in as_feat_cols:
            feat_col = f"{col}_feat"
            processed_df[feat_col] = processed_df[col]
            new_cols.append(feat_col)

        config.setdefault("cols", [])
        if categorical_cols:
            processed_df = processed_df.drop(columns=categorical_cols)
            config["cols"] = [
                col for col in config["cols"] if col not in categorical_cols
            ]
        config["cols"].extend([
            col for col in new_cols if col not in config["cols"]
        ])
        return processed_df
    
    def save_stand_norm_info(self, df, config, standardisation_path, normalisation01_path, normalisation11_path):
        excluded_cols = set(config.get("non_feat", config.get("no_norm", [])))
        stat_cols = [
            col
            for col in df.columns
            if col not in excluded_cols
        ]

        standardisation = {}
        normalisation01 = {}
        normalisation11 = {}

        for col in stat_cols:
            values = pd.to_numeric(df[col], errors="raise")
            finite_values = values.dropna()
            if finite_values.empty:
                raise ValueError(f"Column {col} has no numeric values for standardisation/normalisation")

            mean = float(finite_values.mean())
            std = float(finite_values.std(ddof=0))
            min_value = float(finite_values.min())
            max_value = float(finite_values.max())
            denominator = max_value - min_value
            if std == 0:
                std = 1.0
            if denominator == 0:
                denominator = 1.0

            standardisation[col] = {
                "mean": mean,
                "std": std,
            }
            normalisation01[col] = {
                "min": min_value,
                "max": max_value,
                "denominator": denominator,
                "target_min": 0.0,
                "target_max": 1.0,
            }
            normalisation11[col] = {
                "min": min_value,
                "max": max_value,
                "denominator": denominator,
                "target_min": -1.0,
                "target_max": 1.0,
            }

        output_paths = [
            (standardisation_path, standardisation),
            (normalisation01_path, normalisation01),
            (normalisation11_path, normalisation11),
        ]
        for output_path, mapping in output_paths:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(mapping, f, indent=2)
    
    def process_n_store_size_node(self, size_node_df, config):
        processed_df = self.feature_preprocessing(size_node_df, config)
        processed_df = self.bfr_save_preprocessing(processed_df, config)
        self.save_stand_norm_info(
            processed_df,
            config,
            self.out_paths["size_node_standardisation_path"],
            self.out_paths["size_node_normalisation01_path"],
            self.out_paths["size_node_normalisation11_path"],
        )
        processed_df.to_csv(self.out_paths["size_node_path"], index=False)
        return processed_df
    
    def max_clip_fillna(self, df, config):
        processed_df = df.copy()
        max_clip_keywords = config.get("max_clip_glop", config.get("max_clip_glo", []))
        max_clip_cols = list(config.get("max_clip_col", []))
        excluded_cols = set(config.get("categorical", [])) | set(config.get("to_drop", []))
        keyword_cols = [
            col
            for col in processed_df.columns
            if col not in excluded_cols
            and any(keyword in col for keyword in max_clip_keywords)
        ]
        max_clip_cols.extend([
            col for col in keyword_cols if col not in max_clip_cols
        ])

        if not max_clip_cols:
            return processed_df

        missing_cols = [
            col for col in max_clip_cols if col not in processed_df.columns
        ]
        if missing_cols:
            raise KeyError(
                f"Missing configured max-clip/fillna columns: {missing_cols}"
            )

        numeric_values = processed_df[max_clip_cols].apply(
            pd.to_numeric,
            errors="raise",
        )
        finite_values = numeric_values.to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            raise ValueError(
                "Configured max-clip/fillna columns have no numeric values"
            )

        fill_value = float(np.ceil(np.max(finite_values)))
        processed_df[max_clip_cols] = numeric_values.clip(upper=fill_value).fillna(fill_value)

        return processed_df

    def process_n_store_project_node(self, project_node_df, config):
        processed_df = self.max_clip_fillna(project_node_df, config)
        processed_df = self.feature_preprocessing(processed_df, config)
        processed_df = self.bfr_save_preprocessing(processed_df, config)
        self.save_stand_norm_info(
            processed_df,
            config,
            self.out_paths["project_node_standardisation_path"],
            self.out_paths["project_node_normalisation01_path"],
            self.out_paths["project_node_normalisation11_path"],
        )
        processed_df.to_csv(self.out_paths["project_node_path"], index=False)
        return processed_df

    def process(self):
        missing_out_paths = [
            path
            for path in self.out_paths.values()
            if not Path(path).exists()
        ]
        if not missing_out_paths:
            logger.info("All output paths exist; loading cached node files")
            proj_node_df = pd.read_csv(self.out_paths["project_node_path"])
            size_node_df = pd.read_csv(self.out_paths["size_node_path"])
            logger.info(
                "Loaded cached nodes: project_node_shape=%s size_node_shape=%s",
                proj_node_df.shape,
                size_node_df.shape,
            )
            return proj_node_df, size_node_df

        df = pd.read_csv(self.in_paths["input_path"])
        logger.info(
            "Loaded input data: path=%s shape=%s",
            self.in_paths["input_path"],
            df.shape,
        )
        df = self.rename(df, self.config["rename"])
        logger.info("Renamed input columns: shape=%s", df.shape)
        df = self.check_std(df)
        logger.info("Validated standard columns: shape=%s", df.shape)
        self.save_meta_info(
            df,
            self.out_paths["project_info_path"],
            self.out_paths["timestep_info_path"],
        )
        logger.info(
            "Saved metadata: project_info_path=%s timestep_info_path=%s",
            self.out_paths["project_info_path"],
            self.out_paths["timestep_info_path"],
        )

        # project-level aggregation
        project_node_df = self.prepare_projlv_df(df, self.config["form_project_node"])
        project_node_df.to_csv(self.out_paths["project_node_path"], index   =False) 
        logger.info("Prepared project-level nodes: shape=%s", project_node_df.shape)

        proj_node_df, size_node_df = self.reindex_nodes_id(project_node_df, df, self.config["reindex_node_id"])

        # proj_node_df.to_csv("project_node_path_raw.csv", index=False)
        # size_node_df.to_csv("size_node_path_raw.csv", index=False)
        # preprocess size node
        size_node_df = self.process_n_store_size_node(size_node_df, self.config["size_node"])
        logger.info("Processed and saved size nodes: shape=%s", size_node_df.shape)
        # preprocess project node
        proj_node_df = self.process_n_store_project_node(proj_node_df, self.config["project_node"])
        logger.info("Processed and saved project nodes: shape=%s", proj_node_df.shape)
        return proj_node_df, size_node_df

    


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config_path = Path("src/config/graph/V260519.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info("Loaded config: path=%s name=%s", config_path, config.get("name"))
    
    node_processor = FormNodeProcessor(config)
    node_processor.process()
