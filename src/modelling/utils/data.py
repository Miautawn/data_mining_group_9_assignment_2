import lightgbm as lgb
import numpy as np
import pandas as pd


def load_parquet_files(x_path, y_path=None, group_path=None):
    """Loads feature, target, and group dataframes, ensuring proper sorting."""
    x_df = pd.read_parquet(x_path).sort_values("srch_id").reset_index(drop=True)

    y_df, groups = None, None
    if y_path:
        y_df = pd.read_parquet(y_path).sort_values("srch_id").reset_index(drop=True)
    if group_path:
        groups = pd.read_parquet(group_path).sort_values("srch_id")

    return x_df, y_df, groups


def extract_features(df, ignore_cols=["srch_id", "prop_id", "relevance"]):
    """Extracts trainable feature columns."""
    return [col for col in df.columns if col not in ignore_cols]


def create_lgb_dataset(x_df, y_df, groups_df, feature_cols, reference=None):
    """Creates a LightGBM dataset applying specific sample weights."""
    weights = (
        np.where(x_df["random_bool"] == 1, 0.3522, 1.0)
        if "random_bool" in x_df.columns
        else None
    )

    return lgb.Dataset(
        data=x_df[feature_cols],
        label=y_df["relevance"] if y_df is not None else None,
        group=groups_df["group_size"],
        weight=weights,
        reference=reference,
        free_raw_data=True,
    )


def split_for_tuning(x_df, y_df, groups_df, split_ratio=0.1):
    """Splits full datasets based on unique search IDs to prevent data leakage."""
    unique_searches = x_df["srch_id"].unique()
    np.random.shuffle(unique_searches)

    split_idx = int(len(unique_searches) * split_ratio)
    train_ids = unique_searches[:split_idx]
    val_ids = unique_searches[split_idx:]

    splits = {}
    for name, ids in [("train", train_ids), ("val", val_ids)]:
        splits[f"x_{name}"] = (
            x_df[x_df["srch_id"].isin(ids)]
            .copy()
            .sort_values(["srch_id", "prop_id"])
            .reset_index(drop=True)
        )
        splits[f"y_{name}"] = (
            y_df[y_df["srch_id"].isin(ids)]
            .copy()
            .sort_values(["srch_id", "prop_id"])
            .reset_index(drop=True)
        )
        splits[f"groups_{name}"] = (
            groups_df[groups_df["srch_id"].isin(ids)]
            .copy()
            .sort_values(["srch_id"])
            .reset_index(drop=True)
        )

    return splits
