"""Build Task 3 data-preparation outputs for the Expedia ranking assignment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_preparation.evaluation import baseline_ndcg_scores
from src.data_preparation.feature_config import (
    COMPETITOR_IDS,
    DEAL_RATIO_CLIP,
    DEFAULT_DTYPE_MAP,
    EPSILON,
    IMPORTANT_MISSING_COLUMNS,
    LEAKAGE_COLUMNS,
    MAX_HISTORICAL_PRICE,
    OUTPUT_DIR,
    PRICE_CAP_QUANTILE,
    RANDOM_SEED,
    SMOOTHING_ALPHA,
    SPARSE_KEY_SMOOTHING_ALPHA,
    TARGET_COLUMNS,
    TEST_FILE,
    TRAIN_FILE,
    VALID_SIZE,
)
from src.data_preparation.leakage_safe_encodings import SmoothedHistoryEncoder
from src.data_preparation.utils import (
    downcast_numeric,
    ensure_dir,
    load_csv,
    make_relevance,
    missing_percentages,
    save_group_sizes,
    save_json,
    save_table,
    setup_logging,
)
from src.data_preparation.validation_split import split_by_srch_id


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=TRAIN_FILE)
    parser.add_argument("--test-path", type=Path, default=TEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--valid-size", type=float, default=VALID_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Read only first N rows from each CSV.",
    )
    parser.add_argument(
        "--synthetic-smoke",
        action="store_true",
        help="Run the pipeline on a tiny synthetic Expedia-like dataset.",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    run_pipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        valid_size=args.valid_size,
        seed=args.seed,
        sample_rows=args.sample_rows,
        synthetic_smoke=args.synthetic_smoke,
    )


def run_pipeline(
    train_path: Path = TRAIN_FILE,
    test_path: Path = TEST_FILE,
    output_dir: Path = OUTPUT_DIR,
    valid_size: float = VALID_SIZE,
    seed: int = RANDOM_SEED,
    sample_rows: int | None = None,
    synthetic_smoke: bool = False,
) -> dict[str, Any]:
    ensure_dir(output_dir)
    LOGGER.info("Loading data")
    if synthetic_smoke:
        train_raw, test_raw = make_synthetic_data()
    else:
        train_raw = load_csv(train_path, nrows=sample_rows, dtype_map=DEFAULT_DTYPE_MAP)
        test_raw = load_csv(test_path, nrows=sample_rows, dtype_map=DEFAULT_DTYPE_MAP)

    missing_meta = {
        "train": missing_percentages(train_raw),
        "test": missing_percentages(test_raw),
    }

    LOGGER.info("Splitting validation by srch_id")
    train_idx, valid_idx = split_by_srch_id(train_raw, valid_size=valid_size, seed=seed)
    train_part_raw = train_raw.loc[train_idx].copy()
    valid_part_raw = train_raw.loc[valid_idx].copy()

    LOGGER.info("Fitting preprocessing statistics")
    split_stats = fit_preprocessing_stats(train_part_raw)
    full_stats = fit_preprocessing_stats(train_raw)

    LOGGER.info("Creating non-target features")
    train_part_features, base_meta = build_base_features(train_part_raw, split_stats)
    valid_features_base, valid_base_meta = build_base_features(
        valid_part_raw, split_stats
    )
    full_train_base, full_base_meta = build_base_features(train_raw, full_stats)
    test_base, test_base_meta = build_base_features(test_raw, full_stats)
    base_meta["validation_columns_seen"] = valid_base_meta["input_columns"]
    base_meta["full_train_columns_seen"] = full_base_meta["input_columns"]
    base_meta["test_columns_seen"] = test_base_meta["input_columns"]

    LOGGER.info("Fitting leakage-safe historical encodings")
    split_encoder = SmoothedHistoryEncoder(
        alpha=SMOOTHING_ALPHA, sparse_alpha=SPARSE_KEY_SMOOTHING_ALPHA
    ).fit(attach_history_price(train_part_raw, train_part_features))
    full_encoder = SmoothedHistoryEncoder(
        alpha=SMOOTHING_ALPHA, sparse_alpha=SPARSE_KEY_SMOOTHING_ALPHA
    ).fit(attach_history_price(train_raw, full_train_base))
    train_hist = split_encoder.transform(train_part_raw)
    valid_hist = split_encoder.transform(valid_part_raw)
    full_train_hist = full_encoder.transform(train_raw)
    test_hist = full_encoder.transform(test_raw)

    train_features = pd.concat(
        [train_part_features.reset_index(drop=True), train_hist.reset_index(drop=True)],
        axis=1,
    )
    valid_features = pd.concat(
        [valid_features_base.reset_index(drop=True), valid_hist.reset_index(drop=True)],
        axis=1,
    )
    full_train_features = pd.concat(
        [
            full_train_base.reset_index(drop=True),
            full_train_hist.reset_index(drop=True),
        ],
        axis=1,
    )
    test_features = pd.concat(
        [test_base.reset_index(drop=True), test_hist.reset_index(drop=True)], axis=1
    )

    train_features, valid_features, _, impute_meta = align_and_impute(
        train_features,
        valid_features,
        test_features,
    )
    full_train_features, test_features, final_impute_meta = align_and_impute_two(
        full_train_features,
        test_features,
    )

    y_train = make_label_frame(train_part_raw).reset_index(drop=True)
    y_valid = make_label_frame(valid_part_raw).reset_index(drop=True)
    y_full_train = make_label_frame(train_raw).reset_index(drop=True)

    train_features, y_train = sort_features_and_labels(train_features, y_train)
    valid_features, y_valid = sort_features_and_labels(valid_features, y_valid)
    full_train_features, y_full_train = sort_features_and_labels(
        full_train_features, y_full_train
    )
    test_features = sort_features(test_features)

    leakage_present = sorted(
        set(train_features.columns) & set(TARGET_COLUMNS + LEAKAGE_COLUMNS)
    )
    if leakage_present:
        raise ValueError(
            f"Leakage columns present in final features: {leakage_present}"
        )

    baseline_scores = baseline_ndcg_scores(y_valid, valid_features, k=5, seed=seed)

    LOGGER.info("Saving outputs")
    output_paths = {
        "X_train": str(
            save_table(downcast_numeric(train_features), output_dir / "X_train.parquet")
        ),
        "X_valid": str(
            save_table(downcast_numeric(valid_features), output_dir / "X_valid.parquet")
        ),
        "X_test": str(
            save_table(downcast_numeric(test_features), output_dir / "X_test.parquet")
        ),
        "full_train_features": str(
            save_table(
                downcast_numeric(full_train_features),
                output_dir / "full_train_features.parquet",
            )
        ),
        "train_labels": str(save_table(y_train, output_dir / "y_train.parquet")),
        "valid_labels": str(save_table(y_valid, output_dir / "y_valid.parquet")),
        "full_train_labels": str(
            save_table(y_full_train, output_dir / "y_full_train.parquet")
        ),
        "train_group_sizes": str(
            save_group_sizes(y_train, output_dir / "groups_train.parquet")
        ),
        "valid_group_sizes": str(
            save_group_sizes(y_valid, output_dir / "groups_valid.parquet")
        ),
        "full_train_group_sizes": str(
            save_group_sizes(y_full_train, output_dir / "groups_full_train.parquet")
        ),
    }

    split_indices = pd.DataFrame(
        {
            "row_index": np.concatenate([train_idx, valid_idx]),
            "split": ["train"] * len(train_idx) + ["valid"] * len(valid_idx),
        }
    )
    output_paths["split_indices"] = str(
        save_table(split_indices, output_dir / "validation_split_indices.parquet")
    )
    baseline_path = output_dir / "baseline_ndcg.json"

    metadata = {
        "input_paths": {
            "train": str(train_path),
            "test": str(test_path),
            "synthetic_smoke": synthetic_smoke,
            "sample_rows": sample_rows,
        },
        "rows": {
            "train_raw": int(len(train_raw)),
            "test_raw": int(len(test_raw)),
            "train_split": int(len(train_features)),
            "valid_split": int(len(valid_features)),
            "full_train_features": int(len(full_train_features)),
        },
        "validation": {
            "split_column": "srch_id",
            "valid_size": valid_size,
            "seed": seed,
            "n_train_searches": int(y_train["srch_id"].nunique()),
            "n_valid_searches": int(y_valid["srch_id"].nunique()),
            "group_order": "feature and label files are sorted by srch_id before group sizes are saved",
        },
        "feature_names": list(train_features.columns),
        "feature_count": len(train_features.columns),
        "dropped_columns": base_meta["dropped_columns"],
        "target_columns_excluded": TARGET_COLUMNS,
        "leakage_columns_excluded": LEAKAGE_COLUMNS,
        "leakage_columns_found_in_features": leakage_present,
        "missing_percentages": missing_meta,
        "missing_indicators": base_meta["missing_indicators"],
        "imputation": impute_meta,
        "final_train_test_imputation": final_impute_meta,
        "fitted_preprocessing": {
            "validation_fit": split_stats["metadata"],
            "final_test_fit": full_stats["metadata"],
        },
        "date_time_features": base_meta["date_time_features"],
        "price_features": base_meta["price_features"],
        "deal_features": base_meta["deal_features"],
        "within_search_features": base_meta["within_search_features"],
        "match_features": base_meta["match_features"],
        "competitor_features": base_meta["competitor_features"],
        "historical_features": list(train_hist.columns),
        "historical_encoding": {
            "validation_fit": "training split only; applied to validation",
            "test_fit": "full training data; applied to test",
            "prop_alpha": SMOOTHING_ALPHA,
            "sparse_key_alpha": SPARSE_KEY_SMOOTHING_ALPHA,
            "fallback": "global training mean for unseen keys",
        },
        "baseline_ndcg_at_5": baseline_scores,
        "output_paths": output_paths | {"baseline_ndcg": str(baseline_path)},
    }
    save_json(baseline_scores, baseline_path)
    save_json(metadata, output_dir / "feature_metadata.json")
    write_report_notes(output_dir / "task3_report_notes.md", metadata)
    LOGGER.info("Done. Outputs written to %s", output_dir)
    return metadata


def make_label_frame(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "srch_id": df["srch_id"].to_numpy(),
            "prop_id": df["prop_id"].to_numpy(),
            "click_bool": df["click_bool"].to_numpy(),
            "booking_bool": df["booking_bool"].to_numpy(),
            "relevance": make_relevance(df).to_numpy(),
        }
    )


def attach_history_price(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    out = raw.copy()
    if "price_usd_capped" in features.columns:
        out["price_usd_capped"] = features["price_usd_capped"].to_numpy()
    return out


def sort_features(features: pd.DataFrame) -> pd.DataFrame:
    if "srch_id" not in features.columns:
        return features.reset_index(drop=True)
    return features.sort_values(["srch_id"], kind="mergesort").reset_index(drop=True)


def sort_features_and_labels(
    features: pd.DataFrame, labels: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "srch_id" not in features.columns:
        return features.reset_index(drop=True), labels.reset_index(drop=True)
    order = features[["srch_id"]].sort_values(["srch_id"], kind="mergesort").index
    return (
        features.loc[order].reset_index(drop=True),
        labels.loc[order].reset_index(drop=True),
    )


def build_base_features(
    df: pd.DataFrame, preprocessing_stats: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    input_columns = list(out.columns)
    date_features = add_date_features(out)
    missing_indicators = add_missing_indicators(out)
    price_features = add_price_features(out, preprocessing_stats)
    deal_features = add_deal_features(out)
    imputed_columns = apply_special_imputation(out, preprocessing_stats)
    within_features = add_within_search_features(out)
    out = out.copy()
    match_features = add_match_features(out)
    out = out.copy()
    competitor_features = add_competitor_features(out)

    raw_competitor_columns = get_competitor_columns(out)
    dropped = [
        c
        for c in [
            *TARGET_COLUMNS,
            *LEAKAGE_COLUMNS,
            "date_time",
            *raw_competitor_columns,
        ]
        if c in out.columns
    ]
    out = out.drop(columns=dropped)

    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = downcast_numeric(out)
    metadata = {
        "input_columns": input_columns,
        "dropped_columns": dropped,
        "missing_indicators": missing_indicators,
        "date_time_features": date_features,
        "price_features": price_features,
        "deal_features": deal_features,
        "special_imputed_columns": imputed_columns,
        "within_search_features": within_features,
        "match_features": match_features,
        "competitor_features": competitor_features,
    }
    return out, metadata


def fit_preprocessing_stats(df: pd.DataFrame) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "price_cap_quantile": PRICE_CAP_QUANTILE,
        "price_cap_group": "site_id",
        "price_cap_fallback": "global 99th percentile fitted on training data",
        "special_imputation": {},
    }

    price = pd.to_numeric(df.get("price_usd"), errors="coerce")
    if "site_id" in df.columns and price is not None:
        site_caps = (
            df.assign(_price_for_cap=price)
            .groupby("site_id")["_price_for_cap"]
            .quantile(PRICE_CAP_QUANTILE)
        )
        stats["site_price_caps"] = {
            int(k): float(v) for k, v in site_caps.dropna().items()
        }
        stats["global_price_cap"] = float(price.quantile(PRICE_CAP_QUANTILE))
    else:
        stats["site_price_caps"] = {}
        stats["global_price_cap"] = (
            float(price.quantile(PRICE_CAP_QUANTILE)) if price is not None else 0.0
        )

    stats["visitor_country_medians"] = {}
    for col in ["visitor_hist_starrating", "visitor_hist_adr_usd"]:
        if col in df.columns and "visitor_location_country_id" in df.columns:
            med = df.groupby("visitor_location_country_id")[col].median().dropna()
            stats["visitor_country_medians"][col] = {
                int(k): float(v) for k, v in med.items()
            }
            stats[f"{col}_global_median"] = float(
                pd.to_numeric(df[col], errors="coerce").median()
            )
            metadata["special_imputation"][col] = (
                "visitor_location_country_id median, then global median"
            )

    if "prop_location_score2" in df.columns and "prop_country_id" in df.columns:
        med = df.groupby("prop_country_id")["prop_location_score2"].median().dropna()
        stats["prop_country_location2_median"] = {
            int(k): float(v) for k, v in med.items()
        }
        stats["prop_location_score2_global_median"] = float(
            pd.to_numeric(df["prop_location_score2"], errors="coerce").median()
        )
        metadata["special_imputation"]["prop_location_score2"] = (
            "prop_country_id median, then global median"
        )

    if "srch_query_affinity_score" in df.columns:
        col = pd.to_numeric(df["srch_query_affinity_score"], errors="coerce")
        observed_min = float(col.min(skipna=True)) if col.notna().any() else -1.0
        stats["srch_query_affinity_score_sentinel"] = observed_min - 1.0
        metadata["special_imputation"]["srch_query_affinity_score"] = (
            "sentinel below observed training minimum"
        )

    if "orig_destination_distance" in df.columns:
        stats["orig_destination_distance_median"] = float(
            pd.to_numeric(df["orig_destination_distance"], errors="coerce").median()
        )
        metadata["special_imputation"]["orig_destination_distance"] = "training median"

    stats["metadata"] = metadata
    return stats


def add_date_features(df: pd.DataFrame) -> list[str]:
    if "date_time" not in df.columns:
        return []
    dt = pd.to_datetime(df["date_time"], errors="coerce")
    df["date_month"] = dt.dt.month.astype("float32")
    df["date_dayofweek"] = dt.dt.dayofweek.astype("float32")
    df["date_hour"] = dt.dt.hour.astype("float32")
    df["date_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype("int8")
    return ["date_month", "date_dayofweek", "date_hour", "date_is_weekend"]


def add_missing_indicators(df: pd.DataFrame) -> list[str]:
    indicators: list[str] = []
    candidates = list(IMPORTANT_MISSING_COLUMNS)
    for i in COMPETITOR_IDS:
        candidates.extend(
            [f"comp{i}_rate", f"comp{i}_inv", f"comp{i}_rate_percent_diff"]
        )
    for col in candidates:
        if col in df.columns and df[col].isna().any():
            new_col = f"is_missing_{col}"
            df[new_col] = df[col].isna().astype("int8")
            indicators.append(new_col)
    return indicators


def add_price_features(df: pd.DataFrame, stats: dict[str, Any]) -> list[str]:
    if "price_usd" not in df.columns:
        return []
    price = pd.to_numeric(df["price_usd"], errors="coerce")
    global_cap = stats.get(
        "global_price_cap", float(price.quantile(PRICE_CAP_QUANTILE))
    )
    if "site_id" in df.columns:
        caps = df["site_id"].map(stats.get("site_price_caps", {})).fillna(global_cap)
    else:
        caps = pd.Series(global_cap, index=df.index)
    df["price_usd_capped"] = price.clip(upper=caps)
    stay = (
        pd.to_numeric(df.get("srch_length_of_stay", 1), errors="coerce")
        .fillna(1)
        .clip(lower=1)
    )
    df["price_per_night"] = df["price_usd_capped"] / stay
    df["log_price_usd_capped"] = np.log1p(df["price_usd_capped"].clip(lower=0))
    return ["price_usd_capped", "price_per_night", "log_price_usd_capped"]


def add_deal_features(df: pd.DataFrame) -> list[str]:
    if (
        "prop_log_historical_price" not in df.columns
        or "price_usd_capped" not in df.columns
    ):
        return []
    hist_log = pd.to_numeric(df["prop_log_historical_price"], errors="coerce")
    valid_hist = hist_log.notna() & np.isfinite(hist_log) & (hist_log > 0)
    df["prop_historical_price_missing_or_zero"] = (~valid_hist).astype("int8")
    clipped_log = hist_log.where(valid_hist).clip(upper=np.log(MAX_HISTORICAL_PRICE))
    df["historical_price"] = np.exp(clipped_log).clip(upper=MAX_HISTORICAL_PRICE)
    df.loc[~valid_hist, "historical_price"] = np.nan
    df["deal_value"] = (df["historical_price"] - df["price_usd_capped"]).clip(
        lower=-MAX_HISTORICAL_PRICE, upper=MAX_HISTORICAL_PRICE
    )
    df["deal_ratio"] = (df["deal_value"] / (df["price_usd_capped"] + EPSILON)).clip(
        lower=-DEAL_RATIO_CLIP, upper=DEAL_RATIO_CLIP
    )
    df["log_historical_price_diff"] = hist_log.where(valid_hist) - np.log1p(
        df["price_usd_capped"].clip(lower=0)
    )
    return [
        "prop_historical_price_missing_or_zero",
        "historical_price",
        "deal_value",
        "deal_ratio",
        "log_historical_price_diff",
    ]


def apply_special_imputation(df: pd.DataFrame, stats: dict[str, Any]) -> list[str]:
    imputed: list[str] = []
    visitor_medians = stats.get("visitor_country_medians", {})
    for col in ["visitor_hist_starrating", "visitor_hist_adr_usd"]:
        if col in df.columns:
            fill = (
                df["visitor_location_country_id"].map(visitor_medians.get(col, {}))
                if "visitor_location_country_id" in df.columns
                else np.nan
            )
            df[col] = (
                df[col].fillna(fill).fillna(stats.get(f"{col}_global_median", 0.0))
            )
            imputed.append(col)
    if "prop_location_score2" in df.columns:
        fill = (
            df["prop_country_id"].map(stats.get("prop_country_location2_median", {}))
            if "prop_country_id" in df.columns
            else np.nan
        )
        df["prop_location_score2"] = (
            df["prop_location_score2"]
            .fillna(fill)
            .fillna(stats.get("prop_location_score2_global_median", 0.0))
        )
        imputed.append("prop_location_score2")
    if "srch_query_affinity_score" in df.columns:
        df["srch_query_affinity_score"] = df["srch_query_affinity_score"].fillna(
            stats.get("srch_query_affinity_score_sentinel", -1.0)
        )
        imputed.append("srch_query_affinity_score")
    if "orig_destination_distance" in df.columns:
        df["orig_destination_distance"] = df["orig_destination_distance"].fillna(
            stats.get("orig_destination_distance_median", 0.0)
        )
        imputed.append("orig_destination_distance")
    return imputed


def add_within_search_features(df: pd.DataFrame) -> list[str]:
    features: list[str] = []
    if "srch_id" not in df.columns:
        return features
    for col, ascending in [
        ("price_usd_capped", True),
        ("price_per_night", True),
        ("prop_starrating", False),
        ("prop_review_score", False),
        ("prop_location_score1", False),
        ("prop_location_score2", False),
        ("prop_log_historical_price", False),
        ("deal_value", False),
    ]:
        if col in df.columns:
            features.extend(add_group_relative_features(df, col, ascending=ascending))

    cheapest_col = (
        "price_per_night" if "price_per_night" in df.columns else "price_usd_capped"
    )
    if cheapest_col in df.columns:
        group = df.groupby("srch_id")[cheapest_col]
        df["is_cheapest_in_search"] = (
            df[cheapest_col] == group.transform("min")
        ).astype("int8")
        features.append("is_cheapest_in_search")
    for col, new_col in [
        ("prop_review_score", "is_highest_review_in_search"),
        ("prop_location_score2", "is_highest_location2_in_search"),
        ("prop_starrating", "is_highest_starrating_in_search"),
    ]:
        if col in df.columns:
            group = df.groupby("srch_id")[col]
            df[new_col] = (df[col] == group.transform("max")).astype("int8")
            features.append(new_col)
    return features


def add_group_relative_features(
    df: pd.DataFrame, col: str, ascending: bool
) -> list[str]:
    group = df.groupby("srch_id")[col]
    mean = group.transform("mean")
    median = group.transform("median")
    std = group.transform("std").replace(0, np.nan)
    prefix = col
    out_cols = [
        f"{prefix}_rank_in_search",
        f"{prefix}_pct_rank_in_search",
        f"{prefix}_diff_from_search_mean",
        f"{prefix}_diff_from_search_median",
        f"{prefix}_zscore_in_search",
    ]
    df[out_cols[0]] = group.rank(method="min", ascending=ascending)
    df[out_cols[1]] = group.rank(method="average", pct=True, ascending=ascending)
    df[out_cols[2]] = df[col] - mean
    df[out_cols[3]] = df[col] - median
    df[out_cols[4]] = (df[col] - mean) / std
    return out_cols


def add_match_features(df: pd.DataFrame) -> list[str]:
    features: list[str] = []
    if {"visitor_hist_starrating", "prop_starrating"}.issubset(df.columns):
        df["star_user_pref_abs_diff"] = (
            df["visitor_hist_starrating"] - df["prop_starrating"]
        ).abs()
        features.append("star_user_pref_abs_diff")
    price_col = "price_usd_capped" if "price_usd_capped" in df.columns else "price_usd"
    if {"visitor_hist_adr_usd", price_col}.issubset(df.columns):
        df["price_user_pref_abs_diff"] = (
            df["visitor_hist_adr_usd"] - df[price_col]
        ).abs()
        features.append("price_user_pref_abs_diff")
    if {"srch_adults_count", "srch_children_count"}.issubset(df.columns):
        df["guest_count"] = df["srch_adults_count"] + df["srch_children_count"]
        df["has_children"] = (df["srch_children_count"] > 0).astype("int8")
        df["is_family_search"] = (
            (df["srch_children_count"] > 0) & (df["srch_adults_count"] >= 1)
        ).astype("int8")
        features.extend(["guest_count", "has_children", "is_family_search"])
        if "srch_room_count" in df.columns:
            df["guests_per_room"] = safe_ratio(df["guest_count"], df["srch_room_count"])
            features.append("guests_per_room")
    if "srch_booking_window" in df.columns:
        df["booking_window_bucket"] = pd.cut(
            df["srch_booking_window"], bins=[-1, 0, 7, 30, 90, np.inf], labels=False
        ).astype("float32")
        features.append("booking_window_bucket")
    if "srch_length_of_stay" in df.columns:
        df["length_of_stay_bucket"] = pd.cut(
            df["srch_length_of_stay"],
            bins=[0, 1, 2, 4, 7, np.inf],
            labels=False,
            include_lowest=True,
        ).astype("float32")
        features.append("length_of_stay_bucket")
    if {"visitor_location_country_id", "prop_country_id"}.issubset(df.columns):
        df["domestic_trip_indicator"] = (
            df["visitor_location_country_id"] == df["prop_country_id"]
        ).astype("int8")
        features.append("domestic_trip_indicator")
    if {"price_usd_capped", "srch_length_of_stay"}.issubset(df.columns):
        df["total_price_proxy"] = df["price_usd_capped"] * df[
            "srch_length_of_stay"
        ].clip(lower=1)
        features.append("total_price_proxy")
    rank_col = (
        "price_per_night_rank_in_search"
        if "price_per_night_rank_in_search" in df.columns
        else "price_usd_capped_rank_in_search"
    )
    if {rank_col, "promotion_flag"}.issubset(df.columns):
        df["promotion_x_price_rank"] = df["promotion_flag"] * df[rank_col]
        features.append("promotion_x_price_rank")
    if {"prop_starrating", "prop_review_score"}.issubset(df.columns):
        df["star_x_review"] = df["prop_starrating"] * df["prop_review_score"]
        features.append("star_x_review")
    return features


def add_competitor_features(df: pd.DataFrame) -> list[str]:
    rate_cols = [
        f"comp{i}_rate" for i in COMPETITOR_IDS if f"comp{i}_rate" in df.columns
    ]
    inv_cols = [f"comp{i}_inv" for i in COMPETITOR_IDS if f"comp{i}_inv" in df.columns]
    pct_cols = [
        f"comp{i}_rate_percent_diff"
        for i in COMPETITOR_IDS
        if f"comp{i}_rate_percent_diff" in df.columns
    ]
    features: list[str] = []
    if rate_cols:
        rates = df[rate_cols]
        df["comp_rate_better_count"] = (rates == 1).sum(axis=1)
        df["comp_rate_equal_count"] = (rates == 0).sum(axis=1)
        df["comp_rate_worse_count"] = (rates == -1).sum(axis=1)
        df["comp_rate_available_count"] = rates.notna().sum(axis=1)
        features.extend(
            [
                "comp_rate_better_count",
                "comp_rate_equal_count",
                "comp_rate_worse_count",
                "comp_rate_available_count",
            ]
        )
    if inv_cols:
        inv = df[inv_cols]
        df["comp_inv_unavailable_count"] = (inv == 1).sum(axis=1)
        df["comp_inv_available_count"] = inv.notna().sum(axis=1)
        features.extend(["comp_inv_unavailable_count", "comp_inv_available_count"])
    if pct_cols:
        pct = df[pct_cols]
        df["comp_percent_diff_mean"] = pct.mean(axis=1)
        df["comp_percent_diff_max"] = pct.max(axis=1)
        df["comp_percent_diff_min"] = pct.min(axis=1)
        features.extend(
            ["comp_percent_diff_mean", "comp_percent_diff_max", "comp_percent_diff_min"]
        )
    all_comp_cols = rate_cols + inv_cols + pct_cols
    if all_comp_cols:
        df["comp_missing_count"] = df[all_comp_cols].isna().sum(axis=1)
        df["comp_any_data_count"] = df[all_comp_cols].notna().sum(axis=1)
        features.extend(["comp_missing_count", "comp_any_data_count"])
    return features


def get_competitor_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for i in COMPETITOR_IDS:
        cols.extend(
            [
                c
                for c in [f"comp{i}_rate", f"comp{i}_inv", f"comp{i}_rate_percent_diff"]
                if c in df.columns
            ]
        )
    return cols


def align_and_impute(
    train_features: pd.DataFrame,
    valid_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    common_cols = [
        c
        for c in train_features.columns
        if c in valid_features.columns and c in test_features.columns
    ]
    train = train_features[common_cols].copy()
    valid = valid_features[common_cols].copy()
    test = test_features[common_cols].copy()
    numeric_cols = list(train.select_dtypes(include=[np.number]).columns)
    medians = (
        train[numeric_cols]
        .median(numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    for frame in [train, valid, test]:
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        frame[numeric_cols] = frame[numeric_cols].fillna(medians)
    meta = {
        "strategy": "column-specific fitted preprocessing first; remaining numeric medians fitted on training split",
        "n_common_features": len(common_cols),
        "columns_dropped_by_alignment": sorted(
            (
                set(train_features.columns)
                | set(valid_features.columns)
                | set(test_features.columns)
            )
            - set(common_cols)
        ),
        "numeric_medians": {k: float(v) for k, v in medians.items()},
    }
    return train, valid, test, meta


def align_and_impute_two(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    common_cols = [c for c in train_features.columns if c in test_features.columns]
    train = train_features[common_cols].copy()
    test = test_features[common_cols].copy()
    numeric_cols = list(train.select_dtypes(include=[np.number]).columns)
    medians = (
        train[numeric_cols]
        .median(numeric_only=True)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )
    for frame in [train, test]:
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        frame[numeric_cols] = frame[numeric_cols].fillna(medians)
    meta = {
        "strategy": "column-specific fitted preprocessing first; remaining numeric medians fitted on full labeled training data",
        "n_common_features": len(common_cols),
        "columns_dropped_by_alignment": sorted(
            (set(train_features.columns) | set(test_features.columns))
            - set(common_cols)
        ),
        "numeric_medians": {k: float(v) for k, v in medians.items()},
    }
    return train, test, meta


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def write_report_notes(path: Path, metadata: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    baseline_lines = "\n".join(
        f"- Validation baseline `{name}` NDCG@5: `{value:.4f}`."
        for name, value in metadata.get("baseline_ndcg_at_5", {}).items()
    )
    text = f"""# Task 3 Data Preparation Notes

- The pipeline creates a group-based validation split by `srch_id` with seed `{metadata["validation"]["seed"]}` and validation share `{metadata["validation"]["valid_size"]}`. This prevents hotels from the same search appearing in both train and validation.
- Leakage columns excluded from model features: `{", ".join(metadata["leakage_columns_excluded"])}`. Target columns excluded: `{", ".join(metadata["target_columns_excluded"])}`.
- `position` is excluded even though it is predictive because it is not available in the test set. It should only be used for EDA, bias analysis, or diagnostic comparisons.
- Relevance labels use `5` for bookings, `1` for clicks without bookings, and `0` otherwise.
- `price_usd` is capped at the `{PRICE_CAP_QUANTILE:.2f}` quantile within `site_id`, using caps fitted only on the relevant training data and a global fallback for unseen sites.
- `price_per_night` and `log_price_usd_capped` are created from the capped price to reduce the influence of extreme price outliers.
- Historical deal features use `prop_log_historical_price`; zero or invalid values are flagged with `prop_historical_price_missing_or_zero`, and unstable deal ratios are clipped.
- Missing values are documented in `feature_metadata.json`; selected high-value sparse columns get binary missing indicators.
- Visitor history is imputed with `visitor_location_country_id` medians where possible, `prop_location_score2` with `prop_country_id` medians, `srch_query_affinity_score` with a low sentinel, and `orig_destination_distance` with a fitted median.
- Within-search features compare hotels only inside the same `srch_id`, including ranks, percentile ranks, mean/median differences, z-scores, and best/cheapest indicators.
- Competitor fields are collapsed into aggregate counts and percent-difference summaries; raw competitor blocks are dropped from final features to reduce sparsity.
- Historical encodings are smoothed with property alpha `{SMOOTHING_ALPHA}` and sparse-key alpha `{SPARSE_KEY_SMOOTHING_ALPHA}`. Validation encodings are fitted only on the training split; test encodings are fitted on the full labeled training data.
- `X_train`/`X_valid` and group files support offline LightGBM LambdaRank validation. `full_train_features`/`X_test` support final modelling and Kaggle submission generation.
- Group-size files are saved after sorting rows by `srch_id`, matching the feature and label row order.
{baseline_lines}
"""
    path.write_text(text, encoding="utf-8")


def make_synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    rows = []
    for srch_id in range(100, 112):
        n = int(rng.integers(3, 7))
        booked = int(rng.integers(0, n))
        clicked = set(rng.choice(n, size=min(2, n), replace=False).tolist())
        clicked.add(booked)
        for pos in range(n):
            prop_id = int(1000 + rng.integers(0, 18))
            rows.append(
                {
                    "srch_id": srch_id,
                    "date_time": f"2013-0{int(rng.integers(1, 7))}-{int(rng.integers(10, 28))} {int(rng.integers(0, 24)):02d}:00:00",
                    "site_id": int(rng.integers(1, 4)),
                    "visitor_location_country_id": int(rng.choice([55, 219, 100])),
                    "prop_country_id": int(rng.choice([55, 219, 100])),
                    "prop_id": prop_id,
                    "prop_starrating": int(rng.integers(1, 6)),
                    "prop_review_score": float(rng.choice([3.5, 4.0, 4.5, np.nan])),
                    "prop_brand_bool": int(rng.integers(0, 2)),
                    "prop_location_score1": float(rng.normal(2.5, 0.7)),
                    "prop_location_score2": float(
                        rng.choice([rng.normal(0.1, 0.05), np.nan])
                    ),
                    "prop_log_historical_price": float(
                        rng.choice([0.0, rng.normal(4.8, 0.4)])
                    ),
                    "price_usd": float(
                        rng.choice([rng.lognormal(4.8, 0.35), 250_000.0])
                    ),
                    "promotion_flag": int(rng.integers(0, 2)),
                    "srch_destination_id": int(rng.integers(1, 6)),
                    "srch_length_of_stay": int(rng.integers(1, 8)),
                    "srch_booking_window": int(rng.integers(0, 120)),
                    "srch_adults_count": int(rng.integers(1, 4)),
                    "srch_children_count": int(rng.integers(0, 3)),
                    "srch_room_count": int(rng.integers(1, 3)),
                    "srch_saturday_night_bool": int(rng.integers(0, 2)),
                    "visitor_hist_starrating": float(rng.choice([3.0, 4.0, np.nan])),
                    "visitor_hist_adr_usd": float(rng.choice([120.0, 180.0, np.nan])),
                    "orig_destination_distance": float(
                        rng.choice([rng.lognormal(5.0, 0.5), np.nan])
                    ),
                    "srch_query_affinity_score": float(
                        rng.choice([-20.0, -5.0, np.nan])
                    ),
                    "random_bool": int(rng.integers(0, 2)),
                    "position": pos + 1,
                    "gross_booking_usd": float(rng.lognormal(6.0, 0.4))
                    if pos == booked
                    else np.nan,
                    "click_bool": int(pos in clicked),
                    "booking_bool": int(pos == booked),
                    "comp1_rate": float(rng.choice([-1, 0, 1, np.nan])),
                    "comp1_inv": float(rng.choice([0, 1, np.nan])),
                    "comp1_rate_percent_diff": float(rng.choice([-20, 5, 30, np.nan])),
                    "comp2_rate": float(rng.choice([-1, 0, 1, np.nan])),
                    "comp2_inv": float(rng.choice([0, 1, np.nan])),
                    "comp2_rate_percent_diff": float(rng.choice([-10, 10, 25, np.nan])),
                }
            )
    train = pd.DataFrame(rows)
    test = (
        train[train["srch_id"] >= 108]
        .drop(columns=["click_bool", "booking_bool", "position", "gross_booking_usd"])
        .copy()
    )
    train = train[train["srch_id"] < 108].reset_index(drop=True)
    test["srch_id"] = test["srch_id"] + 1000
    return train.reset_index(drop=True), test.reset_index(drop=True)


if __name__ == "__main__":
    main()
