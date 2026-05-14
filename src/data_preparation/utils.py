"""Utility helpers for the Task 3 data-preparation pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns in place where pandas can do so safely."""
    for col in df.select_dtypes(include=["integer", "int"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["floating", "float"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def missing_percentages(df: pd.DataFrame) -> dict[str, float]:
    missing = df.isna().mean().sort_values(ascending=False)
    return {str(k): float(v) for k, v in missing.items() if v > 0}


def save_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """Save a dataframe, preferring parquet and falling back to CSV."""
    ensure_dir(path.parent)
    try:
        df.to_parquet(path, index=False)
        return path
    except Exception as exc:
        fallback = path.with_suffix(".csv")
        LOGGER.warning(
            "Could not write parquet %s (%s); writing %s", path, exc, fallback
        )
        df.to_csv(fallback, index=False)
        return fallback


def load_csv(
    path: Path, nrows: int | None = None, dtype_map: dict[str, str] | None = None
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing expected data file: {path}. Run `dvc pull` or pass --synthetic-smoke."
        )
    if dtype_map is None:
        return pd.read_csv(path, nrows=nrows)
    header = pd.read_csv(path, nrows=0)
    usable_dtypes = {k: v for k, v in dtype_map.items() if k in header.columns}
    return pd.read_csv(path, nrows=nrows, dtype=usable_dtypes)


def make_relevance(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            df["booking_bool"].to_numpy() == 1,
            5,
            np.where(df["click_bool"].to_numpy() == 1, 1, 0),
        ),
        index=df.index,
        name="relevance",
    ).astype("int8")


def save_group_sizes(df: pd.DataFrame, path: Path) -> Path:
    groups = df.groupby("srch_id", sort=True).size().rename("group_size").reset_index()
    return save_table(groups, path)
