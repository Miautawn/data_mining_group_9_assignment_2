"""Validation split utilities for ranking data."""

from __future__ import annotations

import numpy as np
import pandas as pd


def split_by_srch_id(
    df: pd.DataFrame,
    valid_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row indices for a deterministic group split by srch_id."""
    unique_ids = df["srch_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(seed)
    shuffled = unique_ids.copy()
    rng.shuffle(shuffled)
    n_valid = max(1, int(round(len(shuffled) * valid_size))) if len(shuffled) > 1 else 0
    valid_ids = set(shuffled[:n_valid])
    is_valid = df["srch_id"].isin(valid_ids).to_numpy()
    return df.index[~is_valid].to_numpy(), df.index[is_valid].to_numpy()
