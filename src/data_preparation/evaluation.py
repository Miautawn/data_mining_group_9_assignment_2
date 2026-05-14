"""Offline ranking metrics for Expedia validation experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd


def dcg_at_k(relevance: np.ndarray, k: int = 5) -> float:
    rel = np.asarray(relevance)[:k]
    if len(rel) == 0:
        return 0.0
    gains = np.power(2.0, rel) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, len(rel) + 2))
    return float(np.sum(gains * discounts))


def ndcg_at_k(relevance: np.ndarray, k: int = 5) -> float:
    actual = dcg_at_k(relevance, k=k)
    ideal = dcg_at_k(np.sort(relevance)[::-1], k=k)
    return actual / ideal if ideal > 0 else 0.0


def mean_ndcg_at_k(
    frame: pd.DataFrame,
    prediction_col: str,
    relevance_col: str = "relevance",
    group_col: str = "srch_id",
    k: int = 5,
) -> float:
    """Compute mean NDCG@k after sorting predictions descending per search."""
    scores: list[float] = []
    for _, group in frame.groupby(group_col, sort=False):
        ordered = group.sort_values(prediction_col, ascending=False)
        scores.append(ndcg_at_k(ordered[relevance_col].to_numpy(), k=k))
    return float(np.mean(scores)) if scores else 0.0


def baseline_ndcg_scores(
    labels: pd.DataFrame,
    features: pd.DataFrame,
    k: int = 5,
    seed: int = 42,
) -> dict[str, float]:
    """Compute simple grouped NDCG@k baselines for validation diagnostics."""
    frame = labels[["srch_id", "prop_id", "relevance"]].reset_index(drop=True).copy()
    aligned = features.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    frame["random_score"] = rng.random(len(frame))
    scores = {"random": mean_ndcg_at_k(frame, "random_score", k=k)}

    price_col = (
        "price_usd_capped" if "price_usd_capped" in aligned.columns else "price_usd"
    )
    if price_col in aligned.columns:
        frame["price_ascending_score"] = -aligned[price_col].to_numpy()
        scores["price_ascending"] = mean_ndcg_at_k(frame, "price_ascending_score", k=k)

    if "prop_booking_rate_smooth" in aligned.columns:
        frame["property_booking_rate_score"] = aligned[
            "prop_booking_rate_smooth"
        ].to_numpy()
        scores["property_booking_rate"] = mean_ndcg_at_k(
            frame, "property_booking_rate_score", k=k
        )

    if "position" in aligned.columns:
        frame["original_position_score"] = -aligned["position"].to_numpy()
        scores["original_position_diagnostic_only"] = mean_ndcg_at_k(
            frame, "original_position_score", k=k
        )

    return {name: float(value) for name, value in scores.items()}
