"""Leakage-safe historical encodings for Task 3."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class SmoothedHistoryEncoder:
    """Fit smoothed historical features on labeled data and apply to new rows."""

    alpha: float = 10.0
    sparse_alpha: float = 50.0
    global_booking_mean: float = 0.0
    global_click_mean: float = 0.0
    global_relevance_mean: float = 0.0
    global_price_mean: float = 0.0
    mappings: dict[str, pd.DataFrame] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame) -> "SmoothedHistoryEncoder":
        work = df.copy()
        work["relevance"] = np.where(
            work["booking_bool"].to_numpy() == 1,
            5,
            np.where(work["click_bool"].to_numpy() == 1, 1, 0),
        )
        self.global_booking_mean = float(work["booking_bool"].mean())
        self.global_click_mean = float(work["click_bool"].mean())
        self.global_relevance_mean = float(work["relevance"].mean())
        if "price_usd_capped" in work.columns:
            self.global_price_mean = float(work["price_usd_capped"].mean())
        elif "price_usd" in work.columns:
            self.global_price_mean = float(work["price_usd"].mean())

        self.mappings = {}
        self._fit_single(work, ["prop_id"], "prop", alpha=self.alpha)
        self._fit_single(work, ["srch_destination_id"], "destination", alpha=self.alpha)
        if {"prop_id", "srch_destination_id"}.issubset(work.columns):
            self._fit_single(
                work,
                ["prop_id", "srch_destination_id"],
                "prop_destination",
                alpha=self.sparse_alpha,
            )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        out["prop_impressions"] = 0.0
        out["prop_click_rate_smooth"] = self.global_click_mean
        out["prop_booking_rate_smooth"] = self.global_booking_mean
        out["prop_relevance_mean_smooth"] = self.global_relevance_mean
        out["prop_mean_price_usd_capped"] = self.global_price_mean
        out["destination_impressions"] = 0.0
        out["destination_booking_rate_smooth"] = self.global_booking_mean
        out["destination_relevance_mean_smooth"] = self.global_relevance_mean
        out["prop_destination_impressions"] = 0.0
        out["prop_destination_booking_rate_smooth"] = self.global_booking_mean
        out["prop_destination_relevance_mean_smooth"] = self.global_relevance_mean

        if "prop" in self.mappings and "prop_id" in df.columns:
            merged = df[["prop_id"]].merge(
                self.mappings["prop"], on="prop_id", how="left"
            )
            out["prop_impressions"] = merged["prop_count"].fillna(0).to_numpy()
            out["prop_click_rate_smooth"] = (
                merged["prop_click_rate_smooth"]
                .fillna(self.global_click_mean)
                .to_numpy()
            )
            out["prop_booking_rate_smooth"] = (
                merged["prop_booking_rate_smooth"]
                .fillna(self.global_booking_mean)
                .to_numpy()
            )
            out["prop_relevance_mean_smooth"] = (
                merged["prop_relevance_mean_smooth"]
                .fillna(self.global_relevance_mean)
                .to_numpy()
            )
            if "prop_mean_price_usd_capped" in merged.columns:
                out["prop_mean_price_usd_capped"] = (
                    merged["prop_mean_price_usd_capped"]
                    .fillna(self.global_price_mean)
                    .to_numpy()
                )

        if "destination" in self.mappings and "srch_destination_id" in df.columns:
            merged = df[["srch_destination_id"]].merge(
                self.mappings["destination"], on="srch_destination_id", how="left"
            )
            out["destination_impressions"] = (
                merged["destination_count"].fillna(0).to_numpy()
            )
            out["destination_booking_rate_smooth"] = (
                merged["destination_booking_rate_smooth"]
                .fillna(self.global_booking_mean)
                .to_numpy()
            )
            out["destination_relevance_mean_smooth"] = (
                merged["destination_relevance_mean_smooth"]
                .fillna(self.global_relevance_mean)
                .to_numpy()
            )

        keys = ["prop_id", "srch_destination_id"]
        if "prop_destination" in self.mappings and set(keys).issubset(df.columns):
            merged = df[keys].merge(
                self.mappings["prop_destination"], on=keys, how="left"
            )
            out["prop_destination_impressions"] = (
                merged["prop_destination_count"].fillna(0).to_numpy()
            )
            out["prop_destination_booking_rate_smooth"] = (
                merged["prop_destination_booking_rate_smooth"]
                .fillna(self.global_booking_mean)
                .to_numpy()
            )
            out["prop_destination_relevance_mean_smooth"] = (
                merged["prop_destination_relevance_mean_smooth"]
                .fillna(self.global_relevance_mean)
                .to_numpy()
            )

        return out

    def _fit_single(
        self, df: pd.DataFrame, keys: list[str], prefix: str, alpha: float
    ) -> None:
        if not set(keys).issubset(df.columns):
            return
        agg_spec = {
            "count": ("booking_bool", "size"),
            "clicks": ("click_bool", "sum"),
            "bookings": ("booking_bool", "sum"),
            "relevance_sum": ("relevance", "sum"),
        }
        if prefix == "prop":
            price_col = (
                "price_usd_capped" if "price_usd_capped" in df.columns else "price_usd"
            )
            if price_col in df.columns:
                agg_spec["mean_price_usd_capped"] = (price_col, "mean")
        grouped = df.groupby(keys, dropna=False).agg(**agg_spec).reset_index()
        grouped[f"{prefix}_count"] = grouped["count"].astype("float32")
        grouped[f"{prefix}_click_rate_smooth"] = (
            grouped["clicks"] + alpha * self.global_click_mean
        ) / (grouped["count"] + alpha)
        grouped[f"{prefix}_booking_rate_smooth"] = (
            grouped["bookings"] + alpha * self.global_booking_mean
        ) / (grouped["count"] + alpha)
        grouped[f"{prefix}_relevance_mean_smooth"] = (
            grouped["relevance_sum"] + alpha * self.global_relevance_mean
        ) / (grouped["count"] + alpha)
        if prefix == "prop" and "mean_price_usd_capped" in grouped.columns:
            grouped["prop_mean_price_usd_capped"] = grouped[
                "mean_price_usd_capped"
            ].astype("float32")

        generated = [
            f"{prefix}_count",
            f"{prefix}_click_rate_smooth",
            f"{prefix}_booking_rate_smooth",
            f"{prefix}_relevance_mean_smooth",
            f"{prefix}_mean_price_usd_capped",
        ]
        keep = keys + [c for c in generated if c in grouped.columns]
        self.mappings[prefix] = grouped[keep]
