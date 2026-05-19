def run_fairness_checks(x_df, y_df, predictions):
    """Executes demographic and exposure fairness reporting."""
    print("\n📊 RUNNING RECONNAISSANCE FAIRNESS CHECKS...")

    df_eval = x_df[
        ["srch_id", "prop_brand_bool", "domestic_trip_indicator", "promotion_flag"]
    ].copy()
    df_eval["booking_bool"] = y_df["booking_bool"]
    df_eval["relevance"] = y_df["relevance"]
    df_eval["score"] = predictions
    df_eval["temp_score"] = predictions
    df_eval["predicted_rank"] = df_eval.groupby("srch_id")["score"].rank(
        ascending=False, method="first"
    )
    df_eval["in_top_5"] = (df_eval["predicted_rank"] <= 5).astype(int)

    # Example Check: Promo Disparity
    base_no_promo = (df_eval["promotion_flag"] == 0).mean()
    top5_no_promo = (df_eval[df_eval["in_top_5"] == 1]["promotion_flag"] == 0).mean()

    print("-" * 65)
    print(f"📉 Non-promoted hotels overall pool share:  {base_no_promo * 100:.2f}%")
    print(f"🚀 Non-promoted hotels Top 5 exposure:      {top5_no_promo * 100:.2f}%")
    print(
        f"👉 Disparity Ratio:                         {top5_no_promo / (base_no_promo + 1e-9):.4f}"
    )
    print("-" * 65)


def apply_calibration(df, base_col="score", boost_factor=0.16):
    """Boosts utility scores for non-promoted properties to fix exposure disparities."""
    df_calibrated = df.copy()
    df_calibrated["calibrated_score"] = df_calibrated[base_col]

    mask = df_calibrated["promotion_flag"] == 0
    df_calibrated.loc[mask, "calibrated_score"] += boost_factor

    return df_calibrated
