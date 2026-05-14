# Task 3 Feature Outline

This file documents the final model features produced by `src/data_preparation/build_features.py`. It is generated from `outputs/features/feature_metadata.json` plus the feature-engineering rules in the pipeline.

- Feature count: `145`
- Labels such as `click_bool`, `booking_bool`, and `relevance` are stored separately in `y_train`, `y_valid`, and `y_full_train`.
- Leakage columns `position`, `gross_booking_usd`, and `gross_bookings_usd` are not final model features.

## Feature Table

| Feature | Category | Meaning |
|---|---|---|
| `srch_id` | Identifier | Search/result-list identifier. Used to group hotels for ranking. |
| `site_id` | Raw context | Expedia site identifier; also used for fitting site-specific price caps. |
| `visitor_location_country_id` | Raw context | Visitor country identifier. Used as coarse user context because no true user id is available. |
| `visitor_hist_starrating` | Raw visitor history | Visitor historical preferred star rating, imputed with visitor-country medians where possible. |
| `visitor_hist_adr_usd` | Raw visitor history | Visitor historical average daily rate in USD, imputed with visitor-country medians where possible. |
| `prop_country_id` | Raw property | Hotel/property country identifier. |
| `prop_id` | Identifier | Hotel/property identifier. Used for property-level historical popularity features. |
| `prop_starrating` | Raw property | Hotel star rating. |
| `prop_review_score` | Raw property | Hotel review score. |
| `prop_brand_bool` | Raw property | Whether the property belongs to a known brand. |
| `prop_location_score1` | Raw property | First Expedia location attractiveness score. |
| `prop_location_score2` | Raw property | Second Expedia location attractiveness score, imputed with property-country medians where possible. |
| `prop_log_historical_price` | Raw property | Logged historical property price signal; also used to derive deal features. |
| `price_usd` | Raw price | Original displayed price in USD. Kept as a raw signal; relative price features use capped variants. |
| `promotion_flag` | Raw offer | Whether the hotel offer had a promotion. |
| `srch_destination_id` | Raw search context | Destination identifier for the search. |
| `srch_length_of_stay` | Raw search context | Requested length of stay. |
| `srch_booking_window` | Raw search context | Days between search date and stay date. |
| `srch_adults_count` | Raw search context | Number of adults in the search. |
| `srch_children_count` | Raw search context | Number of children in the search. |
| `srch_room_count` | Raw search context | Number of requested rooms. |
| `srch_saturday_night_bool` | Raw search context | Whether the stay includes a Saturday night. |
| `srch_query_affinity_score` | Raw sparse signal | Search-query affinity score, imputed with a low sentinel below the training minimum. |
| `orig_destination_distance` | Raw sparse signal | Distance between origin and destination, imputed with a fitted median. |
| `random_bool` | Raw experiment flag | Whether the result list came from randomized Expedia sorting. |
| `date_month` | Date/time | Month extracted from date_time. |
| `date_dayofweek` | Date/time | Day of week extracted from date_time. |
| `date_hour` | Date/time | Hour extracted from date_time. |
| `date_is_weekend` | Date/time | Indicator for Saturday or Sunday search timestamp. |
| `is_missing_visitor_hist_starrating` | Missing indicator | Indicator that visitor_hist_starrating was missing before imputation. |
| `is_missing_visitor_hist_adr_usd` | Missing indicator | Indicator that visitor_hist_adr_usd was missing before imputation. |
| `is_missing_prop_review_score` | Missing indicator | Indicator that prop_review_score was missing before imputation. |
| `is_missing_prop_location_score2` | Missing indicator | Indicator that prop_location_score2 was missing before imputation. |
| `is_missing_srch_query_affinity_score` | Missing indicator | Indicator that srch_query_affinity_score was missing before imputation. |
| `is_missing_orig_destination_distance` | Missing indicator | Indicator that orig_destination_distance was missing before imputation. |
| `is_missing_comp1_rate` | Missing indicator | Indicator that comp1_rate was missing before imputation. |
| `is_missing_comp1_inv` | Missing indicator | Indicator that comp1_inv was missing before imputation. |
| `is_missing_comp1_rate_percent_diff` | Missing indicator | Indicator that comp1_rate_percent_diff was missing before imputation. |
| `is_missing_comp2_rate` | Missing indicator | Indicator that comp2_rate was missing before imputation. |
| `is_missing_comp2_inv` | Missing indicator | Indicator that comp2_inv was missing before imputation. |
| `is_missing_comp2_rate_percent_diff` | Missing indicator | Indicator that comp2_rate_percent_diff was missing before imputation. |
| `is_missing_comp3_rate` | Missing indicator | Indicator that comp3_rate was missing before imputation. |
| `is_missing_comp3_inv` | Missing indicator | Indicator that comp3_inv was missing before imputation. |
| `is_missing_comp3_rate_percent_diff` | Missing indicator | Indicator that comp3_rate_percent_diff was missing before imputation. |
| `is_missing_comp4_rate` | Missing indicator | Indicator that comp4_rate was missing before imputation. |
| `is_missing_comp4_inv` | Missing indicator | Indicator that comp4_inv was missing before imputation. |
| `is_missing_comp4_rate_percent_diff` | Missing indicator | Indicator that comp4_rate_percent_diff was missing before imputation. |
| `is_missing_comp5_rate` | Missing indicator | Indicator that comp5_rate was missing before imputation. |
| `is_missing_comp5_inv` | Missing indicator | Indicator that comp5_inv was missing before imputation. |
| `is_missing_comp5_rate_percent_diff` | Missing indicator | Indicator that comp5_rate_percent_diff was missing before imputation. |
| `is_missing_comp6_rate` | Missing indicator | Indicator that comp6_rate was missing before imputation. |
| `is_missing_comp6_inv` | Missing indicator | Indicator that comp6_inv was missing before imputation. |
| `is_missing_comp6_rate_percent_diff` | Missing indicator | Indicator that comp6_rate_percent_diff was missing before imputation. |
| `is_missing_comp7_rate` | Missing indicator | Indicator that comp7_rate was missing before imputation. |
| `is_missing_comp7_inv` | Missing indicator | Indicator that comp7_inv was missing before imputation. |
| `is_missing_comp7_rate_percent_diff` | Missing indicator | Indicator that comp7_rate_percent_diff was missing before imputation. |
| `is_missing_comp8_rate` | Missing indicator | Indicator that comp8_rate was missing before imputation. |
| `is_missing_comp8_inv` | Missing indicator | Indicator that comp8_inv was missing before imputation. |
| `is_missing_comp8_rate_percent_diff` | Missing indicator | Indicator that comp8_rate_percent_diff was missing before imputation. |
| `price_usd_capped` | Price | price_usd capped at the training-fitted 99th percentile within site_id, with a global fallback. |
| `price_per_night` | Price | Capped price divided by max(srch_length_of_stay, 1). |
| `log_price_usd_capped` | Price | log1p transform of price_usd_capped. |
| `prop_historical_price_missing_or_zero` | Deal | Indicator that prop_log_historical_price was zero, missing, or invalid for deal construction. |
| `historical_price` | Deal | exp(prop_log_historical_price), clipped and set missing when historical price is invalid or zero. |
| `deal_value` | Deal | historical_price minus price_usd_capped, clipped to avoid unstable extremes. |
| `deal_ratio` | Deal | deal_value divided by price_usd_capped plus epsilon, clipped to avoid unstable extremes. |
| `log_historical_price_diff` | Deal | prop_log_historical_price minus log1p(price_usd_capped). |
| `price_usd_capped_rank_in_search` | Within-search | price_usd_capped rank within the same srch_id. |
| `price_usd_capped_pct_rank_in_search` | Within-search | price_usd_capped_pct rank within the same srch_id. |
| `price_usd_capped_diff_from_search_mean` | Within-search | price_usd_capped difference from the mean within the same srch_id. |
| `price_usd_capped_diff_from_search_median` | Within-search | price_usd_capped difference from the median within the same srch_id. |
| `price_usd_capped_zscore_in_search` | Within-search | price_usd_capped z-score within the same srch_id. |
| `price_per_night_rank_in_search` | Within-search | price_per_night rank within the same srch_id. |
| `price_per_night_pct_rank_in_search` | Within-search | price_per_night_pct rank within the same srch_id. |
| `price_per_night_diff_from_search_mean` | Within-search | price_per_night difference from the mean within the same srch_id. |
| `price_per_night_diff_from_search_median` | Within-search | price_per_night difference from the median within the same srch_id. |
| `price_per_night_zscore_in_search` | Within-search | price_per_night z-score within the same srch_id. |
| `prop_starrating_rank_in_search` | Within-search | prop_starrating rank within the same srch_id. |
| `prop_starrating_pct_rank_in_search` | Within-search | prop_starrating_pct rank within the same srch_id. |
| `prop_starrating_diff_from_search_mean` | Within-search | prop_starrating difference from the mean within the same srch_id. |
| `prop_starrating_diff_from_search_median` | Within-search | prop_starrating difference from the median within the same srch_id. |
| `prop_starrating_zscore_in_search` | Within-search | prop_starrating z-score within the same srch_id. |
| `prop_review_score_rank_in_search` | Within-search | prop_review_score rank within the same srch_id. |
| `prop_review_score_pct_rank_in_search` | Within-search | prop_review_score_pct rank within the same srch_id. |
| `prop_review_score_diff_from_search_mean` | Within-search | prop_review_score difference from the mean within the same srch_id. |
| `prop_review_score_diff_from_search_median` | Within-search | prop_review_score difference from the median within the same srch_id. |
| `prop_review_score_zscore_in_search` | Within-search | prop_review_score z-score within the same srch_id. |
| `prop_location_score1_rank_in_search` | Within-search | prop_location_score1 rank within the same srch_id. |
| `prop_location_score1_pct_rank_in_search` | Within-search | prop_location_score1_pct rank within the same srch_id. |
| `prop_location_score1_diff_from_search_mean` | Within-search | prop_location_score1 difference from the mean within the same srch_id. |
| `prop_location_score1_diff_from_search_median` | Within-search | prop_location_score1 difference from the median within the same srch_id. |
| `prop_location_score1_zscore_in_search` | Within-search | prop_location_score1 z-score within the same srch_id. |
| `prop_location_score2_rank_in_search` | Within-search | prop_location_score2 rank within the same srch_id. |
| `prop_location_score2_pct_rank_in_search` | Within-search | prop_location_score2_pct rank within the same srch_id. |
| `prop_location_score2_diff_from_search_mean` | Within-search | prop_location_score2 difference from the mean within the same srch_id. |
| `prop_location_score2_diff_from_search_median` | Within-search | prop_location_score2 difference from the median within the same srch_id. |
| `prop_location_score2_zscore_in_search` | Within-search | prop_location_score2 z-score within the same srch_id. |
| `prop_log_historical_price_rank_in_search` | Within-search | prop_log_historical_price rank within the same srch_id. |
| `prop_log_historical_price_pct_rank_in_search` | Within-search | prop_log_historical_price_pct rank within the same srch_id. |
| `prop_log_historical_price_diff_from_search_mean` | Within-search | prop_log_historical_price difference from the mean within the same srch_id. |
| `prop_log_historical_price_diff_from_search_median` | Within-search | prop_log_historical_price difference from the median within the same srch_id. |
| `prop_log_historical_price_zscore_in_search` | Within-search | prop_log_historical_price z-score within the same srch_id. |
| `deal_value_rank_in_search` | Within-search | deal_value rank within the same srch_id. |
| `deal_value_pct_rank_in_search` | Within-search | deal_value_pct rank within the same srch_id. |
| `deal_value_diff_from_search_mean` | Within-search | deal_value difference from the mean within the same srch_id. |
| `deal_value_diff_from_search_median` | Within-search | deal_value difference from the median within the same srch_id. |
| `deal_value_zscore_in_search` | Within-search | deal_value z-score within the same srch_id. |
| `is_cheapest_in_search` | Within-search | Indicator for the cheapest hotel within the same srch_id, based on price_per_night when available. |
| `is_highest_review_in_search` | Within-search | Indicator for the highest prop_review_score within the same srch_id. |
| `is_highest_location2_in_search` | Within-search | Indicator for the highest prop_location_score2 within the same srch_id. |
| `is_highest_starrating_in_search` | Within-search | Indicator for the highest prop_starrating within the same srch_id. |
| `star_user_pref_abs_diff` | Search-property match | Absolute difference between visitor_hist_starrating and prop_starrating. |
| `price_user_pref_abs_diff` | Search-property match | Absolute difference between visitor_hist_adr_usd and capped current price. |
| `guest_count` | Search-property match | srch_adults_count plus srch_children_count. |
| `has_children` | Search-property match | Indicator that srch_children_count is greater than zero. |
| `is_family_search` | Search-property match | Indicator for searches with children and at least one adult. |
| `guests_per_room` | Search-property match | guest_count divided by srch_room_count. |
| `booking_window_bucket` | Search-property match | Bucketed srch_booking_window. |
| `length_of_stay_bucket` | Search-property match | Bucketed srch_length_of_stay. |
| `domestic_trip_indicator` | Search-property match | Indicator that visitor_location_country_id equals prop_country_id. |
| `total_price_proxy` | Search-property match | price_usd_capped multiplied by srch_length_of_stay. |
| `promotion_x_price_rank` | Search-property match | promotion_flag multiplied by within-search price rank. |
| `star_x_review` | Search-property match | prop_starrating multiplied by prop_review_score. |
| `comp_rate_better_count` | Competitor aggregate | Count of comp*_rate values equal to 1. |
| `comp_rate_equal_count` | Competitor aggregate | Count of comp*_rate values equal to 0. |
| `comp_rate_worse_count` | Competitor aggregate | Count of comp*_rate values equal to -1. |
| `comp_rate_available_count` | Competitor aggregate | Number of non-missing comp*_rate observations. |
| `comp_inv_unavailable_count` | Competitor aggregate | Count of comp*_inv values equal to 1. |
| `comp_inv_available_count` | Competitor aggregate | Number of non-missing comp*_inv observations. |
| `comp_percent_diff_mean` | Competitor aggregate | Mean of available comp*_rate_percent_diff values. |
| `comp_percent_diff_max` | Competitor aggregate | Maximum available comp*_rate_percent_diff value. |
| `comp_percent_diff_min` | Competitor aggregate | Minimum available comp*_rate_percent_diff value. |
| `comp_missing_count` | Competitor aggregate | Total missing values across retained raw competitor source columns before dropping them. |
| `comp_any_data_count` | Competitor aggregate | Total non-missing values across raw competitor source columns. |
| `prop_impressions` | Historical popularity | Number of property impressions in the fitted labelled training data. |
| `prop_click_rate_smooth` | Historical popularity | Smoothed historical click rate for prop_id. |
| `prop_booking_rate_smooth` | Historical popularity | Smoothed historical booking rate for prop_id. |
| `prop_relevance_mean_smooth` | Historical popularity | Smoothed historical mean relevance for prop_id. |
| `prop_mean_price_usd_capped` | Historical popularity | Mean capped price for prop_id in the fitted training data. |
| `destination_impressions` | Historical popularity | Number of destination impressions in the fitted labelled training data. |
| `destination_booking_rate_smooth` | Historical popularity | Smoothed historical booking rate for srch_destination_id. |
| `destination_relevance_mean_smooth` | Historical popularity | Smoothed historical mean relevance for srch_destination_id. |
| `prop_destination_impressions` | Historical popularity | Number of prop_id and srch_destination_id pair impressions in fitted training data. |
| `prop_destination_booking_rate_smooth` | Historical popularity | Smoothed historical booking rate for prop_id and srch_destination_id pair. |
| `prop_destination_relevance_mean_smooth` | Historical popularity | Smoothed historical mean relevance for prop_id and srch_destination_id pair. |

## Output Tables

| File | Purpose |
|---|---|
| `outputs/features/X_train.parquet` | Training-split feature matrix for offline validation. |
| `outputs/features/X_valid.parquet` | Validation-split feature matrix for offline validation. |
| `outputs/features/X_test.parquet` | Test feature matrix for prediction. |
| `outputs/features/full_train_features.parquet` | Full labelled training feature matrix for final model fitting. |
| `outputs/features/y_train.parquet` | Labels for `X_train`, including `click_bool`, `booking_bool`, and `relevance`. |
| `outputs/features/y_valid.parquet` | Labels for `X_valid`, including `click_bool`, `booking_bool`, and `relevance`. |
| `outputs/features/groups_train.parquet` | Query group sizes for `X_train`, ordered by `srch_id`. |
| `outputs/features/groups_valid.parquet` | Query group sizes for `X_valid`, ordered by `srch_id`. |
