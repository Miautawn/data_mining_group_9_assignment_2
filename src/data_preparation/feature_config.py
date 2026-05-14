"""Configuration for the Task 3 Expedia feature pipeline."""

from __future__ import annotations

from pathlib import Path


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs") / "features"

TRAIN_FILE = DATA_DIR / "training_set_VU_DM.csv"
TEST_FILE = DATA_DIR / "test_set_VU_DM.csv"

TARGET_COLUMNS = ["click_bool", "booking_bool"]
LEAKAGE_COLUMNS = ["position", "gross_booking_usd", "gross_bookings_usd"]
ID_COLUMNS = ["srch_id", "prop_id"]

RANDOM_SEED = 42
VALID_SIZE = 0.2
SMOOTHING_ALPHA = 10.0
SPARSE_KEY_SMOOTHING_ALPHA = 50.0
PRICE_CAP_QUANTILE = 0.99
DEAL_RATIO_CLIP = 100.0
MAX_HISTORICAL_PRICE = 100_000.0
EPSILON = 1e-6

IMPORTANT_MISSING_COLUMNS = [
    "visitor_hist_starrating",
    "visitor_hist_adr_usd",
    "prop_review_score",
    "prop_location_score2",
    "srch_query_affinity_score",
    "orig_destination_distance",
]

COMPETITOR_IDS = range(1, 9)

DEFAULT_DTYPE_MAP = {
    "srch_id": "int32",
    "site_id": "int16",
    "visitor_location_country_id": "int16",
    "prop_country_id": "int16",
    "prop_id": "int32",
    "srch_destination_id": "int32",
    "srch_length_of_stay": "int16",
    "srch_booking_window": "int16",
    "srch_adults_count": "int8",
    "srch_children_count": "int8",
    "srch_room_count": "int8",
    "prop_brand_bool": "int8",
    "promotion_flag": "int8",
    "srch_saturday_night_bool": "int8",
    "random_bool": "int8",
    "click_bool": "int8",
    "booking_bool": "int8",
    "position": "int16",
}
