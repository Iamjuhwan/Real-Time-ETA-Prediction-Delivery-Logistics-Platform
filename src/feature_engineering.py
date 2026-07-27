

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_raw(data_dir: Path):
    orders = pd.read_csv(data_dir / "orders.csv", parse_dates=["order_ts"])
    restaurants = pd.read_csv(data_dir / "restaurants.csv")
    riders = pd.read_csv(data_dir / "riders.csv")
    return orders, restaurants, riders


def time_based_split(orders: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    
    orders = orders.sort_values("order_ts").reset_index(drop=True)
    n = len(orders)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    split = np.empty(n, dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"
    split[val_end:] = "test"
    orders["split"] = split

    for s in ["train", "val", "test"]:
        sub = orders[orders["split"] == s]
        log.info(f"{s}: {len(sub)} orders, {sub['order_ts'].min()} -> {sub['order_ts'].max()}")
    return orders


def add_restaurant_historical_features(orders: pd.DataFrame) -> pd.DataFrame:

   
    train = orders[orders["split"] == "train"]

    rest_stats = train.groupby("restaurant_id").agg(
        rest_hist_avg_prep=("prep_time_min", "mean"),
        rest_hist_std_prep=("prep_time_min", "std"),
        rest_hist_order_count=("order_id", "count"),
    ).reset_index()

    # Global fallback for restaurants with no training-period orders (cold start —
    # a real scenario for newly onboarded restaurants) so the model never sees NaN
    global_avg_prep = train["prep_time_min"].mean()
    global_std_prep = train["prep_time_min"].std()

    orders = orders.merge(rest_stats, on="restaurant_id", how="left")
    orders["rest_hist_avg_prep"] = orders["rest_hist_avg_prep"].fillna(global_avg_prep)
    orders["rest_hist_std_prep"] = orders["rest_hist_std_prep"].fillna(global_std_prep)
    orders["rest_hist_order_count"] = orders["rest_hist_order_count"].fillna(0)
    orders["rest_is_cold_start"] = (orders["rest_hist_order_count"] == 0).astype(int)

    return orders


def add_zone_historical_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Zone x hour historical average total delivery time — becomes both a
    standalone feature and the input to the baseline model in the next
    script. Same leakage-safe pattern: train-period only."""
    train = orders[orders["split"] == "train"]

    zone_hour_stats = train.groupby(["restaurant_zone", "hour"]).agg(
        zone_hour_avg_total=("actual_total_min", "mean"),
    ).reset_index()

    global_avg_total = train["actual_total_min"].mean()

    orders = orders.merge(zone_hour_stats, on=["restaurant_zone", "hour"], how="left")
    orders["zone_hour_avg_total"] = orders["zone_hour_avg_total"].fillna(global_avg_total)
    return orders


def build_features(orders: pd.DataFrame, restaurants: pd.DataFrame) -> pd.DataFrame:
    orders = orders.merge(
        restaurants[["restaurant_id", "cuisine", "popularity_weight"]],
        on="restaurant_id", how="left",
    )
    orders["same_zone"] = (orders["restaurant_zone"] == orders["customer_zone"]).astype(int)
    orders["is_weekend"] = (orders["day_of_week"] >= 5).astype(int)
    orders["is_rain"] = orders["is_rain"].astype(int)

    # Cyclical encoding for hour — a raw integer 0-23 tells the model 23 and 0
    # are far apart when they're actually adjacent (11pm/midnight)
    orders["hour_sin"] = np.sin(2 * np.pi * orders["hour"] / 24)
    orders["hour_cos"] = np.cos(2 * np.pi * orders["hour"] / 24)

    return orders


FEATURE_COLS = [
    "distance_km", "rider_to_restaurant_km", "hour_sin", "hour_cos",
    "day_of_week", "is_weekend", "is_rain", "item_count",
    "rider_experience_months", "rider_daily_order_seq",
    "rest_hist_avg_prep", "rest_hist_std_prep", "rest_hist_order_count",
    "rest_is_cold_start", "popularity_weight", "same_zone",
    "zone_hour_avg_total", "cuisine", "restaurant_zone", "customer_zone",
]
TARGET_COL = "actual_total_min"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--out_dir", type=str, default="../data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    log.info("Loading raw data...")
    orders, restaurants, riders = load_raw(data_dir)

    log.info("Applying time-based split...")
    orders = time_based_split(orders)

    log.info("Adding leakage-safe restaurant historical features...")
    orders = add_restaurant_historical_features(orders)

    log.info("Adding leakage-safe zone x hour historical features...")
    orders = add_zone_historical_features(orders)

    log.info("Building final feature set...")
    orders = build_features(orders, restaurants)

    keep_cols = ["order_id", "order_ts", "split"] + FEATURE_COLS + [TARGET_COL]
    final = orders[keep_cols]

    out_path = out_dir / "features.csv"
    final.to_csv(out_path, index=False)
    log.info(f"Saved {len(final)} rows x {len(FEATURE_COLS)} features -> {out_path}")

    # Confirm no leakage sanity check: cold-start rate in test should be very
    # low (restaurants seen in train are almost always the ones seeing test
    # orders too, since we only run 150 restaurants over 60 days)
    test_cold_start_rate = final[orders["split"] == "test"]["rest_is_cold_start"].mean()
    log.info(f"Cold-start rate in test split: {test_cold_start_rate:.1%} "
              f"(sanity check — should be low/zero for a stable restaurant list)")


if __name__ == "__main__":
    main()