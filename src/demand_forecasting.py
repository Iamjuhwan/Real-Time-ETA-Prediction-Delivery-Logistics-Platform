"""
Zone x Hour Demand Forecasting

Forecasts order volume per (zone, hour) one day ahead. This feeds 
rider pre-positioning logic: if the model expects a demand spike in Lekki at
7pm tomorrow, riders should be nudged toward Lekki before it happens, not
after orders start queueing.

Key discipline carried over from feature engineering: only use features actually knowable
at forecast time. In particular, `is_rain`  is
NOT used here — you don't know tomorrow's actual rain, only a forecast
probability, which has real but imperfect skill. This script simulates that
distinction explicitly rather than quietly using ground-truth rain, which
would make the forecast look better than it could ever be in production.

Baseline: seasonal-naive (predict this week using last week's same zone/hour
count) — the simplest reasonable heuristic and a real thing ops teams do by
gut feel. The model has to beat it to justify its existence.

Usage:
    python  demand_forecasting.py --data_dir ../data --out_dir ../data
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(7)


def build_demand_grid(orders: pd.DataFrame) -> pd.DataFrame:
    """Full date x zone x hour grid, filling true zeros where no orders
    occurred — a missing combination IS signal (zero demand), not missing
    data, and must not be silently dropped."""
    orders["date"] = orders["order_ts"].dt.date
    counts = (orders.groupby(["date", "restaurant_zone", "hour"])
              .size().reset_index(name="order_count"))

    all_dates = pd.date_range(orders["date"].min(), orders["date"].max(), freq="D").date
    zones = orders["restaurant_zone"].unique()
    hours = range(24)
    full_index = pd.MultiIndex.from_product([all_dates, zones, hours],
                                             names=["date", "restaurant_zone", "hour"])
    grid = pd.DataFrame(index=full_index).reset_index()

    grid = grid.merge(counts, on=["date", "restaurant_zone", "hour"], how="left")
    grid["order_count"] = grid["order_count"].fillna(0).astype(int)
    grid["date"] = pd.to_datetime(grid["date"])
    return grid.sort_values(["restaurant_zone", "date", "hour"]).reset_index(drop=True)


def add_rain_and_forecast(grid: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Attach the actual daily rain flag (for reference/eval only — NEVER
    used as a model feature), plus a simulated 'day-ahead weather forecast'
    that's correlated with actual rain but imperfect, which IS a legitimate
    feature since real forecasts are genuinely available in advance."""
    daily_rain = orders.groupby("date")["is_rain"].first().reset_index()
    daily_rain["date"] = pd.to_datetime(daily_rain["date"])
    grid = grid.merge(daily_rain, on="date", how="left")

    # Simulate imperfect forecast skill: correlated with actual, not equal to it
    forecast_prob = np.where(
        grid["is_rain"],
        RNG.beta(6, 2, size=len(grid)),   # skewed high when it will actually rain
        RNG.beta(2, 6, size=len(grid)),   # skewed low when it won't
    )
    grid["rain_forecast_prob"] = forecast_prob
    return grid


def add_lag_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Lag/rolling features computed strictly from the past, per zone x hour
    series — this is what makes it a real forecasting setup rather than
    same-day regression."""
    grid = grid.sort_values(["restaurant_zone", "hour", "date"]).reset_index(drop=True)
    g = grid.groupby(["restaurant_zone", "hour"])["order_count"]

    grid["lag_1day"] = g.shift(1)
    grid["lag_7day"] = g.shift(7)
    grid["rolling_7day_mean"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).mean())
    grid["rolling_7day_std"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=3).std())

    grid["day_of_week"] = grid["date"].dt.dayofweek
    grid["is_weekend"] = (grid["day_of_week"] >= 5).astype(int)
    grid["hour_sin"] = np.sin(2 * np.pi * grid["hour"] / 24)
    grid["hour_cos"] = np.cos(2 * np.pi * grid["hour"] / 24)

    return grid


def time_based_split(grid: pd.DataFrame, test_days=14):
    grid = grid.dropna(subset=["lag_7day"]).reset_index(drop=True)  # need 7 days history first
    cutoff = grid["date"].max() - pd.Timedelta(days=test_days)
    grid["split"] = np.where(grid["date"] <= cutoff, "train", "test")
    for s in ["train", "test"]:
        sub = grid[grid["split"] == s]
        log.info(f"{s}: {len(sub)} rows, {sub['date'].min().date()} -> {sub['date'].max().date()}")
    return grid


FEATURE_COLS = [
    "lag_1day", "lag_7day", "rolling_7day_mean", "rolling_7day_std",
    "day_of_week", "is_weekend", "hour_sin", "hour_cos",
    "rain_forecast_prob", "restaurant_zone",
]
TARGET_COL = "order_count"


def run_seasonal_naive_baseline(grid: pd.DataFrame):
    test = grid[grid["split"] == "test"]
    mae = mean_absolute_error(test[TARGET_COL], test["lag_7day"])
    log.info(f"[BASELINE: seasonal-naive, same zone/hour last week] Test MAE = {mae:.2f} orders")
    return mae


def train_demand_model(grid: pd.DataFrame, feature_cols: list):
    train = grid[grid["split"] == "train"]
    # Small internal validation slice from the tail of train, for early stopping
    val_cutoff = train["date"].max() - pd.Timedelta(days=7)
    tr, val = train[train["date"] <= val_cutoff], train[train["date"] > val_cutoff]

    train_set = lgb.Dataset(tr[feature_cols], label=tr[TARGET_COL],
                             categorical_feature=["restaurant_zone"], free_raw_data=False)
    val_set = lgb.Dataset(val[feature_cols], label=val[TARGET_COL],
                           categorical_feature=["restaurant_zone"], reference=train_set,
                           free_raw_data=False)

    params = {
        "objective": "poisson",  # order counts are non-negative integers — Poisson fits the data shape
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }
    model = lgb.train(
        params, train_set, num_boost_round=1000, valid_sets=[val_set],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                   lgb.log_evaluation(period=0)],
    )
    log.info(f"[demand model] best iteration: {model.best_iteration}")
    return model


def evaluate(model, grid: pd.DataFrame, feature_cols: list):
    test = grid[grid["split"] == "test"]
    pred = np.clip(model.predict(test[feature_cols], num_iteration=model.best_iteration), 0, None)
    mae = mean_absolute_error(test[TARGET_COL], pred)
    log.info(f"[demand model] Test MAE = {mae:.2f} orders")
    return pred, mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--out_dir", type=str, default="../data")
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orders = pd.read_csv(data_dir / "orders.csv", parse_dates=["order_ts"])

    log.info("Building zone x hour demand grid...")
    grid = build_demand_grid(orders)
    grid = add_rain_and_forecast(grid, orders)
    grid = add_lag_features(grid)
    grid = time_based_split(grid)

    grid["restaurant_zone"] = grid["restaurant_zone"].astype("category")

    baseline_mae = run_seasonal_naive_baseline(grid)

    log.info("\n=== Training demand forecasting model ===")
    model = train_demand_model(grid, FEATURE_COLS)
    pred, model_mae = evaluate(model, grid, FEATURE_COLS)

    log.info(f"\n=== Summary ===\nSeasonal-naive baseline MAE: {baseline_mae:.2f} orders\n"
              f"LightGBM demand model MAE: {model_mae:.2f} orders\n"
              f"Improvement: {(1 - model_mae/baseline_mae)*100:.1f}%")

    test = grid[grid["split"] == "test"].copy()
    test["predicted_demand"] = pred
    test[["date", "restaurant_zone", "hour", "order_count", "predicted_demand"]].to_csv(
        out_dir / "demand_forecast.csv", index=False
    )
    model.save_model(str(out_dir.parent / "models" / "demand_model.txt")) if (out_dir.parent / "models").exists() else None
    log.info(f"Saved forecasts -> {out_dir / 'demand_forecast.csv'}")


if __name__ == "__main__":
    main()