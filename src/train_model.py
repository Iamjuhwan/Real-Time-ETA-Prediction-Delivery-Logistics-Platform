

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

CATEGORICAL_COLS = ["cuisine", "restaurant_zone", "customer_zone"]
NUMERIC_COLS = [
    "distance_km", "rider_to_restaurant_km", "hour_sin", "hour_cos",
    "day_of_week", "is_weekend", "is_rain", "item_count",
    "rider_experience_months", "rider_daily_order_seq",
    "rest_hist_avg_prep", "rest_hist_std_prep", "rest_hist_order_count",
    "rest_is_cold_start", "popularity_weight", "same_zone",
    "zone_hour_avg_total",
]
TARGET_COL = "actual_total_min"


def pinball_loss(y_true, y_pred, quantile):
    """Manual pinball loss — the metric quantile models are actually
    optimizing, and the right one to report (MAE alone doesn't tell you if
    a 'P90' model is actually covering 90% of outcomes)."""
    diff = y_true - y_pred
    return np.mean(np.maximum(quantile * diff, (quantile - 1) * diff))


def coverage(y_true, y_pred_quantile, quantile):
    """% of actual outcomes at or below the predicted quantile. For a
    well-calibrated P90 model, this should land close to 0.90."""
    return np.mean(y_true <= y_pred_quantile)


def load_features(data_dir: Path):
    df = pd.read_csv(data_dir / "features.csv", parse_dates=["order_ts"])
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")
    return df


def run_baseline(df: pd.DataFrame):
    """Zone x hour historical average, already computed in feature
    engineering as `zone_hour_avg_total` — using it directly AS the
    prediction is the baseline the model must beat."""
    test = df[df["split"] == "test"]
    y_true = test[TARGET_COL].values
    y_pred = test["zone_hour_avg_total"].values

    mae = mean_absolute_error(y_true, y_pred)
    log.info(f"[BASELINE: zone x hour historical avg] Test MAE = {mae:.2f} min")
    return mae


def train_quantile_model(df: pd.DataFrame, quantile: float, feature_cols: list):
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]

    train_set = lgb.Dataset(train[feature_cols], label=train[TARGET_COL],
                             categorical_feature=CATEGORICAL_COLS, free_raw_data=False)
    val_set = lgb.Dataset(val[feature_cols], label=val[TARGET_COL],
                           categorical_feature=CATEGORICAL_COLS, reference=train_set,
                           free_raw_data=False)

    params = {
        "objective": "quantile",
        "alpha": quantile,
        "metric": "quantile",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
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
    log.info(f"[P{int(quantile*100)} model] best iteration: {model.best_iteration}")
    return model


def evaluate_model(model, df: pd.DataFrame, quantile: float, feature_cols: list):
    test = df[df["split"] == "test"]
    y_true = test[TARGET_COL].values
    y_pred = model.predict(test[feature_cols], num_iteration=model.best_iteration)

    pb_loss = pinball_loss(y_true, y_pred, quantile)
    cov = coverage(y_true, y_pred, quantile)
    mae = mean_absolute_error(y_true, y_pred) if quantile == 0.5 else None

    log.info(f"[P{int(quantile*100)} model] Test pinball loss = {pb_loss:.3f} | "
             f"coverage = {cov:.1%} (target ~{int(quantile*100)}%)"
             + (f" | MAE = {mae:.2f} min" if mae is not None else ""))
    return y_pred, pb_loss, cov


def error_analysis(df: pd.DataFrame, y_pred_p50: np.ndarray):
    """Break down P50 error by rain, hour bucket, and zone — this is what
    you actually discuss in an interview, not the headline MAE number."""
    test = df[df["split"] == "test"].copy()
    test["abs_error"] = np.abs(test[TARGET_COL].values - y_pred_p50)

    log.info("\n--- Error analysis (P50 model, test split) ---")
    log.info(f"By rain:\n{test.groupby('is_rain')['abs_error'].mean().round(2)}")

    test["hour_bucket"] = pd.cut(
        test["order_ts"].dt.hour, bins=[-1, 6, 11, 15, 19, 23],
        labels=["late_night", "morning", "afternoon", "evening_rush", "night"],
    )
    log.info(f"By hour bucket:\n{test.groupby('hour_bucket', observed=True)['abs_error'].mean().round(2)}")

    log.info(f"By zone (top 5 worst):\n"
             f"{test.groupby('restaurant_zone', observed=True)['abs_error'].mean().sort_values(ascending=False).head(5).round(2)}")


def feature_importance(model, feature_cols: list, top_n=10):
    imp = pd.Series(model.feature_importance(importance_type="gain"), index=feature_cols)
    imp = imp.sort_values(ascending=False).head(top_n)
    log.info(f"\n--- Top {top_n} features (P50 model, gain) ---\n{imp.round(1)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--out_dir", type=str, default="../models")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_features(data_dir)
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS

    baseline_mae = run_baseline(df)

    log.info("\n=== Training P50 (median) model ===")
    model_p50 = train_quantile_model(df, 0.5, feature_cols)
    y_pred_p50, _, _ = evaluate_model(model_p50, df, 0.5, feature_cols)

    log.info("\n=== Training P90 model ===")
    model_p90 = train_quantile_model(df, 0.9, feature_cols)
    evaluate_model(model_p90, df, 0.9, feature_cols)

    error_analysis(df, y_pred_p50)
    feature_importance(model_p50, feature_cols)

    p50_mae = mean_absolute_error(df[df["split"] == "test"][TARGET_COL], y_pred_p50)
    log.info(f"\n=== Summary ===\nBaseline (zone x hour avg) MAE: {baseline_mae:.2f} min\n"
              f"LightGBM P50 MAE: {p50_mae:.2f} min\n"
              f"Improvement: {(1 - p50_mae/baseline_mae)*100:.1f}%")

    model_p50.save_model(str(out_dir / "eta_model_p50.txt"))
    model_p90.save_model(str(out_dir / "eta_model_p90.txt"))
    log.info(f"Saved models -> {out_dir}")


if __name__ == "__main__":
    main()