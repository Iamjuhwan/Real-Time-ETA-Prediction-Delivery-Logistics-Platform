"""
 Model Monitoring / Drift Detection

Reads the JSONL prediction log written by serving/app.py and compares the
distribution of both INPUT features and OUTPUT predictions against the
training-time baseline (features.csv, train split). Two-sample
Kolmogorov-Smirnov tests flag statistically significant distribution shift
per feature.

Why both input AND output drift matter, and why they're different signals:
  - INPUT drift (e.g. distance_km distribution shifts) means the population
    of orders being served has changed — maybe a new zone launched, maybe
    a marketing push changed order mix. The model might still be accurate,
    but it's now extrapolating outside what it was trained on.
  - OUTPUT/prediction drift (predicted ETAs trending up or down over time)
    is often the first visible symptom of a real operational problem — e.g.
    a rainy season starting, or restaurant prep times degrading — and is
    worth alerting on even before you know the root cause.

This is designed to be run on a schedule (cron, Airflow, whatever) against
an accumulating log file — see the CI workflow for how this plugs into an
automated retraining trigger.

Usage:
    python drift_check.py --data_dir ../data --log_path ../logs/predictions.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Features worth drift-checking — numeric, high-signal ones. Categorical
# drift (e.g. a new zone appearing) is checked separately via value-count
# comparison, since KS-test doesn't apply to categoricals.
DRIFT_CHECK_NUMERIC = [
    "distance_km", "rider_to_restaurant_km", "item_count",
    "rider_experience_months", "rest_hist_avg_prep",
]
KS_ALPHA = 0.01  # significance threshold — conservative, to avoid alert fatigue
MIN_SAMPLES_FOR_TEST = 30  # don't run KS-test on tiny/noisy log samples


def load_baseline(data_dir: Path) -> pd.DataFrame:
    """Training-period feature distribution — the reference point everything
    live gets compared against."""
    features = pd.read_csv(data_dir / "features.csv")
    return features[features["split"] == "train"]


def load_prediction_log(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        raise FileNotFoundError(
            f"No prediction log at {log_path} — run the serving app and make "
            f"some requests first, or point --log_path at a real log file."
        )
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            flat = {**rec["request"], "eta_p50_min": rec["eta_p50_min"],
                    "eta_p90_min": rec["eta_p90_min"], "timestamp": rec["timestamp"]}
            records.append(flat)
    return pd.DataFrame(records)


def check_numeric_drift(baseline: pd.DataFrame, live: pd.DataFrame) -> list:
    results = []
    for col in DRIFT_CHECK_NUMERIC:
        if col not in live.columns:
            continue
        base_vals = baseline[col].dropna().values
        live_vals = live[col].dropna().values
        if len(live_vals) < MIN_SAMPLES_FOR_TEST:
            results.append({
                "feature": col, "status": "insufficient_data",
                "n_live": len(live_vals), "p_value": None, "drift_detected": False,
            })
            continue

        stat, p_value = ks_2samp(base_vals, live_vals)
        drift = p_value < KS_ALPHA
        results.append({
            "feature": col, "status": "checked", "n_live": len(live_vals),
            "ks_statistic": round(float(stat), 4), "p_value": round(float(p_value), 6),
            "drift_detected": bool(drift),
            "baseline_mean": round(float(np.mean(base_vals)), 2),
            "live_mean": round(float(np.mean(live_vals)), 2),
        })
    return results


def check_prediction_drift(baseline: pd.DataFrame, live: pd.DataFrame) -> dict:
    """Compares live P50 predictions against the training-period actual
    delivery times — the closest available reference for 'what ETAs should
    typically look like'."""
    base_actuals = baseline["actual_total_min"].dropna().values
    live_preds = live["eta_p50_min"].dropna().values

    if len(live_preds) < MIN_SAMPLES_FOR_TEST:
        return {"status": "insufficient_data", "n_live": len(live_preds)}

    stat, p_value = ks_2samp(base_actuals, live_preds)
    return {
        "status": "checked", "n_live": len(live_preds),
        "ks_statistic": round(float(stat), 4), "p_value": round(float(p_value), 6),
        "drift_detected": bool(p_value < KS_ALPHA),
        "baseline_mean_min": round(float(np.mean(base_actuals)), 2),
        "live_mean_min": round(float(np.mean(live_preds)), 2),
    }


def check_categorical_novelty(baseline: pd.DataFrame, live: pd.DataFrame, col: str) -> dict:
    """Flags categories appearing in live traffic that never appeared in
    training — e.g. a new zone or cuisine the model has never priced."""
    if col not in live.columns:
        return {"feature": col, "status": "not_available"}
    known = set(baseline[col].dropna().unique())
    seen = set(live[col].dropna().unique())
    novel = seen - known
    return {
        "feature": col, "status": "checked",
        "novel_categories": sorted(novel), "novel_count": len(novel),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--log_path", type=str, default="../logs/predictions.jsonl")
    args = parser.parse_args()

    data_dir, log_path = Path(args.data_dir), Path(args.log_path)

    log.info("Loading training baseline...")
    baseline = load_baseline(data_dir)
    log.info(f"Baseline: {len(baseline)} training-period orders")

    log.info("Loading live prediction log...")
    live = load_prediction_log(log_path)
    log.info(f"Live log: {len(live)} logged predictions")

    log.info("\n=== Numeric input drift (KS-test, alpha=%.2f) ===" % KS_ALPHA)
    numeric_results = check_numeric_drift(baseline, live)
    any_input_drift = False
    for r in numeric_results:
        if r["status"] == "insufficient_data":
            log.info(f"  {r['feature']}: SKIPPED (only {r['n_live']} samples, need {MIN_SAMPLES_FOR_TEST}+)")
            continue
        flag = "*** DRIFT DETECTED ***" if r["drift_detected"] else "ok"
        log.info(f"  {r['feature']}: p={r['p_value']:.4g} | baseline_mean={r['baseline_mean']} "
                  f"live_mean={r['live_mean']} -> {flag}")
        any_input_drift = any_input_drift or r["drift_detected"]

    log.info("\n=== Prediction (output) drift ===")
    pred_result = check_prediction_drift(baseline, live)
    if pred_result["status"] == "checked":
        flag = "*** DRIFT DETECTED ***" if pred_result["drift_detected"] else "ok"
        log.info(f"  P50 predictions vs training actuals: p={pred_result['p_value']:.4g} | "
                  f"baseline_mean={pred_result['baseline_mean_min']} min "
                  f"live_mean={pred_result['live_mean_min']} min -> {flag}")
    else:
        log.info(f"  SKIPPED (only {pred_result['n_live']} samples)")

    log.info("\n=== Categorical novelty check ===")
    for col in ["cuisine", "restaurant_zone", "customer_zone"]:
        result = check_categorical_novelty(baseline, live, col)
        if result["status"] == "checked" and result["novel_count"] > 0:
            log.info(f"  {col}: {result['novel_count']} NEW categories seen live: {result['novel_categories']}")
        elif result["status"] == "checked":
            log.info(f"  {col}: no novel categories")

    any_drift = any_input_drift or pred_result.get("drift_detected", False)
    log.info(f"\n=== SUMMARY: {'DRIFT DETECTED — consider retraining' if any_drift else 'No significant drift'} ===")

    # Exit code signals downstream automation (e.g. CI job) whether to
    # trigger a retraining pipeline — see .github/workflows/ml_pipeline.yml
    exit(1 if any_drift else 0)


if __name__ == "__main__":
    main()