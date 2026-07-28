"""
Live Dispatch Simulator + Batch Rider-Order Assignment

Why this exists (and what it's NOT reusing from Phase 1):
`orders.csv` already contains a rider assignment, but it's a static,
post-hoc one — for every order, the generator sampled 8 random riders and
picked the nearest, with no notion of rider availability, busy state, or
queueing. It's a fine GREEDY BASELINE, but it isn't a live dispatch stream,
so we can't just re-label those rows to claim "optimization."

This script replays orders.csv chronologically as an ARRIVAL stream (using
only order_ts, restaurant, customer_zone, distance_km — ignoring the
baked-in rider_id) against a live rider-state simulation, and runs two
dispatch policies over the identical stream so they're honestly comparable:

  1. GREEDY: as each order arrives, assign the nearest currently-idle rider
     immediately (this is what a naive real-time system does, and is
     structurally the same idea as Phase 1's generator logic — just now
     respecting busy/idle state, which Phase 1 didn't model at all).
  2. BATCH-OPTIMIZED: every `dispatch_interval` seconds of simulated time,
     collect all pending orders + idle riders and solve a joint assignment
     (Hungarian algorithm) minimizing predicted ETA plus a fairness penalty
     that discourages repeatedly giving the worst orders to the same rider.

Both policies use the SAME trained P50 LightGBM model to predict
ETA for a candidate (rider, order) pair — the model is doing the same job
it would do in production, just now feeding a decision instead of only
being evaluated after the fact.

Known simplification: orders.csv doesn't store exact customer lat/lon (only
customer_zone), so rider position after a delivery is approximated as a
jittered point within the customer's zone, using the same zone centroids as
Phase 1. This is an approximation, not the exact original customer location
— acceptable for comparing dispatch POLICIES against each other, since both
policies see the same approximated geometry.

Usage:
    python dispatch_simulator.py --data_dir ../data --model_dir ../models \
        --dispatch_interval 20 --limit_orders 5000
"""

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize import linear_sum_assignment

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Same zone centroids as Phase 1 — needed to approximate rider/customer
# locations that aren't stored explicitly in orders.csv.
LAGOS_ZONES = {
    "Ikeja": (6.6018, 3.3515), "Yaba": (6.5095, 3.3711), "Surulere": (6.5010, 3.3592),
    "Lekki": (6.4432, 3.4726), "Victoria Island": (6.4281, 3.4219), "Ajah": (6.4698, 3.5852),
    "Ikoyi": (6.4541, 3.4316), "Apapa": (6.4550, 3.3599), "Gbagada": (6.5480, 3.3835),
    "Ojota": (6.5764, 3.3789),
}

RNG = np.random.default_rng(123)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def jitter_point(zone_name, spread_km=2.0):
    lat0, lon0 = LAGOS_ZONES[zone_name]
    dlat = RNG.normal(0, spread_km * 0.009)
    dlon = RNG.normal(0, spread_km * 0.009 / np.cos(np.radians(lat0)))
    return lat0 + dlat, lon0 + dlon


# --------------------------------------------------------------------------
# Rider state
# --------------------------------------------------------------------------

@dataclass
class RiderState:
    rider_id: str
    experience_months: int
    lat: float
    lon: float
    busy_until: pd.Timestamp = None       # None == idle
    current_day: int = -1
    daily_order_seq: int = 0              # resets each simulated day, matches Phase 2 feature
    recent_costs: list = field(default_factory=list)  # rolling window, for fairness term

    def is_idle(self, now) -> bool:
        return self.busy_until is None or self.busy_until <= now

    def rolling_avg_cost(self) -> float:
        if not self.recent_costs:
            return 0.0
        return float(np.mean(self.recent_costs[-20:]))  # last 20 trips


def init_riders(riders_df: pd.DataFrame) -> dict:
    riders = {}
    for row in riders_df.itertuples():
        lat, lon = jitter_point(row.home_zone, spread_km=1.0)
        riders[row.rider_id] = RiderState(
            rider_id=row.rider_id,
            experience_months=row.experience_months,
            lat=lat, lon=lon,
        )
    return riders


# --------------------------------------------------------------------------
# Feature assembly for the ETA model — mirrors Phase 2's feature_engineering
# exactly, except rider_to_restaurant_km, rider_experience_months, and
# rider_daily_order_seq are computed live per candidate (rider, order) pair
# instead of being fixed columns in a static CSV.
# --------------------------------------------------------------------------

NUMERIC_COLS = [
    "distance_km", "rider_to_restaurant_km", "hour_sin", "hour_cos",
    "day_of_week", "is_weekend", "is_rain", "item_count",
    "rider_experience_months", "rider_daily_order_seq",
    "rest_hist_avg_prep", "rest_hist_std_prep", "rest_hist_order_count",
    "rest_is_cold_start", "popularity_weight", "same_zone",
    "zone_hour_avg_total",
]
CATEGORICAL_COLS = ["cuisine", "restaurant_zone", "customer_zone"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def order_static_features(order_row, restaurant_row) -> dict:
    """Everything about the order/restaurant that does NOT depend on which
    rider ends up assigned. Precompute once per order, reused across all
    candidate riders in the cost matrix — avoids recomputing the same
    feature_engineering.py-derived quantities inside a hot loop."""
    return {
        "distance_km": order_row.distance_km,
        "hour_sin": np.sin(2 * np.pi * order_row.hour / 24),
        "hour_cos": np.cos(2 * np.pi * order_row.hour / 24),
        "day_of_week": order_row.day_of_week,
        "is_weekend": int(order_row.day_of_week >= 5),
        "is_rain": int(order_row.is_rain),
        "item_count": order_row.item_count,
        "rest_hist_avg_prep": order_row.rest_hist_avg_prep,
        "rest_hist_std_prep": order_row.rest_hist_std_prep,
        "rest_hist_order_count": order_row.rest_hist_order_count,
        "rest_is_cold_start": order_row.rest_is_cold_start,
        "popularity_weight": order_row.popularity_weight,
        "same_zone": int(order_row.restaurant_zone == order_row.customer_zone),
        "zone_hour_avg_total": order_row.zone_hour_avg_total,
        "cuisine": order_row.cuisine,
        "restaurant_zone": order_row.restaurant_zone,
        "customer_zone": order_row.customer_zone,
        "restaurant_lat": restaurant_row.lat,
        "restaurant_lon": restaurant_row.lon,
    }


def candidate_row(static_feats: dict, rider: RiderState, day_idx: int) -> dict:
    """Combine order-static features with rider-dependent ones for a single
    candidate (rider, order) pair."""
    rider_to_restaurant_km = haversine_km(
        rider.lat, rider.lon, static_feats["restaurant_lat"], static_feats["restaurant_lon"]
    )
    next_seq = rider.daily_order_seq + 1 if rider.current_day == day_idx else 1
    row = dict(static_feats)
    row["rider_to_restaurant_km"] = rider_to_restaurant_km
    row["rider_experience_months"] = rider.experience_months
    row["rider_daily_order_seq"] = next_seq
    return row


def predict_eta_batch(model, rows: list) -> np.ndarray:
    df = pd.DataFrame(rows)
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")
    return model.predict(df[FEATURE_COLS], num_iteration=model.best_iteration)


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------

def solve_batch_assignment(cost: np.ndarray):
    """Hungarian algorithm with padding for unequal rider/order counts.
    Padded cells get a very high cost so the solver never prefers a dummy
    match over a real one."""
    n_riders, n_orders = cost.shape
    n = max(n_riders, n_orders)
    padded = np.full((n, n), cost.max() * 2 + 1 if cost.size else 1.0)
    padded[:n_riders, :n_orders] = cost
    row_ind, col_ind = linear_sum_assignment(padded)
    return [(r, c) for r, c in zip(row_ind, col_ind) if r < n_riders and c < n_orders]


def fairness_penalty(rider: RiderState, cohort_avg: float) -> float:
    """Riders currently below the cohort's average recent trip cost (i.e.
    who've been getting easier/faster trips, or fewer of them) get a
    penalty REDUCTION — making them relatively more attractive to the
    solver — nudging future assignments back toward balance."""
    return max(0.0, cohort_avg - rider.rolling_avg_cost())


# --------------------------------------------------------------------------
# Dispatch policies
# --------------------------------------------------------------------------

def run_greedy(orders: pd.DataFrame, restaurants: pd.DataFrame, riders_df: pd.DataFrame, model):
    """Baseline: as each order arrives, assign nearest idle rider immediately.
    No batching, no fairness term — the naive real-time policy."""
    riders = init_riders(riders_df)
    rest_lookup = restaurants.set_index("restaurant_id")

    log_rows = []
    for order in orders.itertuples():
        now = order.order_ts
        day_idx = (now.normalize() - orders["order_ts"].min().normalize()).days
        for r in riders.values():
            if r.current_day != day_idx:
                r.current_day, r.daily_order_seq = day_idx, 0

        idle = [r for r in riders.values() if r.is_idle(now)]
        if not idle:
            log_rows.append({"order_id": order.order_id, "assigned": False})
            continue

        rest_row = rest_lookup.loc[order.restaurant_id]
        static_feats = order_static_features(order, rest_row)
        rows = [candidate_row(static_feats, r, day_idx) for r in idle]
        preds = predict_eta_batch(model, rows)
        best_i = int(np.argmin(preds))
        chosen = idle[best_i]

        eta_pred = float(preds[best_i])
        chosen.busy_until = now + pd.Timedelta(minutes=eta_pred)
        chosen.daily_order_seq += 1
        chosen.recent_costs.append(eta_pred)
        clat, clon = jitter_point(order.customer_zone, spread_km=1.0)
        chosen.lat, chosen.lon = clat, clon

        log_rows.append({"order_id": order.order_id, "rider_id": chosen.rider_id,
                          "assigned": True, "predicted_eta": eta_pred})

    return pd.DataFrame(log_rows), riders


def run_batch_optimized(orders: pd.DataFrame, restaurants: pd.DataFrame, riders_df: pd.DataFrame,
                         model, dispatch_interval_sec: int, w_fairness: float = 0.3):
    """Every dispatch_interval seconds, batch all pending orders + idle
    riders and solve jointly. Falls back to leaving unmatched orders queued
    for the next interval (real systems do exactly this during demand
    spikes)."""
    riders = init_riders(riders_df)
    rest_lookup = restaurants.set_index("restaurant_id")

    orders_sorted = orders.sort_values("order_ts").reset_index(drop=True)
    sim_start, sim_end = orders_sorted["order_ts"].min(), orders_sorted["order_ts"].max()
    step = pd.Timedelta(seconds=dispatch_interval_sec)

    pending = []
    next_order_idx = 0
    log_rows = []
    now = sim_start

    while now <= sim_end or pending:
        day_idx = (now.normalize() - sim_start.normalize()).days
        for r in riders.values():
            if r.current_day != day_idx:
                r.current_day, r.daily_order_seq = day_idx, 0

        while next_order_idx < len(orders_sorted) and orders_sorted.loc[next_order_idx, "order_ts"] <= now:
            pending.append(orders_sorted.iloc[next_order_idx])
            next_order_idx += 1

        idle = [r for r in riders.values() if r.is_idle(now)]

        if pending and idle:
            cohort_avg = float(np.mean([r.rolling_avg_cost() for r in idle])) if idle else 0.0
            cost = np.zeros((len(idle), len(pending)))
            eta_lookup = np.zeros((len(idle), len(pending)))

            for j, order in enumerate(pending):
                rest_row = rest_lookup.loc[order.restaurant_id]
                static_feats = order_static_features(order, rest_row)
                rows = [candidate_row(static_feats, r, day_idx) for r in idle]
                preds = predict_eta_batch(model, rows)
                eta_lookup[:, j] = preds
                for i, r in enumerate(idle):
                    cost[i, j] = preds[i] - w_fairness * fairness_penalty(r, cohort_avg)

            pairs = solve_batch_assignment(cost)
            matched_order_idxs, matched_rider_ids = set(), set()

            for i, j in pairs:
                rider, order = idle[i], pending[j]
                eta_pred = float(eta_lookup[i, j])
                rider.busy_until = now + pd.Timedelta(minutes=eta_pred)
                rider.daily_order_seq += 1
                rider.recent_costs.append(eta_pred)
                clat, clon = jitter_point(order.customer_zone, spread_km=1.0)
                rider.lat, rider.lon = clat, clon

                log_rows.append({"order_id": order.order_id, "rider_id": rider.rider_id,
                                  "assigned": True, "predicted_eta": eta_pred,
                                  "wait_before_assignment_sec": (now - order.order_ts).total_seconds()})
                matched_order_idxs.add(j)
                matched_rider_ids.add(rider.rider_id)

            pending = [o for j, o in enumerate(pending) if j not in matched_order_idxs]

        now += step

    return pd.DataFrame(log_rows), riders


# --------------------------------------------------------------------------
# Comparison metrics
# --------------------------------------------------------------------------

def gini(values: np.ndarray) -> float:
    """Standard Gini coefficient — 0 = perfectly equal trip distribution
    across riders, higher = more concentrated on fewer riders. This is the
    number that quantifies the 'fairness' claim, not just an assertion."""
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def summarize(name: str, log_df: pd.DataFrame):
    assigned = log_df[log_df.get("assigned", True) == True]
    trip_counts = assigned["rider_id"].value_counts().values if "rider_id" in assigned else np.array([])
    unassigned_rate = 1 - len(assigned) / len(log_df) if len(log_df) else 0.0

    log.info(f"\n--- {name} ---")
    log.info(f"Orders: {len(log_df)} | Assigned: {len(assigned)} | Never assigned: {unassigned_rate:.1%}")
    if "predicted_eta" in assigned:
        log.info(f"Mean predicted ETA: {assigned['predicted_eta'].mean():.2f} min | "
                  f"P90: {assigned['predicted_eta'].quantile(0.9):.2f} min")
    if "wait_before_assignment_sec" in assigned:
        log.info(f"Mean queue wait before assignment: {assigned['wait_before_assignment_sec'].mean():.1f} sec")
    if len(trip_counts):
        log.info(f"Rider trip-count Gini: {gini(trip_counts):.3f} "
                  f"(active riders: {len(trip_counts)}, mean trips/rider: {trip_counts.mean():.1f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--model_dir", type=str, default="../models")
    parser.add_argument("--dispatch_interval", type=int, default=20,
                         help="Batch dispatch interval in seconds")
    parser.add_argument("--limit_orders", type=int, default=5000,
                         help="Truncate order stream for faster iteration during development")
    args = parser.parse_args()

    data_dir, model_dir = Path(args.data_dir), Path(args.model_dir)

    log.info("Loading data + trained P50 ETA model...")
    features = pd.read_csv(data_dir / "features.csv", parse_dates=["order_ts"])
    orders_raw = pd.read_csv(data_dir / "orders.csv", parse_dates=["order_ts"])
    restaurants = pd.read_csv(data_dir / "restaurants.csv")
    riders_df = pd.read_csv(data_dir / "riders.csv")
    model = lgb.Booster(model_file=str(model_dir / "eta_model_p50.txt"))

    # Use the TEST split only — the ETA model has never seen this period,
    # which is the honest way to evaluate a dispatch policy built on top of it.
    orders = features[features["split"] == "test"].merge(
        orders_raw[["order_id", "restaurant_id", "hour", "day_of_week", "is_rain", "item_count"]],
        on="order_id", suffixes=("", "_raw"),
    )
    orders = orders.sort_values("order_ts").head(args.limit_orders).reset_index(drop=True)
    log.info(f"Simulating dispatch over {len(orders)} test-period orders "
             f"({orders['order_ts'].min()} -> {orders['order_ts'].max()})")

    log.info("\n=== Running GREEDY baseline (nearest idle rider) ===")
    greedy_log, _ = run_greedy(orders, restaurants, riders_df, model)
    summarize("GREEDY baseline", greedy_log)

    log.info(f"\n=== Running BATCH-OPTIMIZED policy (interval={args.dispatch_interval}s) ===")
    batch_log, _ = run_batch_optimized(orders, restaurants, riders_df, model, args.dispatch_interval)
    summarize("BATCH-OPTIMIZED", batch_log)

    out_dir = data_dir
    greedy_log.to_csv(out_dir / "dispatch_greedy_log.csv", index=False)
    batch_log.to_csv(out_dir / "dispatch_batch_log.csv", index=False)
    log.info(f"\nSaved dispatch logs -> {out_dir}")


if __name__ == "__main__":
    main()