"""
Project A — Phase 1: Synthetic Delivery Operations Data Generator

Simulates a Lagos food-delivery marketplace: restaurants, riders, and orders
over a configurable date range, with realistic operational dynamics baked in:
  - Lunch/dinner demand peaks that vary by day of week
  - Rush-hour traffic multipliers (distinct from raw distance)
  - Random rain days that slow deliveries city-wide
  - Restaurant-specific prep-time distributions (some kitchens are just slower)
  - Rider heterogeneity (experience -> speed) and fatigue (queue backlog -> delay)

The point of building these dynamics in deliberately is that a model trained
on pure-noise data can't demonstrate anything interesting — this generator
creates real, learnable structure (and realistic irreducible noise) so the
downstream ETA model has something genuine to find, and so evaluation
(feature importance, error analysis) tells an honest story.

Usage:
    python generate_synthetic_data.py --days 60 --out_dir ./data
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

RNG = np.random.default_rng(42)

# Approximate real Lagos area centers (lat, lon) — used as hubs for
# restaurant/customer location jitter so distances are geographically
# plausible rather than uniform-random.
LAGOS_ZONES = {
    "Ikeja":     (6.6018, 3.3515),
    "Yaba":      (6.5095, 3.3711),
    "Surulere":  (6.5010, 3.3592),
    "Lekki":     (6.4432, 3.4726),
    "Victoria Island": (6.4281, 3.4219),
    "Ajah":      (6.4698, 3.5852),
    "Ikoyi":     (6.4541, 3.4316),
    "Apapa":     (6.4550, 3.3599),
    "Gbagada":   (6.5480, 3.3835),
    "Ojota":     (6.5764, 3.3789),
}
ZONE_NAMES = list(LAGOS_ZONES.keys())


@dataclass
class SimConfig:
    n_restaurants: int = 150
    n_riders: int = 300
    days: int = 60
    orders_per_day_base: int = 800
    seed: int = 42


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km — vectorized for array inputs."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def jitter_location(zone_name, spread_km=2.0):
    """Random point near a named zone center, spread controlled in km."""
    lat0, lon0 = LAGOS_ZONES[zone_name]
    # ~0.009 degrees latitude per km; longitude scaling adjusted for Lagos latitude
    dlat = RNG.normal(0, spread_km * 0.009)
    dlon = RNG.normal(0, spread_km * 0.009 / np.cos(np.radians(lat0)))
    return lat0 + dlat, lon0 + dlon


def generate_restaurants(cfg: SimConfig) -> pd.DataFrame:
    cuisines = ["Nigerian", "Fast Food", "Continental", "Chinese", "Grills/BBQ", "Pastries"]
    rows = []
    for i in range(cfg.n_restaurants):
        zone = RNG.choice(ZONE_NAMES)
        lat, lon = jitter_location(zone, spread_km=1.5)
        # Some kitchens are structurally slower — a real, persistent effect
        base_prep_mean = RNG.gamma(shape=4, scale=2) + 6  # ~6-25 min typical
        base_prep_std = RNG.uniform(1.5, 4.5)
        popularity = RNG.pareto(a=2.0) + 0.5  # long-tail: a few very popular spots
        rows.append({
            "restaurant_id": f"R{i:04d}",
            "zone": zone,
            "lat": lat,
            "lon": lon,
            "cuisine": RNG.choice(cuisines),
            "base_prep_mean_min": round(base_prep_mean, 1),
            "base_prep_std_min": round(base_prep_std, 1),
            "popularity_weight": round(popularity, 3),
        })
    return pd.DataFrame(rows)


def generate_riders(cfg: SimConfig) -> pd.DataFrame:
    rows = []
    for i in range(cfg.n_riders):
        home_zone = RNG.choice(ZONE_NAMES)
        experience_months = RNG.integers(1, 48)
        # More experienced riders are faster, with diminishing returns
        base_speed_kmh = 22 + 12 * (1 - np.exp(-experience_months / 12)) + RNG.normal(0, 2)
        base_speed_kmh = np.clip(base_speed_kmh, 15, 38)
        rows.append({
            "rider_id": f"D{i:04d}",
            "home_zone": home_zone,
            "experience_months": int(experience_months),
            "base_speed_kmh": round(base_speed_kmh, 1),
        })
    return pd.DataFrame(rows)


def hourly_demand_multiplier(hour: int, dow: int) -> float:
    """Lunch (12-14h) and dinner (18-21h) peaks; weekends shift dinner later
    and add a light weekend-brunch bump."""
    is_weekend = dow >= 5
    lunch = np.exp(-0.5 * ((hour - 13) / 1.3) ** 2) * 2.2
    dinner_center = 20 if is_weekend else 19
    dinner = np.exp(-0.5 * ((hour - dinner_center) / 1.8) ** 2) * 2.6
    late_night = np.exp(-0.5 * ((hour - 23) / 1.5) ** 2) * (1.3 if is_weekend else 0.5)
    baseline = 0.25
    return baseline + lunch + dinner + late_night


def traffic_multiplier(hour: int, dow: int, is_rain: bool) -> float:
    """Multiplier applied to travel time — rush hour + rain compound."""
    is_weekend = dow >= 5
    morning_rush = np.exp(-0.5 * ((hour - 8) / 1.0) ** 2) * (0.0 if is_weekend else 0.9)
    evening_rush = np.exp(-0.5 * ((hour - 17.5) / 1.3) ** 2) * (0.0 if is_weekend else 1.1)
    base = 1.0 + morning_rush + evening_rush
    if is_rain:
        base *= RNG.uniform(1.3, 1.7)  # rain slows Lagos traffic substantially
    return base


def generate_orders(cfg: SimConfig, restaurants: pd.DataFrame, riders: pd.DataFrame) -> pd.DataFrame:
    start_date = pd.Timestamp("2026-01-05")  # a Monday
    rest_weights = restaurants["popularity_weight"].values
    rest_weights = rest_weights / rest_weights.sum()

    # One rain-day flag per calendar day (~20% of days, clustered by season is
    # overkill for this scope — simple iid flag is a reasonable simplification)
    rain_days = {d: RNG.random() < 0.20 for d in range(cfg.days)}

    # Track running order backlog per rider per day to simulate fatigue/queueing
    rider_daily_count = {}

    rows = []
    order_id = 0
    for day in range(cfg.days):
        date = start_date + pd.Timedelta(days=day)
        dow = date.dayofweek
        is_rain = rain_days[day]
        day_order_count = int(cfg.orders_per_day_base * RNG.uniform(0.85, 1.15))

        # Sample hour for each order using the demand curve as weights
        hours = np.arange(24)
        hour_weights = np.array([hourly_demand_multiplier(h, dow) for h in hours])
        hour_weights = hour_weights / hour_weights.sum()
        order_hours = RNG.choice(hours, size=day_order_count, p=hour_weights)

        for hour in order_hours:
            minute = RNG.integers(0, 60)
            order_ts = date + pd.Timedelta(hours=int(hour), minutes=int(minute))

            rest_idx = RNG.choice(len(restaurants), p=rest_weights)
            restaurant = restaurants.iloc[rest_idx]

            # Customer located in same or nearby zone (most orders are local —
            # real delivery marketplaces bias matching toward proximity)
            if RNG.random() < 0.82:
                cust_zone = restaurant["zone"]
            else:
                cust_zone = RNG.choice(ZONE_NAMES)
            cust_lat, cust_lon = jitter_location(cust_zone, spread_km=2.0)

            distance_km = haversine_km(restaurant["lat"], restaurant["lon"], cust_lat, cust_lon)
            distance_km = max(distance_km, 0.3)

            # Assign nearest-ish rider: sample a small candidate pool, pick
            # shortest distance from rider's home zone centroid (simplification
            # for prep-time-vs-assignment realism; true optimization is Phase 4)
            candidates = riders.sample(n=min(8, len(riders)), random_state=None)
            cand_lat, cand_lon = zip(*[LAGOS_ZONES[z] for z in candidates["home_zone"]])
            cand_dist = haversine_km(np.array(cand_lat), np.array(cand_lon), restaurant["lat"], restaurant["lon"])
            rider = candidates.iloc[np.argmin(cand_dist)]
            rider_to_restaurant_km = float(np.min(cand_dist))

            # Prep time: restaurant baseline + load effect (busier hour = slower kitchen)
            load_factor = 1.0 + 0.25 * hourly_demand_multiplier(hour, dow) / 2.6
            prep_time = max(3.0, RNG.normal(restaurant["base_prep_mean_min"] * load_factor,
                                             restaurant["base_prep_std_min"]))

            # Rider fatigue: nth order today for this rider adds small cumulative delay
            key = (rider["rider_id"], day)
            rider_daily_count[key] = rider_daily_count.get(key, 0) + 1
            fatigue_delay = min(rider_daily_count[key] * 0.4, 8.0)

            traffic_mult = traffic_multiplier(hour, dow, is_rain)
            effective_speed = rider["base_speed_kmh"] / traffic_mult

            travel_to_restaurant_min = (rider_to_restaurant_km / effective_speed) * 60
            travel_to_customer_min = (distance_km / effective_speed) * 60

            noise = RNG.normal(0, 2.5)  # irreducible randomness (parking, lift wait, etc.)
            actual_total_min = (prep_time + travel_to_restaurant_min
                                 + travel_to_customer_min + fatigue_delay + noise)
            actual_total_min = max(actual_total_min, 5.0)

            item_count = RNG.integers(1, 6)

            rows.append({
                "order_id": f"O{order_id:07d}",
                "order_ts": order_ts,
                "hour": int(hour),
                "day_of_week": int(dow),
                "is_rain": is_rain,
                "restaurant_id": restaurant["restaurant_id"],
                "restaurant_zone": restaurant["zone"],
                "customer_zone": cust_zone,
                "distance_km": round(distance_km, 3),
                "rider_id": rider["rider_id"],
                "rider_to_restaurant_km": round(rider_to_restaurant_km, 3),
                "rider_experience_months": int(rider["experience_months"]),
                "item_count": int(item_count),
                "rider_daily_order_seq": rider_daily_count[key],
                "prep_time_min": round(prep_time, 2),
                "actual_total_min": round(actual_total_min, 2),
            })
            order_id += 1

    df = pd.DataFrame(rows).sort_values("order_ts").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--n_restaurants", type=int, default=150)
    parser.add_argument("--n_riders", type=int, default=300)
    parser.add_argument("--orders_per_day", type=int, default=800)
    parser.add_argument("--out_dir", type=str, default="./data")
    args = parser.parse_args()

    cfg = SimConfig(n_restaurants=args.n_restaurants, n_riders=args.n_riders,
                     days=args.days, orders_per_day_base=args.orders_per_day)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Generating restaurants...")
    restaurants = generate_restaurants(cfg)
    log.info("Generating riders...")
    riders = generate_riders(cfg)
    log.info(f"Generating {cfg.days} days of orders (~{cfg.orders_per_day_base}/day)...")
    orders = generate_orders(cfg, restaurants, riders)

    restaurants.to_csv(out_dir / "restaurants.csv", index=False)
    riders.to_csv(out_dir / "riders.csv", index=False)
    orders.to_csv(out_dir / "orders.csv", index=False)

    log.info(f"Saved {len(restaurants)} restaurants, {len(riders)} riders, {len(orders)} orders -> {out_dir}")

    # Sanity checks — confirm the dynamics we built in are actually visible
    log.info("\n--- Sanity checks ---")
    log.info(f"Mean ETA by hour (should peak ~13h and ~19-20h):\n"
              f"{orders.groupby('hour')['actual_total_min'].mean().round(1).to_dict()}")
    log.info(f"Mean ETA rain vs no rain: "
              f"{orders.groupby('is_rain')['actual_total_min'].mean().round(2).to_dict()}")
    log.info(f"Overall ETA stats: mean={orders['actual_total_min'].mean():.1f} min, "
              f"p50={orders['actual_total_min'].median():.1f}, "
              f"p90={orders['actual_total_min'].quantile(0.9):.1f}, "
              f"p99={orders['actual_total_min'].quantile(0.99):.1f}")


if __name__ == "__main__":
    main()
