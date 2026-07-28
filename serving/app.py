"""
 ETA Model Serving Layer

Serves the LightGBM P50/P90 models behind a FastAPI endpoint.

Design choices worth being able to explain:
  - Models loaded ONCE at import time (module-level), not per-request —
    reloading a LightGBM booster on every call would be a major,
    easy-to-miss latency bug in a real system. This used to be wired up
    via @app.on_event("startup"), but that hook only fires when something
    actually drives the app's lifespan (uvicorn, or a test using
    `with TestClient(app) as client:`). A plain `TestClient(app)` with no
    `with` block silently skipped model loading, leaving every
    /predict_eta call 503ing. Calling load_models() directly at import
    time makes "loaded once at startup" true regardless of how the app
    is invoked.
  - Pydantic request/response schemas — validation happens before the model
    ever sees the input, so malformed requests fail fast with a clear 422
    instead of an opaque model exception.
  - Every prediction is logged (features + output + latency) to a JSONL file
    that `monitoring/drift_check.py` reads later. This is the minimum viable
    version of what a feature/prediction store does in production.
  - Categorical feature values not seen during training are coerced to a
    known "UNKNOWN" bucket rather than left to crash LightGBM's categorical
    handling — a realistic production hazard (a new zone or cuisine type
    added after the model was trained).

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
LOG_PATH = Path(__file__).parent.parent / "logs" / "predictions.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

KNOWN_CUISINES = ["Nigerian", "Fast Food", "Continental", "Chinese", "Grills/BBQ", "Pastries"]
KNOWN_ZONES = ["Ikeja", "Yaba", "Surulere", "Lekki", "Victoria Island", "Ajah",
               "Ikoyi", "Apapa", "Gbagada", "Ojota"]

FEATURE_COLS = [
    "distance_km", "rider_to_restaurant_km", "hour_sin", "hour_cos",
    "day_of_week", "is_weekend", "is_rain", "item_count",
    "rider_experience_months", "rider_daily_order_seq",
    "rest_hist_avg_prep", "rest_hist_std_prep", "rest_hist_order_count",
    "rest_is_cold_start", "popularity_weight", "same_zone",
    "zone_hour_avg_total", "cuisine", "restaurant_zone", "customer_zone",
]
CATEGORICAL_COLS = ["cuisine", "restaurant_zone", "customer_zone"]

app = FastAPI(title="Chowdeck-style ETA Prediction Service", version="1.0.0")

# --- Models loaded once at import time, not per-request ---
_model_p50: Optional[lgb.Booster] = None
_model_p90: Optional[lgb.Booster] = None


def load_models():
    global _model_p50, _model_p90

    p50_path = MODEL_DIR / "eta_model_p50.txt"
    p90_path = MODEL_DIR / "eta_model_p90.txt"

    log.info("MODEL_DIR: %s", MODEL_DIR)
    log.info("P50 exists: %s %s", p50_path.exists(), p50_path)
    log.info("P90 exists: %s %s", p90_path.exists(), p90_path)

    try:
        _model_p50 = lgb.Booster(model_file=str(p50_path))
        log.info("Loaded P50")
    except Exception as e:
        log.error("Failed to load P50: %r", e)
        raise

    try:
        _model_p90 = lgb.Booster(model_file=str(p90_path))
        log.info("Loaded P90")
    except Exception as e:
        log.error("Failed to load P90: %r", e)
        raise


# Load at import time — guarantees this runs whether the app is started via
# uvicorn, a plain TestClient(app), or anything else that imports this module.
load_models()


class ETARequest(BaseModel):
    distance_km: float = Field(..., gt=0, description="Restaurant-to-customer distance in km")
    rider_to_restaurant_km: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Monday ... 6=Sunday")
    is_rain: bool = False
    item_count: int = Field(..., ge=1)
    rider_experience_months: int = Field(..., ge=0)
    rider_daily_order_seq: int = Field(..., ge=1, description="This rider's Nth order today")
    rest_hist_avg_prep: float = Field(..., gt=0, description="Restaurant's historical avg prep time (min)")
    rest_hist_std_prep: float = Field(..., ge=0)
    rest_hist_order_count: int = Field(..., ge=0)
    popularity_weight: float = Field(..., ge=0)
    cuisine: str
    restaurant_zone: str
    customer_zone: str


class ETAResponse(BaseModel):
    eta_p50_min: float
    eta_p90_min: float
    eta_range_display: str
    model_latency_ms: float


def _safe_categorical(value: str, known_values: list) -> str:
    """Guard against unseen categories at serving time — a new zone or
    cuisine added after training shouldn't crash the request."""
    return value if value in known_values else "UNKNOWN"


def build_feature_row(req: ETARequest) -> pd.DataFrame:
    row = {
        "distance_km": req.distance_km,
        "rider_to_restaurant_km": req.rider_to_restaurant_km,
        "hour_sin": np.sin(2 * np.pi * req.hour / 24),
        "hour_cos": np.cos(2 * np.pi * req.hour / 24),
        "day_of_week": req.day_of_week,
        "is_weekend": int(req.day_of_week >= 5),
        "is_rain": int(req.is_rain),
        "item_count": req.item_count,
        "rider_experience_months": req.rider_experience_months,
        "rider_daily_order_seq": req.rider_daily_order_seq,
        "rest_hist_avg_prep": req.rest_hist_avg_prep,
        "rest_hist_std_prep": req.rest_hist_std_prep,
        "rest_hist_order_count": req.rest_hist_order_count,
        "rest_is_cold_start": int(req.rest_hist_order_count == 0),
        "popularity_weight": req.popularity_weight,
        "same_zone": int(req.restaurant_zone == req.customer_zone),
        "zone_hour_avg_total": req.rest_hist_avg_prep,  # fallback proxy if not separately provided
        "cuisine": _safe_categorical(req.cuisine, KNOWN_CUISINES),
        "restaurant_zone": _safe_categorical(req.restaurant_zone, KNOWN_ZONES),
        "customer_zone": _safe_categorical(req.customer_zone, KNOWN_ZONES),
    }
    df = pd.DataFrame([row])
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("category")
    return df[FEATURE_COLS]


def log_prediction(req: ETARequest, p50: float, p90: float, latency_ms: float):
    """Append-only JSONL log — the minimum viable prediction store. Read by
    monitoring/drift_check.py to compare live input/output distributions
    against the training-time baseline."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request": req.model_dump(),
        "eta_p50_min": p50,
        "eta_p90_min": p90,
        "latency_ms": latency_ms,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    ok = _model_p50 is not None and _model_p90 is not None
    return {"status": "ok" if ok else "models not loaded"}


@app.post("/predict_eta", response_model=ETAResponse)
def predict_eta(req: ETARequest):
    if _model_p50 is None or _model_p90 is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    t0 = time.perf_counter()
    X = build_feature_row(req)
    try:
        p50 = float(_model_p50.predict(X, num_iteration=_model_p50.best_iteration)[0])
        p90 = float(_model_p90.predict(X, num_iteration=_model_p90.best_iteration)[0])
    except Exception as e:
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # P90 should never be below P50 by construction, but quantile models
    # trained independently can occasionally cross — guard against a
    # nonsensical response rather than silently serving it
    p90 = max(p90, p50)
    latency_ms = (time.perf_counter() - t0) * 1000

    log_prediction(req, p50, p90, latency_ms)

    return ETAResponse(
        eta_p50_min=round(p50, 1),
        eta_p90_min=round(p90, 1),
        eta_range_display=f"{round(p50)}-{round(p90)} min",
        model_latency_ms=round(latency_ms, 2),
    )