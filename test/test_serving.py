"""
Test suite for the ETA serving API.

Run locally (after `pip install -r requirements.txt` and training the
some of the models so serving/app.py can load them at import time):

    pytest tests/test_serving.py -v

 
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "serving"))
from app import app  # noqa: E402

client = TestClient(app)

VALID_PAYLOAD = {
    "distance_km": 3.5,
    "rider_to_restaurant_km": 1.2,
    "hour": 18,
    "day_of_week": 5,
    "is_rain": False,
    "item_count": 2,
    "rider_experience_months": 12,
    "rider_daily_order_seq": 3,
    "rest_hist_avg_prep": 15.0,
    "rest_hist_std_prep": 3.0,
    "rest_hist_order_count": 200,
    "popularity_weight": 1.1,
    "cuisine": "Nigerian",
    "restaurant_zone": "Lekki",
    "customer_zone": "Lekki",
}


def test_health_check():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid_request():
    resp = client.post("/predict_eta", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "eta_p50_min" in body and "eta_p90_min" in body
    assert body["eta_p50_min"] > 0
    # P90 must never be below P50 — the guard rail added in app.py for
    # when independently-trained quantile models cross
    assert body["eta_p90_min"] >= body["eta_p50_min"]


def test_predict_rejects_negative_distance():
    bad = {**VALID_PAYLOAD, "distance_km": -2.0}
    resp = client.post("/predict_eta", json=bad)
    assert resp.status_code == 422  # Pydantic validation should catch this before the model ever runs


def test_predict_rejects_invalid_hour():
    bad = {**VALID_PAYLOAD, "hour": 25}
    resp = client.post("/predict_eta", json=bad)
    assert resp.status_code == 422


def test_predict_handles_unseen_cuisine_gracefully():
    """A cuisine type that didn't exist at training time (e.g. a newly
    onboarded restaurant category) must not crash the request — it should
    be coerced to the UNKNOWN bucket and still return a prediction."""
    novel = {**VALID_PAYLOAD, "cuisine": "Molecular Gastronomy"}
    resp = client.post("/predict_eta", json=novel)
    assert resp.status_code == 200
    assert resp.json()["eta_p50_min"] > 0


def test_predict_handles_unseen_zone_gracefully():
    novel = {**VALID_PAYLOAD, "restaurant_zone": "Epe", "customer_zone": "Epe"}
    resp = client.post("/predict_eta", json=novel)
    assert resp.status_code == 200


def test_predict_cold_start_restaurant():
    """rest_hist_order_count=0 should be handled (new restaurant, no
    history yet) rather than producing a nonsensical or failed prediction."""
    cold_start = {**VALID_PAYLOAD, "rest_hist_order_count": 0}
    resp = client.post("/predict_eta", json=cold_start)
    assert resp.status_code == 200


def test_eta_range_display_format():
    resp = client.post("/predict_eta", json=VALID_PAYLOAD)
    display = resp.json()["eta_range_display"]
    assert "-" in display and "min" in display