# Real-Time ETA Prediction & Delivery Logistics Platform

An end-to-end machine learning system for food delivery ETA prediction, demand forecasting, and rider-order dispatch simulation — built to mirror the production ML surface area of a delivery logistics company (ETA prediction, demand forecasting, and dispatch/logistics optimization).

**Live API:** [https://real-time-eta-prediction-delivery.onrender.com](https://real-time-eta-prediction-delivery.onrender.com)
**Interactive docs (Swagger UI):** [/docs](https://real-time-eta-prediction-delivery.onrender.com/docs)

> Hosted on Render's free tier — the instance spins down after inactivity, so the first request after idle may take 30–60 seconds to respond.

---

## What this does

Given a delivery request (distance, restaurant history, time of day, cuisine, zone, etc.), the service predicts a **P50 and P90 ETA range** in minutes using two independently trained LightGBM quantile regression models, and serves that prediction behind a production-style FastAPI layer.

Beyond the core prediction, this project also includes:

- **Demand forecasting** — a LightGBM-based model for predicting order volume by zone/hour
- **Dispatch simulation** — rider-to-order assignment using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`), comparing greedy vs. optimal batch assignment strategies
- **Drift monitoring** — a script that compares live prediction inputs/outputs (logged per-request) against the training-time baseline distribution

## Why these design choices

- **Quantile regression (P50/P90), not a single point estimate** — delivery ETAs are inherently uncertain; giving a range is more honest and more useful to a customer-facing product than a single number.
- **Models loaded once at import time, not per-request** — reloading a LightGBM booster on every call would be a significant, easy-to-miss latency bug in a real system.
- **Unseen categorical values are handled gracefully** — a new zone or cuisine type added after training won't crash the service; it's coerced to an `UNKNOWN` bucket so the model still returns a usable prediction. Same for cold-start restaurants (zero order history).
- **Every prediction is logged** to a JSONL file — the minimum viable version of a feature/prediction store, and what the drift monitor reads.
- **Validation before the model ever sees input** — Pydantic schemas reject malformed requests with a clear 422 instead of an opaque model exception.

## API

### `GET /health`

Returns service status and whether both models are loaded.

```json
{ "status": "ok" }
```

### `POST /predict_eta`

Predicts a delivery ETA range.

**Request body:**

```json
{
  "distance_km": 3.5,
  "rider_to_restaurant_km": 1.2,
  "hour": 18,
  "day_of_week": 5,
  "is_rain": false,
  "item_count": 2,
  "rider_experience_months": 12,
  "rider_daily_order_seq": 3,
  "rest_hist_avg_prep": 15.0,
  "rest_hist_std_prep": 3.0,
  "rest_hist_order_count": 200,
  "popularity_weight": 1.1,
  "cuisine": "Nigerian",
  "restaurant_zone": "Lekki",
  "customer_zone": "Lekki"
}
```

**Response:**

```json
{
  "eta_p50_min": 23.9,
  "eta_p90_min": 29.4,
  "eta_range_display": "24-29 min",
  "model_latency_ms": 12.4
}
```

Try it live:

```bash
curl -X POST https://real-time-eta-prediction-delivery.onrender.com/predict_eta \
  -H "Content-Type: application/json" \
  -d '{"distance_km": 3.5, "rider_to_restaurant_km": 1.2, "hour": 18, "day_of_week": 5, "is_rain": false, "item_count": 2, "rider_experience_months": 12, "rider_daily_order_seq": 3, "rest_hist_avg_prep": 15.0, "rest_hist_std_prep": 3.0, "rest_hist_order_count": 200, "popularity_weight": 1.1, "cuisine": "Nigerian", "restaurant_zone": "Lekki", "customer_zone": "Lekki"}'
```

Full interactive documentation with a "try it out" UI is available at [`/docs`](https://real-time-eta-prediction-delivery.onrender.com/docs).

## Project structure

```
.
├── src/
│   ├── generate_synthetic_data.py   # Synthetic delivery order dataset generation
│   ├── feature_engineering.py       # Feature pipeline (time, zone, restaurant history features)
│   ├── train_model.py               # LightGBM P50/P90 quantile model training + evaluation
│   ├── demand_forecasting.py        # Zone/hour order-volume forecasting model
│   └── dispatch_simulator.py        # Rider-order assignment simulation (Hungarian algorithm)
├── serving/
│   └── app.py                       # FastAPI serving layer
├── test/
│   └── test_serving.py              # API test suite (health, validation, edge cases)
├── monitoring/
│   └── drift_check.py               # Compares live prediction logs against training baseline
├── models/                          # Trained LightGBM boosters (.txt)
├── data/                            # Generated/synthetic datasets and simulation logs
├── logs/                            # Per-request prediction logs (JSONL)
├── Dockerfile
└── requirements.txt
```

## Running locally

```bash
git clone https://github.com/Iamjuhwan/Real-Time-ETA-Prediction-Delivery-Logistics-Platform.git
cd Real-Time-ETA-Prediction-Delivery-Logistics-Platform
pip install -r requirements.txt
uvicorn serving.app:app --reload
```

The service will be available at `http://localhost:8000`, with docs at `http://localhost:8000/docs`.

## Running with Docker

```bash
docker build -t eta-prediction-service .
docker run -p 8000:8000 eta-prediction-service
```

## Testing

```bash
python -m pytest test/ -v
```

The test suite covers health checks, valid predictions, input validation (rejecting invalid distance/hour values), and graceful handling of unseen cuisines, unseen zones, and cold-start restaurants.

## Deployment

Deployed on [Render](https://render.com) directly from this repo's `Dockerfile`, with auto-deploy on push to `main`.

## Tech stack

Python · FastAPI · LightGBM · scikit-learn · scipy · pandas · NumPy · Docker · pytest
