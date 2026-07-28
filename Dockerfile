FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (separate layer) so code changes don't bust the
# dependency cache on every rebuild — a small but real thing that matters
# once you're rebuilding images many times a day in CI.
COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ ./serving/
COPY models/ ./models/

EXPOSE 8000

# --workers 1 for local/dev; in real deployment this would be tuned to CPU
# count and fronted by a process manager (gunicorn) — noted here rather
# than silently hard-coded as if it were a production-ready value.
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
