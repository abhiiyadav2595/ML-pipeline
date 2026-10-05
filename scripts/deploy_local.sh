#!/usr/bin/env bash
set -e

echo "=========================================================="
echo " Launching ML Model Deployment & Monitoring Platform"
echo "=========================================================="

# 1. Check Docker status
echo -e "\n[1/4] Checking Docker environment..."
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker daemon is not running."
    exit 1
fi
echo "Docker daemon is active."

# 2. Check local artifacts
echo -e "\n[2/4] Verifying model artifacts..."
if [ ! -f "artifacts/model.joblib" ]; then
    echo "Artifacts not found. Training baseline model..."
    python3 -m src.model.train
else
    echo "Model artifacts verified."
fi

# 3. Start Docker Compose Stack
echo -e "\n[3/4] Launching containers via Docker Compose..."
docker compose up -d --build

# 4. Display Status & URLs
echo -e "\n[4/4] Verifying running services..."
sleep 5
docker compose ps

echo -e "\n=========================================================="
echo " MLOps Platform Successfully Deployed!"
echo "=========================================================="
echo " ML Serving API (Swagger UI):  http://localhost:8000/docs"
echo " API Health Probe:             http://localhost:8000/health"
echo " Prometheus Metrics Endpoint:  http://localhost:8000/metrics"
echo " Visual Drift HTML Report:     http://localhost:8000/drift/report"
echo " Prometheus Server:            http://localhost:9090"
echo " Grafana Dashboard:            http://localhost:3000 (admin / admin)"
echo -e "==========================================================\n"
