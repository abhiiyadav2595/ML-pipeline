# PowerShell Local Deployment Script for MLOps Stack
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Launching ML Model Deployment & Monitoring Platform" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Docker status
Write-Host "`n[1/4] Checking Docker environment..." -ForegroundColor Yellow
docker info > $null 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Docker daemon is not running. Please start Docker Desktop." -ForegroundColor Red
    exit 1
}
Write-Host "Docker daemon is active." -ForegroundColor Green

# 2. Check local artifacts
Write-Host "`n[2/4] Verifying model artifacts..." -ForegroundColor Yellow
if (-not (Test-Path "artifacts/model.joblib")) {
    Write-Host "Artifacts not found. Training baseline model..." -ForegroundColor Yellow
    & .venv/Scripts/python.exe -m src.model.train
} else {
    Write-Host "Model artifacts verified (artifacts/model.joblib, reference_data.csv)." -ForegroundColor Green
}

# 3. Start Docker Compose Stack
Write-Host "`n[3/4] Launching containers via Docker Compose..." -ForegroundColor Yellow
docker compose up -d --build

# 4. Display Status & URLs
Write-Host "`n[4/4] Verifying running services..." -ForegroundColor Yellow
Start-Sleep -Seconds 5
docker compose ps

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " MLOps Platform Successfully Deployed!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " ML Serving API (Swagger UI):  http://localhost:8000/docs" -ForegroundColor White
Write-Host " API Health Probe:             http://localhost:8000/health" -ForegroundColor White
Write-Host " Prometheus Metrics Endpoint:  http://localhost:8000/metrics" -ForegroundColor White
Write-Host " Visual Drift HTML Report:     http://localhost:8000/drift/report" -ForegroundColor White
Write-Host " Prometheus Server:            http://localhost:9090" -ForegroundColor White
Write-Host " Grafana Dashboard:            http://localhost:3000 (admin / admin)" -ForegroundColor White
Write-Host "==========================================================`n" -ForegroundColor Green

Write-Host "To simulate live drift traffic, run:" -ForegroundColor Cyan
Write-Host "  python scripts/simulate_traffic.py --mode drift --count 150`n" -ForegroundColor White
