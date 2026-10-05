# Production ML Model Deployment & Real-Time Drift Monitoring Platform

[![CI/CD Pipeline](https://github.com/abhiiyadav2595/ML-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/abhiiyadav2595/ML-pipeline/actions)
[![Vercel Deployment](https://img.shields.io/badge/Deployed%20on-Vercel-black.svg?logo=vercel)](https://vercel.com/)
[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-20.10+-2496ED.svg?logo=docker)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-v1.28+-326CE5.svg?logo=kubernetes)](https://kubernetes.io/)
[![Prometheus](https://img.shields.io/badge/Prometheus-v2.51+-E6522C.svg?logo=prometheus)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-v10.4+-F46800.svg?logo=grafana)](https://grafana.com/)
[![Coverage](https://img.shields.io/badge/coverage-81%25-brightgreen.svg)]()

An enterprise-grade, end-to-end Machine Learning deployment and operational monitoring system. The platform serves a financial credit default and risk prediction model with **sub-millisecond inference**, automated **CI/CD quality gates**, **Kubernetes orchestration with auto-scaling**, and **real-time statistical data & prediction drift detection** instrumented with **Prometheus** and pre-configured **Grafana** dashboards.

---

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Statistical Drift Detection Engine](#statistical-drift-detection-engine)
3. [Prometheus Metrics & Grafana Dashboards](#prometheus-metrics--grafana-dashboards)
4. [Kubernetes & Containerization](#kubernetes--containerization)
5. [CI/CD Practices](#cicd-practices)
6. [Repository Structure](#repository-structure)
7. [Quickstart Guide](#quickstart-guide)
8. [Simulating Live Traffic & Inducing Drift](#simulating-live-traffic--inducing-drift)
9. [API Documentation](#api-documentation)
10. [Test Suite & Verification](#test-suite--verification)

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Client Traffic
        A[Web / Mobile Clients] -->|REST / JSON| B[Kubernetes Ingress / Load Balancer]
        Sim[Traffic & Drift Simulator] -->|Baseline / Drift Requests| B
    end

    subgraph Kubernetes Cluster (Namespace: ml-serving)
        B --> Svc[Kubernetes Service: ClusterIP]
        Svc --> Pod1[ML Serving Pod 1: FastAPI]
        Svc --> Pod2[ML Serving Pod 2: FastAPI]
        Svc --> Pod3[ML Serving Pod 3: FastAPI]
        
        HPA[Horizontal Pod Autoscaler] -.->|Scales Replicas 2 - 8| Pods[Deployment: ml-serving]
        
        subgraph Inside ML Pod
            API[FastAPI Application]
            Pipeline[Scikit-Learn Model Pipeline]
            Buffer[Sliding Window Buffer: 1000 items]
            Detector[Statistical Drift Detector Engine]
            PromClient[Prometheus Client Registry]
            
            API --> Pipeline
            API --> Buffer
            Buffer --> Detector
            Detector -->|Updates Gauges| PromClient
        end
    end

    subgraph Observability Stack
        PromClient -->|Scraped every 5s on /metrics| Prom[Prometheus Server]
        Prom -->|Evaluates Alert Rules| Alerts[Alertmanager / Notification Channels]
        Prom -->|PromQL Datasource| Graf[Grafana Dashboards]
        User[ML Engineer / SRE] -->|Visual Inspection| Graf
        User -->|Interactive HTML Drift Report| Rep[http://localhost:8000/drift/report]
    end
```

---

## Statistical Drift Detection Engine

Machine learning models degrade in production due to **Covariate Shift** (changes in input feature distributions $P(X)$) and **Concept Drift** (changes in the relationship between features and outcomes $P(Y|X)$).

This platform implements a dual-layer statistical monitoring engine operating over a thread-safe sliding window buffer:

### 1. Continuous Numerical Features (Two-Sample Kolmogorov-Smirnov Test)
For numerical features (`income`, `credit_score`, `debt_to_income`, `loan_amount`, `age`):
- **Algorithm**: Two-sample Kolmogorov-Smirnov (KS) test (`scipy.stats.ks_2samp`) comparing sample window against the training reference baseline.
- **Metric**: KS-statistic $D = \sup_x |F_{\text{ref}}(x) - F_{\text{current}}(x)|$.
- **Detection Criteria**: Feature is flagged as drifted when both $p\text{-value} < 0.05$ and $D \ge 0.12$ (to avoid false positives on small sample noise).
- **Secondary Metric**: Earth Mover's (Wasserstein) Distance measuring geometric shift magnitude.

### 2. Categorical Features (Population Stability Index - PSI)
For categorical features (`employment_status`, `loan_purpose`):
- **Algorithm**: Population Stability Index (PSI):
  $$\text{PSI} = \sum_{i=1}^{k} \left( P_i - Q_i \right) \times \ln\left(\frac{P_i}{Q_i}\right)$$
  *(where $P_i$ is current proportion and $Q_i$ is reference proportion with Laplace smoothing $\epsilon = 10^{-4}$)*.
- **Thresholds**:
  - $\text{PSI} < 0.10$: **Stable** (No action needed).
  - $0.10 \le \text{PSI} < 0.20$: **Moderate Shift** (Warning).
  - $\text{PSI} \ge 0.20$: **Significant Drift** (Alert triggered).

### 3. Prediction Probability Drift (Concept Drift Proxy)
Monitors the distribution of model predicted probabilities $P(\hat{Y}=1)$ against reference baseline predictions using combined KS-test and PSI. A shift here indicates the model is classifying significantly more or fewer positive cases.

### 4. Dataset Drift Decision
If $\ge 33\%$ of monitored features exhibit statistically significant drift, the platform asserts:
$$\text{model\_dataset\_drift\_detected} = 1$$
This instantly raises an alert in Prometheus and flashes the Grafana dashboard status panel to **RED**.

---

## Prometheus Metrics & Grafana Dashboards

### Exposed Prometheus Metrics (`/metrics`)

| Metric Name | Type | Labels | Description |
|---|---|---|---|
| `model_prediction_requests_total` | Counter | `endpoint`, `status`, `prediction_class` | Total count of inference requests processed |
| `model_prediction_latency_seconds` | Histogram | `endpoint` | End-to-end inference latency with tuned buckets |
| `model_requests_in_flight` | Gauge | - | Active concurrent requests being processed |
| `model_prediction_score_distribution` | Histogram | `le` | Binned distribution of predicted default probabilities |
| `model_predictions_by_class_total` | Counter | `prediction_class` | Total outputs classified by class (`default` vs `non_default`) |
| `model_feature_drift_score` | Gauge | `feature`, `test_type` | Real-time drift test statistic (KS statistic or PSI value) |
| `model_feature_drift_pvalue` | Gauge | `feature` | Statistical p-value from Kolmogorov-Smirnov test |
| `model_feature_drift_detected` | Gauge | `feature` | Binary indicator (1 if drifted, 0 otherwise) |
| `model_dataset_drift_detected` | Gauge | - | Overall dataset drift alert flag (1 or 0) |
| `model_prediction_drift_detected` | Gauge | - | Binary indicator for output probability distribution drift |
| `model_drifted_features_ratio` | Gauge | - | Percentage of monitored features in drift state |
| `model_drift_evaluation_sample_size`| Gauge | - | Sample count currently in sliding window buffer |

### Prometheus Alerting Rules (`prometheus/alert.rules.yml`)
- `ModelDatasetDriftDetected`: Fires if `model_dataset_drift_detected == 1` for 15 seconds.
- `ModelPredictionDriftDetected`: Fires if `model_prediction_drift_detected == 1` for 20 seconds.
- `HighModelPredictionLatency`: Fires if P95 latency exceeds 200ms for 30 seconds.
- `HighInferenceErrorRate`: Fires if error percentage exceeds 5% over 1 minute.
- `MLModelAPIDown`: Fires if the model serving API becomes unreachable.

### Provisioned Grafana Dashboard (`grafana/dashboards/model_monitoring.json`)
The pre-configured Grafana dashboard includes:
1. **Live Dataset Drift Status Stat Panel**: Green (`HEALTHY`) / Red (`DRIFT DETECTED`).
2. **Total Inference Volume & Throughput Gauge**: Real-time request counters.
3. **P95 Latency & SLA Monitor**: Latency tracking against SLA bounds.
4. **Drifted Features Ratio**: Proportion of drifted features.
5. **Real-time Throughput (RPS) by Class**: Live line chart of defaults vs non-defaults.
6. **Latency Percentiles (P50, P90, P99)**: Sub-millisecond latency distribution over time.
7. **Feature Drift Scores Timeline**: Dynamic KS-statistic & PSI values per feature.
8. **Kolmogorov-Smirnov P-Values**: Tracks statistical significance (<0.05 highlighted).
9. **Live Feature Drift Status Table**: Instant summary matrix of all input dimensions.
10. **Prediction Probability Distribution**: Real-time histogram of output probabilities.

---

## Kubernetes & Containerization

### Hardened Production Container (`Dockerfile`)
- **Multi-Stage Build**: Separates compile-time dependencies from the runtime image to reduce attack surface and minimize image footprint.
- **Principle of Least Privilege**: Runs under an unprivileged non-root user (`mluser`, UID `10001`).
- **Health Probing**: Embedded container `HEALTHCHECK` with curl validation against `/health`.

### Kubernetes Production Manifests (`k8s/`)
- `namespace.yaml`: Isolated namespace `ml-serving`.
- `deployment.yaml`:
  - 3 replicas with zero-downtime `RollingUpdate` (`maxSurge: 1`, `maxUnavailable: 0`).
  - Strict resource `requests` (CPU 250m, Mem 512Mi) and `limits` (CPU 1000m, Mem 1Gi).
  - Configured `livenessProbe` and `readinessProbe` with appropriate start periods.
  - Security context dropping all Linux capabilities (`drop: [ALL]`).
- `service.yaml`: ClusterIP service annotated for Prometheus auto-discovery (`prometheus.io/scrape: "true"`).
- `hpa.yaml`: HorizontalPodAutoscaler automatically scaling from 2 to 8 pods based on CPU (70%) and memory (80%) thresholds.
- `ingress.yaml`: NGINX ingress routing external traffic to the service.
- `monitoring/prometheus-k8s.yaml` & `monitoring/grafana-k8s.yaml`: Standalone cluster monitoring stack.
- `kustomization.yaml`: Kustomize configuration enabling clean overlay management across environments (`dev`, `staging`, `prod`).

---

## CI/CD Practices

The platform includes production GitHub Actions workflows implementing industry best practices:

### Continuous Integration (`.github/workflows/ci.yml`)
1. **Code Quality & Linting**: Runs `ruff` for ultra-fast static analysis, formatting checks, and security rule enforcement.
2. **Automated Pytest Suite**: Executes all 14 unit and integration tests with code coverage verification (`pytest-cov`).
3. **Automated ML Quality Gate**:
   - Trains candidate model and verifies validation/test metrics against strict minimum performance criteria:
     - $\text{ROC-AUC} \ge 0.75$
     - $\text{F1-Score} \ge 0.60$
   - Prevents flawed or degrading model weights from ever reaching containerization!

### Continuous Deployment (`.github/workflows/cd.yml`)
1. **Docker Buildx & Multi-Arch Build**: Builds and packages the container.
2. **Container Security Scan (Trivy)**: Scans OS packages and application dependencies for CVE vulnerabilities.
3. **Container Registry Push**: Pushes signed images to GitHub Container Registry (`ghcr.io`) tagged with Git SHA and `latest`.
4. **Manifest Validation**: Runs `kubectl kustomize` dry-run validation.
5. **Automated Deployment**: Applies manifests to Kubernetes cluster and verifies `kubectl rollout status`.

---

## Repository Structure

```
ML-PIPELINE/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # CI: Linting, Unit/Integration Tests, ML Quality Gate
│       └── cd.yml                 # CD: Container Build, Trivy Scan, K8s Deploy
├── api/
│   └── index.py                   # Vercel Serverless Function entrypoint
├── artifacts/                     # Model weights & reference baseline
│   ├── model.joblib               # Trained Scikit-Learn Pipeline
│   ├── reference_data.csv         # Baseline dataset for drift detection
│   └── metadata.json              # Model parameters & validation metrics
├── data/
│   └── generate_data.py           # Realistic credit risk dataset generator
├── grafana/
│   ├── dashboards/
│   │   └── model_monitoring.json  # Comprehensive Grafana dashboard JSON
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboards.yml     # Auto-provision dashboards
│       └── datasources/
│           └── prometheus.yml     # Auto-provision Prometheus datasource
├── k8s/
│   ├── monitoring/
│   │   ├── grafana-k8s.yaml       # Grafana Deployment & Service for K8s
│   │   └── prometheus-k8s.yaml    # Prometheus Deployment & Service for K8s
│   ├── configmap.yaml             # Application configuration
│   ├── deployment.yaml            # Rolling update deployment with probes & limits
│   ├── hpa.yaml                   # Horizontal Pod Autoscaler (2-8 pods)
│   ├── ingress.yaml               # Ingress routing definition
│   ├── kustomization.yaml         # Kustomize manifest bundle
│   ├── namespace.yaml             # Kubernetes namespace
│   ├── secret.yaml                # Secrets configuration
│   └── service.yaml               # Service with Prometheus scrape annotations
├── prometheus/
│   ├── alert.rules.yml            # Alerting rules (drift, latency, errors)
│   └── prometheus.yml             # Prometheus scraping configuration
├── scripts/
│   ├── deploy_local.ps1           # Windows PowerShell one-click launch
│   ├── deploy_local.sh            # Linux/macOS one-click launch
│   ├── run_drift_check.py         # Batch offline drift CLI reporter
│   └── simulate_traffic.py        # Live traffic generator (normal & drift modes)
├── src/
│   ├── app/
│   │   ├── dashboard.py           # Interactive glassmorphic web dashboard
│   │   ├── main.py                # FastAPI serving, health, and monitoring API
│   │   ├── metrics.py             # Prometheus metrics registry
│   │   └── schemas.py             # Pydantic v2 validation models
│   ├── drift/
│   │   ├── detector.py            # Kolmogorov-Smirnov, PSI, and sliding window buffer
│   │   └── report.py              # Visual HTML & JSON report generator
│   └── model/
│       ├── evaluate.py            # Classification metrics & latency profiling
│       ├── pipeline.py            # Preprocessor & Estimator pipeline factory
│       └── train.py               # Rigorous train/val/test training pipeline
├── tests/
│   ├── test_api.py                # FastAPI endpoints & Prometheus integration tests
│   ├── test_drift.py              # Statistical drift algorithms unit tests
│   └── test_pipeline.py           # Preprocessing, imputation, and model tests
├── .dockerignore
├── .gitignore
├── Dockerfile                     # Multi-stage production container
├── docker-compose.yml             # Full local stack (API, Prometheus, Grafana, Simulator)
├── pyproject.toml                 # Tool configurations (pytest, ruff)
├── requirements.txt               # Pinned Python dependencies
├── vercel.json                    # Vercel serverless deployment routing config
└── README.md                      # Comprehensive system documentation
```

---

## Quickstart Guide

### Option 1: One-Click Local Deployment with Docker Compose (Recommended)

To launch the complete stack (ML API + Prometheus + Grafana + Traffic Simulator) in under 60 seconds:

**Windows (PowerShell):**
```powershell
.\scripts\deploy_local.ps1
```

**Linux / macOS:**
```bash
chmod +x scripts/deploy_local.sh
./scripts/deploy_local.sh
```

**Or directly with Docker Compose:**
```bash
docker compose up -d --build
```

#### Access Points:
- **Interactive Web App Dashboard**: [http://localhost:8000/](http://localhost:8000/)
- **FastAPI Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **API Health & Probe**: [http://localhost:8000/health](http://localhost:8000/health)
- **Prometheus Server**: [http://localhost:9090](http://localhost:9090)
- **Grafana Monitoring Dashboard**: [http://localhost:3000](http://localhost:3000) *(User: `admin`, Password: `admin`)*
- **Visual HTML Drift Report**: [http://localhost:8000/drift/report](http://localhost:8000/drift/report)
- **Raw Prometheus Metrics**: [http://localhost:8000/metrics](http://localhost:8000/metrics)

---

### Option 2: Instant Vercel Serverless Deployment

This repository includes native Vercel configuration (`vercel.json` + `api/index.py`), enabling instantaneous zero-cold-start cloud deployment:

1. **Via GitHub Integration (Recommended - Automated CI/CD)**:
   - Go to [Vercel New Project](https://vercel.com/new).
   - Connect or select your GitHub repository: `abhiiyadav2595/ML-pipeline`.
   - Vercel automatically detects the configuration in `vercel.json` and uses the Python runtime.
   - Click **Deploy**. Vercel will build and assign a public `https://<project-name>.vercel.app` URL.

2. **Via Vercel CLI**:
   ```bash
   vercel --prod
   ```

Once deployed, your live production URL provides:
- **Live Interactive ML Prediction Dashboard** at `/`
- **Swagger OpenAPI Explorer** at `/docs`
- **Statistical Drift Visual Reports** at `/drift/report`
- **System Health Status** at `/health`

---

### Option 3: Kubernetes Deployment (Minikube / Kind / Production Cluster)

Deploy all manifests using `kubectl`:

```bash
# 1. Apply all resources via Kustomize
kubectl apply -k k8s/

# 2. Verify deployment rollout
kubectl rollout status deployment/ml-serving-deployment -n ml-serving

# 3. Check running pods and services
kubectl get pods,svc,hpa -n ml-serving

# 4. Port forward to access locally
kubectl port-forward svc/ml-serving-service 8000:8000 -n ml-serving &
kubectl port-forward svc/prometheus-service 9090:9090 -n ml-serving &
kubectl port-forward svc/grafana-service 3000:3000 -n ml-serving &
```

---

### Option 3: Local Python Virtual Environment

```bash
# 1. Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model & generate reference baseline
python -m src.model.train

# 4. Run tests
pytest tests/ -v --cov=src

# 5. Launch FastAPI service
uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Simulating Live Traffic & Inducing Drift

The platform provides a dedicated traffic generation utility to simulate live production traffic scenarios:

### 1. Send Normal Baseline Traffic (Healthy State)
Sends 150 requests sampled from the baseline reference distribution:
```bash
python scripts/simulate_traffic.py --mode normal --count 150 --delay 0.03
```
- In Grafana: Status remains **GREEN (`HEALTHY`)**, P-values > 0.05, PSI < 0.10.

### 2. Induce Macroeconomic Shock / Data Drift
Simulates a real-world financial crisis where applicant debt-to-income spikes by 40%, credit scores plummet by ~80 points, and unemployment surges:
```bash
python scripts/simulate_traffic.py --mode drift --count 150 --delay 0.03
```
- In Grafana: Status turns **RED (`DRIFT DETECTED`)**, feature drift scores spike, KS test p-values drop to `< 0.0001`, and alert `ModelDatasetDriftDetected` triggers in Prometheus!

### 3. Continuous Traffic Loop (Demonstration Mode)
Runs an automated loop alternating between normal and drifted traffic to keep dashboards live:
```bash
python scripts/simulate_traffic.py --loop --count 80 --delay 0.05
```

### 4. Interactive HTML Drift Dashboard
Open [http://localhost:8000/drift/report](http://localhost:8000/drift/report) in your browser to view a comprehensive visual diagnostic breakdown of every feature, test statistics, and p-values.

---

## API Documentation

### 1. Single Prediction (`POST /predict`)
**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "features": {
         "age": 42,
         "income": 72000.0,
         "credit_score": 690.0,
         "debt_to_income": 0.28,
         "loan_amount": 18000.0,
         "employment_status": "employed",
         "loan_purpose": "debt_consolidation"
       }
     }'
```
**Response (200 OK):**
```json
{
  "prediction": 0,
  "prediction_label": "non_default",
  "probability": 0.1842,
  "model_version": "1.0.0",
  "timestamp": "2026-09-25T07:35:00.000Z"
}
```

### 2. Batch Prediction (`POST /predict/batch`)
Accepts an array of application objects for high-throughput batch scoring.

### 3. Health Probe (`GET /health`)
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "reference_data_loaded": true,
  "model_version": "1.0.0",
  "uptime_seconds": 342.5
}
```

### 4. On-Demand Drift Evaluation (`POST /drift/evaluate`)
Calculates drift across current sliding window observations and updates Prometheus gauges immediately.

---

## Test Suite & Verification

The project includes an automated test suite verifying data pipelines, mathematical drift tests, and API endpoints:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Test Coverage Summary:
- `tests/test_pipeline.py`: Tests schema generation, missing value imputation, ColumnTransformer fitting, and evaluation metrics.
- `tests/test_drift.py`: Validates Kolmogorov-Smirnov test, Population Stability Index (PSI), Wasserstein distance, and sliding window buffer state transitions.
- `tests/test_api.py`: Validates FastAPI endpoints, input schemas, batch processing, `/metrics` exposition, and HTML report serving.

**Result: 14 passed in 9.6s with 81%+ total code coverage.**
