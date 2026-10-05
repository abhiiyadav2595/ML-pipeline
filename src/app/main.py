"""
FastAPI Model Serving & Real-Time Monitoring Service.
Features:
  - Real-time single & batch prediction.
  - Automatic fallback training if artifacts are missing.
  - Prometheus metrics instrumentation (latency, throughput, drift gauges).
  - Background & on-demand statistical drift evaluation.
  - Interactive HTML drift report dashboard.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.app.dashboard import get_dashboard_html
from src.app.metrics import (
    DATASET_DRIFT_DETECTED,
    DRIFT_EVALUATION_SAMPLE_SIZE,
    DRIFTED_FEATURES_RATIO,
    FEATURE_DRIFT_DETECTED,
    FEATURE_DRIFT_PVALUE,
    FEATURE_DRIFT_SCORE,
    PREDICTION_DRIFT_DETECTED,
    PREDICTION_LATENCY_SECONDS,
    PREDICTION_PROBABILITY_HISTOGRAM,
    PREDICTION_REQUESTS_TOTAL,
    PREDICTIONS_BY_CLASS,
    REQUESTS_IN_FLIGHT,
)
from src.app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.drift.detector import DriftDetector, DriftResult
from src.drift.report import generate_html_drift_report
from src.model.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from src.model.train import run_training_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml_api")

# Application state holder
app_state: Dict[str, Any] = {
    "model": None,
    "metadata": None,
    "drift_detector": None,
    "start_time": time.time(),
    "model_version": "1.0.0",
    "request_counter": 0,
}


def ensure_initialized() -> None:
    """
    Ensure ML model and drift detector are loaded in memory.
    Resilient to serverless cold starts across any working directory.
    """
    if app_state["model"] is not None and app_state["drift_detector"] is not None:
        return

    base_dir = Path(__file__).resolve().parent.parent.parent
    default_artifacts = str(base_dir / "artifacts")
    artifacts_dir = os.getenv("ARTIFACTS_DIR", default_artifacts)

    # Fallback to local artifacts if directory not found
    if not os.path.exists(artifacts_dir):
        if os.path.exists("artifacts"):
            artifacts_dir = "artifacts"
        else:
            artifacts_dir = default_artifacts

    model_path = os.path.join(artifacts_dir, "model.joblib")
    ref_path = os.path.join(artifacts_dir, "reference_data.csv")

    if not os.path.exists(model_path) or not os.path.exists(ref_path):
        logger.warning(f"Artifacts missing in {artifacts_dir}! Running training pipeline...")
        run_training_pipeline(artifacts_dir=artifacts_dir)

    logger.info(f"Loading trained model from {model_path}...")
    app_state["model"] = joblib.load(model_path)

    logger.info(f"Loading reference dataset from {ref_path}...")
    ref_df = pd.read_csv(ref_path)

    app_state["drift_detector"] = DriftDetector(
        reference_data=ref_df,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        window_size=1000,
        ks_alpha=0.05,
        ks_stat_threshold=0.12,
        psi_threshold=0.20,
    )
    logger.info("Drift Detector successfully initialized.")


def update_prometheus_drift_metrics(drift_result: DriftResult) -> None:
    """Update all Prometheus gauges with latest statistical drift computation."""
    DATASET_DRIFT_DETECTED.set(1 if drift_result.dataset_drift_detected else 0)
    DRIFTED_FEATURES_RATIO.set(drift_result.drift_ratio)
    DRIFT_EVALUATION_SAMPLE_SIZE.set(drift_result.sample_size)

    for feat_name, metric in drift_result.feature_metrics.items():
        FEATURE_DRIFT_SCORE.labels(feature=feat_name, test_type=metric.test_name).set(metric.drift_score)
        FEATURE_DRIFT_DETECTED.labels(feature=feat_name).set(1 if metric.drift_detected else 0)
        if metric.p_value is not None:
            FEATURE_DRIFT_PVALUE.labels(feature=feat_name).set(metric.p_value)

    if drift_result.prediction_drift:
        PREDICTION_DRIFT_DETECTED.set(1 if drift_result.prediction_drift.drift_detected else 0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for loading artifacts and setting up drift monitoring."""
    logger.info("Initializing ML Model Serving & Monitoring Service...")
    ensure_initialized()
    yield
    logger.info("Shutting down ML Model Serving Service...")


app = FastAPI(
    title="ML Model Deployment & Real-Time Monitoring API",
    description="High-performance credit risk prediction service instrumented with Prometheus metrics and automated statistical drift detection.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def track_requests(request: Request, call_next):
    REQUESTS_IN_FLIGHT.inc()
    try:
        response = await call_next(request)
        return response
    finally:
        REQUESTS_IN_FLIGHT.dec()


@app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
def root_dashboard():
    """
    Interactive web application dashboard for real-time model inference and drift monitoring.
    """
    ensure_initialized()
    return HTMLResponse(content=get_dashboard_html())


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Liveness & Readiness probe for Kubernetes."""
    ensure_initialized()
    uptime = time.time() - app_state["start_time"]
    model_loaded = app_state["model"] is not None
    ref_loaded = app_state["drift_detector"] is not None
    healthy = model_loaded and ref_loaded

    status_str = "healthy" if healthy else "degraded"
    return HealthResponse(
        status=status_str,
        model_loaded=model_loaded,
        reference_data_loaded=ref_loaded,
        model_version=app_state["model_version"],
        uptime_seconds=round(uptime, 2),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest):
    """
    Perform real-time credit default risk prediction for a single application.
    Also streams observation into the real-time drift detection window.
    """
    ensure_initialized()
    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    start_time = time.perf_counter()
    features_dict = request.features.model_dump()
    df_instance = pd.DataFrame([features_dict])

    try:
        model = app_state["model"]
        probas = model.predict_proba(df_instance)
        default_prob = float(probas[0, 1])
        prediction = int(default_prob >= 0.5)
        label = "default" if prediction == 1 else "non_default"

        # Record in drift buffer
        detector: DriftDetector = app_state["drift_detector"]
        if detector:
            detector.add_observation(features_dict, default_prob)

        # Update Prometheus metrics
        elapsed = time.perf_counter() - start_time
        PREDICTION_LATENCY_SECONDS.labels(endpoint="/predict").observe(elapsed)
        PREDICTION_REQUESTS_TOTAL.labels(
            endpoint="/predict", status="success", prediction_class=label
        ).inc()
        PREDICTIONS_BY_CLASS.labels(prediction_class=label).inc()
        PREDICTION_PROBABILITY_HISTOGRAM.observe(default_prob)

        # Periodic background drift gauge refresh
        app_state["request_counter"] += 1
        if detector and app_state["request_counter"] % 50 == 0:
            drift_res = detector.evaluate_drift()
            update_prometheus_drift_metrics(drift_res)

        return PredictionResponse(
            prediction=prediction,
            prediction_label=label,
            probability=round(default_prob, 4),
            model_version=app_state["model_version"],
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except Exception as e:
        PREDICTION_REQUESTS_TOTAL.labels(
            endpoint="/predict", status="error", prediction_class="unknown"
        ).inc()
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}") from e


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(request: BatchPredictionRequest):
    """
    Perform batch inference for high-throughput workloads.
    """
    ensure_initialized()
    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    start_time = time.perf_counter()
    records = [inst.model_dump() for inst in request.instances]
    df_batch = pd.DataFrame(records)

    try:
        model = app_state["model"]
        probas = model.predict_proba(df_batch)[:, 1]
        predictions = (probas >= 0.5).astype(int)

        results: List[PredictionResponse] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for pred, prob in zip(predictions, probas, strict=False):
            lbl = "default" if pred == 1 else "non_default"
            results.append(
                PredictionResponse(
                    prediction=int(pred),
                    prediction_label=lbl,
                    probability=round(float(prob), 4),
                    model_version=app_state["model_version"],
                    timestamp=now_iso,
                )
            )
            PREDICTIONS_BY_CLASS.labels(prediction_class=lbl).inc()
            PREDICTION_PROBABILITY_HISTOGRAM.observe(float(prob))

        # Record batch in drift detector
        detector: DriftDetector = app_state["drift_detector"]
        if detector:
            detector.add_observations_batch(df_batch, [float(p) for p in probas])
            drift_res = detector.evaluate_drift()
            update_prometheus_drift_metrics(drift_res)

        elapsed = time.perf_counter() - start_time
        PREDICTION_LATENCY_SECONDS.labels(endpoint="/predict/batch").observe(elapsed)
        PREDICTION_REQUESTS_TOTAL.labels(
            endpoint="/predict/batch", status="success", prediction_class="batch"
        ).inc()

        return BatchPredictionResponse(
            predictions=results,
            count=len(results),
            model_version=app_state["model_version"],
        )

    except Exception as e:
        PREDICTION_REQUESTS_TOTAL.labels(
            endpoint="/predict/batch", status="error", prediction_class="batch"
        ).inc()
        logger.error(f"Batch inference error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}") from e


@app.post("/drift/evaluate", response_model=DriftResult, tags=["Drift Monitoring"])
def evaluate_drift_endpoint():
    """
    On-demand trigger to compute statistical data & prediction drift against baseline.
    Updates Prometheus drift gauges immediately.
    """
    ensure_initialized()
    detector: DriftDetector = app_state["drift_detector"]
    if detector is None:
        raise HTTPException(status_code=503, detail="Drift detector is not initialized")

    drift_res = detector.evaluate_drift()
    update_prometheus_drift_metrics(drift_res)
    return drift_res


@app.get("/drift/report", response_class=HTMLResponse, tags=["Drift Monitoring"])
def drift_report():
    """
    Generate and serve an interactive visual HTML dashboard for data & prediction drift.
    """
    ensure_initialized()
    detector: DriftDetector = app_state["drift_detector"]
    if detector is None:
        return HTMLResponse("<h3>Drift detector not initialized.</h3>", status_code=503)

    drift_res = detector.evaluate_drift()
    html_content = generate_html_drift_report(drift_res)
    return HTMLResponse(content=html_content)


@app.get("/drift/metrics", response_model=DriftResult, tags=["Drift Monitoring"])
def drift_metrics_json():
    """
    Return latest drift evaluation results in structured JSON format.
    """
    ensure_initialized()
    detector: DriftDetector = app_state["drift_detector"]
    if detector is None:
        raise HTTPException(status_code=503, detail="Drift detector not initialized")
    return detector.evaluate_drift()


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Prometheus scraping endpoint exposing all operational and drift metrics.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
