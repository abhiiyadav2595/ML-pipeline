"""
Prometheus Metrics Instrumentation for ML Model Serving & Real-Time Drift.
Exposes standard latency/throughput metrics plus real-time drift gauges.
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
)

# 1. Operational & Latency Metrics
PREDICTION_REQUESTS_TOTAL = Counter(
    "model_prediction_requests_total",
    "Total count of model inference requests",
    ["endpoint", "status", "prediction_class"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "model_prediction_latency_seconds",
    "Latency of model prediction in seconds",
    ["endpoint"],
    buckets=(0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

REQUESTS_IN_FLIGHT = Gauge(
    "model_requests_in_flight",
    "Current number of requests being processed",
)

# 2. Prediction Distribution & Output Metrics
PREDICTION_PROBABILITY_HISTOGRAM = Histogram(
    "model_prediction_probability_distribution",
    "Distribution of predicted default probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

PREDICTIONS_BY_CLASS = Counter(
    "model_predictions_by_class_total",
    "Total predictions classified by output outcome",
    ["prediction_class"],
)

# 3. Drift Detection Metrics
FEATURE_DRIFT_SCORE = Gauge(
    "model_feature_drift_score",
    "Current drift score (KS statistic for numerical, PSI for categorical)",
    ["feature", "test_type"],
)

FEATURE_DRIFT_PVALUE = Gauge(
    "model_feature_drift_pvalue",
    "P-value from statistical drift hypothesis test (numerical features)",
    ["feature"],
)

FEATURE_DRIFT_DETECTED = Gauge(
    "model_feature_drift_detected",
    "Binary indicator (1 if feature drift detected, 0 otherwise)",
    ["feature"],
)

DATASET_DRIFT_DETECTED = Gauge(
    "model_dataset_drift_detected",
    "Binary indicator (1 if overall dataset drift detected, 0 otherwise)",
)

PREDICTION_DRIFT_DETECTED = Gauge(
    "model_prediction_drift_detected",
    "Binary indicator (1 if prediction probability distribution drift detected, 0 otherwise)",
)

DRIFTED_FEATURES_RATIO = Gauge(
    "model_drifted_features_ratio",
    "Ratio of monitored features that currently show statistically significant drift",
)

DRIFT_EVALUATION_SAMPLE_SIZE = Gauge(
    "model_drift_evaluation_sample_size",
    "Number of samples in the current drift evaluation window",
)
