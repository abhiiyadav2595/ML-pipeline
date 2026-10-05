"""
Model Evaluation Suite.
Computes comprehensive classification metrics, calibration scores, and operational latency.
Adheres to strict evaluation best practices.
"""

import time
from typing import Any, Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Evaluate trained pipeline on a held-out test split.

    Args:
        model: Trained pipeline with predict and predict_proba.
        X_test: Feature matrix.
        y_test: Ground truth labels.
        threshold: Decision threshold for positive class.

    Returns:
        Dictionary of performance metrics, confusion matrix, and latency.
    """
    # Measure latency for operational profiling
    start_time = time.perf_counter()
    y_proba = model.predict_proba(X_test)[:, 1]
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    latency_per_sample_us = (elapsed_ms / len(X_test)) * 1000.0

    y_pred = (y_proba >= threshold).astype(int)

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = average_precision_score(y_test, y_proba)
    except ValueError:
        pr_auc = 0.0

    brier = brier_score_loss(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4),
        "confusion_matrix": cm,
        "latency_per_sample_us": round(float(latency_per_sample_us), 2),
        "samples_evaluated": int(len(X_test)),
        "decision_threshold": threshold,
    }
