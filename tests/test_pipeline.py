"""
Unit Tests for Machine Learning Pipeline, Preprocessing, and Evaluation.
"""

import numpy as np
import pytest

from data.generate_data import generate_credit_dataset
from src.model.evaluate import evaluate_model
from src.model.pipeline import (
    ALL_FEATURES,
    TARGET_COLUMN,
    build_model_pipeline,
)


@pytest.fixture
def sample_dataset():
    """Generates small reproducible dataset for testing."""
    return generate_credit_dataset(n_samples=500, random_state=42, drift=False)


def test_data_generator_schema(sample_dataset):
    """Ensure data generator outputs expected schema and data types."""
    assert len(sample_dataset) == 500
    for col in ALL_FEATURES + [TARGET_COLUMN]:
        assert col in sample_dataset.columns
    assert set(sample_dataset[TARGET_COLUMN].unique()).issubset({0, 1})


def test_pipeline_handles_missing_values(sample_dataset):
    """Test that pipeline's ColumnTransformer imputes missing values cleanly without error."""
    X = sample_dataset[ALL_FEATURES].copy()
    y = sample_dataset[TARGET_COLUMN]

    # Inject random NaNs into numerical and categorical columns
    X.loc[0:20, "income"] = np.nan
    X.loc[10:30, "credit_score"] = np.nan
    X.loc[15:35, "employment_status"] = np.nan

    pipeline = build_model_pipeline()
    pipeline.fit(X, y)

    # Prediction should succeed without NaN errors
    probas = pipeline.predict_proba(X)
    assert probas.shape == (len(X), 2)
    assert not np.isnan(probas).any()


def test_model_evaluation_metrics(sample_dataset):
    """Test evaluation function returns all required statistical and operational metrics."""
    X = sample_dataset[ALL_FEATURES]
    y = sample_dataset[TARGET_COLUMN]

    pipeline = build_model_pipeline()
    pipeline.fit(X, y)

    metrics = evaluate_model(pipeline, X, y)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "brier_score" in metrics
    assert "confusion_matrix" in metrics
    assert "latency_per_sample_us" in metrics

    # Sanity checks on metric bounds
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["latency_per_sample_us"] > 0
