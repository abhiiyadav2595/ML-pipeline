"""
Unit Tests for Statistical Drift Detection Engine.
Validates Kolmogorov-Smirnov test, PSI calculation, and DriftDetector stateful buffer.
"""

import numpy as np
import pandas as pd

from data.generate_data import generate_credit_dataset
from src.drift.detector import (
    DriftDetector,
    DriftResult,
    calculate_categorical_psi,
    calculate_psi,
)
from src.model.pipeline import CATEGORICAL_FEATURES, NUMERICAL_FEATURES


def test_psi_identical_distributions():
    """Identical distributions should have near-zero PSI (< 0.05)."""
    np.random.seed(42)
    dist_a = np.random.normal(loc=100, scale=15, size=2000)
    dist_b = np.random.normal(loc=100, scale=15, size=2000)

    psi_score = calculate_psi(dist_a, dist_b)
    assert psi_score < 0.05


def test_psi_drifted_distributions():
    """Significantly shifted distribution should produce high PSI (> 0.25)."""
    np.random.seed(42)
    dist_a = np.random.normal(loc=100, scale=15, size=2000)
    dist_b = np.random.normal(loc=135, scale=25, size=2000)  # Major shift

    psi_score = calculate_psi(dist_a, dist_b)
    assert psi_score > 0.25


def test_categorical_psi_shift():
    """Test PSI for shifted categorical distributions."""
    series_a = pd.Series(["employed"] * 80 + ["unemployed"] * 20)
    series_b = pd.Series(["employed"] * 30 + ["unemployed"] * 70)  # Massive unemployment rise

    psi_score = calculate_categorical_psi(series_a, series_b)
    assert psi_score > 0.25


def test_drift_detector_clean_baseline():
    """Test that baseline traffic generates NO drift alert."""
    ref_df = generate_credit_dataset(n_samples=1000, random_state=42, drift=False)
    cur_df = generate_credit_dataset(n_samples=500, random_state=43, drift=False)

    detector = DriftDetector(
        reference_data=ref_df,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        ks_stat_threshold=0.12,
        psi_threshold=0.20,
    )

    result = detector.evaluate_drift(current_data=cur_df)
    assert isinstance(result, DriftResult)
    assert result.dataset_drift_detected is False
    assert result.drift_ratio < 0.33


def test_drift_detector_severe_drift():
    """Test that drifted macroeconomic shock triggers dataset drift alert."""
    ref_df = generate_credit_dataset(n_samples=1000, random_state=42, drift=False)
    drifted_df = generate_credit_dataset(n_samples=500, random_state=99, drift=True)

    detector = DriftDetector(
        reference_data=ref_df,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        ks_stat_threshold=0.12,
        psi_threshold=0.20,
    )

    result = detector.evaluate_drift(current_data=drifted_df)
    assert isinstance(result, DriftResult)
    assert result.dataset_drift_detected is True
    assert result.drifted_features_count >= 2
    # Verify specific key drifted features
    assert result.feature_metrics["debt_to_income"].drift_detected is True
    assert result.feature_metrics["credit_score"].drift_detected is True
