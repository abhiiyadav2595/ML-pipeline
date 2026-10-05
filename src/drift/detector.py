"""
Statistical Drift Detection Engine.
Computes:
  - 2-Sample Kolmogorov-Smirnov (KS) test and Wasserstein distance for continuous features.
  - Population Stability Index (PSI) and Chi-Square test for categorical features.
  - Prediction Probability Drift (Target/Concept Drift proxy).
  - Sliding Window buffer for real-time inference monitoring.
"""

import threading
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.stats import ks_2samp, wasserstein_distance


class FeatureDriftResult(BaseModel):
    feature_name: str
    feature_type: str  # 'numerical' or 'categorical'
    test_name: str     # 'ks_test' or 'psi'
    drift_score: float # KS statistic or PSI value
    p_value: Optional[float] = None
    drift_detected: bool
    threshold: float
    message: str


class DriftResult(BaseModel):
    timestamp: str
    sample_size: int
    dataset_drift_detected: bool
    drifted_features_count: int
    total_features_count: int
    drift_ratio: float
    feature_metrics: Dict[str, FeatureDriftResult]
    prediction_drift: Optional[FeatureDriftResult] = None


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate Population Stability Index (PSI) between two distributions.

    Interpretation:
      - PSI < 0.1: No significant change / stable
      - 0.1 <= PSI < 0.25: Moderate shift / warning
      - PSI >= 0.25: Significant drift / action required
    """
    expected = expected[~pd.isna(expected)]
    actual = actual[~pd.isna(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine bin edges from reference (expected) distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges = np.unique(bin_edges)  # Remove identical percentiles

    if len(bin_edges) < 2:
        return 0.0

    # Count occurrences in bins
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to proportions with smoothing epsilon
    expected_pct = (expected_counts + epsilon) / (np.sum(expected_counts) + epsilon * len(expected_counts))
    actual_pct = (actual_counts + epsilon) / (np.sum(actual_counts) + epsilon * len(actual_counts))

    # Calculate PSI formula
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_val, 5))


def calculate_categorical_psi(
    expected_series: pd.Series,
    actual_series: pd.Series,
    epsilon: float = 1e-4,
) -> float:
    """
    Calculate PSI for categorical features across all unique categories.
    """
    all_categories = sorted(list(set(expected_series.dropna().unique()) | set(actual_series.dropna().unique())))
    if not all_categories:
        return 0.0

    exp_counts = expected_series.value_counts(normalize=True).to_dict()
    act_counts = actual_series.value_counts(normalize=True).to_dict()

    psi_total = 0.0
    for cat in all_categories:
        e = exp_counts.get(cat, 0.0) + epsilon
        a = act_counts.get(cat, 0.0) + epsilon
        psi_total += (a - e) * np.log(a / e)

    return float(np.round(psi_total, 5))


class DriftDetector:
    """
    Stateful Drift Detector maintaining reference baseline distributions
    and a thread-safe sliding window buffer of live inference observations.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str],
        window_size: int = 500,
        ks_alpha: float = 0.05,
        ks_stat_threshold: float = 0.12,
        psi_threshold: float = 0.20,
        dataset_drift_threshold: float = 0.33,
    ):
        self.reference_data = reference_data.copy()
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.window_size = window_size
        self.ks_alpha = ks_alpha
        self.ks_stat_threshold = ks_stat_threshold
        self.psi_threshold = psi_threshold
        self.dataset_drift_threshold = dataset_drift_threshold

        # Sliding window buffer for real-time observations
        self.lock = threading.Lock()
        self.buffer: deque = deque(maxlen=window_size)
        self.prediction_buffer: deque = deque(maxlen=window_size)

    def add_observation(self, feature_dict: Dict[str, Any], prediction_proba: Optional[float] = None) -> None:
        """
        Record a live inference sample into the sliding window.
        """
        with self.lock:
            self.buffer.append(feature_dict)
            if prediction_proba is not None:
                self.prediction_buffer.append(prediction_proba)

    def add_observations_batch(self, df_batch: pd.DataFrame, probas: Optional[List[float]] = None) -> None:
        """
        Record multiple inference samples into the buffer.
        """
        records = df_batch.to_dict(orient="records")
        with self.lock:
            for i, rec in enumerate(records):
                self.buffer.append(rec)
                if probas is not None and i < len(probas):
                    self.prediction_buffer.append(probas[i])

    def get_current_window_df(self) -> pd.DataFrame:
        """
        Get snapshot of observations currently in buffer.
        """
        with self.lock:
            if not self.buffer:
                return pd.DataFrame()
            return pd.DataFrame(list(self.buffer))

    def evaluate_drift(
        self,
        current_data: Optional[pd.DataFrame] = None,
        current_probas: Optional[np.ndarray] = None,
    ) -> DriftResult:
        """
        Run statistical drift hypothesis tests comparing current data to reference baseline.
        If current_data is None, uses the internal sliding window buffer.
        """
        from datetime import datetime, timezone

        if current_data is None:
            current_data = self.get_current_window_df()
            with self.lock:
                if self.prediction_buffer:
                    current_probas = np.array(list(self.prediction_buffer))

        sample_size = len(current_data)
        metrics: Dict[str, FeatureDriftResult] = {}
        drifted_count = 0

        total_features = len(self.numerical_features) + len(self.categorical_features)

        # Handle empty or small sample case
        if sample_size < 30:
            for num_col in self.numerical_features:
                metrics[num_col] = FeatureDriftResult(
                    feature_name=num_col,
                    feature_type="numerical",
                    test_name="ks_test",
                    drift_score=0.0,
                    p_value=1.0,
                    drift_detected=False,
                    threshold=self.ks_stat_threshold,
                    message="Insufficient sample size (< 30 samples) in window",
                )
            for cat_col in self.categorical_features:
                metrics[cat_col] = FeatureDriftResult(
                    feature_name=cat_col,
                    feature_type="categorical",
                    test_name="psi",
                    drift_score=0.0,
                    drift_detected=False,
                    threshold=self.psi_threshold,
                    message="Insufficient sample size (< 30 samples) in window",
                )

            return DriftResult(
                timestamp=datetime.now(timezone.utc).isoformat(),
                sample_size=sample_size,
                dataset_drift_detected=False,
                drifted_features_count=0,
                total_features_count=total_features,
                drift_ratio=0.0,
                feature_metrics=metrics,
                prediction_drift=None,
            )

        # 1. Evaluate Numerical Features with 2-Sample Kolmogorov-Smirnov Test
        for col in self.numerical_features:
            if col not in current_data.columns or col not in self.reference_data.columns:
                continue

            ref_vals = pd.to_numeric(self.reference_data[col], errors="coerce").dropna().values
            cur_vals = pd.to_numeric(current_data[col], errors="coerce").dropna().values

            if len(cur_vals) < 10 or len(ref_vals) < 10:
                continue

            ks_res = ks_2samp(ref_vals, cur_vals)
            stat = float(np.round(ks_res.statistic, 4))
            p_val = float(np.round(ks_res.pvalue, 6))

            # Drift rule: p-value < alpha AND KS statistic exceeds minimum practical difference threshold
            drift_detected = bool(p_val < self.ks_alpha and stat >= self.ks_stat_threshold)
            if drift_detected:
                drifted_count += 1

            w_dist = float(np.round(wasserstein_distance(ref_vals, cur_vals), 4))

            msg = (
                f"Drift detected! KS-stat={stat} (threshold={self.ks_stat_threshold}), p-val={p_val}, Wasserstein={w_dist}"
                if drift_detected
                else f"Stable. KS-stat={stat}, p-val={p_val}"
            )

            metrics[col] = FeatureDriftResult(
                feature_name=col,
                feature_type="numerical",
                test_name="ks_test",
                drift_score=stat,
                p_value=p_val,
                drift_detected=drift_detected,
                threshold=self.ks_stat_threshold,
                message=msg,
            )

        # 2. Evaluate Categorical Features with Population Stability Index (PSI)
        for col in self.categorical_features:
            if col not in current_data.columns or col not in self.reference_data.columns:
                continue

            ref_series = self.reference_data[col].astype(str)
            cur_series = current_data[col].astype(str)

            psi_score = calculate_categorical_psi(ref_series, cur_series)
            drift_detected = bool(psi_score >= self.psi_threshold)
            if drift_detected:
                drifted_count += 1

            msg = (
                f"Categorical drift detected! PSI={psi_score} (threshold={self.psi_threshold})"
                if drift_detected
                else f"Stable. PSI={psi_score}"
            )

            metrics[col] = FeatureDriftResult(
                feature_name=col,
                feature_type="categorical",
                test_name="psi",
                drift_score=psi_score,
                p_value=None,
                drift_detected=drift_detected,
                threshold=self.psi_threshold,
                message=msg,
            )

        # 3. Evaluate Prediction Distribution Drift (Target/Concept Drift Proxy)
        pred_drift_result = None
        if "predicted_probability" in self.reference_data.columns and current_probas is not None and len(current_probas) >= 20:
            ref_probas = self.reference_data["predicted_probability"].dropna().values
            cur_probas = np.array(current_probas)

            ks_pred = ks_2samp(ref_probas, cur_probas)
            pred_stat = float(np.round(ks_pred.statistic, 4))
            pred_pval = float(np.round(ks_pred.pvalue, 6))
            pred_psi = calculate_psi(ref_probas, cur_probas, num_bins=10)

            is_pred_drift = bool(pred_pval < self.ks_alpha and (pred_stat >= self.ks_stat_threshold or pred_psi >= self.psi_threshold))

            pred_drift_result = FeatureDriftResult(
                feature_name="predicted_probability",
                feature_type="prediction",
                test_name="ks_and_psi",
                drift_score=pred_psi,
                p_value=pred_pval,
                drift_detected=is_pred_drift,
                threshold=self.psi_threshold,
                message=f"Prediction Drift: PSI={pred_psi}, KS-stat={pred_stat}, p-val={pred_pval}",
            )

        # Overall dataset drift decision
        drift_ratio = (drifted_count / total_features) if total_features > 0 else 0.0
        dataset_drift = bool(drift_ratio >= self.dataset_drift_threshold)

        return DriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            sample_size=sample_size,
            dataset_drift_detected=dataset_drift,
            drifted_features_count=drifted_count,
            total_features_count=total_features,
            drift_ratio=round(drift_ratio, 4),
            feature_metrics=metrics,
            prediction_drift=pred_drift_result,
        )
