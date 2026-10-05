"""
Batch Offline Drift Detector & CLI Reporter.
Can be executed as a scheduled batch job (cron / Airflow / K8s Job)
to analyze historical batch predictions against baseline.
"""

import os
import argparse
import pandas as pd
from data.generate_data import generate_credit_dataset
from src.drift.detector import DriftDetector
from src.drift.report import generate_html_drift_report
from src.model.pipeline import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


def run_batch_drift_analysis(
    reference_path: str = "artifacts/reference_data.csv",
    current_data_path: str = None,
    output_html_path: str = "artifacts/drift_report.html",
    drift_simulated: bool = False,
):
    """Run batch statistical drift analysis between reference and target datasets."""
    print("=" * 60)
    print(" Starting Batch Drift Analysis Job")
    print("=" * 60)

    if not os.path.exists(reference_path):
        raise FileNotFoundError(f"Reference data not found at {reference_path}. Run training first.")

    ref_df = pd.read_csv(reference_path)
    print(f"Loaded reference baseline: {len(ref_df)} rows")

    if current_data_path and os.path.exists(current_data_path):
        cur_df = pd.read_csv(current_data_path)
        print(f"Loaded target dataset from {current_data_path}: {len(cur_df)} rows")
    else:
        print(f"Generating synthetic target data (drift={drift_simulated})...")
        cur_df = generate_credit_dataset(n_samples=1500, random_state=123, drift=drift_simulated)

    detector = DriftDetector(
        reference_data=ref_df,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        ks_stat_threshold=0.12,
        psi_threshold=0.20,
    )

    result = detector.evaluate_drift(current_data=cur_df)

    print("\n--- Drift Assessment Results ---")
    print(f"Dataset Drift Flag: {'DRIFT DETECTED!' if result.dataset_drift_detected else 'STABLE'}")
    print(f"Drifted Features: {result.drifted_features_count} / {result.total_features_count} ({result.drift_ratio * 100:.1f}%)")
    print("\nFeature Details:")
    for feat, met in result.feature_metrics.items():
        status_txt = "DRIFT" if met.drift_detected else "OK"
        p_val_txt = f"{met.p_value:.4f}" if met.p_value is not None else "N/A"
        print(f"  - {feat:<20}: [{status_txt}] (Test: {met.test_name}, Score: {met.drift_score:.4f}, p-val: {p_val_txt})")

    # Generate HTML report
    html_report = generate_html_drift_report(result)
    os.makedirs(os.path.dirname(output_html_path) or ".", exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"\nHTML Drift Report saved to: {output_html_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Drift Detection CLI")
    parser.add_argument("--ref", type=str, default="artifacts/reference_data.csv", help="Path to reference CSV")
    parser.add_argument("--cur", type=str, default=None, help="Path to current CSV (optional)")
    parser.add_argument("--out", type=str, default="artifacts/drift_report.html", help="Output HTML report path")
    parser.add_argument("--simulate-drift", action="store_true", help="Generate synthetic drifted data")

    args = parser.parse_args()
    run_batch_drift_analysis(
        reference_path=args.ref,
        current_data_path=args.cur,
        output_html_path=args.out,
        drift_simulated=args.simulate_drift,
    )
