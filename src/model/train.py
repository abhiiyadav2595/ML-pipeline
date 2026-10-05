"""
Model Training and Comparison Pipeline.
Implements rigorous ML validation:
  1. Chronological/Fixed train-val-test split prior to preprocessing.
  2. Imputation and feature scaling within Scikit-learn Pipeline (zero data leakage).
  3. Comparison of baseline (Logistic Regression) vs Production candidate (Gradient Boosting).
  4. Model artifact serialization and reference data storage for drift monitoring.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from data.generate_data import generate_credit_dataset
from src.model.evaluate import evaluate_model
from src.model.pipeline import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    build_model_pipeline,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_training_pipeline(
    data_path: str = None,
    artifacts_dir: str = "artifacts",
    n_samples: int = 12000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Execute full training, model comparison, validation, and serialization workflow.
    """
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Acquire Data
    if data_path and os.path.exists(data_path):
        logger.info(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        logger.info(f"Generating synthetic baseline dataset ({n_samples} samples)...")
        df = generate_credit_dataset(n_samples=n_samples, random_state=random_state, drift=False)

    logger.info(f"Dataset shape: {df.shape}. Target distribution: {df[TARGET_COLUMN].value_counts(normalize=True).to_dict()}")

    # Verify schema integrity
    for col in ALL_FEATURES + [TARGET_COLUMN]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from dataset!")

    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    # 2. Strict Train / Validation / Test Splitting
    # Split train vs temp (80% train, 20% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    # Split temp into validation and test (15% val, 15% test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    logger.info(f"Split sizes - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 3. Model Comparison
    logger.info("Comparing candidate models on validation split...")

    # Baseline Model: Logistic Regression
    baseline_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    baseline_pipe = build_model_pipeline(baseline_clf)
    baseline_pipe.fit(X_train, y_train)
    baseline_val_metrics = evaluate_model(baseline_pipe, X_val, y_val)
    logger.info(f"Baseline (Logistic Regression) Val ROC-AUC: {baseline_val_metrics['roc_auc']:.4f}, F1: {baseline_val_metrics['f1_score']:.4f}")

    # Production Candidate: HistGradientBoosting
    prod_clf = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=random_state,
    )
    prod_pipe = build_model_pipeline(prod_clf)
    prod_pipe.fit(X_train, y_train)
    prod_val_metrics = evaluate_model(prod_pipe, X_val, y_val)
    logger.info(f"Champion (Gradient Boosting) Val ROC-AUC: {prod_val_metrics['roc_auc']:.4f}, F1: {prod_val_metrics['f1_score']:.4f}")

    # Decision logic
    if prod_val_metrics["roc_auc"] >= baseline_val_metrics["roc_auc"]:
        selected_model_name = "HistGradientBoostingClassifier"
        selected_pipeline = prod_pipe
    else:
        selected_model_name = "LogisticRegression"
        selected_pipeline = baseline_pipe

    logger.info(f"Selected Champion Model: {selected_model_name}")

    # 4. Final Evaluation on Unseen Test Split
    test_metrics = evaluate_model(selected_pipeline, X_test, y_test)
    logger.info(f"Final Test Evaluation: ROC-AUC={test_metrics['roc_auc']}, F1={test_metrics['f1_score']}, Accuracy={test_metrics['accuracy']}")

    # 5. Save Reference Dataset (for Drift Monitoring)
    # The reference dataset contains clean baseline feature distributions and predicted probabilities
    reference_df = X_train.copy()
    reference_df["predicted_probability"] = selected_pipeline.predict_proba(X_train)[:, 1]
    reference_path = os.path.join(artifacts_dir, "reference_data.csv")
    reference_df.to_csv(reference_path, index=False)
    logger.info(f"Saved reference dataset for drift detection to {reference_path}")

    # 6. Save Model Artifact
    model_path = os.path.join(artifacts_dir, "model.joblib")
    joblib.dump(selected_pipeline, model_path)
    logger.info(f"Saved trained model pipeline to {model_path}")

    # 7. Save Metadata
    metadata = {
        "model_name": selected_model_name,
        "model_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": {
            "all": ALL_FEATURES,
            "numerical": NUMERICAL_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "target": TARGET_COLUMN,
        },
        "validation_metrics": {
            "baseline": baseline_val_metrics,
            "champion": prod_val_metrics,
        },
        "test_metrics": test_metrics,
        "model_path": model_path,
        "reference_path": reference_path,
    }

    metadata_path = os.path.join(artifacts_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    return metadata


if __name__ == "__main__":
    run_training_pipeline()
