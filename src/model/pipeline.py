"""
ML Pipeline Architecture.
Encapsulates data preprocessing (imputation, scaling, one-hot encoding)
and classification into an atomic scikit-learn Pipeline.
"""

from typing import List, Optional

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Feature definitions
NUMERICAL_FEATURES: List[str] = [
    "age",
    "income",
    "credit_score",
    "debt_to_income",
    "loan_amount",
]

CATEGORICAL_FEATURES: List[str] = [
    "employment_status",
    "loan_purpose",
]

ALL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN: str = "default"


def build_preprocessor() -> ColumnTransformer:
    """
    Construct preprocessing pipeline for numerical and categorical features.
    Guarantees strict separation of concerns and prevents data leakage.
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_model_pipeline(estimator: Optional[BaseEstimator] = None) -> Pipeline:
    """
    Create a complete Pipeline with preprocessing and estimator.
    """
    if estimator is None:
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=150,
            max_leaf_nodes=31,
            random_state=42,
            class_weight="balanced",
        )

    preprocessor = build_preprocessor()

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator),
    ])
    return pipeline
