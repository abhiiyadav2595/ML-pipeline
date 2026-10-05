"""
Synthetic Data Generator for Credit Risk & Loan Default Prediction.
Generates realistic financial features with non-linear relationships.
"""


import numpy as np
import pandas as pd


def generate_credit_dataset(
    n_samples: int = 10000,
    random_state: int = 42,
    drift: bool = False,
    drift_type: str = "sudden",
) -> pd.DataFrame:
    """
    Generate synthetic credit dataset.

    Args:
        n_samples: Number of sample records to generate.
        random_state: Seed for reproducibility.
        drift: Whether to induce data/feature drift.
        drift_type: 'sudden' or 'macroeconomic_shock'.

    Returns:
        pd.DataFrame containing feature columns and the binary 'default' target.
    """
    rng = np.random.default_rng(random_state)

    if not drift:
        # Baseline distributions
        age = rng.integers(low=21, high=70, size=n_samples)
        income = rng.lognormal(mean=10.8, sigma=0.55, size=n_samples)  # Mean ~ $55,000
        income = np.clip(income, 15000, 300000)

        credit_score = rng.normal(loc=680, scale=60, size=n_samples)
        credit_score = np.clip(credit_score, 300, 850)

        debt_to_income = rng.beta(a=2, b=5, size=n_samples) * 0.8  # ~ 0.22 average
        debt_to_income = np.clip(debt_to_income, 0.01, 1.2)

        loan_amount = rng.lognormal(mean=9.5, sigma=0.7, size=n_samples)  # Mean ~ $15,000
        loan_amount = np.clip(loan_amount, 1000, 60000)

        employment_probs = [0.72, 0.15, 0.08, 0.05]
        loan_purpose_probs = [0.45, 0.25, 0.15, 0.10, 0.05]
    else:
        # DRIFTED distributions (e.g. severe macroeconomic downturn / inflation shock)
        age = rng.integers(low=21, high=70, size=n_samples)
        # Income drops and credit scores decline
        income = rng.lognormal(mean=10.4, sigma=0.6, size=n_samples)  # Mean ~ $38,000
        income = np.clip(income, 12000, 250000)

        # Severe drop in credit scores (shifted left by ~80 points)
        credit_score = rng.normal(loc=605, scale=75, size=n_samples)
        credit_score = np.clip(credit_score, 300, 850)

        # Debt to income surges dramatically
        debt_to_income = rng.beta(a=3, b=3, size=n_samples) * 1.1  # ~ 0.55 average
        debt_to_income = np.clip(debt_to_income, 0.05, 1.5)

        loan_amount = rng.lognormal(mean=9.8, sigma=0.8, size=n_samples)
        loan_amount = np.clip(loan_amount, 2000, 75000)

        # Higher unemployment and distressed loan purposes
        employment_probs = [0.50, 0.12, 0.32, 0.06]
        loan_purpose_probs = [0.60, 0.10, 0.15, 0.05, 0.10]

    employment_categories = ["employed", "self_employed", "unemployed", "retired"]
    employment_status = rng.choice(employment_categories, size=n_samples, p=employment_probs)

    loan_purpose_categories = [
        "debt_consolidation",
        "home_improvement",
        "business",
        "education",
        "medical",
    ]
    loan_purpose = rng.choice(loan_purpose_categories, size=n_samples, p=loan_purpose_probs)

    # Compute realistic default probability using logistic latent equation
    # High DTI, low credit score, high loan/income ratio, and unemployment raise risk
    loan_to_income = loan_amount / (income + 1e-5)

    employment_risk = {
        "employed": -0.4,
        "self_employed": 0.1,
        "unemployed": 1.2,
        "retired": -0.2,
    }
    purpose_risk = {
        "debt_consolidation": 0.3,
        "home_improvement": -0.2,
        "business": 0.4,
        "education": 0.0,
        "medical": 0.5,
    }

    emp_risk_arr = np.array([employment_risk[e] for e in employment_status])
    purp_risk_arr = np.array([purpose_risk[p] for p in loan_purpose])

    logits = (
        -3.2
        + 3.5 * debt_to_income
        - 0.008 * (credit_score - 600)
        + 1.8 * np.clip(loan_to_income, 0, 2)
        - 0.02 * (age - 35)
        + emp_risk_arr
        + purp_risk_arr
        + rng.normal(0, 0.35, size=n_samples)
    )

    probs = 1.0 / (1.0 + np.exp(-logits))
    default = (rng.uniform(size=n_samples) < probs).astype(int)

    df = pd.DataFrame({
        "age": np.round(age, 0).astype(int),
        "income": np.round(income, 2),
        "credit_score": np.round(credit_score, 1),
        "debt_to_income": np.round(debt_to_income, 4),
        "loan_amount": np.round(loan_amount, 2),
        "employment_status": employment_status,
        "loan_purpose": loan_purpose,
        "default": default,
    })

    return df


if __name__ == "__main__":
    baseline_df = generate_credit_dataset(n_samples=5000, random_state=42, drift=False)
    drift_df = generate_credit_dataset(n_samples=2000, random_state=99, drift=True)
    print("Baseline dataset shape:", baseline_df.shape)
    print("Baseline default rate:", baseline_df["default"].mean())
    print("Drift dataset shape:", drift_df.shape)
    print("Drift default rate:", drift_df["default"].mean())
