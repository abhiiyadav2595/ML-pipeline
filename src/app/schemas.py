"""
Pydantic Data Schemas for Inference and Drift Monitoring.
Provides strict validation, descriptive error messages, and schema examples.
"""

from typing import List, Literal

from pydantic import BaseModel, Field


class CreditFeatures(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    income: float = Field(..., ge=0.0, description="Annual gross income in USD")
    credit_score: float = Field(..., ge=300.0, le=850.0, description="FICO credit score")
    debt_to_income: float = Field(..., ge=0.0, le=5.0, description="Debt-to-income ratio (e.g., 0.28)")
    loan_amount: float = Field(..., ge=100.0, le=250000.0, description="Requested loan principal amount")
    employment_status: Literal["employed", "self_employed", "unemployed", "retired"] = Field(
        ..., description="Current employment status"
    )
    loan_purpose: Literal[
        "debt_consolidation", "home_improvement", "business", "education", "medical"
    ] = Field(..., description="Stated purpose of the loan")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 38,
                "income": 65000.0,
                "credit_score": 710.0,
                "debt_to_income": 0.25,
                "loan_amount": 15000.0,
                "employment_status": "employed",
                "loan_purpose": "debt_consolidation",
            }
        }
    }


class PredictionRequest(BaseModel):
    features: CreditFeatures


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for Non-Default, 1 for Default")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability of default")
    model_version: str = Field(..., description="Active deployed model version")
    timestamp: str = Field(..., description="Inference timestamp (ISO 8601)")


class BatchPredictionRequest(BaseModel):
    instances: List[CreditFeatures]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    count: int
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    reference_data_loaded: bool
    model_version: str
    uptime_seconds: float
