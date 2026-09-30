from typing import Literal, Optional
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time churn risk scoring using Scikit-Learn and XGBoost.",
    version="1.0.0",
)

MODEL_PATH = "artifacts/model_pipeline.joblib"
model = None


@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        model = None
        print(f"Warning: Model artifact not loaded: {e}")


class CustomerPayload(BaseModel):
    tenure: int = Field(..., ge=0, le=120, example=12)
    MonthlyCharges: float = Field(..., ge=0.0, example=65.5)
    TotalCharges: Optional[float] = Field(None, ge=0.0, example=786.0)
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    PaperlessBilling: Literal["Yes", "No"] = "Yes"


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_tier: str


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerPayload):
    if model is None:
        raise HTTPException(status_code=503, detail="Model pipeline is not ready or trained.")

    data = pd.DataFrame([customer.dict()])
    prob = float(model.predict_proba(data)[0, 1])
    pred = int(prob >= 0.5)
    risk_tier = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=pred,
        risk_tier=risk_tier,
    )