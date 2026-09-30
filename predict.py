import joblib
import pandas as pd


class ChurnPredictor:

    def __init__(self, model_path: str = "artifacts/model_pipeline.joblib"):
        self.model = joblib.load(model_path)

    def predict_record(self, record: dict) -> dict:
        df = pd.DataFrame([record])
        prob = float(self.model.predict_proba(df)[0, 1])
        prediction = int(prob >= 0.5)

        risk_tier = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")
        return {
            "churn_prediction": prediction,
            "churn_probability": round(prob, 4),
            "risk_tier": risk_tier,
        }