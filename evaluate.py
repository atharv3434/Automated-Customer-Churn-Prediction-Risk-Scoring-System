from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def evaluate():
    test_path = Path("artifacts/test_set.csv")
    model_path = Path("artifacts/model_pipeline.joblib")

    if not test_path.exists() or not model_path.exists():
        raise FileNotFoundError("Artifacts missing. Run `src/train.py` first.")

    test_data = pd.read_csv(test_path)
    X_test = test_data.drop(columns=["Churn"])
    y_test = test_data["Churn"]

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    matrix = confusion_matrix(y_test, y_pred)

    print("================== Evaluation Metrics ==================")
    print(f"ROC-AUC: {auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:")
    print(matrix)
    print("========================================================")


if __name__ == "__main__":
    evaluate()