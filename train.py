from pathlib import Path
import joblib
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_loader import load_raw_data
from src.preprocessing import create_preprocessor


def train_model(save_path: str = "artifacts/model_pipeline.joblib"):
    df = load_raw_data()
    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save test set for dedicated evaluation step
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    X_test.assign(Churn=y_test).to_csv(artifacts_dir / "test_set.csv", index=False)

    preprocessor = create_preprocessor()
    base_model = XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False,
    )

    full_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", base_model),
        ]
    )

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__scale_pos_weight": [1.0, 2.0],  # Address class imbalance
    }

    grid = GridSearchCV(
        full_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
    )

    print("Running Hyperparameter Search...")
    grid.fit(X_train, y_train)

    best_pipeline = grid.best_estimator_
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    print(f"Optimal Parameters: {grid.best_params_}")

    joblib.dump(best_pipeline, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    train_model()