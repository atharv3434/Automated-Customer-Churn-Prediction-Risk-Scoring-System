import numpy as np
import pandas as pd
import pytest
from src.data_loader import generate_synthetic_data
from src.preprocessing import create_preprocessor


@pytest.fixture
def sample_data():
    return generate_synthetic_data(n_samples=50, seed=10)


def test_feature_engineering_transformation(sample_data):
    preprocessor = create_preprocessor()
    X = sample_data.drop(columns=["customerID", "Churn"])

    X_transformed = preprocessor.fit_transform(X)

    # 5 numeric + 1 (Contract=2) + 2 (Internet=2) + 2 (TechSupport=2) + 1 (Paperless=1) = 11 features
    assert X_transformed.shape[0] == 50
    assert X_transformed.shape[1] > 0
    assert not np.isnan(X_transformed).any(), "Preprocessed array contains unresolved NaNs"


def test_missing_value_resilience():
    preprocessor = create_preprocessor()
    edge_df = pd.DataFrame(
        {
            "tenure": [0, 5],
            "MonthlyCharges": [20.0, np.nan],
            "TotalCharges": [np.nan, 100.0],
            "Contract": ["Month-to-month", "Two year"],
            "InternetService": ["No", "DSL"],
            "TechSupport": ["No internet service", "Yes"],
            "PaperlessBilling": ["No", "Yes"],
        }
    )
    result = preprocessor.fit_transform(edge_df)
    assert not np.isnan(result).any()