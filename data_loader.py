from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic telco customer churn dataset."""
    np.random.seed(seed)

    tenure = np.random.exponential(scale=20, size=n_samples).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(loc=70, scale=30, size=n_samples).clip(18.0, 120.0)
    total_charges = (tenure * monthly_charges) + np.random.normal(0, 25, size=n_samples)
    total_charges = np.clip(total_charges, 18.0, None)

    # Introduce ~2% missing data in TotalCharges to handle realistically
    missing_mask = np.random.rand(n_samples) < 0.02
    total_charges[missing_mask] = np.nan

    contracts = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.25, 0.20],
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_samples,
        p=[0.35, 0.45, 0.20],
    )
    tech_support = np.random.choice(
        ["Yes", "No", "No internet service"],
        size=n_samples,
        p=[0.30, 0.50, 0.20],
    )
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])

    # Churn probability logic
    logit = (
        -0.05 * tenure
        + 0.03 * (monthly_charges - 60)
        + 1.2 * (contracts == "Month-to-month")
        - 0.8 * (contracts == "Two year")
        + 0.6 * (internet_service == "Fiber optic")
        - 0.5 * (tech_support == "Yes")
        - 1.0
    )
    prob = 1 / (1 + np.exp(-logit))
    churn = (np.random.rand(n_samples) < prob).astype(int)

    df = pd.DataFrame(
        {
            "customerID": [f"CUST-{i:05d}" for i in range(n_samples)],
            "tenure": tenure,
            "MonthlyCharges": np.round(monthly_charges, 2),
            "TotalCharges": np.round(total_charges, 2),
            "Contract": contracts,
            "InternetService": internet_service,
            "TechSupport": tech_support,
            "PaperlessBilling": paperless_billing,
            "Churn": churn,
        }
    )
    return df


def load_raw_data(data_path: str = "data/raw/churn_data.csv") -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_synthetic_data()
        df.to_csv(path, index=False)
        return df
    return pd.read_csv(path)