from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Derives domain-specific ratios and flags."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        # Avoid zero division
        safe_tenure = X_out["tenure"].replace(0, 1)
        X_out["ChargesPerMonthRatio"] = X_out["TotalCharges"] / (safe_tenure * X_out["MonthlyCharges"])
        X_out["IsLongTerm"] = (X_out["tenure"] >= 24).astype(int)
        return X_out


def create_preprocessor():
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges", "ChargesPerMonthRatio", "IsLongTerm"]
    categorical_features = ["Contract", "InternetService", "TechSupport", "PaperlessBilling"]

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ]
    )

    full_pipeline = Pipeline(
        [
            ("feature_engineer", FeatureEngineer()),
            ("transformer", transformer),
        ]
    )
    return full_pipeline