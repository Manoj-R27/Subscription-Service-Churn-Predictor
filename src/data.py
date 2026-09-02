from pathlib import Path

import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
TARGET = "Churn"
ID_COLUMN = "customerID"
NUMERIC_COLUMNS = [
    "tenure", "MonthlyCharges", "TotalCharges", "AverageCharges",
    "IsMonthToMonth", "HasInternet", "TenureGroup",
]
BASE_CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURE_COLUMNS = BASE_CATEGORICAL_COLUMNS + NUMERIC_COLUMNS


def load_data(source: str = DATA_URL) -> pd.DataFrame:
    """Load the public IBM Telco data, or a local CSV path."""
    path = Path(source)
    frame = pd.read_csv(path if path.exists() else source)
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    frame = frame.dropna(subset=["TotalCharges"]).copy()
    frame[TARGET] = frame[TARGET].map({"Yes": 1, "No": 0}).astype(int)
    return engineer_features(frame.drop(columns=[ID_COLUMN], errors="ignore"))


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic, non-target feature engineering used downstream.

    AverageCharges is historical billed amount per active month. The remaining
    derived indicators are interpretable convenience features and use no target data.
    """
    result = frame.copy()
    result["AverageCharges"] = result["TotalCharges"] / result["tenure"].clip(lower=1)
    result["IsMonthToMonth"] = (result["Contract"] == "Month-to-month").astype(int)
    result["HasInternet"] = (result["InternetService"] != "No").astype(int)
    result["TenureGroup"] = pd.cut(
        result["tenure"], bins=[-1, 12, 24, 48, 60, 100], labels=False
    ).astype(int)
    return result


def split_xy(frame: pd.DataFrame):
    return frame[FEATURE_COLUMNS].copy(), frame[TARGET].copy()


def customer_row(values: dict) -> pd.DataFrame:
    """Build one raw customer row and apply the shared feature engineering."""
    return engineer_features(pd.DataFrame([values]))[FEATURE_COLUMNS]
