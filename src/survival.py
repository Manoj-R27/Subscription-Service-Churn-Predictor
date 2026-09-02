import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

from .data import BASE_CATEGORICAL_COLUMNS

SURVIVAL_FEATURES = BASE_CATEGORICAL_COLUMNS + ["MonthlyCharges", "TotalCharges"]


def fit_survival(frame):
    km = KaplanMeierFitter().fit(frame["tenure"], event_observed=frame["Churn"], label="All customers")
    grouped_curves = {}
    for column in ["Contract", "InternetService"]:
        grouped_curves[column] = {}
        for value, group in frame.groupby(column):
            estimator = KaplanMeierFitter().fit(group["tenure"], group["Churn"], label=str(value))
            grouped_curves[column][value] = estimator
    cox_data = pd.get_dummies(frame[SURVIVAL_FEATURES], drop_first=True, dtype=float)
    cox_data["tenure"] = frame["tenure"]
    cox_data["Churn"] = frame["Churn"]
    cox = CoxPHFitter(penalizer=0.1)
    cox.fit(cox_data, duration_col="tenure", event_col="Churn")
    hazards = cox.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(
        columns={"exp(coef)": "hazard_ratio", "exp(coef) lower 95%": "lower_95", "exp(coef) upper 95%": "upper_95", "p": "p_value"}
    ).reset_index(names="feature")
    return km, grouped_curves, cox, hazards


def cox_row(frame):
    """Encode a customer row with the same columns used to fit the Cox model."""
    encoded = pd.get_dummies(frame[SURVIVAL_FEATURES], drop_first=True, dtype=float)
    return encoded


def survival_estimate(cox, customer_features, horizons=(6, 12, 24)):
    row = pd.DataFrame([customer_features])
    survival = cox.predict_survival_function(row, times=list(horizons)).iloc[:, 0]
    median = cox.predict_median(row)
    median_value = median.iloc[0] if hasattr(median, "iloc") else median
    return {"survival_probability": {str(horizon): float(survival.loc[horizon]) for horizon in horizons}, "median_months": None if pd.isna(median_value) else float(median_value)}


def estimate_median_time(cox, customer_features, high_risk_probability, risk_cutoff=0.5):
    if high_risk_probability < risk_cutoff:
        return None
    median = cox.predict_median(pd.DataFrame([customer_features]))
    return float(median.iloc[0] if hasattr(median, "iloc") else median)
