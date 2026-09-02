import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data import customer_row, engineer_features
from src.evaluate import BUSINESS_PARAMS, expected_net_benefit
from src.features import build_pipeline
from src.survival import cox_row
from src.uplift import fit_t_learner, simulate_retention_offer


def raw_customer():
    return {"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "No", "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service", "DeviceProtection": "No internet service", "TechSupport": "No internet service", "StreamingTV": "No internet service", "StreamingMovies": "No internet service", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 50.0, "TotalCharges": 0.0}


def test_average_charges_uses_total_charges_and_zero_is_safe():
    result = engineer_features(pd.DataFrame([{"tenure": 0, "MonthlyCharges": 80, "TotalCharges": 120, "Contract": "Month-to-month", "InternetService": "DSL"}]))
    assert result.loc[0, "AverageCharges"] == 120


def test_customer_row_has_expected_engineered_features():
    row = customer_row(raw_customer())
    assert row.loc[0, "AverageCharges"] == 0
    assert row.loc[0, "TenureGroup"] == 0


def test_pipeline_accepts_raw_customer_rows():
    X = engineer_features(pd.DataFrame([raw_customer(), {**raw_customer(), "tenure": 24, "TotalCharges": 1200}]))
    y = [0, 1]
    pipeline = build_pipeline(LogisticRegression(max_iter=1000)).fit(X, y)
    assert pipeline.predict_proba(X).shape == (2, 2)


def test_net_benefit_includes_targeting_costs():
    result = expected_net_benefit([1, 0], [0.9, 0.9], 0.5)
    expected = 70 * .30 * 12 - 2 * (BUSINESS_PARAMS["intervention_cost"] + BUSINESS_PARAMS["contact_cost"])
    assert result["net_benefit"] == expected


def test_uplift_simulation_is_reproducible_and_randomized():
    frame = pd.DataFrame({"Churn": [0, 1] * 20, "Contract": ["Month-to-month", "Two year"] * 20, "MonthlyCharges": [90, 40] * 20, "tenure": [4, 50] * 20})
    first = simulate_retention_offer(frame, 42); second = simulate_retention_offer(frame, 42)
    assert np.array_equal(first["treatment"], second["treatment"])
    assert first["treatment"].nunique() == 2


def test_uplift_uses_positive_retention_benefit_convention():
    frame = pd.DataFrame({"Churn": [0, 1] * 30, "Contract": ["Month-to-month", "Two year"] * 30, "MonthlyCharges": [90, 40] * 30, "tenure": [4, 50] * 30, "TotalCharges": [300, 2000] * 30, "AverageCharges": [75, 40] * 30, "SeniorCitizen": [1, 0] * 30, "IsMonthToMonth": [1, 0] * 30, "HasInternet": [1, 1] * 30})
    simulated = simulate_retention_offer(frame, 42)
    _, _, scored = fit_t_learner(simulated, ["tenure", "MonthlyCharges", "TotalCharges", "AverageCharges", "SeniorCitizen", "IsMonthToMonth", "HasInternet"], random_state=42)
    assert "predicted_retention_uplift" in scored
