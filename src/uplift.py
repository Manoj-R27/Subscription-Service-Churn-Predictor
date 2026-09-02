import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def simulate_retention_offer(frame, random_state=42):
    """Simulate treatment and outcomes for demonstrating causal ML only.

    This is not historical intervention data: treatment is randomized and the
    effect is deliberately stronger for month-to-month, high-charge customers.
    """
    rng = np.random.default_rng(random_state)
    result = frame.copy()
    result["treatment"] = rng.binomial(1, 0.5, len(result))
    responsiveness = (
        0.04 + 0.08 * (result["Contract"] == "Month-to-month")
        + 0.05 * (result["MonthlyCharges"] >= result["MonthlyCharges"].median())
        - 0.03 * (result["tenure"] >= 36)
    )
    base_probability = np.clip(0.10 + 0.45 * result["Churn"] + rng.normal(0, 0.08, len(result)), 0.01, 0.99)
    outcome_probability = np.clip(base_probability - result["treatment"] * responsiveness, 0.01, 0.99)
    result["simulated_outcome"] = rng.binomial(1, outcome_probability)
    result["simulated_effect"] = responsiveness
    return result


def fit_t_learner(frame, feature_columns, risk_scores=None, random_state=42):
    train, test = train_test_split(frame, test_size=0.25, stratify=frame["treatment"], random_state=random_state)
    control = RandomForestRegressor(n_estimators=250, min_samples_leaf=8, random_state=random_state, n_jobs=-1)
    treated = RandomForestRegressor(n_estimators=250, min_samples_leaf=8, random_state=random_state + 1, n_jobs=-1)
    control.fit(train.loc[train.treatment == 0, feature_columns], train.loc[train.treatment == 0, "simulated_outcome"])
    treated.fit(train.loc[train.treatment == 1, feature_columns], train.loc[train.treatment == 1, "simulated_outcome"])
    test = test.copy()
    test["predicted_retention_uplift"] = control.predict(test[feature_columns]) - treated.predict(test[feature_columns])
    test["risk_score"] = test.index.map(risk_scores).astype(float) if risk_scores is not None else 0.0
    return control, treated, test


def uplift_metrics(frame):
    """Compare predicted churn reduction with the known simulated effect."""
    error = frame["predicted_retention_uplift"] - frame["simulated_effect"]
    return {"mean_predicted_retention_uplift": float(frame["predicted_retention_uplift"].mean()), "true_effect_mean": float(frame["simulated_effect"].mean()), "correlation_with_true_effect": float(frame["predicted_retention_uplift"].corr(frame["simulated_effect"])), "mae": float(error.abs().mean()), "rmse": float(np.sqrt((error ** 2).mean()))}


def cumulative_gain(frame, score_column, outcome="simulated_outcome", treatment="treatment"):
    ordered = frame.sort_values(score_column, ascending=False).reset_index(drop=True)
    gains, treated_count, control_count, treated_outcome, control_outcome = [], 0, 0, 0, 0
    for _, row in ordered.iterrows():
        if row[treatment] == 1:
            treated_count += 1; treated_outcome += row[outcome]
        else:
            control_count += 1; control_outcome += row[outcome]
        effect = (treated_outcome / treated_count - control_outcome / control_count) if treated_count and control_count else 0
        gains.append(effect * (treated_count + control_count))
    return np.arange(1, len(ordered) + 1) / len(ordered), np.asarray(gains)
