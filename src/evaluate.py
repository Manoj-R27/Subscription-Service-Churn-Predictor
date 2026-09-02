import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SEARCH_SPACE = {"model__max_depth": [3, 4, 5, 6, 8], "model__learning_rate": [0.02, 0.05, 0.1, 0.2], "model__n_estimators": [150, 250, 400, 600], "model__subsample": [0.7, 0.8, 0.9, 1.0], "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0], "model__min_child_weight": [1, 2, 5, 10]}
SCORING = {"roc_auc": "roc_auc", "average_precision": "average_precision", "accuracy": "accuracy", "recall": "recall", "precision": "precision", "f1": "f1", "brier": "neg_brier_score"}
BUSINESS_PARAMS = {"monthly_revenue": 70.0, "retention_effect": 0.30, "intervention_cost": 12.0, "contact_cost": 2.0, "customer_lifetime_months": 12, "false_positive_cost": 0.0}


def compare_models(models, X, y):
    rows = []
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=CV, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        for metric in SCORING:
            values = -scores[f"test_{metric}"] if metric == "brier" else scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows.append(row)
    return pd.DataFrame(rows)


def tune_xgb(estimator, X, y):
    search = RandomizedSearchCV(
        estimator, SEARCH_SPACE, n_iter=6, scoring="roc_auc", cv=CV,
        random_state=42, n_jobs=1, refit=True, return_train_score=False,
    )
    search.fit(X, y)
    return search


def expected_net_benefit(y_true, probabilities, threshold, params=None):
    assumptions = {**BUSINESS_PARAMS, **(params or {})}
    target = np.asarray(probabilities) >= threshold; actual = np.asarray(y_true).astype(int)
    tp = int(((target == 1) & (actual == 1)).sum()); fp = int(((target == 1) & (actual == 0)).sum()); targeted = int(target.sum())
    saved = tp * assumptions["monthly_revenue"] * assumptions["retention_effect"] * assumptions["customer_lifetime_months"]
    intervention = targeted * assumptions["intervention_cost"]; contact = targeted * assumptions["contact_cost"]; opportunity = fp * assumptions["false_positive_cost"]
    return {"threshold": float(threshold), "precision": float(precision_score(actual, target, zero_division=0)), "recall": float(recall_score(actual, target, zero_division=0)), "f1": float(f1_score(actual, target, zero_division=0)), "number_targeted": targeted, "true_positives": tp, "false_positives": fp, "expected_revenue_saved": float(saved), "intervention_cost": float(intervention), "contact_cost": float(contact), "net_benefit": float(saved - intervention - contact - opportunity)}


def optimize_threshold(y_true, probabilities):
    outcomes = threshold_analysis(y_true, probabilities)
    return outcomes.loc[outcomes["net_benefit"].idxmax()].to_dict()


def threshold_analysis(y_true, probabilities, params=None):
    return pd.DataFrame([expected_net_benefit(y_true, probabilities, threshold, params) for threshold in np.linspace(0.05, 0.95, 91)])


def calibration_points(y_true, probabilities):
    fraction, mean_predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    return mean_predicted.tolist(), fraction.tolist()


def summary_metrics(y_true, probabilities, threshold=0.5):
    predicted = np.asarray(probabilities) >= threshold
    return {"roc_auc": float(roc_auc_score(y_true, probabilities)), "average_precision": float(average_precision_score(y_true, probabilities)), "accuracy": float(accuracy_score(y_true, predicted)), "precision": float(precision_score(y_true, predicted, zero_division=0)), "recall": float(recall_score(y_true, predicted, zero_division=0)), "f1": float(f1_score(y_true, predicted, zero_division=0)), "brier_score": float(brier_score_loss(y_true, probabilities)), "confusion_matrix": confusion_matrix(y_true, predicted).tolist()}
