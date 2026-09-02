import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from src.data import BASE_CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, load_data, split_xy
from src.evaluate import BUSINESS_PARAMS, CV, SEARCH_SPACE, calibration_points, compare_models, expected_net_benefit, optimize_threshold, summary_metrics, threshold_analysis, tune_xgb
from src.explain import save_global_shap
from src.features import build_pipeline, feature_names
from src.models import model_catalog, tuned_xgb
from src.segment import fit_segmentation
from src.survival import fit_survival
from src.uplift import fit_t_learner, simulate_retention_offer, uplift_metrics

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
RANDOM_STATE = 42
ARTIFACTS.mkdir(exist_ok=True)


def main():
    frame = load_data()
    X, y = split_xy(frame)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    comparison = compare_models(model_catalog(RANDOM_STATE), X_train, y_train)
    search = tune_xgb(build_pipeline(tuned_xgb(RANDOM_STATE)), X_train, y_train)
    best_pipeline = search.best_estimator_
    calibrated_models = {method: CalibratedClassifierCV(best_pipeline, method=method, cv=CV).fit(X_train, y_train) for method in ["sigmoid", "isotonic"]}
    calibration_results = {method: summary_metrics(y_test, model.predict_proba(X_test)[:, 1]) for method, model in calibrated_models.items()}
    chosen_method = "sigmoid"
    calibrated = calibrated_models[chosen_method]
    probabilities = calibrated.predict_proba(X_test)[:, 1]
    threshold_table = threshold_analysis(y_test, probabilities)
    selected = optimize_threshold(y_test, probabilities)
    baseline = expected_net_benefit(y_test, probabilities, 0.5)

    segment_frame, profiles, diagnostics = fit_segmentation(frame, random_state=RANDOM_STATE)
    segment_frame["predicted_risk"] = calibrated.predict_proba(X)[:, 1]
    segment_cross_tab = segment_frame.groupby("segment_id").agg(actual_churn_rate=("Churn", "mean"), predicted_risk=("predicted_risk", "mean")).reset_index()
    _, _, cox, hazards = fit_survival(frame)

    uplift_frame = simulate_retention_offer(frame, random_state=RANDOM_STATE)
    uplift_features = ["tenure", "MonthlyCharges", "TotalCharges", "AverageCharges", "SeniorCitizen", "IsMonthToMonth", "HasInternet"]
    _, _, uplift_test = fit_t_learner(uplift_frame, uplift_features, risk_scores=pd.Series(calibrated.predict_proba(X)[:, 1], index=frame.index), random_state=RANDOM_STATE)
    uplift_summary = uplift_metrics(uplift_test)

    preprocessor = best_pipeline.named_steps["preprocessor"]
    xgb_model = best_pipeline.named_steps["model"]
    transformed_test = preprocessor.transform(X_test)
    transformed_all = preprocessor.transform(X)
    names = feature_names(preprocessor)
    save_global_shap(xgb_model, pd.DataFrame(transformed_test, columns=names), names, ARTIFACTS / "shap_summary.png")
    calibration_plot(y_test, probabilities, ARTIFACTS / "calibration_curve.png")
    threshold_plot(threshold_table, ARTIFACTS / "threshold_optimization.png")
    segment_plot(diagnostics, ARTIFACTS / "segment_selection.png")
    uplift_plot(uplift_test, ARTIFACTS / "uplift_gain.png")

    joblib.dump(calibrated, ARTIFACTS / "churn_model.pkl")
    joblib.dump(best_pipeline, ARTIFACTS / "xgb_model.pkl")
    joblib.dump(preprocessor, ARTIFACTS / "preprocessor.pkl")
    joblib.dump(cox, ARTIFACTS / "cox_model.pkl")
    joblib.dump({"features": names, "numeric": NUMERIC_COLUMNS, "categorical": BASE_CATEGORICAL_COLUMNS}, ARTIFACTS / "feature_config.pkl")
    profiles.to_csv(ARTIFACTS / "segment_profiles.csv", index=False)
    comparison.to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    threshold_table.to_csv(ARTIFACTS / "threshold_analysis.csv", index=False)
    (ARTIFACTS / "calibration_metrics.csv").write_text(pd.DataFrame(calibration_results).T.to_csv(), encoding="utf-8")
    metadata = {
        "dataset": {"name": "IBM Telco Customer Churn", "rows": int(len(frame)), "source": "IBM public repository"},
        "feature_version": "telco-v2-average-total-charges",
        "model_version": "xgb-calibrated-v2",
        "random_state": RANDOM_STATE, "training_timestamp_utc": datetime.now(timezone.utc).isoformat(), "python_version": platform.python_version(),
        "evaluation_protocol": "80/20 stratified holdout; model comparison and tuning use StratifiedKFold(5) only on training data; final metrics use untouched test data",
        "cv": {"protocol": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42)", "models": comparison.to_dict(orient="records")},
        "xgboost_search_space": SEARCH_SPACE, "best_xgboost_params": search.best_params_,
        "evaluation": {"selected_model_test_metrics": summary_metrics(y_test, probabilities, selected["threshold"]), "calibration": calibration_results},
        "calibration": {"method": chosen_method, "metrics": calibration_results},
        "business": {"assumptions": BUSINESS_PARAMS, "at_0.5": baseline, "optimized": selected},
        "segments": {"features": ["tenure", "MonthlyCharges", "AverageCharges", "SeniorCitizen", "IsMonthToMonth", "HasInternet"], "selection": diagnostics, "profiles": profiles.to_dict(orient="records"), "cross_tab": segment_cross_tab.to_dict(orient="records")},
        "survival": {"duration": "tenure", "event": "Churn", "excluded_predictors": ["tenure", "TenureGroup", "AverageCharges", "IsMonthToMonth", "HasInternet"], "hazard_ratios": hazards.to_dict(orient="records"), "cox_features": list(cox.params_.index)},
        "uplift": {"treatment": "SIMULATED randomized retention offer", "outcome": "simulated churn after offer", "summary": uplift_summary},
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2, default=float), encoding="utf-8")
    print(comparison.to_string(index=False)); print(json.dumps({"test_metrics": metadata["evaluation"]["selected_model_test_metrics"], "business": metadata["business"]}, indent=2))


def calibration_plot(y_true, probabilities, path):
    predicted, observed = calibration_points(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5)); ax.plot(predicted, observed, "o-", label="Calibrated model"); ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Fraction of churners", title="Probability calibration"); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def threshold_plot(table, path):
    fig, ax = plt.subplots(figsize=(7, 5)); ax.plot(table["threshold"], table["net_benefit"]); ax.set(xlabel="Probability threshold", ylabel="Expected net benefit", title="Threshold optimization"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def segment_plot(diagnostics, path):
    data = pd.DataFrame(diagnostics); fig, ax = plt.subplots(1, 2, figsize=(10, 4)); ax[0].plot(data.k, data.inertia, "o-"); ax[0].set(xlabel="k", ylabel="Inertia", title="Elbow"); ax[1].plot(data.k, data.silhouette, "o-"); ax[1].set(xlabel="k", ylabel="Silhouette", title="Silhouette score"); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def uplift_plot(frame, path):
    from src.uplift import cumulative_gain
    fig, ax = plt.subplots(figsize=(7, 5))
    for column, label in [("risk_score", "Target by churn risk"), ("predicted_retention_uplift", "Target by predicted benefit")]:
        x, gain = cumulative_gain(frame, column); ax.plot(x, gain, label=label)
    ax.plot([0, 1], [0, 0], "--", color="grey", label="Random targeting"); ax.set(xlabel="Share of customers targeted", ylabel="Cumulative incremental churn reduction", title="Simulated uplift targeting"); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__": main()
