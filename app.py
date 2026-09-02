from pathlib import Path

import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.data import BASE_CATEGORICAL_COLUMNS, customer_row, load_data
from src.evaluate import BUSINESS_PARAMS
from src.segment import fit_segmentation
from src.survival import cox_row, survival_estimate

ROOT = Path(__file__).parent
ARTIFACTS = ROOT / "artifacts"
st.set_page_config(page_title="Telco Churn Intelligence", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 2rem;}
[data-testid="stMetric"] {background: #f4f7f5; border-left: 4px solid #1f6f5b; padding: .8rem;}
.risk-note {border-radius: 6px; padding: .7rem 1rem; background: #eef6f2; color: #173b32;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    return {"model": joblib.load(ARTIFACTS / "churn_model.pkl"), "xgb": joblib.load(ARTIFACTS / "xgb_model.pkl"), "cox": joblib.load(ARTIFACTS / "cox_model.pkl"), "config": joblib.load(ARTIFACTS / "feature_config.pkl")}


@st.cache_data
def get_data(): return load_data()


def customer_inputs(frame):
    defaults = {"gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 70.0, "TotalCharges": 1000.0}
    labels = {"gender": "Gender", "SeniorCitizen": "Senior citizen", "Partner": "Partner", "Dependents": "Dependents", "tenure": "Tenure (months)", "PhoneService": "Phone service", "MultipleLines": "Multiple lines", "InternetService": "Internet service", "OnlineSecurity": "Online security", "OnlineBackup": "Online backup", "DeviceProtection": "Device protection", "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies", "Contract": "Contract", "PaperlessBilling": "Paperless billing", "PaymentMethod": "Payment method", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges"}
    values = {}
    columns = st.columns(3)
    input_columns = BASE_CATEGORICAL_COLUMNS + ["tenure", "MonthlyCharges", "TotalCharges"]
    for index, column in enumerate(input_columns):
        default = defaults[column]; container = columns[index % 3]
        if column == "SeniorCitizen": values[column] = 1 if container.selectbox(labels[column], ["No", "Yes"], index=default, help="Stored as 0 for No and 1 for Yes.") == "Yes" else 0
        elif isinstance(default, (int, float)): values[column] = container.number_input(labels[column], value=default)
        else:
            options = sorted(frame[column].dropna().unique()); values[column] = container.selectbox(labels[column], options, index=options.index(default))
    return values


def display_feature_name(name):
    """Convert transformed pipeline names into customer-facing explanations."""
    clean = name.replace("num__", "").replace("cat__", "")
    replacements = {"IsMonthToMonth": "Month-to-month contract", "MonthlyCharges": "Monthly charges", "TotalCharges": "Total charges", "AverageCharges": "Average charges per month", "tenure": "Tenure", "SeniorCitizen": "Senior citizen", "PaymentMethod_": "Payment method: ", "InternetService_": "Internet service: ", "OnlineSecurity_": "Online security: ", "TechSupport_": "Tech support: ", "Contract_": "Contract: ", "MultipleLines_": "Multiple lines: "}
    for source, target in replacements.items(): clean = clean.replace(source, target)
    return clean.replace("_", " ")


def display_column_name(name):
    labels = {"segment_id": "Segment", "customers": "Customers", "avg_tenure": "Average tenure (months)", "avg_monthly_charges": "Average monthly charges", "avg_total_charges": "Average total charges", "churn_rate": "Actual churn rate", "dominant_Contract": "Dominant contract", "dominant_InternetService": "Dominant internet service", "name": "Business label", "threshold": "Probability threshold", "precision": "Precision", "recall": "Recall", "f1": "F1 score", "number_targeted": "Customers targeted", "true_positives": "Churners identified", "false_positives": "False positives", "expected_revenue_saved": "Expected retained revenue", "intervention_cost": "Intervention cost", "contact_cost": "Contact cost", "net_benefit": "Expected net benefit", "roc_auc_mean": "ROC-AUC mean", "roc_auc_std": "ROC-AUC std", "average_precision_mean": "PR-AUC mean", "average_precision_std": "PR-AUC std", "accuracy_mean": "Accuracy mean", "accuracy_std": "Accuracy std", "recall_mean": "Recall mean", "recall_std": "Recall std", "precision_mean": "Precision mean", "precision_std": "Precision std", "f1_mean": "F1 mean", "f1_std": "F1 std", "brier_mean": "Brier score mean", "brier_std": "Brier score std", "model": "Model"}
    return labels.get(name, name.replace("_", " ").title())


artifacts = load_artifacts(); frame = get_data()
metadata = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
selected_threshold = float(metadata["business"]["optimized"]["threshold"])
st.title("Telco Churn Intelligence")
st.caption("A calibrated, explainable retention decision-support system")
risk_tab, business_tab, segment_tab, survival_tab, uplift_tab, evidence_tab = st.tabs(["Customer Risk", "Business Decision", "Customer Segments", "Survival Analysis", "Uplift", "Model Evidence"])

with risk_tab:
    st.subheader("Score a customer")
    with st.form("customer_form"):
        values = customer_inputs(frame); submitted = st.form_submit_button("Score customer", type="primary")
    if submitted:
        row = customer_row(values); probability = float(artifacts["model"].predict_proba(row)[0, 1]); st.metric("Calibrated churn probability", f"{probability:.1%}"); st.progress(probability)
        risk_label = "High risk" if probability >= selected_threshold else "Lower risk"
        decision = "Prioritize retention outreach" if probability >= selected_threshold else "Monitor customer"
        st.markdown(f'<div class="risk-note"><strong>{risk_label}</strong> | {decision} | Business operating threshold: {selected_threshold:.0%}</div>', unsafe_allow_html=True)
        cox_features = cox_row(row).reindex(columns=artifacts["cox"].params_.index, fill_value=0)
        estimate = survival_estimate(artifacts["cox"], cox_features.iloc[0].to_dict())
        st.write("Estimated probability this customer remains active")
        survival_columns = st.columns(3)
        for container, horizon in zip(survival_columns, ["6", "12", "24"]): container.metric(f"After {horizon} months", f"{estimate['survival_probability'][horizon]:.1%}")
        if estimate["median_months"] is not None: st.metric("Median predicted survival", f"{estimate['median_months']:.0f} months")
        else: st.info("Median survival time was not reached within the observed follow-up.")
        transformed = artifacts["xgb"].named_steps["preprocessor"].transform(row); names = artifacts["xgb"].named_steps["preprocessor"].get_feature_names_out().tolist()
        explain = __import__("src.explain", fromlist=["local_explanation"]).local_explanation(artifacts["xgb"].named_steps["model"], transformed, names)
        explain_frame = pd.DataFrame(explain)
        explain_frame["feature"] = explain_frame["feature"].map(display_feature_name)
        explain_frame["direction"] = explain_frame["shap_value"].map(lambda value: "Toward churn" if value > 0 else "Away from churn")
        explain_frame = explain_frame.rename(columns={"feature": "Driver", "shap_value": "Contribution", "direction": "Effect"})
        st.caption("Positive contribution increases predicted churn; negative contribution lowers it.")
        st.dataframe(explain_frame, use_container_width=True, hide_index=True)

with business_tab:
    st.subheader("Net-benefit threshold")
    st.write("Business assumptions, not observed causal effects")
    assumption_labels = {"monthly_revenue": "Monthly revenue", "retention_effect": "Expected retention effect", "intervention_cost": "Intervention cost", "contact_cost": "Contact cost", "customer_lifetime_months": "Customer lifetime (months)", "false_positive_cost": "False-positive opportunity cost"}
    st.dataframe(pd.DataFrame({"Assumption": [assumption_labels[key] for key in BUSINESS_PARAMS], "Value": list(BUSINESS_PARAMS.values())}), use_container_width=True, hide_index=True)
    table = pd.read_csv(ARTIFACTS / "threshold_analysis.csv"); selected = table.loc[table.net_benefit.idxmax()]
    st.metric("Selected threshold", f"{selected.threshold:.2f}"); st.metric("Expected net benefit", f"{selected.net_benefit:,.0f}")
    threshold_view = table[table.threshold.isin([.5, selected.threshold])].rename(columns=display_column_name)
    st.dataframe(threshold_view, use_container_width=True, hide_index=True)
    st.image(str(ARTIFACTS / "threshold_optimization.png"), caption="Expected net benefit by threshold")

with segment_tab:
    st.subheader("Customer segments")
    _, profiles, _ = fit_segmentation(frame); profile_view = profiles.rename(columns=display_column_name); st.dataframe(profile_view, use_container_width=True, hide_index=True); st.bar_chart(profiles.set_index("name")[["customers", "churn_rate"]]); st.image(str(ARTIFACTS / "segment_selection.png"), caption="Elbow and silhouette diagnostics")

with survival_tab:
    st.subheader("When might churn happen?")
    st.write("Classification estimates if churn occurs; survival analysis estimates when it may occur, supporting contact timing.")
    lifelines = __import__("lifelines", fromlist=["KaplanMeierFitter"]); group_by = st.selectbox("Compare by", ["Contract", "InternetService"]); fig, ax = plt.subplots(figsize=(8, 4)); lifelines.KaplanMeierFitter().fit(frame.tenure, frame.Churn, label="All customers").plot_survival_function(ax=ax)
    for value, group in frame.groupby(group_by): lifelines.KaplanMeierFitter().fit(group.tenure, group.Churn, label=str(value)).plot_survival_function(ax=ax)
    ax.set(xlabel="Tenure (months)", ylabel="Probability remaining", title=f"Kaplan-Meier survival by {group_by}"); st.pyplot(fig); plt.close(fig)

with uplift_tab:
    st.subheader("Retention uplift prioritization")
    st.error("SIMULATED - NOT HISTORICAL TREATMENT DATA")
    st.write("Positive predicted retention uplift means the offer is expected to reduce churn more for that customer. This is a T-learner demonstration using randomized synthetic treatment.")
    st.image(str(ARTIFACTS / "uplift_gain.png"), caption="Risk targeting vs predicted benefit targeting")
    uplift_summary = metadata["uplift"]["summary"]
    uplift_view = pd.DataFrame([{"Metric": "Mean predicted churn reduction", "Value": uplift_summary["mean_predicted_retention_uplift"]}, {"Metric": "Mean simulated treatment effect", "Value": uplift_summary["true_effect_mean"]}, {"Metric": "Correlation with simulated effect", "Value": uplift_summary["correlation_with_true_effect"]}, {"Metric": "Mean absolute error", "Value": uplift_summary["mae"]}, {"Metric": "Root mean squared error", "Value": uplift_summary["rmse"]}])
    st.dataframe(uplift_view, use_container_width=True, hide_index=True)

with evidence_tab:
    st.subheader("Model evidence")
    evidence_view = pd.DataFrame(metadata["cv"]["models"]).rename(columns=display_column_name); st.dataframe(evidence_view, use_container_width=True, hide_index=True); st.write("Final holdout metrics"); st.dataframe(pd.DataFrame([metadata["evaluation"]["selected_model_test_metrics"]]).rename(columns=display_column_name), use_container_width=True, hide_index=True); st.image(str(ARTIFACTS / "calibration_curve.png"), caption="Calibration curve"); st.image(str(ARTIFACTS / "shap_summary.png"), caption="Global SHAP importance")
