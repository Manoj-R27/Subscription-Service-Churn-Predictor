import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Exclude TotalCharges: it mostly repeats tenure x monthly billing and would
# give tenure-related information disproportionate weight in clustering.
SEGMENT_FEATURES = ["tenure", "MonthlyCharges", "AverageCharges", "SeniorCitizen", "IsMonthToMonth", "HasInternet"]


def fit_segmentation(frame, max_k=8, random_state=42):
    values = StandardScaler().fit_transform(frame[SEGMENT_FEATURES])
    candidates = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(values)
        candidates.append({"k": k, "inertia": float(model.inertia_), "silhouette": float(silhouette_score(values, model.labels_, sample_size=min(2000, len(values)), random_state=random_state))})
    chosen_k = max(candidates, key=lambda row: row["silhouette"])["k"]
    model = KMeans(n_clusters=chosen_k, n_init=20, random_state=random_state).fit(values)
    result = frame.copy()
    result["segment_id"] = model.labels_
    profile = result.groupby("segment_id").agg(
        customers=("segment_id", "size"), avg_tenure=("tenure", "mean"),
        avg_monthly_charges=("MonthlyCharges", "mean"), avg_total_charges=("TotalCharges", "mean"),
        churn_rate=("Churn", "mean"),
    ).reset_index()
    for column in ["Contract", "InternetService"]:
        dominant = result.groupby("segment_id")[column].agg(lambda values: values.mode().iat[0])
        profile[f"dominant_{column}"] = profile["segment_id"].map(dominant)
    profile["name"] = profile.apply(_segment_name, axis=1)
    return result, profile.sort_values("segment_id"), candidates


def _segment_name(row):
    if row["dominant_Contract"] == "Month-to-month" and row["avg_monthly_charges"] >= 70:
        return "High-Value At-Risk"
    if row["avg_tenure"] >= 40:
        return "Loyal Long-Tenure"
    if row["dominant_Contract"] == "Two year":
        return "Committed Contract"
    return "Emerging / Watchlist"
