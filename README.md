# Subscription Service Churn Predictor

An end-to-end customer churn project on the real IBM Telco Customer Churn dataset. It combines leakage-safe classification, probability calibration, business threshold optimization, interpretable segmentation, survival analysis, SHAP explanations, and a clearly simulated uplift demonstration.

## Problem

Retention teams need both a risk estimate and a practical decision rule: who should be contacted, when, and whether the expected value justifies the intervention. This project uses real tabular customer, service, contract, billing, and churn fields only. It adds no NLP or fabricated text.

## Dataset and Features

The IBM Telco dataset contains 7,043 customer records; 7,032 remain after converting `TotalCharges` to numeric and removing blank values. `src/data.py` is the single source of truth. `AverageCharges = TotalCharges / max(tenure, 1)`. `TenureGroup`, `IsMonthToMonth`, and `HasInternet` are documented, non-target convenience features; they are not used in Cox survival predictors when they duplicate duration or service information.

## Architecture

```text
src/data.py -> src/features.py -> src/models.py -> src/evaluate.py -> artifacts/
                         |-> src/explain.py
                         |-> src/segment.py
                         |-> src/survival.py
                         |-> src/uplift.py
train.py -> app.py (Streamlit decision-support UI)
```

- `data.py`: loading, cleaning, target encoding, and feature engineering.
- `features.py`: ColumnTransformer and complete sklearn pipelines.
- `models.py`: Logistic Regression, Random Forest, and XGBoost definitions.
- `evaluate.py`: 5-fold CV, randomized tuning, calibration metrics, and net-benefit thresholds.
- `explain.py`: global and local SHAP explanations.
- `segment.py`: KMeans, sampled silhouette diagnostics, profiles, and descriptive labels.
- `survival.py`: Kaplan-Meier curves and Cox hazards using tenure only as duration.
- `uplift.py`: simulated randomized treatment and T-learner retention benefit.

## Evaluation Methodology

Model comparison and XGBoost tuning use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the training partition. Preprocessing is inside each pipeline, so validation folds never fit scaling or category encoders. A stratified 80/20 holdout is untouched until final evaluation. This practical holdout design avoids reporting model-selection performance as final performance.

### Actual Fresh Results

| Model | CV ROC-AUC | CV PR-AUC | CV F1 | CV Brier |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.846 +/- 0.005 | 0.659 +/- 0.012 | 0.632 +/- 0.008 | 0.165 +/- 0.004 |
| Random Forest | 0.845 +/- 0.006 | 0.655 +/- 0.018 | 0.633 +/- 0.007 | 0.153 +/- 0.003 |
| XGBoost | 0.847 +/- 0.005 | 0.662 +/- 0.009 | 0.585 +/- 0.016 | 0.135 +/- 0.001 |

The calibrated XGBoost model scored ROC-AUC `0.837`, PR-AUC `0.652`, and Brier score `0.141` on the untouched test set. At the selected threshold `0.07`, recall was `0.989`, precision `0.338`, and expected net benefit `77,910`. At the default `0.50`, recall was `0.505`, precision `0.643`, and expected net benefit `43,512`. These values are generated in `artifacts/metadata.json` and `artifacts/model_comparison.csv`.

## Probability Calibration

Sigmoid and isotonic calibration are evaluated on the untouched test set and both metrics are retained in metadata; sigmoid is the configured final method. The calibration curve is saved as `artifacts/calibration_curve.png`. Probabilities shown in Streamlit come from the calibrated model.

## Business Threshold Optimization

The decision rule maximizes expected net benefit, not raw revenue. Centralized assumptions are monthly revenue `70`, retention effect `30%`, intervention cost `12`, contact cost `2`, and 12-month customer lifetime. They are assumptions, not observed causal effects. Full threshold results are in `artifacts/threshold_analysis.csv`; the plot is `artifacts/threshold_optimization.png`.

## Segmentation

KMeans uses tenure, monthly billing, historical average billing, senior-citizen status, month-to-month status, and internet presence. `TotalCharges` is excluded because it largely duplicates tenure and monthly billing. The selected k uses both elbow inertia and a reproducible 2,000-row silhouette sample. KMeans discovers clusters; business-facing names are assigned afterward from profile statistics and are not algorithmically discovered labels.

## Survival Analysis

Kaplan-Meier curves are reported overall and by Contract and InternetService. Cox uses `tenure` as duration and `Churn` as event, but never as a predictor. Baseline customer characteristics such as charges, contract, payment method, services, and demographics are encoded with a saved feature schema. This answers **when** churn risk may materialize, unlike classification, which answers **if** churn occurs. That supports contact timing and prioritization.

## SHAP Explainability

Global SHAP importance is generated from the final XGBoost model. Local explanations are shown for scored customers. Positive SHAP values push the prediction toward churn; negative values push it away from churn.

## Uplift Modeling

The dataset has no historical retention treatment. `src/uplift.py` therefore creates a reproducible randomized synthetic offer, synthetic post-offer churn outcome, and heterogeneous simulated effect. A T-learner estimates individual retention benefit as `P(churn | control) - P(churn | treatment)`, so positive uplift means predicted churn reduction. The project reports correlation, MAE, and RMSE against the known simulated effect and compares risk targeting with uplift targeting in `artifacts/uplift_gain.png`. These are methodological demonstrations, not measured causal effects from a telecom campaign.

## Streamlit Application

Run `streamlit run app.py` to open six tabs: Customer Risk, Business Decision, Customer Segments, Survival Analysis, Uplift, and Model Evidence. The app displays assumptions, calibrated probability, SHAP drivers, individual survival horizons, segment profiles, threshold net benefit, and the simulated-data warning.

## Reproducibility

```powershell
.\.venv\Scripts\Activate.ps1
python train.py
pytest -q
streamlit run app.py
```

Generated artifacts are traceable through `metadata.json`, which records dataset source and row count, feature/model versions, CV protocol, search space and best parameters, metrics, business assumptions, calibration, segmentation, survival features, uplift settings, random seed, Python version, and training timestamp. The virtual environment and caches are ignored by git.

## Limitations

The IBM Telco dataset is observational. Business calculations depend on configurable assumptions. Retention treatment and outcomes are simulated, so uplift results do not establish real causal effectiveness. Survival estimates have the usual proportional-hazards and censoring assumptions. Nothing here claims deployment in a real telecom retention campaign; production use would require prospective experiments, real intervention outcomes, monitoring, and governance.
