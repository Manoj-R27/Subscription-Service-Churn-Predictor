# Customer Churn Prediction Dashboard

An end-to-end ML project predicting customer churn using XGBoost with SHAP explainability and business impact analysis.

## Results
| Metric | Logistic Regression | XGBoost |
|--------|-------------------|---------|
| Accuracy | 80.4% | 79.1% |
| ROC-AUC | 0.811 | 0.836 |
| Recall (Churn) | 43.6% | 59.0% |

## Business Impact
- True Positives (churners correctly identified): 219
- Estimated annual revenue saved: INR 45.8 Lakhs/year
- Based on 30% retention success rate on identified churners

## Key Insights (SHAP)
- IsMonthToMonth contract is the strongest churn driver
- Longer tenure customers are less likely to churn
- Higher average charges increase churn probability

## Tech Stack
Python, XGBoost, SHAP, Scikit-learn, SMOTE, Streamlit, Pandas, NumPy

## Project Structure
- `train_model.py` — data cleaning, feature engineering, XGBoost training, SHAP
- `app.py` — Streamlit dashboard with 3 tabs
- `Subscription_Service_Churn_Predictor.ipynb` — full EDA and model development notebook

## Dataset
IBM Telco Customer Churn — 7,032 customers, 20 features
Source: IBM/telco-customer-churn-on-icp4d

## Author
Manoj-R27 | github.com/Manoj-R27
