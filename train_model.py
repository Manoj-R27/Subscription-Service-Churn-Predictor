
import pandas as pd, numpy as np, joblib, shap, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
warnings.filterwarnings('ignore')

# Load
df = pd.read_csv('https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df.drop('customerID', axis=1, inplace=True)

# Feature engineering
df['AverageCharges'] = df['MonthlyCharges'] / df['tenure'].replace(0, 1)
df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
df['HasInternet'] = (df['InternetService'] != 'No').astype(int)
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0,12,24,48,60,100], labels=[0,1,2,3,4]).astype(int)

# Split
X = df.drop('Churn', axis=1)
y = df['Churn']
numeric_cols = ['tenure','MonthlyCharges','TotalCharges','AverageCharges','IsMonthToMonth','HasInternet','TenureGroup']
categorical_cols = [c for c in X.columns if c not in numeric_cols]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess
preprocessor = ColumnTransformer([
    ('num', 'passthrough', numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)
cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols).tolist()
all_features = numeric_cols + cat_features

# SMOTE
X_train_bal, y_train_bal = SMOTE(random_state=42).fit_resample(X_train_proc, y_train)
print(f"After SMOTE: {dict(pd.Series(y_train_bal).value_counts())}")

# XGBoost
xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, random_state=42, eval_metric='logloss')
xgb.fit(X_train_bal, y_train_bal)
print("XGBoost trained!")

# Evaluate
y_pred = xgb.predict(X_test_proc)
y_proba = xgb.predict_proba(X_test_proc)[:,1]
roc_auc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {round(roc_auc, 4)}")
print(classification_report(y_test, y_pred))

# Business impact
tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel()
rev = tp*70*0.3*83*12
print(f"True Positives: {tp}")
print(f"Annual revenue saved: INR {rev:,.0f}")
print(f"Approx: INR {rev/100000:.1f} Lakhs/year")

# SHAP
X_test_df = pd.DataFrame(X_test_proc, columns=all_features)
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_test_df)
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_test_df, plot_type='bar', max_display=10, show=False)
plt.title('Top 10 Features Driving Churn (SHAP)')
plt.tight_layout()
plt.savefig('shap_plot.png', dpi=150, bbox_inches='tight')
print("SHAP saved!")

# Save all files
joblib.dump(xgb, 'XGB_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')
joblib.dump(all_features, 'features.pkl')
joblib.dump({'numeric': numeric_cols, 'categorical': categorical_cols}, 'col_config.pkl')
print("All files saved!")
print(f"FINAL ROC-AUC: {round(roc_auc, 4)}")
