# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# 1️⃣ Load Dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# 2️⃣ Clean and preprocess
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 3️⃣ Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Identify numeric and categorical columns
numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = [col for col in X.columns if col not in numeric_cols]

# 4️⃣ Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# 5️⃣ Build Logistic Regression pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# 6️⃣ Split train-test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 7️⃣ Train the model
model.fit(X_train, y_train)
print("✅ Logistic Regression model trained successfully!")

# 8️⃣ Evaluate briefly
score = model.score(X_test, y_test)
print(f"📈 Model Accuracy: {score:.4f}")

# 9️⃣ Save model
joblib.dump(model, "logistic_churn_model.pkl")
print("💾 Model saved as 'logistic_churn_model.pkl'")
