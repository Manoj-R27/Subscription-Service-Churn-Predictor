# app.py — Streamlit Churn Prediction App

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("logistic_churn_model.pkl")

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

st.title("🔮 Customer Churn Prediction System")
st.markdown("Enter customer details below to predict if they are likely to **churn**.")

# --- Input fields for customer data ---
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Has Partner?", ["Yes", "No"])
Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
tenure = st.number_input("Tenure (in months)", min_value=0, max_value=100, value=12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=1000.0)

# Prepare DataFrame
data = pd.DataFrame({
    "customerID": ["0000"],  # placeholder — pipeline expects this column
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

if st.button("🔍 Predict Churn"):
    try:
        probability = model.predict_proba(data)[0][1]
        prediction = model.predict(data)[0]

        st.subheader("🎯 Prediction Result:")
        st.write(f"**Predicted Churn:** {'Yes' if prediction == 1 else 'No'}")
        st.write(f"**Churn Probability:** {probability:.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Tip: model was trained with a different set of columns. Best fix: remove 'customerID' from training data in train_model.py and retrain the model.")
