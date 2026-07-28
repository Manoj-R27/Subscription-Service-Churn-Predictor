
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Prediction Dashboard", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    xgb = joblib.load("XGB_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    features = joblib.load("features.pkl")
    col_config = joblib.load("col_config.pkl")
    return xgb, preprocessor, features, col_config

xgb, preprocessor, features, col_config = load_model()

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("**XGBoost + SHAP Explainability + Business Impact** | by Manoj-R27")

tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Churn",
    "🧠 SHAP Explainability",
    "💰 Business Impact"
])

with tab1:
    st.header("Predict Customer Churn")
    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.number_input("Tenure (months)", 0, 100, 12)
        MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col2:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

    with col3:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Has Partner?", ["Yes", "No"])
        Dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    if st.button("🔍 Predict Churn", type="primary"):
        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges,
            'AverageCharges': MonthlyCharges / max(tenure, 1),
            'IsMonthToMonth': 1 if Contract == 'Month-to-month' else 0,
            'HasInternet': 1 if InternetService != 'No' else 0,
            'TenureGroup': min(int(tenure // 12), 4),
        }])

        try:
            input_proc = preprocessor.transform(input_data[col_config['numeric'] + col_config['categorical']])
            proba = xgb.predict_proba(input_proc)[0][1]
            pred = xgb.predict(input_proc)[0]

            st.subheader("Prediction Result:")
            col_a, col_b = st.columns(2)
            with col_a:
                if pred == 1:
                    st.error("⚠️ HIGH CHURN RISK")
                else:
                    st.success("✅ LOW CHURN RISK")
            with col_b:
                st.metric("Churn Probability", f"{proba*100:.1f}%")

            st.progress(float(proba))

            if proba > 0.7:
                st.warning("🚨 Immediate retention action recommended!")
            elif proba > 0.4:
                st.info("⚠️ Monitor this customer closely")
            else:
                st.success("✅ Customer likely to stay")

        except Exception as e:
            st.error(f"Prediction error: {e}")

with tab2:
    st.header("🧠 SHAP Feature Importance")
    st.markdown("Why does the model predict churn? SHAP values explain each feature's contribution.")
    st.image("shap_plot.png", caption="Top 10 features driving churn", use_column_width=True)
    st.subheader("Key Insights:")
    st.markdown("""
    - **IsMonthToMonth** — strongest churn driver. Month-to-month customers churn most.
    - **Tenure** — longer tenure = lower churn risk. New customers are highest risk.
    - **AverageCharges** — higher average charges increase churn probability.
    - **TotalCharges** — customers who have paid more overall tend to stay longer.
    - **InternetService_Fiber optic** — fiber optic users churn more than DSL users.
    """)

with tab3:
    st.header("💰 Business Impact Calculator")
    st.markdown("Estimate revenue saved by deploying this model in production")

    col1, col2 = st.columns(2)
    with col1:
        total_customers = st.number_input("Total customers", 1000, 1000000, 7032)
        avg_monthly_revenue = st.number_input("Avg monthly revenue per customer ($)", 10, 500, 70)
    with col2:
        churn_rate = st.slider("Expected churn rate (%)", 5, 50, 26)
        retention_rate = st.slider("Retention success rate (%)", 10, 50, 30)

    if st.button("💰 Calculate Impact", type="primary"):
        expected_churners = int(total_customers * churn_rate / 100)
        correctly_identified = int(expected_churners * 0.836)
        customers_saved = int(correctly_identified * retention_rate / 100)
        monthly_saved = customers_saved * avg_monthly_revenue
        annual_saved = monthly_saved * 12

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Churners identified", f"{correctly_identified:,}")
        with col_b:
            st.metric("Customers retained", f"{customers_saved:,}")
        with col_c:
            st.metric("Annual revenue saved", f"INR {annual_saved*83:,.0f}")

        st.success(f"This model saves approximately **INR {annual_saved*83/100000:.1f} Lakhs/year**!")
