# Subscription Service Churn Predictor

A machine learning project using logistic regression to predict customer churn for a subscription-based service. This solution includes data preprocessing, model training, evaluation, and feature importance analysis.

---

## 📦 Project Structure

```
.
├── data/                  # Raw and processed datasets
├── notebook.ipynb         # Jupyter notebook for analysis and modeling
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
└── churn_predictor.py     # (Optional) Script to run the model
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

Open `notebook.ipynb` in Jupyter Notebook or JupyterLab:

```bash
jupyter notebook notebook.ipynb
```

---

## 📊 Data

- **Source**: The dataset typically includes customer demographics, account information, and service usage.
- **Key Columns**: `tenure`, `TotalCharges`, `MonthlyCharges`, `Contract`, `InternetService`, `PaymentMethod`, `SeniorCitizen`, etc.
- **Target**: `Churn` (Yes/No)

---

## 🔬 Model Overview

- **Algorithm**: Logistic Regression
- **Objective**: Predict probability of customer churn
- **Steps**:
  1. Data cleaning & preprocessing
  2. Feature engineering & encoding
  3. Model training & validation
  4. Evaluation: Accuracy, ROC-AUC, Confusion Matrix
  5. Feature importance analysis

---

## 💡 Feature Importance

Top features **increasing** churn:
- High `TotalCharges`
- Month-to-month contracts
- Fiber optic internet service
- Electronic check payment method
- Streaming services enabled

Top features **reducing** churn:
- Long tenure
- Two-year contracts
- DSL or no internet
- Lower monthly charges

(See notebook for full details.)

---

## 📈 Results

- **Accuracy**: _e.g., 82%_
- **ROC-AUC**: _e.g., 0.85_
- **Confusion Matrix**: _see notebook output_
- **Interpretation**: Key drivers of churn identified; actionable insights for retention.

---

## 🛠️ Usage

You can use the notebook to:
- Explore and preprocess your own data
- Train and evaluate the churn prediction model
- Analyze which features most influence customer retention

---

## 📝 How to Adapt

- Replace `data/` with your own dataset (CSV recommended)
- Adjust feature engineering steps to suit your data
- Tune model parameters and try other algorithms (e.g., Random Forest)

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [Kaggle: Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

---

## 👤 Author

- **Your Name:** Manoj-R27  
- **GitHub:** [Manoj-R27](https://github.com/Manoj-R27)

---

## 📄 License

This project is licensed under the MIT License.
