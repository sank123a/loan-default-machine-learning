import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# -----------------------------
# 1. Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "UCI_Credit_Card.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_default_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
TARGET_COL = "default.payment.next.month"

# -----------------------------
# 2. Load artifacts at startup
# -----------------------------
@st.cache_resource
def load_artifacts():
    # Load dataset to get full feature names and medians
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET_COL])
    feature_names = X.columns.tolist()
    median_values = X.median(numeric_only=True)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler, feature_names, median_values

model, scaler, feature_names, median_values = load_artifacts()

# -----------------------------
# 3. Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Loan Default Risk Predictor",
    layout="centered"
)

st.title("Loan Default Risk Prediction (UCI Credit Card)")
st.write(
    "This app uses a machine learning model trained on the "
    "UCI Credit Card Default dataset to estimate the probability "
    "that a customer will default next month."
)

st.markdown("---")

# -----------------------------
# 4. Sidebar inputs
# -----------------------------
st.sidebar.header("Customer Inputs")

LIMIT_BAL = st.sidebar.number_input(
    "Credit Limit (LIMIT_BAL)",
    min_value=10000,
    max_value=1000000,
    value=200000,
    step=10000
)

SEX = st.sidebar.selectbox(
    "Sex (1 = Male, 2 = Female)",
    options=[1, 2],
    index=0
)

EDUCATION = st.sidebar.selectbox(
    "Education (1=Grad,2=Uni,3=HS,4=Other)",
    options=[1, 2, 3, 4],
    index=1
)

MARRIAGE = st.sidebar.selectbox(
    "Marital Status (1=Married,2=Single,3=Other)",
    options=[1, 2, 3],
    index=0
)

AGE = st.sidebar.slider(
    "Age",
    min_value=18,
    max_value=80,
    value=35
)

st.sidebar.markdown("### Repayment Status (PAY_0..PAY_6)")
st.sidebar.write("−2: No consumption, −1: Paid duly, 0: Use of revolving, 1–6: Months delay")

PAY_0 = st.sidebar.selectbox("PAY_0 (last month)", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6], index=2)
PAY_2 = st.sidebar.selectbox("PAY_2", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6], index=2)
PAY_3 = st.sidebar.selectbox("PAY_3", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6], index=2)
PAY_4 = st.sidebar.selectbox("PAY_4", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6], index=2)
PAY_5 = st.sidebar.selectbox("PAY_5", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6], index=2)
PAY_6 = st.sidebar.selectbox("PAY_6", options=[-2, -1, 0, 1, 2, 3, 4, 5, 6], index=2)

st.sidebar.markdown("### Billing & Payment (last month)")
BILL_AMT1 = st.sidebar.number_input(
    "Last Month Bill Amount (BILL_AMT1)",
    min_value=0,
    max_value=2000000,
    value=50000,
    step=5000
)

PAY_AMT1 = st.sidebar.number_input(
    "Last Month Payment Amount (PAY_AMT1)",
    min_value=0,
    max_value=2000000,
    value=20000,
    step=5000
)

# -----------------------------
# 5. Build complete feature vector
# -----------------------------
def build_feature_vector():
    """
    Start from median values for all features,
    then overwrite with the user inputs (if feature exists).
    This guarantees the same feature set & order as training.
    """
    input_series = median_values.copy()

    # Overwrite with user inputs where applicable
    if "LIMIT_BAL" in feature_names:
        input_series["LIMIT_BAL"] = LIMIT_BAL
    if "SEX" in feature_names:
        input_series["SEX"] = SEX
    if "EDUCATION" in feature_names:
        input_series["EDUCATION"] = EDUCATION
    if "MARRIAGE" in feature_names:
        input_series["MARRIAGE"] = MARRIAGE
    if "AGE" in feature_names:
        input_series["AGE"] = AGE

    if "PAY_0" in feature_names:
        input_series["PAY_0"] = PAY_0
    if "PAY_2" in feature_names:
        input_series["PAY_2"] = PAY_2
    if "PAY_3" in feature_names:
        input_series["PAY_3"] = PAY_3
    if "PAY_4" in feature_names:
        input_series["PAY_4"] = PAY_4
    if "PAY_5" in feature_names:
        input_series["PAY_5"] = PAY_5
    if "PAY_6" in feature_names:
        input_series["PAY_6"] = PAY_6

    if "BILL_AMT1" in feature_names:
        input_series["BILL_AMT1"] = BILL_AMT1
    if "PAY_AMT1" in feature_names:
        input_series["PAY_AMT1"] = PAY_AMT1

    # Build DataFrame in *exact* feature order used during training
    input_df = pd.DataFrame(
        [input_series[feature_names].values],
        columns=feature_names
    )
    return input_df

# -----------------------------
# 6. Prediction logic
# -----------------------------
st.markdown("### Prediction")

if st.button("Predict Default Risk"):
    # Build full feature row
    input_df = build_feature_vector()
    # Scale with same scaler used in training
    input_scaled = scaler.transform(input_df)
    # Predict probability of default (class 1)
    prob_default = model.predict_proba(input_scaled)[0, 1]

    # Risk band
    if prob_default < 0.3:
        risk_category = "LOW RISK"
    elif prob_default < 0.6:
        risk_category = "MEDIUM RISK"
    else:
        risk_category = "HIGH RISK"

    st.subheader("Result")
    st.metric(
        label="Estimated Default Probability (Next Month)",
        value=f"{prob_default * 100:.2f} %"
    )
    st.write(f"**Risk Category:** {risk_category}")

    st.markdown("#### Full Feature Vector Used")
    st.dataframe(input_df)

    st.info(
        "Note: For features not shown in the sidebar (e.g., BILL_AMT2-6, PAY_AMT2-6), "
        "median values from the training dataset were used."
    )
