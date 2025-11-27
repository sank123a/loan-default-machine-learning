import os
import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import shap

# -----------------------------
# 1. Paths and configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "UCI_Credit_Card.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "loan_default_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

TARGET_COL = "default.payment.next.month"

# -----------------------------
# 2. Load data, model, scaler
# -----------------------------
print("Loading dataset, model, and scaler...")

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

feature_names = X.columns.tolist()

# Scale using same scaler as training
X_scaled = scaler.transform(X)

# To make SHAP faster, take a sample
# You can increase n_sample if your PC is strong
n_sample = 1000
if X_scaled.shape[0] > n_sample:
    np.random.seed(42)
    idx = np.random.choice(X_scaled.shape[0], size=n_sample, replace=False)
    X_sample = X_scaled[idx]
    y_sample = y.iloc[idx]
else:
    X_sample = X_scaled
    y_sample = y

print("Sample shape for SHAP:", X_sample.shape)

# Convert to DataFrame for better plotting labels
X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

# -----------------------------
# 3. Build SHAP explainer
# -----------------------------
print("Building SHAP explainer...")

# For tree-based models like RandomForest, XGBoost, etc.
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample_df)

# We're interested in the SHAP values for the "default" class (class index 1)
if isinstance(shap_values, list):
    # Classic SHAP for tree models: list of [class0, class1]
    shap_values_default = shap_values[1]
else:
    # Newer SHAP: shape (n_samples, n_features, n_classes)
    # Take SHAP values for class 1
    shap_values_default = shap_values[:, :, 1]

print("SHAP values (class 1) shape:", np.array(shap_values_default).shape)

# -----------------------------
# 4. Global feature importance
# -----------------------------
print("Creating global SHAP summary plots...")

# a) Summary dot plot (impact of features)
plt.figure()
shap.summary_plot(
    shap_values_default,
    X_sample_df,
    show=False
)
plt.tight_layout()
summary_plot_path = os.path.join(OUTPUT_DIR, "shap_summary_dot.png")
plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", summary_plot_path)

# b) Summary bar plot (mean |SHAP| value)
plt.figure()
shap.summary_plot(
    shap_values_default,
    X_sample_df,
    plot_type="bar",
    show=False
)
plt.tight_layout()
bar_plot_path = os.path.join(OUTPUT_DIR, "shap_summary_bar.png")
plt.savefig(bar_plot_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", bar_plot_path)

# -----------------------------
# 5. Local explanation for one example
# -----------------------------
# -----------------------------
# 5. Local explanation for one example
# -----------------------------
print("Creating local SHAP explanation for a single customer...")

# Pick one example – you can change index here
sample_index = 0
x_one = X_sample_df.iloc[sample_index:sample_index + 1]

# Get prediction probability for this example (class 1 = default)
proba_default = model.predict_proba(x_one)[0, 1]
print(f"Example index: {sample_index}, Predicted default probability: {proba_default:.3f}")
print("True label (from sample y):", y_sample.iloc[sample_index])

# Compute SHAP values for this single row
shap_values_one_full = explainer.shap_values(x_one)

if isinstance(shap_values_one_full, list):
    # List version: [class0, class1]
    shap_one = shap_values_one_full[1][0]  # first sample, class 1
    # expected_value is also list-like [class0, class1]
    base_val = explainer.expected_value[1]
else:
    # Array version: shape (1, n_features, n_classes)
    # Take class 1 for the first (and only) sample
    shap_one = shap_values_one_full[0, :, 1]  # shape (n_features,)
    # expected_value may be scalar or array of length 2
    ev = explainer.expected_value
    if np.ndim(ev) == 0:
        base_val = ev
    else:
        base_val = ev[1]

# Build Explanation object for waterfall
explanation = shap.Explanation(
    values=shap_one,
    base_values=base_val,
    data=x_one.values[0],
    feature_names=feature_names
)

plt.figure()
shap.plots.waterfall(explanation, show=False)
waterfall_path = os.path.join(OUTPUT_DIR, "shap_local_waterfall_example0.png")
plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
plt.close()
print("Saved:", waterfall_path)

print("Done. Check the 'reports' folder for SHAP plots.")
