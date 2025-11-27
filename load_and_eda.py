import os

import matplotlib.pyplot as plt
import pandas as pd

# 1. Configure paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "UCI_Credit_Card.csv")

print("Base directory:", BASE_DIR)
print("Data path:", DATA_PATH)

# 2. Load dataset
df = pd.read_csv(DATA_PATH)
print("Dataset loaded. Shape:", df.shape)

print("\nColumns:")
print(df.columns)

# 3. Basic info
print("\nData types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isna().sum())

# 4. Check target distribution
# Usually target column is something like 'default.payment.next.month'
target_col = 'default.payment.next.month'
if target_col in df.columns:
    print("\nTarget value counts:")
    print(df[target_col].value_counts(normalize=True) * 100)
else:
    print(f"\nWARNING: Target column '{target_col}' not found. Check actual column name.")
    print("Available columns:", df.columns)

# 5. Basic histogram of age
if 'AGE' in df.columns:
    plt.figure()
    df['AGE'].hist(bins=30)
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# 6. Correlation with target (if target exists and is numeric)
if target_col in df.columns:
    # Only numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
    print("\nCorrelation with target:")
    print(corr.head(15))
