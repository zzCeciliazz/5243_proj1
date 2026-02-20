import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, skew, probplot

# ====================================================
# 1️⃣ Load Data
# ====================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "employee_survey_merge.csv")

print("==============================================")
print("Data Cleaning + SleepHours Normality Analysis")
print("==============================================")

print("\nLoading file from:", file_path)

df = pd.read_csv(file_path)

print("\nInitial dataset shape:", df.shape)

# ====================================================
# 2️⃣ Standardize Column Names
# ====================================================

df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

print("\nColumn names standardized.")

# ====================================================
# 3️⃣ Basic Data Checks
# ====================================================

print("\nDescriptive statistics:")
print(df.describe(include='all'))

print("\nMissing values check:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# ====================================================
# 4️⃣ Convert Binary Variables to 0-1
# ====================================================

binary_map = {
    "true": 1, "false": 0,
    "male": 1, "female": 0,
    "yes": 1, "no": 0
}

for col in df.columns:

    # Boolean type
    if df[col].dtype == "bool":
        df[col] = df[col].astype(int)

    # Object type
    elif df[col].dtype == "object":
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(binary_map)
            .fillna(df[col])
        )

# ====================================================
# 5️⃣ Convert Numeric-like Strings
# ====================================================

id_columns = ["empid"]

for col in df.columns:
    if col not in id_columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

print("\nData types after cleaning:")
print(df.dtypes)

# ====================================================
# 6️⃣ SleepHours Normality Analysis
# ====================================================

print("\n==============================================")
print("SleepHours Normality Analysis")
print("==============================================")

if "sleephours" not in df.columns:
    print("Column 'sleephours' not found.")
    print("Available columns:")
    print(df.columns)

else:

    sleep = df["sleephours"].dropna()

    print("\nSleepHours Summary:")
    print(sleep.describe())

    sleep_skew = skew(sleep)
    print("\nSkewness:", sleep_skew)

    # Histogram
    plt.figure(figsize=(8,5))
    sns.histplot(sleep, kde=True)
    plt.title("Distribution of SleepHours")
    plt.xlabel("SleepHours")
    plt.ylabel("Frequency")
    plt.show()

    # Q-Q Plot
    plt.figure(figsize=(6,6))
    probplot(sleep, dist="norm", plot=plt)
    plt.title("Q-Q Plot of SleepHours")
    plt.show()

    # Shapiro Test
    if len(sleep) <= 5000:
        stat, p_value = shapiro(sleep)

        print("\nShapiro-Wilk Test:")
        print("Statistic:", stat)
        print("p-value:", p_value)

        # Decision Rule
        if p_value < 0.05 and abs(sleep_skew) > 1:
            print("\n⚠ Significant deviation from normality detected.")
            print("Applying 15%-85% quantile trimming...")

            lower = sleep.quantile(0.15)
            upper = sleep.quantile(0.85)

            df["sleephours"] = df["sleephours"].clip(lower, upper)

            print("Trimming applied.")
            print("Lower bound:", lower)
            print("Upper bound:", upper)

        else:
            print("\n✅ SleepHours approximately normal or mildly skewed.")
            print("No trimming applied to preserve data integrity.")

    else:
        print("\nSample too large for Shapiro test.")
        print("Using skewness and visual inspection only.")

# ====================================================
# 7️⃣ Save Final Dataset
# ====================================================

output_path = os.path.join(current_dir, "employee_survey_sleephours_final.csv")

if os.path.exists(output_path):
    os.remove(output_path)

df.to_csv(output_path, index=False)

print("\n==============================================")
print("Final dataset saved successfully.")
print("Saved to:", output_path)
print("==============================================")
