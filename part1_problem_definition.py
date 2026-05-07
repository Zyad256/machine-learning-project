"""
=============================================================
 Part 1: Problem Definition - Credit Card Fraud Detection
=============================================================
"""

import pandas as pd
import os

# -------------------------------------------------------
# 1. Load the Dataset
# -------------------------------------------------------
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

df_train = pd.read_csv(os.path.join(SCRIPT_DIR, "fraudTrain.csv"))
df_test = pd.read_csv(os.path.join(SCRIPT_DIR, "fraudTest.csv"))

print("=" * 60)
print("   PART 1: PROBLEM DEFINITION")
print("=" * 60)

# -------------------------------------------------------
# 2. Business Objective
# -------------------------------------------------------
print("""
+----------------------------------------------------------+
|              BUSINESS OBJECTIVE                          |
+----------------------------------------------------------+
|                                                          |
|  The goal of this project is to build a machine          |
|  learning model that can accurately detect               |
|  fraudulent credit card transactions.                    |
|                                                          |
|  Financial fraud causes billions of dollars in           |
|  losses every year. By identifying fraudulent            |
|  transactions in real-time, banks and financial          |
|  institutions can:                                       |
|                                                          |
|  1. Protect customers from unauthorized charges          |
|  2. Reduce financial losses                              |
|  3. Improve trust and customer satisfaction              |
|  4. Comply with regulatory requirements                  |
|                                                          |
|  This is a BINARY CLASSIFICATION problem:                |
|  -> Classify each transaction as Fraud (1) or            |
|     Not Fraud (0)                                        |
|                                                          |
+----------------------------------------------------------+
""")

# -------------------------------------------------------
# 3. Target Variable Definition
# -------------------------------------------------------
print("=" * 60)
print("   TARGET VARIABLE: is_fraud")
print("=" * 60)
print(f"""
  Column Name : is_fraud
  Data Type   : Integer (Binary)
  Values      : 0 = Legitimate Transaction
                1 = Fraudulent Transaction
""")

# -------------------------------------------------------
# 4. Target Variable Distribution
# -------------------------------------------------------
print("=" * 60)
print("   TARGET VARIABLE DISTRIBUTION (Train Set)")
print("=" * 60)

target_counts = df_train["is_fraud"].value_counts()
target_percent = df_train["is_fraud"].value_counts(normalize=True) * 100

print(f"""
  Total Transactions : {len(df_train):,}
  
  Not Fraud (0)      : {target_counts[0]:,}  ({target_percent[0]:.2f}%)
  Fraud (1)          : {target_counts[1]:,}  ({target_percent[1]:.2f}%)
  
  [!] Note: The dataset is HIGHLY IMBALANCED
      -> Fraud cases are a very small percentage
      -> This will be addressed in preprocessing
""")

# -------------------------------------------------------
# 5. Dataset Overview
# -------------------------------------------------------
print("=" * 60)
print("   DATASET OVERVIEW")
print("=" * 60)
print(f"""
  Train Set Size   : {df_train.shape[0]:,} rows × {df_train.shape[1]} columns
  Test Set Size    : {df_test.shape[0]:,} rows × {df_test.shape[1]} columns
  Total Records    : {df_train.shape[0] + df_test.shape[0]:,}
""")

# -------------------------------------------------------
# 6. Feature Description
# -------------------------------------------------------
print("=" * 60)
print("   FEATURE DESCRIPTIONS")
print("=" * 60)

features_info = {
    "trans_date_trans_time": "Transaction date and time",
    "cc_num": "Credit card number (customer identifier)",
    "merchant": "Merchant name where transaction occurred",
    "category": "Transaction category (e.g., grocery, gas, shopping)",
    "amt": "Transaction amount in USD",
    "first": "Cardholder first name",
    "last": "Cardholder last name",
    "gender": "Cardholder gender (M/F)",
    "street": "Cardholder street address",
    "city": "Cardholder city",
    "state": "Cardholder state",
    "zip": "Cardholder zip code",
    "lat": "Cardholder latitude",
    "long": "Cardholder longitude",
    "city_pop": "Population of cardholder's city",
    "job": "Cardholder job/occupation",
    "dob": "Cardholder date of birth",
    "trans_num": "Unique transaction ID",
    "unix_time": "Transaction time in unix format",
    "merch_lat": "Merchant latitude",
    "merch_long": "Merchant longitude",
    "is_fraud": "TARGET -> 0 = Not Fraud, 1 = Fraud",
}

for col, desc in features_info.items():
    print(f"  {col:<26} -> {desc}")

# -------------------------------------------------------
# 7. Data Types Summary
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   DATA TYPES SUMMARY")
print("=" * 60)

numerical_cols = df_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df_train.select_dtypes(include=["object"]).columns.tolist()

print(f"""
  Numerical Features  ({len(numerical_cols)}): {numerical_cols}
  
  Categorical Features ({len(categorical_cols)}): {categorical_cols}
""")

# -------------------------------------------------------
# 8. Missing Values Check
# -------------------------------------------------------
print("=" * 60)
print("   MISSING VALUES CHECK")
print("=" * 60)

missing = df_train.isnull().sum()
total_missing = missing.sum()

if total_missing == 0:
    print("\n  [OK] No missing values found in the dataset!\n")
else:
    print(f"\n  [!] Total missing values: {total_missing}")
    print(missing[missing > 0])

print("=" * 60)
print("   END OF PART 1: PROBLEM DEFINITION")
print("=" * 60)
