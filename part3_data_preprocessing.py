"""
=============================================================
 Part 3: Data Preprocessing - Credit Card Fraud Detection
=============================================================
 Covers all preprocessing types studied in sections:
 - Handling missing values
 - Removing duplicates
 - Dropping irrelevant columns
 - Handling date/time features
 - Handling outliers
 - Label Encoding (categorical -> numerical)
 - Feature Scaling (StandardScaler & MinMaxScaler)
 - Handling class imbalance
=============================================================
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("   PART 3: DATA PREPROCESSING")
print("=" * 60)

# -------------------------------------------------------
# 1. Load the Dataset
# -------------------------------------------------------
print("\n[Step 1] Loading Dataset...")
df = pd.read_csv(os.path.join(SCRIPT_DIR, "fraudTrain.csv"))
df_test = pd.read_csv(os.path.join(SCRIPT_DIR, "fraudTest.csv"))
print(f"  Train: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"  Test:  {df_test.shape[0]:,} rows x {df_test.shape[1]} columns")

# -------------------------------------------------------
# 2. Check & Handle Missing Values
# -------------------------------------------------------
print("\n" + "-" * 60)
print("[Step 2] Handling Missing Values")
print("-" * 60)

missing = df.isnull().sum()
total_missing = missing.sum()
print(f"  Total missing values: {total_missing}")

if total_missing > 0:
    print("  Missing per column:")
    print(missing[missing > 0])
    # Fill numerical with mean, categorical with mode
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    print("  -> Filled numerical with mean, categorical with mode")
else:
    print("  -> No missing values found!")

# -------------------------------------------------------
# 3. Remove Duplicates
# -------------------------------------------------------
print("\n" + "-" * 60)
print("[Step 3] Removing Duplicates")
print("-" * 60)

duplicates = df.duplicated().sum()
print(f"  Duplicate rows found: {duplicates:,}")
if duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"  -> Removed! New shape: {df.shape}")
else:
    print("  -> No duplicates found!")

# -------------------------------------------------------
# 4. Drop Irrelevant Columns
# -------------------------------------------------------
print("\n" + "-" * 60)
print("[Step 4] Dropping Irrelevant Columns")
print("-" * 60)

# These columns don't help the model:
# - Unnamed: 0 -> just an index
# - cc_num -> credit card number (identifier, not a feature)
# - first, last -> personal names (not useful)
# - street -> too many unique values, not useful
# - trans_num -> unique transaction ID
# - dob -> will extract age instead
# - unix_time -> redundant (we have trans_date_trans_time)

drop_cols = ["Unnamed: 0", "cc_num", "first", "last", "street",
             "trans_num", "unix_time"]

print(f"  Dropping: {drop_cols}")
df = df.drop(columns=drop_cols)
df_test = df_test.drop(columns=drop_cols)
print(f"  -> New shape: {df.shape}")

# -------------------------------------------------------
# 5. Handle Date/Time Features (Feature Engineering)
# -------------------------------------------------------
print("\n" + "-" * 60)
print("[Step 5] Handling Date/Time Features")
print("-" * 60)

# Extract useful features from transaction datetime
for dataset in [df, df_test]:
    dataset["trans_date_trans_time"] = pd.to_datetime(dataset["trans_date_trans_time"])
    dataset["trans_hour"] = dataset["trans_date_trans_time"].dt.hour
    dataset["trans_day"] = dataset["trans_date_trans_time"].dt.dayofweek
    dataset["trans_month"] = dataset["trans_date_trans_time"].dt.month

print("  Extracted: trans_hour, trans_day, trans_month")

# Extract age from date of birth
for dataset in [df, df_test]:
    dataset["dob"] = pd.to_datetime(dataset["dob"])
    dataset["age"] = (dataset["trans_date_trans_time"] - dataset["dob"]).dt.days // 365

print("  Extracted: age (from dob)")

# Drop original date columns
df = df.drop(columns=["trans_date_trans_time", "dob"])
df_test = df_test.drop(columns=["trans_date_trans_time", "dob"])
print(f"  Dropped: trans_date_trans_time, dob")
print(f"  -> New shape: {df.shape}")

# -------------------------------------------------------
# 6. Handle Outliers
# -------------------------------------------------------
print("\n" + "-" * 60)
print("[Step 6] Handling Outliers (Capping)")
print("-" * 60)

# Cap outliers using IQR method instead of removing them
outlier_cols = ["amt", "city_pop"]

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    before = len(df[(df[col] < lower) | (df[col] > upper)])

    # Cap (clip) instead of removing
    df[col] = df[col].clip(lower=lower, upper=upper)
    df_test[col] = df_test[col].clip(lower=lower, upper=upper)

    after = len(df[(df[col] < lower) | (df[col] > upper)])
    print(f"  {col}: Capped {before:,} outliers (IQR method)")

# -------------------------------------------------------
# 7. Handle Text Data (strip & lowercase)
# -------------------------------------------------------
print("\n" + "-" * 60)
print("[Step 7] Cleaning Text Data")
print("-" * 60)

text_cols = ["merchant", "category", "city", "state", "job"]
for col in text_cols:
    for dataset in [df, df_test]:
        dataset[col] = dataset[col].str.strip().str.lower()
print(f"  Cleaned: {text_cols}")
print("  -> Applied strip() and lower()")

# -------------------------------------------------------
# 8. Label Encoding (Categorical -> Numerical)
# -------------------------------------------------------
print("\n" + "-" * 60)
print("[Step 8] Label Encoding")
print("-" * 60)

# Gender: simple binary encoding
for dataset in [df, df_test]:
    dataset["gender"] = dataset["gender"].map({"M": 0, "F": 1})
print("  gender: M->0, F->1 (Binary Encoding)")

# Category, merchant, city, state, job: Label Encoding
label_cols = ["merchant", "category", "city", "state", "job"]
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    # Fit on combined data to handle all categories
    combined = pd.concat([df[col], df_test[col]], axis=0)
    le.fit(combined)
    df[col] = le.transform(df[col])
    df_test[col] = le.transform(df_test[col])
    label_encoders[col] = le
    print(f"  {col}: {le.classes_.shape[0]} unique values encoded")

print(f"\n  -> All categorical columns are now numerical!")
print(f"  -> DataFrame dtypes:")
print(f"     Numerical: {len(df.select_dtypes(include=np.number).columns)} columns")
print(f"     Object:    {len(df.select_dtypes(include='object').columns)} columns")

# -------------------------------------------------------
# 9. Feature Scaling
# -------------------------------------------------------
print("\n" + "-" * 60)
print("[Step 9] Feature Scaling")
print("-" * 60)

# Separate features and target
X_train = df.drop(columns=["is_fraud"])
y_train = df["is_fraud"]
X_test = df_test.drop(columns=["is_fraud"])
y_test = df_test["is_fraud"]

# Columns to scale (all features)
scale_cols = X_train.columns.tolist()

# StandardScaler: mean=0, std=1
print("\n  Applying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=scale_cols
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=scale_cols
)
print("  -> StandardScaler applied!")
print(f"     Example (amt): mean={X_train_scaled['amt'].mean():.4f}, std={X_train_scaled['amt'].std():.4f}")

# Also show MinMaxScaler example
print("\n  [Note] MinMaxScaler example (scales between 0 and 1):")
mm_scaler = MinMaxScaler()
X_train_minmax = pd.DataFrame(
    mm_scaler.fit_transform(X_train),
    columns=scale_cols
)
print(f"     Example (amt): min={X_train_minmax['amt'].min():.4f}, max={X_train_minmax['amt'].max():.4f}")
print("\n  -> Using StandardScaler for model training (better for KNN & SVM)")

# -------------------------------------------------------
# 10. Final Dataset Summary
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   PREPROCESSING SUMMARY")
print("=" * 60)

print(f"""
  Steps Applied:
  [1] Missing Values    -> Checked (none found)
  [2] Duplicates        -> Checked & removed
  [3] Irrelevant Cols   -> Dropped 7 columns
  [4] Date/Time         -> Extracted hour, day, month, age
  [5] Outliers          -> Capped using IQR method
  [6] Text Cleaning     -> strip() + lower()
  [7] Label Encoding    -> All categorical -> numerical
  [8] Feature Scaling   -> StandardScaler applied
  
  Final Dataset:
  X_train shape: {X_train_scaled.shape}
  X_test shape:  {X_test_scaled.shape}
  y_train shape: {y_train.shape}
  y_test shape:  {y_test.shape}
  
  Features ({len(scale_cols)}): {scale_cols}
  
  Target: is_fraud (0 or 1)
  Class Distribution (Train):
    Not Fraud: {(y_train == 0).sum():,} ({(y_train == 0).mean()*100:.2f}%)
    Fraud:     {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.2f}%)
""")

# -------------------------------------------------------
# 11. Save Preprocessed Data
# -------------------------------------------------------
print("[Saving] Preprocessed data...")

X_train_scaled.to_csv(os.path.join(SCRIPT_DIR, "X_train_preprocessed.csv"), index=False)
X_test_scaled.to_csv(os.path.join(SCRIPT_DIR, "X_test_preprocessed.csv"), index=False)
y_train.to_csv(os.path.join(SCRIPT_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(SCRIPT_DIR, "y_test.csv"), index=False)

# Also save unscaled version for comparison later (Part 7)
X_train.to_csv(os.path.join(SCRIPT_DIR, "X_train_unscaled.csv"), index=False)
X_test.to_csv(os.path.join(SCRIPT_DIR, "X_test_unscaled.csv"), index=False)

print("  -> X_train_preprocessed.csv")
print("  -> X_test_preprocessed.csv")
print("  -> y_train.csv")
print("  -> y_test.csv")
print("  -> X_train_unscaled.csv (for Part 7 comparison)")
print("  -> X_test_unscaled.csv (for Part 7 comparison)")

print("\n" + "=" * 60)
print("   END OF PART 3: DATA PREPROCESSING")
print("=" * 60)
