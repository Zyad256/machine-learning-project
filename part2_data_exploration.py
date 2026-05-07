"""
=============================================================
 Part 2: Advanced Data Exploration - Credit Card Fraud Detection
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Use a clean style for all plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100

# -------------------------------------------------------
# 1. Load the Dataset
# -------------------------------------------------------
print("=" * 60)
print("   PART 2: ADVANCED DATA EXPLORATION")
print("=" * 60)
print("\nLoading dataset...")

df = pd.read_csv(os.path.join(SCRIPT_DIR, "fraudTrain.csv"))
print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

# -------------------------------------------------------
# 2. Advanced Statistics (mean, std, skewness)
# -------------------------------------------------------
print("=" * 60)
print("   2.1 ADVANCED STATISTICS")
print("=" * 60)

# Select numerical columns (excluding index and identifiers)
numerical_cols = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long"]

# Basic stats
print("\n--- Basic Statistics ---")
print(df[numerical_cols].describe().round(2).to_string())

# Skewness
print("\n--- Skewness ---")
skewness = df[numerical_cols].skew().round(4)
for col, val in skewness.items():
    skew_type = "Right-skewed" if val > 1 else "Left-skewed" if val < -1 else "Approximately Normal"
    print(f"  {col:<15} : {val:>10}  ({skew_type})")

# Kurtosis
print("\n--- Kurtosis ---")
kurtosis = df[numerical_cols].kurtosis().round(4)
for col, val in kurtosis.items():
    kurt_type = "Heavy-tailed" if val > 3 else "Light-tailed" if val < -3 else "Normal-tailed"
    print(f"  {col:<15} : {val:>10}  ({kurt_type})")

# -------------------------------------------------------
# 3. Correlation Matrix
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   2.2 CORRELATION MATRIX")
print("=" * 60)

corr_cols = numerical_cols + ["is_fraud"]
correlation = df[corr_cols].corr().round(4)

print("\n--- Correlation with Target (is_fraud) ---")
fraud_corr = correlation["is_fraud"].drop("is_fraud").sort_values(ascending=False)
for col, val in fraud_corr.items():
    strength = "Strong" if abs(val) > 0.5 else "Moderate" if abs(val) > 0.2 else "Weak"
    print(f"  {col:<15} : {val:>10}  ({strength})")

# Plot: Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap="RdBu_r", center=0,
            fmt=".3f", square=True, linewidths=0.5)
plt.title("Correlation Matrix - Numerical Features", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
plt.close()
print("\n  [Saved] correlation_heatmap.png")

# -------------------------------------------------------
# 4. Outlier Detection
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   2.3 OUTLIER DETECTION (IQR Method)")
print("=" * 60)

outlier_cols = ["amt", "city_pop"]

for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"\n  Column: {col}")
    print(f"    Q1 = {Q1:.2f}, Q3 = {Q3:.2f}, IQR = {IQR:.2f}")
    print(f"    Lower Bound = {lower:.2f}, Upper Bound = {upper:.2f}")
    print(f"    Outliers Count = {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")

# Plot: Boxplots for outlier detection
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, col in enumerate(outlier_cols):
    sns.boxplot(data=df, x=col, ax=axes[i], color="skyblue")
    axes[i].set_title(f"Boxplot of {col}", fontsize=12, fontweight="bold")
fig.suptitle("Outlier Detection using Boxplots", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "outlier_boxplots.png"))
plt.close()
print("\n  [Saved] outlier_boxplots.png")

# -------------------------------------------------------
# 5. Class Imbalance Analysis
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   2.4 CLASS IMBALANCE ANALYSIS")
print("=" * 60)

fraud_counts = df["is_fraud"].value_counts()
fraud_pct = df["is_fraud"].value_counts(normalize=True) * 100

print(f"\n  Class 0 (Not Fraud) : {fraud_counts[0]:>10,}  ({fraud_pct[0]:.2f}%)")
print(f"  Class 1 (Fraud)     : {fraud_counts[1]:>10,}  ({fraud_pct[1]:.2f}%)")
print(f"  Imbalance Ratio     : 1:{fraud_counts[0]//fraud_counts[1]}")
print(f"\n  --> Dataset is HIGHLY IMBALANCED!")
print(f"  --> Will need techniques like oversampling/undersampling in Part 3")

# Plot: Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart
colors = ["#2ecc71", "#e74c3c"]
axes[0].bar(["Not Fraud (0)", "Fraud (1)"], fraud_counts.values, color=colors)
axes[0].set_title("Class Distribution (Count)", fontsize=12, fontweight="bold")
axes[0].set_ylabel("Count")
for i, v in enumerate(fraud_counts.values):
    axes[0].text(i, v + 5000, f"{v:,}", ha="center", fontweight="bold")

# Pie chart
axes[1].pie(fraud_counts.values, labels=["Not Fraud", "Fraud"],
            autopct="%1.2f%%", colors=colors, startangle=90,
            explode=(0, 0.1), shadow=True)
axes[1].set_title("Class Distribution (%)", fontsize=12, fontweight="bold")

fig.suptitle("Target Variable (is_fraud) - Imbalance Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "class_imbalance.png"))
plt.close()
print("  [Saved] class_imbalance.png")

# -------------------------------------------------------
# 6. Data Visualizations
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   2.5 DATA VISUALIZATIONS")
print("=" * 60)

# --- 6.1 Transaction Amount Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# All transactions
axes[0].hist(df["amt"], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
axes[0].set_title("Transaction Amount Distribution (All)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Amount ($)")
axes[0].set_ylabel("Frequency")

# Fraud vs Not Fraud
axes[1].hist(df[df["is_fraud"] == 0]["amt"], bins=50, alpha=0.6, label="Not Fraud", color="#2ecc71")
axes[1].hist(df[df["is_fraud"] == 1]["amt"], bins=50, alpha=0.6, label="Fraud", color="#e74c3c")
axes[1].set_title("Amount Distribution by Class", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Amount ($)")
axes[1].set_ylabel("Frequency")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "amount_distribution.png"))
plt.close()
print("  [Saved] amount_distribution.png")

# --- 6.2 Top 10 Fraud Categories ---
fraud_by_category = df[df["is_fraud"] == 1]["category"].value_counts().head(10)

plt.figure(figsize=(12, 6))
bars = plt.barh(fraud_by_category.index, fraud_by_category.values, color="coral", edgecolor="black")
plt.xlabel("Number of Fraud Cases")
plt.title("Top 10 Categories with Most Fraud Cases", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
for bar in bars:
    plt.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
             f"{int(bar.get_width()):,}", va="center", fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "top_fraud_categories.png"))
plt.close()
print("  [Saved] top_fraud_categories.png")

# --- 6.3 Gender Distribution in Fraud ---
gender_fraud = df.groupby(["gender", "is_fraud"]).size().unstack(fill_value=0)

plt.figure(figsize=(8, 5))
gender_fraud.plot(kind="bar", color=colors, edgecolor="black", ax=plt.gca())
plt.title("Fraud Distribution by Gender", fontsize=14, fontweight="bold")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(["Not Fraud", "Fraud"])
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "gender_fraud.png"))
plt.close()
print("  [Saved] gender_fraud.png")

# --- 6.4 Transaction Amount: Fraud vs Not Fraud (Boxplot) ---
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="is_fraud", y="amt", palette=colors)
plt.title("Transaction Amount: Fraud vs Not Fraud", fontsize=14, fontweight="bold")
plt.xlabel("Is Fraud (0=No, 1=Yes)")
plt.ylabel("Amount ($)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fraud_amount_boxplot.png"))
plt.close()
print("  [Saved] fraud_amount_boxplot.png")

# --- 6.5 Fraud by Hour of Day ---
df["trans_hour"] = pd.to_datetime(df["trans_date_trans_time"]).dt.hour

fraud_by_hour = df.groupby(["trans_hour", "is_fraud"]).size().unstack(fill_value=0)
fraud_rate_by_hour = (fraud_by_hour[1] / (fraud_by_hour[0] + fraud_by_hour[1]) * 100)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count of fraud per hour
axes[0].bar(fraud_by_hour.index, fraud_by_hour[1], color="tomato", edgecolor="black")
axes[0].set_title("Fraud Count by Hour of Day", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Hour")
axes[0].set_ylabel("Fraud Count")

# Fraud rate per hour
axes[1].plot(fraud_rate_by_hour.index, fraud_rate_by_hour.values,
             color="darkred", marker="o", linewidth=2)
axes[1].fill_between(fraud_rate_by_hour.index, fraud_rate_by_hour.values, alpha=0.2, color="red")
axes[1].set_title("Fraud Rate (%) by Hour of Day", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Hour")
axes[1].set_ylabel("Fraud Rate (%)")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "fraud_by_hour.png"))
plt.close()
print("  [Saved] fraud_by_hour.png")

# Clean up temp column
df.drop(columns=["trans_hour"], inplace=True)

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   EXPLORATION SUMMARY")
print("=" * 60)
print(f"""
  Key Findings:
  
  1. IMBALANCE: Only {fraud_pct[1]:.2f}% of transactions are fraud
     -> Need oversampling or class weights in modeling
     
  2. AMOUNT: Fraud transactions tend to have higher amounts
     -> 'amt' is an important feature
     
  3. SKEWNESS: 'amt' and 'city_pop' are right-skewed
     -> May need log transformation
     
  4. OUTLIERS: Present in 'amt' and 'city_pop'
     -> Will be handled in preprocessing
     
  5. TIME: Fraud rate varies by hour of day
     -> Hour feature could be useful
     
  6. CATEGORIES: Some categories have more fraud than others
     -> Category is an important feature
""")

print("  All plots saved to: " + PLOTS_DIR)
print("\n" + "=" * 60)
print("   END OF PART 2: DATA EXPLORATION")
print("=" * 60)
