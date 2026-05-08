"""
=============================================================
 Part 7: Improvement & Experimentation
=============================================================
 Comparisons:
 1. Before Scaling vs After Scaling
 2. Before Tuning vs After Tuning
 3. Before SMOTE vs After SMOTE (oversampling)
 4. Final Best Model Summary
=============================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("   PART 7: IMPROVEMENT & EXPERIMENTATION")
print("=" * 60)

# -------------------------------------------------------
# 1. Load data
# -------------------------------------------------------
print("\n[Step 1] Loading data...")

# Load Part 4 results (before tuning)
with open(os.path.join(SCRIPT_DIR, "part4_results.pkl"), "rb") as f:
    part4 = pickle.load(f)

# Load Part 5 results (after tuning)
with open(os.path.join(SCRIPT_DIR, "part5_results.pkl"), "rb") as f:
    part5 = pickle.load(f)

X_train = part5["train_data"]["X_train"]
y_train = part5["train_data"]["y_train"]
X_test = part5["test_data"]["X_test"]
y_test = part5["test_data"]["y_test"]

# Load unscaled data for comparison
X_train_unscaled = pd.read_csv(os.path.join(SCRIPT_DIR, "X_train_unscaled.csv"))
X_test_unscaled = pd.read_csv(os.path.join(SCRIPT_DIR, "X_test_unscaled.csv"))

# Subsample unscaled data to match
SAMPLE_SIZE = 30000
TEST_SAMPLE = 10000
y_train_full = pd.read_csv(os.path.join(SCRIPT_DIR, "y_train.csv")).values.ravel()
y_test_full = pd.read_csv(os.path.join(SCRIPT_DIR, "y_test.csv")).values.ravel()

X_train_unsub, _, y_train_unsub, _ = train_test_split(
    X_train_unscaled, y_train_full, train_size=SAMPLE_SIZE,
    random_state=42, stratify=y_train_full
)
X_test_unsub, _, y_test_unsub, _ = train_test_split(
    X_test_unscaled, y_test_full, train_size=TEST_SAMPLE,
    random_state=42, stratify=y_test_full
)

print(f"  Loaded all data successfully")

# Helper function
def get_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, zero_division=0),
        "rec": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": confusion_matrix(y_true, y_pred)[1][1] if confusion_matrix(y_true, y_pred).shape == (2,2) else 0
    }

# ==========================================================
# EXPERIMENT 1: Before Scaling vs After Scaling
# ==========================================================
print("\n" + "=" * 60)
print("   EXPERIMENT 1: BEFORE SCALING vs AFTER SCALING")
print("=" * 60)
print("  Training Random Forest on unscaled vs scaled data")
print("  (with class_weight='balanced')")

# Train on UNSCALED data
rf_unscaled = RandomForestClassifier(n_estimators=200, max_depth=10,
                                      class_weight="balanced", random_state=42, n_jobs=-1)
rf_unscaled.fit(X_train_unsub, y_train_unsub)
pred_unscaled = rf_unscaled.predict(X_test_unsub)
metrics_unscaled = get_metrics(y_test_unsub, pred_unscaled)

# Train on SCALED data (already have from Part 5)
pred_scaled = part5["predictions"]["rf"]
metrics_scaled = get_metrics(y_test, pred_scaled)

print(f"\n  {'':>20} {'Before Scaling':>15} {'After Scaling':>15}")
print("  " + "-" * 50)
print(f"  {'Accuracy':>20} {metrics_unscaled['acc']:>15.4f} {metrics_scaled['acc']:>15.4f}")
print(f"  {'Precision':>20} {metrics_unscaled['prec']:>15.4f} {metrics_scaled['prec']:>15.4f}")
print(f"  {'Recall':>20} {metrics_unscaled['rec']:>15.4f} {metrics_scaled['rec']:>15.4f}")
print(f"  {'F1-Score':>20} {metrics_unscaled['f1']:>15.4f} {metrics_scaled['f1']:>15.4f}")
print(f"  {'Fraud Detected (TP)':>20} {metrics_unscaled['tp']:>15} {metrics_scaled['tp']:>15}")

# ==========================================================
# EXPERIMENT 2: Before Tuning vs After Tuning
# ==========================================================
print("\n" + "=" * 60)
print("   EXPERIMENT 2: BEFORE TUNING vs AFTER TUNING")
print("=" * 60)

model_names = ["KNN", "SVM", "Random Forest"]
model_keys = ["knn", "svm", "rf"]

print(f"\n  {'Model':>20} {'Before F1':>12} {'After F1':>12} {'Before TP':>12} {'After TP':>12}")
print("  " + "-" * 68)

before_f1s = []
after_f1s = []

for name, key in zip(model_names, model_keys):
    before = get_metrics(y_test, part4["predictions"][key])
    after = get_metrics(y_test, part5["predictions"][key])
    before_f1s.append(before["f1"])
    after_f1s.append(after["f1"])
    print(f"  {name:>20} {before['f1']:>12.4f} {after['f1']:>12.4f} {before['tp']:>12} {after['tp']:>12}")

# ==========================================================
# EXPERIMENT 3: With SMOTE (Oversampling)
# ==========================================================
print("\n" + "=" * 60)
print("   EXPERIMENT 3: APPLYING SMOTE (Oversampling)")
print("=" * 60)

try:
    from imblearn.over_sampling import SMOTE

    print("\n  SMOTE creates synthetic fraud samples to balance classes")
    print(f"  Before SMOTE: {(y_train == 0).sum()} Not Fraud, {(y_train == 1).sum()} Fraud")

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"  After SMOTE:  {(y_train_smote == 0).sum()} Not Fraud, {(y_train_smote == 1).sum()} Fraud")

    smote_results = {}

    # Train all 3 models with SMOTE data
    # KNN with SMOTE
    print("\n  Training KNN with SMOTE...")
    knn_smote = KNeighborsClassifier(n_neighbors=3)
    knn_smote.fit(X_train_smote, y_train_smote)
    knn_smote_pred = knn_smote.predict(X_test)
    smote_results["KNN"] = get_metrics(y_test, knn_smote_pred)

    # SVM with SMOTE (use smaller sample for speed)
    print("  Training SVM with SMOTE...")
    svm_smote = SVC(kernel="rbf", C=10, random_state=42)
    svm_smote.fit(X_train_smote, y_train_smote)
    svm_smote_pred = svm_smote.predict(X_test)
    smote_results["SVM"] = get_metrics(y_test, svm_smote_pred)

    # RF with SMOTE
    print("  Training Random Forest with SMOTE...")
    rf_smote = RandomForestClassifier(n_estimators=200, max_depth=10,
                                       random_state=42, n_jobs=-1)
    rf_smote.fit(X_train_smote, y_train_smote)
    rf_smote_pred = rf_smote.predict(X_test)
    smote_results["Random Forest"] = get_metrics(y_test, rf_smote_pred)

    print(f"\n  {'Model':>20} {'Without SMOTE F1':>18} {'With SMOTE F1':>18} {'SMOTE TP':>12}")
    print("  " + "-" * 68)

    smote_f1s = []
    for name, key in zip(model_names, model_keys):
        without = get_metrics(y_test, part5["predictions"][key])
        with_s = smote_results[name]
        smote_f1s.append(with_s["f1"])
        print(f"  {name:>20} {without['f1']:>18.4f} {with_s['f1']:>18.4f} {with_s['tp']:>12}")

    smote_available = True

except ImportError:
    print("\n  [!] imblearn not installed. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "imbalanced-learn", "-q"])
    print("  [!] Please re-run this script after installation.")
    smote_available = False
    smote_f1s = [0, 0, 0]

# ==========================================================
# VISUALIZATION: All Experiments Comparison
# ==========================================================
print("\n" + "=" * 60)
print("   VISUALIZATION: ALL EXPERIMENTS")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Before vs After Scaling (RF only)
labels_1 = ["Before\nScaling", "After\nScaling"]
vals_1 = [metrics_unscaled["f1"], metrics_scaled["f1"]]
bars1 = axes[0].bar(labels_1, vals_1, color=["#e74c3c", "#2ecc71"], edgecolor="black")
axes[0].set_title("Experiment 1: Scaling Effect (RF)", fontsize=13, fontweight="bold")
axes[0].set_ylabel("F1-Score")
axes[0].set_ylim(0, max(vals_1) * 1.5 if max(vals_1) > 0 else 0.5)
for bar, val in zip(bars1, vals_1):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", fontweight="bold")

# Plot 2: Before vs After Tuning
x = np.arange(len(model_names))
width = 0.35
bars2a = axes[1].bar(x - width/2, before_f1s, width, label="Before Tuning",
                      color="#e74c3c", edgecolor="black")
bars2b = axes[1].bar(x + width/2, after_f1s, width, label="After Tuning",
                      color="#2ecc71", edgecolor="black")
axes[1].set_title("Experiment 2: Tuning Effect", fontsize=13, fontweight="bold")
axes[1].set_ylabel("F1-Score")
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names, fontsize=9)
axes[1].legend()
axes[1].set_ylim(0, max(max(before_f1s), max(after_f1s)) * 1.5 if max(max(before_f1s), max(after_f1s)) > 0 else 0.5)

# Plot 3: Without vs With SMOTE
bars3a = axes[2].bar(x - width/2, after_f1s, width, label="Without SMOTE",
                      color="#e74c3c", edgecolor="black")
bars3b = axes[2].bar(x + width/2, smote_f1s, width, label="With SMOTE",
                      color="#2ecc71", edgecolor="black")
axes[2].set_title("Experiment 3: SMOTE Effect", fontsize=13, fontweight="bold")
axes[2].set_ylabel("F1-Score")
axes[2].set_xticks(x)
axes[2].set_xticklabels(model_names, fontsize=9)
axes[2].legend()
axes[2].set_ylim(0, max(max(after_f1s), max(smote_f1s)) * 1.5 if max(max(after_f1s), max(smote_f1s)) > 0 else 0.5)

fig.suptitle("Improvement Experiments - F1-Score Comparison", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "improvement_experiments.png"))
plt.close()
print("  [Saved] improvement_experiments.png")

# ==========================================================
# FINAL SUMMARY
# ==========================================================
print("\n" + "=" * 60)
print("   FINAL PROJECT SUMMARY")
print("=" * 60)

print("""
  IMPROVEMENT JOURNEY:
  
  Step 1: Base Models (Part 4)
    - KNN:  F1=0.0000 | SVM: F1=0.0000 | RF: F1=0.2273
    - Problem: Models ignore fraud due to class imbalance
  
  Step 2: Hyperparameter Tuning (Part 5)
    - Added class_weight='balanced'
    - SVM improved: 0.0000 -> 0.1818
    - RF Recall improved: 12.8% -> 38.5%
""")

if smote_available:
    print(f"""  Step 3: SMOTE Oversampling (Part 7)
    - KNN with SMOTE: F1={smote_results['KNN']['f1']:.4f} (TP={smote_results['KNN']['tp']})
    - SVM with SMOTE: F1={smote_results['SVM']['f1']:.4f} (TP={smote_results['SVM']['tp']})
    - RF with SMOTE:  F1={smote_results['Random Forest']['f1']:.4f} (TP={smote_results['Random Forest']['tp']})
""")

    # Find overall best
    all_results = {
        "RF (Tuned)": get_metrics(y_test, part5["predictions"]["rf"]),
        "SVM (Tuned)": get_metrics(y_test, part5["predictions"]["svm"]),
        "KNN (SMOTE)": smote_results["KNN"],
        "SVM (SMOTE)": smote_results["SVM"],
        "RF (SMOTE)": smote_results["Random Forest"],
    }

    best_name = max(all_results, key=lambda k: all_results[k]["f1"])
    best = all_results[best_name]

    print(f"  BEST OVERALL MODEL: {best_name}")
    print(f"    Accuracy:  {best['acc']:.4f}")
    print(f"    Precision: {best['prec']:.4f}")
    print(f"    Recall:    {best['rec']:.4f}")
    print(f"    F1-Score:  {best['f1']:.4f}")
    print(f"    Fraud Detected: {best['tp']} out of {y_test.sum()}")

print(f"""
  KEY LESSONS LEARNED:
  
  1. Accuracy is misleading with imbalanced data
  2. class_weight='balanced' is essential for rare class detection
  3. SMOTE oversampling can further improve minority class detection
  4. F1-Score is the right metric for fraud detection
  5. Scaling is important for distance-based models (KNN, SVM)
  6. Random Forest generally handles imbalanced data best
""")

print("=" * 60)
print("   END OF PART 7: IMPROVEMENT & EXPERIMENTATION")
print("=" * 60)
print("\n  PROJECT COMPLETE!")
