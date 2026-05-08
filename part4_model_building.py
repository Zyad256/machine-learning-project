"""
=============================================================
 Part 4: Model Building - Credit Card Fraud Detection
=============================================================
 Models:
 1. K-Nearest Neighbors (KNN)
 2. Support Vector Machine (SVM)
 3. Random Forest (Additional Model)
=============================================================
"""

import pandas as pd
import numpy as np
import os
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("   PART 4: MODEL BUILDING")
print("=" * 60)

# -------------------------------------------------------
# 1. Load Preprocessed Data
# -------------------------------------------------------
print("\n[Step 1] Loading Preprocessed Data...")

X_train_full = pd.read_csv(os.path.join(SCRIPT_DIR, "X_train_preprocessed.csv"))
X_test_full = pd.read_csv(os.path.join(SCRIPT_DIR, "X_test_preprocessed.csv"))
y_train_full = pd.read_csv(os.path.join(SCRIPT_DIR, "y_train.csv")).values.ravel()
y_test_full = pd.read_csv(os.path.join(SCRIPT_DIR, "y_test.csv")).values.ravel()

print(f"  X_train: {X_train_full.shape}")
print(f"  X_test:  {X_test_full.shape}")

# -------------------------------------------------------
# 2. Subsampling (for KNN & SVM performance)
# -------------------------------------------------------
print("\n[Step 2] Subsampling for efficient training...")
print("  Note: KNN and SVM are slow on 1.3M rows")
print("  Using stratified subsample to keep class balance")

# Take a stratified subsample for training
from sklearn.model_selection import train_test_split

# Use 30,000 samples for training (keeps it fast but meaningful)
SAMPLE_SIZE = 30000

X_train_sample, _, y_train_sample, _ = train_test_split(
    X_train_full, y_train_full,
    train_size=SAMPLE_SIZE,
    random_state=42,
    stratify=y_train_full
)

# Use 10,000 samples for testing
TEST_SAMPLE = 10000
X_test_sample, _, y_test_sample, _ = train_test_split(
    X_test_full, y_test_full,
    train_size=TEST_SAMPLE,
    random_state=42,
    stratify=y_test_full
)

print(f"  Train sample: {X_train_sample.shape}")
print(f"  Test sample:  {X_test_sample.shape}")
print(f"  Train fraud ratio: {y_train_sample.mean()*100:.2f}%")
print(f"  Test fraud ratio:  {y_test_sample.mean()*100:.2f}%")

# -------------------------------------------------------
# 3. Model 1: K-Nearest Neighbors (KNN)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   MODEL 1: K-NEAREST NEIGHBORS (KNN)")
print("=" * 60)

print("\n  Training KNN with K=5...")
start_time = time.time()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_sample, y_train_sample)
knn_pred = knn.predict(X_test_sample)

knn_time = time.time() - start_time

print(f"  Training + Prediction time: {knn_time:.2f} seconds")
print(f"\n  --- KNN Results ---")
print(f"  Accuracy:  {accuracy_score(y_test_sample, knn_pred):.4f}")
print(f"  Precision: {precision_score(y_test_sample, knn_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test_sample, knn_pred, zero_division=0):.4f}")
print(f"  F1-Score:  {f1_score(y_test_sample, knn_pred, zero_division=0):.4f}")

knn_cm = confusion_matrix(y_test_sample, knn_pred)
print(f"\n  Confusion Matrix:")
print(f"  TN={knn_cm[0][0]:>5}  FP={knn_cm[0][1]:>5}")
print(f"  FN={knn_cm[1][0]:>5}  TP={knn_cm[1][1]:>5}")

# -------------------------------------------------------
# 4. Model 2: Support Vector Machine (SVM)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   MODEL 2: SUPPORT VECTOR MACHINE (SVM)")
print("=" * 60)

print("\n  Training SVM with RBF kernel...")
start_time = time.time()

svm = SVC(kernel="rbf", random_state=42)
svm.fit(X_train_sample, y_train_sample)
svm_pred = svm.predict(X_test_sample)

svm_time = time.time() - start_time

print(f"  Training + Prediction time: {svm_time:.2f} seconds")
print(f"\n  --- SVM Results ---")
print(f"  Accuracy:  {accuracy_score(y_test_sample, svm_pred):.4f}")
print(f"  Precision: {precision_score(y_test_sample, svm_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test_sample, svm_pred, zero_division=0):.4f}")
print(f"  F1-Score:  {f1_score(y_test_sample, svm_pred, zero_division=0):.4f}")

svm_cm = confusion_matrix(y_test_sample, svm_pred)
print(f"\n  Confusion Matrix:")
print(f"  TN={svm_cm[0][0]:>5}  FP={svm_cm[0][1]:>5}")
print(f"  FN={svm_cm[1][0]:>5}  TP={svm_cm[1][1]:>5}")

# -------------------------------------------------------
# 5. Model 3: Random Forest (Additional Model)
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   MODEL 3: RANDOM FOREST (Additional)")
print("=" * 60)

print("\n  Training Random Forest with 100 trees...")
start_time = time.time()

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_sample, y_train_sample)
rf_pred = rf.predict(X_test_sample)

rf_time = time.time() - start_time

print(f"  Training + Prediction time: {rf_time:.2f} seconds")
print(f"\n  --- Random Forest Results ---")
print(f"  Accuracy:  {accuracy_score(y_test_sample, rf_pred):.4f}")
print(f"  Precision: {precision_score(y_test_sample, rf_pred, zero_division=0):.4f}")
print(f"  Recall:    {recall_score(y_test_sample, rf_pred, zero_division=0):.4f}")
print(f"  F1-Score:  {f1_score(y_test_sample, rf_pred, zero_division=0):.4f}")

rf_cm = confusion_matrix(y_test_sample, rf_pred)
print(f"\n  Confusion Matrix:")
print(f"  TN={rf_cm[0][0]:>5}  FP={rf_cm[0][1]:>5}")
print(f"  FN={rf_cm[1][0]:>5}  TP={rf_cm[1][1]:>5}")

# -------------------------------------------------------
# 6. Model Comparison
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   MODEL COMPARISON")
print("=" * 60)

models = {
    "KNN": {"pred": knn_pred, "time": knn_time},
    "SVM": {"pred": svm_pred, "time": svm_time},
    "Random Forest": {"pred": rf_pred, "time": rf_time},
}

print(f"\n  {'Model':<17} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Time(s)':>10}")
print("  " + "-" * 67)

for name, data in models.items():
    acc = accuracy_score(y_test_sample, data["pred"])
    prec = precision_score(y_test_sample, data["pred"], zero_division=0)
    rec = recall_score(y_test_sample, data["pred"], zero_division=0)
    f1 = f1_score(y_test_sample, data["pred"], zero_division=0)
    print(f"  {name:<17} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f} {data['time']:>10.2f}")

# Find best model based on F1-Score
best_model = max(models.keys(), key=lambda x: f1_score(y_test_sample, models[x]["pred"], zero_division=0))
best_f1 = f1_score(y_test_sample, models[best_model]["pred"], zero_division=0)

print(f"\n  --> Best Model: {best_model} (F1-Score: {best_f1:.4f})")
print("  --> F1-Score is used because dataset is imbalanced")
print("      (Accuracy alone is misleading with 99.4% vs 0.6%)")

# -------------------------------------------------------
# 7. Save results for later parts
# -------------------------------------------------------
import pickle

results = {
    "models": {
        "knn": knn,
        "svm": svm,
        "rf": rf
    },
    "predictions": {
        "knn": knn_pred,
        "svm": svm_pred,
        "rf": rf_pred
    },
    "test_data": {
        "X_test": X_test_sample,
        "y_test": y_test_sample
    },
    "train_data": {
        "X_train": X_train_sample,
        "y_train": y_train_sample
    }
}

with open(os.path.join(SCRIPT_DIR, "part4_results.pkl"), "wb") as f:
    pickle.dump(results, f)

print("\n  [Saved] part4_results.pkl (models + predictions for Part 5-7)")

print("\n" + "=" * 60)
print("   END OF PART 4: MODEL BUILDING")
print("=" * 60)
