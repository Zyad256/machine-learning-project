"""
=============================================================
 Part 5: Hyperparameter Tuning - Credit Card Fraud Detection
=============================================================
 Using:
 - GridSearchCV / RandomizedSearchCV
 - class_weight='balanced' to handle imbalance
 - Finding best K for KNN
 - Finding best Kernel for SVM
 - Tuning Random Forest
=============================================================
"""

import pandas as pd
import numpy as np
import os
import time
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("   PART 5: HYPERPARAMETER TUNING")
print("=" * 60)

# -------------------------------------------------------
# 1. Load saved data from Part 4
# -------------------------------------------------------
print("\n[Step 1] Loading data from Part 4...")

with open(os.path.join(SCRIPT_DIR, "part4_results.pkl"), "rb") as f:
    part4 = pickle.load(f)

X_train = part4["train_data"]["X_train"]
y_train = part4["train_data"]["y_train"]
X_test = part4["test_data"]["X_test"]
y_test = part4["test_data"]["y_test"]

print(f"  X_train: {X_train.shape}")
print(f"  X_test:  {X_test.shape}")
print(f"  Fraud in train: {y_train.sum()} ({y_train.mean()*100:.2f}%)")

# -------------------------------------------------------
# Helper function to print results
# -------------------------------------------------------
def print_results(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  --- {name} Results ---")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"  TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
    print(f"  FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

# -------------------------------------------------------
# 2. KNN Tuning - Find Best K
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   TUNING KNN: Finding Best K")
print("=" * 60)

# KNN doesn't have class_weight, so we use different K values
# and pick the one with best F1
knn_params = {"n_neighbors": [3, 5, 7, 9, 11]}

print(f"\n  Grid Search with params: {knn_params}")
print("  Scoring: F1 (because data is imbalanced)")

start_time = time.time()
knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    scoring="f1",
    cv=3,
    n_jobs=-1
)
knn_grid.fit(X_train, y_train)
knn_time = time.time() - start_time

print(f"  Time: {knn_time:.2f}s")
print(f"\n  Results for each K:")
for i, k in enumerate(knn_params["n_neighbors"]):
    score = knn_grid.cv_results_["mean_test_score"][i]
    print(f"    K={k:>2}  ->  F1 = {score:.4f}")

print(f"\n  --> Best K = {knn_grid.best_params_['n_neighbors']}")
print(f"  --> Best CV F1 = {knn_grid.best_score_:.4f}")

knn_tuned_pred = knn_grid.predict(X_test)
knn_tuned_results = print_results("KNN Tuned", y_test, knn_tuned_pred)

# -------------------------------------------------------
# 3. SVM Tuning - Find Best Kernel + class_weight
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   TUNING SVM: Kernel + Class Weight")
print("=" * 60)

svm_params = {
    "kernel": ["rbf", "linear"],
    "C": [0.1, 1, 10],
    "class_weight": ["balanced"]  # Handles imbalance!
}

print(f"\n  Grid Search with params: {svm_params}")
print("  class_weight='balanced' adjusts for imbalanced classes")

start_time = time.time()
svm_grid = GridSearchCV(
    SVC(random_state=42),
    svm_params,
    scoring="f1",
    cv=3,
    n_jobs=-1
)
svm_grid.fit(X_train, y_train)
svm_time = time.time() - start_time

print(f"  Time: {svm_time:.2f}s")
print(f"\n  Results for each combination:")
for params, score in zip(svm_grid.cv_results_["params"], svm_grid.cv_results_["mean_test_score"]):
    print(f"    {params}  ->  F1 = {score:.4f}")

print(f"\n  --> Best params: {svm_grid.best_params_}")
print(f"  --> Best CV F1 = {svm_grid.best_score_:.4f}")

svm_tuned_pred = svm_grid.predict(X_test)
svm_tuned_results = print_results("SVM Tuned", y_test, svm_tuned_pred)

# -------------------------------------------------------
# 4. Random Forest Tuning
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   TUNING RANDOM FOREST")
print("=" * 60)

rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [10, 20, None],
    "class_weight": ["balanced"]  # Handles imbalance!
}

print(f"\n  RandomizedSearchCV with params: {rf_params}")

start_time = time.time()
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params,
    n_iter=6,
    scoring="f1",
    cv=3,
    random_state=42,
    n_jobs=-1
)
rf_search.fit(X_train, y_train)
rf_time = time.time() - start_time

print(f"  Time: {rf_time:.2f}s")
print(f"\n  Results for each combination:")
for params, score in zip(rf_search.cv_results_["params"], rf_search.cv_results_["mean_test_score"]):
    print(f"    {params}  ->  F1 = {score:.4f}")

print(f"\n  --> Best params: {rf_search.best_params_}")
print(f"  --> Best CV F1 = {rf_search.best_score_:.4f}")

rf_tuned_pred = rf_search.predict(X_test)
rf_tuned_results = print_results("Random Forest Tuned", y_test, rf_tuned_pred)

# -------------------------------------------------------
# 5. Before vs After Tuning Comparison
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   BEFORE vs AFTER TUNING")
print("=" * 60)

# Load Part 4 predictions for comparison
knn_before = part4["predictions"]["knn"]
svm_before = part4["predictions"]["svm"]
rf_before = part4["predictions"]["rf"]

print(f"\n  {'Model':<20} {'Before F1':>12} {'After F1':>12} {'Improvement':>12}")
print("  " + "-" * 56)

for name, before, after_results in [
    ("KNN", knn_before, knn_tuned_results),
    ("SVM", svm_before, svm_tuned_results),
    ("Random Forest", rf_before, rf_tuned_results)
]:
    f1_before = f1_score(y_test, before, zero_division=0)
    f1_after = after_results["f1"]
    improvement = f1_after - f1_before
    sign = "+" if improvement >= 0 else ""
    print(f"  {name:<20} {f1_before:>12.4f} {f1_after:>12.4f} {sign}{improvement:>11.4f}")

# Find overall best
best_name = max(
    [("KNN", knn_tuned_results), ("SVM", svm_tuned_results), ("RF", rf_tuned_results)],
    key=lambda x: x[1]["f1"]
)
print(f"\n  --> Best Model After Tuning: {best_name[0]} (F1: {best_name[1]['f1']:.4f})")

# -------------------------------------------------------
# 6. Save tuned models
# -------------------------------------------------------
tuned_results = {
    "models": {
        "knn": knn_grid.best_estimator_,
        "svm": svm_grid.best_estimator_,
        "rf": rf_search.best_estimator_
    },
    "predictions": {
        "knn": knn_tuned_pred,
        "svm": svm_tuned_pred,
        "rf": rf_tuned_pred
    },
    "best_params": {
        "knn": knn_grid.best_params_,
        "svm": svm_grid.best_params_,
        "rf": rf_search.best_params_
    },
    "test_data": {"X_test": X_test, "y_test": y_test},
    "train_data": {"X_train": X_train, "y_train": y_train}
}

with open(os.path.join(SCRIPT_DIR, "part5_results.pkl"), "wb") as f:
    pickle.dump(tuned_results, f)

print("\n  [Saved] part5_results.pkl")

print("\n" + "=" * 60)
print("   END OF PART 5: HYPERPARAMETER TUNING")
print("=" * 60)
