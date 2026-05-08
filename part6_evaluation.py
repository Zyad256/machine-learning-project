"""
=============================================================
 Part 6: Evaluation - Credit Card Fraud Detection
=============================================================
 Metrics:
 - Accuracy, Precision, Recall, F1-Score
 - Confusion Matrix (with visualization)
 - Overfitting / Underfitting Analysis
 - Classification Report
=============================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# -------------------------------------------------------
# Setup
# -------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("   PART 6: EVALUATION")
print("=" * 60)

# -------------------------------------------------------
# 1. Load tuned models from Part 5
# -------------------------------------------------------
print("\n[Step 1] Loading tuned models from Part 5...")

with open(os.path.join(SCRIPT_DIR, "part5_results.pkl"), "rb") as f:
    results = pickle.load(f)

models = results["models"]
predictions = results["predictions"]
X_test = results["test_data"]["X_test"]
y_test = results["test_data"]["y_test"]
X_train = results["train_data"]["X_train"]
y_train = results["train_data"]["y_train"]

print(f"  Loaded 3 tuned models: KNN, SVM, Random Forest")
print(f"  Test set: {len(y_test)} samples ({y_test.sum()} fraud)")

# -------------------------------------------------------
# 2. Detailed Evaluation for Each Model
# -------------------------------------------------------
model_names = ["KNN", "SVM", "Random Forest"]
model_keys = ["knn", "svm", "rf"]
all_metrics = {}

for name, key in zip(model_names, model_keys):
    print("\n" + "=" * 60)
    print(f"   EVALUATION: {name}")
    print("=" * 60)

    y_pred = predictions[key]

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    all_metrics[name] = {"acc": acc, "prec": prec, "rec": rec, "f1": f1}

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  +----------+-----------+")
    print(f"  |          | Predicted |")
    print(f"  |          |  0  |  1  |")
    print(f"  +----------+-----+-----+")
    print(f"  | Actual 0 | {cm[0][0]:>4}| {cm[0][1]:>4}|")
    print(f"  | Actual 1 | {cm[1][0]:>4}| {cm[1][1]:>4}|")
    print(f"  +----------+-----+-----+")
    print(f"\n  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

    # --- Classification Report ---
    print(f"\n  Classification Report:")
    report = classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"], zero_division=0)
    for line in report.split("\n"):
        print(f"  {line}")

# -------------------------------------------------------
# 3. Confusion Matrix Visualizations
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   CONFUSION MATRIX VISUALIZATIONS")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, (name, key) in enumerate(zip(model_names, model_keys)):
    cm = confusion_matrix(y_test, predictions[key])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"])
    axes[i].set_title(f"{name}", fontsize=14, fontweight="bold")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

fig.suptitle("Confusion Matrices - All Models (After Tuning)", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrices.png"))
plt.close()
print("  [Saved] confusion_matrices.png")

# -------------------------------------------------------
# 4. Metrics Comparison Bar Chart
# -------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
metrics_list = ["acc", "prec", "rec", "f1"]
metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
colors = ["#3498db", "#e74c3c", "#2ecc71"]

for i, (metric, label) in enumerate(zip(metrics_list, metrics_labels)):
    values = [all_metrics[name][metric] for name in model_names]
    bars = axes[i].bar(model_names, values, color=colors, edgecolor="black")
    axes[i].set_title(label, fontsize=13, fontweight="bold")
    axes[i].set_ylim(0, 1.1)
    for bar, val in zip(bars, values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")

fig.suptitle("Model Comparison - All Metrics", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "metrics_comparison.png"))
plt.close()
print("  [Saved] metrics_comparison.png")

# -------------------------------------------------------
# 5. Overfitting / Underfitting Analysis
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   OVERFITTING / UNDERFITTING ANALYSIS")
print("=" * 60)

print("""
  Overfitting: Model performs well on training data but poorly on test data.
  Underfitting: Model performs poorly on both training and test data.
  Good Fit: Model performs well on both training and test data.
""")

print(f"  {'Model':<17} {'Train F1':>10} {'Test F1':>10} {'Diff':>10} {'Diagnosis':>15}")
print("  " + "-" * 62)

for name, key in zip(model_names, model_keys):
    model = models[key]
    y_train_pred = model.predict(X_train)
    y_test_pred = predictions[key]

    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    diff = train_f1 - test_f1

    if train_f1 < 0.3 and test_f1 < 0.3:
        diagnosis = "Underfitting"
    elif diff > 0.15:
        diagnosis = "Overfitting"
    else:
        diagnosis = "Acceptable"

    print(f"  {name:<17} {train_f1:>10.4f} {test_f1:>10.4f} {diff:>10.4f} {diagnosis:>15}")

# Train vs Test visualization
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(model_names))
width = 0.35

train_f1s = []
test_f1s = []
for key in model_keys:
    y_train_pred = models[key].predict(X_train)
    train_f1s.append(f1_score(y_train, y_train_pred, zero_division=0))
    test_f1s.append(f1_score(y_test, predictions[key], zero_division=0))

bars1 = ax.bar(x - width/2, train_f1s, width, label="Train F1", color="#3498db", edgecolor="black")
bars2 = ax.bar(x + width/2, test_f1s, width, label="Test F1", color="#e74c3c", edgecolor="black")

ax.set_xlabel("Model")
ax.set_ylabel("F1-Score")
ax.set_title("Train vs Test F1-Score (Overfitting Check)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()
ax.set_ylim(0, 1.0)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.4f}", ha="center", fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f"{bar.get_height():.4f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "overfitting_check.png"))
plt.close()
print("\n  [Saved] overfitting_check.png")

# -------------------------------------------------------
# 6. Final Summary
# -------------------------------------------------------
print("\n" + "=" * 60)
print("   EVALUATION SUMMARY")
print("=" * 60)

print(f"\n  {'Model':<17} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("  " + "-" * 57)
for name in model_names:
    m = all_metrics[name]
    print(f"  {name:<17} {m['acc']:>10.4f} {m['prec']:>10.4f} {m['rec']:>10.4f} {m['f1']:>10.4f}")

best = max(model_names, key=lambda n: all_metrics[n]["f1"])
print(f"\n  --> Best Model: {best} (F1: {all_metrics[best]['f1']:.4f})")

print("""
  Analysis:
  - All models have high accuracy (>99%) but this is misleading
    due to extreme class imbalance (99.4% Not Fraud).
  - F1-Score and Recall are the most important metrics for
    fraud detection because missing fraud is costly.
  - Random Forest has the best Recall (catches more fraud).
  - SVM has better balance between Precision and Recall.
  - KNN failed to detect any fraud cases.
""")

print("  Plots saved:")
print("  - confusion_matrices.png")
print("  - metrics_comparison.png")
print("  - overfitting_check.png")

print("\n" + "=" * 60)
print("   END OF PART 6: EVALUATION")
print("=" * 60)
