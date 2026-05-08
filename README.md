# 💳 Credit Card Fraud Detection — Machine Learning Project

A complete end-to-end machine learning pipeline for detecting fraudulent credit card transactions. The project covers the full ML lifecycle from problem definition to model improvement across **7 structured parts**.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Models Used](#models-used)
- [Results](#results)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Key Findings](#key-findings)

---

## Overview

Financial fraud causes billions in losses annually. This project builds a system to **automatically detect suspicious credit card transactions** using supervised machine learning.

**Problem Type:** Binary Classification  
**Target Variable:** `is_fraud` (0 = Legitimate, 1 = Fraudulent)  
**Main Challenge:** Extreme class imbalance — only **0.58%** of transactions are fraud.

---

## Dataset

- **Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- **Train Set:** 1,296,675 transactions × 23 features
- **Test Set:** 555,719 transactions × 23 features
- **Fraud Ratio:** 0.58% (1:171 imbalance)

> ⚠️ The CSV files are too large for GitHub (~500MB). Download them from Kaggle and place `fraudTrain.csv` and `fraudTest.csv` in the `project/` directory.

### Features

| Feature | Description |
|---------|-------------|
| `trans_date_trans_time` | Transaction timestamp |
| `merchant` | Merchant name |
| `category` | Transaction category (grocery, gas, etc.) |
| `amt` | Transaction amount (USD) |
| `gender` | Cardholder gender |
| `city`, `state`, `zip` | Cardholder location |
| `lat`, `long` | Customer coordinates |
| `merch_lat`, `merch_long` | Merchant coordinates |
| `city_pop` | City population |
| `job` | Cardholder occupation |
| `dob` | Date of birth |
| `is_fraud` | **Target** (0 or 1) |

---

## Project Structure

```
project/
├── main.py                          # Runs all 7 parts sequentially
├── part1_problem_definition.py      # Problem definition & dataset overview
├── part2_data_exploration.py        # Advanced EDA & visualizations
├── part3_data_preprocessing.py      # Cleaning, encoding, scaling
├── part4_model_building.py          # Training KNN, SVM, Random Forest
├── part5_hyperparameter_tuning.py   # GridSearchCV & RandomizedSearchCV
├── part6_evaluation.py              # Metrics, confusion matrices, overfitting
├── part7_improvement.py             # Scaling, tuning & SMOTE experiments
├── full_report.tex                  # Complete LaTeX report (all 7 parts)
├── .gitignore                       # Excludes CSVs and large files
├── plots/                           # Generated visualizations
│   ├── correlation_heatmap.png
│   ├── amount_distribution.png
│   ├── top_fraud_categories.png
│   ├── gender_fraud.png
│   ├── fraud_amount_boxplot.png
│   ├── fraud_by_hour.png
│   ├── confusion_matrices.png
│   ├── metrics_comparison.png
│   ├── overfitting_check.png
│   └── improvement_experiments.png
├── fraudTrain.csv                   # (not uploaded - download from Kaggle)
└── fraudTest.csv                    # (not uploaded - download from Kaggle)
```

---

## Pipeline Overview

| Part | Stage | Description |
|------|-------|-------------|
| **1** | Problem Definition | Define business objective, identify target variable, analyze class distribution |
| **2** | Data Exploration | Statistical analysis, skewness, kurtosis, correlation, outlier detection, 6 visualizations |
| **3** | Data Preprocessing | Handle missing values, drop irrelevant columns, feature engineering (hour/day/month/age), outlier capping, label encoding, StandardScaler |
| **4** | Model Building | Train KNN, SVM, Random Forest on stratified 30K subsample |
| **5** | Hyperparameter Tuning | GridSearchCV/RandomizedSearchCV with `class_weight="balanced"` |
| **6** | Evaluation | Accuracy, Precision, Recall, F1, Confusion Matrix, Overfitting analysis |
| **7** | Improvement | 3 experiments: Scaling effect, Tuning effect, SMOTE oversampling |

---

## Models Used

| Model | Algorithm | Key Parameters |
|-------|-----------|----------------|
| **KNN** | K-Nearest Neighbors | `n_neighbors=3` |
| **SVM** | Support Vector Machine | `kernel=rbf`, `C=10`, `class_weight=balanced` |
| **Random Forest** | Ensemble of Decision Trees | `n_estimators=200`, `max_depth=10`, `class_weight=balanced` |

---

## Results

### Base Models (Part 4 — Before Tuning)

| Model | Accuracy | Precision | Recall | F1-Score | Fraud Detected |
|-------|----------|-----------|--------|----------|----------------|
| KNN | 99.61% | 0.00% | 0.00% | 0.0000 | 0 / 39 |
| SVM | 99.61% | 0.00% | 0.00% | 0.0000 | 0 / 39 |
| Random Forest | 99.66% | 100% | 12.8% | 0.2273 | 5 / 39 |

### After Tuning (Part 5)

| Model | Accuracy | Precision | Recall | F1-Score | Fraud Detected |
|-------|----------|-----------|--------|----------|----------------|
| KNN | 99.59% | 0.00% | 0.00% | 0.0000 | 0 / 39 |
| SVM | 99.37% | 18.4% | 17.9% | 0.1818 | **7 / 39** |
| **Random Forest** | 98.74% | 12.8% | **38.5%** | **0.1923** | **15 / 39** |

### After SMOTE (Part 7)

| Model | F1-Score | Fraud Detected |
|-------|----------|----------------|
| KNN + SMOTE | 0.0963 | **9 / 39** |
| SVM + SMOTE | 0.1136 | 5 / 39 |
| RF + SMOTE | 0.1600 | **20 / 39** |

### 🏆 Best Model: **Random Forest (Tuned)** — F1 = 0.1923, detects 15/39 fraud

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Zyad256/machine-learning-project.git
cd machine-learning-project/project
```

### 2. Download the dataset

Download from [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place `fraudTrain.csv` and `fraudTest.csv` in the `project/` directory.

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
```

### 4. Run the full pipeline

```bash
python main.py
```

Or run individual parts:

```bash
python part1_problem_definition.py
python part2_data_exploration.py
python part3_data_preprocessing.py
python part4_model_building.py
python part5_hyperparameter_tuning.py
python part6_evaluation.py
python part7_improvement.py
```

> ⏱️ Full pipeline takes approximately **10-15 minutes** depending on hardware.

---

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn

---

## Key Findings

1. **Accuracy is misleading** — A model predicting "Not Fraud" for everything achieves 99.4% accuracy but detects zero fraud.
2. **`class_weight="balanced"`** is the most effective technique for handling class imbalance in SVM and Random Forest.
3. **SMOTE** helps models without built-in class weighting (like KNN) by generating synthetic minority samples.
4. **F1-Score** is the correct metric for fraud detection, balancing precision and recall.
5. **Random Forest** consistently outperforms KNN and SVM on imbalanced data.
6. **Feature engineering** (extracting hour, day, month, age from dates) creates valuable predictive features.
7. **All models showed overfitting** due to very few fraud training samples (174 out of 30,000).
