# Machine Learning Model Evaluation Platform

An interactive Streamlit web application that trains, evaluates, and compares multiple Machine Learning classification algorithms on a multi-feature dataset.

## Problem Statement
This project benchmarks 5 classification models across 6 standard performance metrics to provide a unified evaluation platform.

### Dataset Specifications
* **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset
* **Instances:** 569 (Meets $\ge 500$ requirement)
* **Features:** 30 (Meets $\ge 12$ requirement)
* **Target Variable:** Binary Classification (Malignant vs. Benign)

## Implemented Algorithms
1. Logistic Regression
2. Decision Tree Classifier
3. k-Nearest Neighbors (kNN)
4. Naive Bayes (Gaussian)
5. Random Forest Classifier

## Evaluation Metrics
* Accuracy
* Area Under ROC Curve (AUC)
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

## Project Structure
├── app.py                   # Streamlit web application frontend & logic
├── model/
│   └── train_models.py      # Script to load data, train models, and export test CSV
├── test_data.csv            # Generated test subset for interactive uploading
├── requirements.txt         # Required Python packages for deployment
└── README.md                # Project documentation
## How to Run Locally

## Github Link
https://github.com/AditiDas1928/VS_ML_Assignment_2

   ## Final Test-Set Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| **kNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Random Forest** | 0.9561 | 0.9927 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Top performer across all metrics (98.25% accuracy, 0.9623 MCC). Standardized features enable linear decision boundaries to cleanly separate malignant and benign cases. |
| **Decision Tree** | Weakest performer overall (92.11% accuracy, 0.8341 MCC). Hyperparameter constraints successfully prevented overfitting, revealing true baseline performance on unseen test data. |
| **kNN** | Strong performer (95.61% accuracy, 0.9054 MCC), leveraging distance-based proximity effectively after feature scaling. |
| **Naive Bayes** | Solid probabilistic performance (93.86% accuracy, 0.9878 AUC), though feature independence assumptions slightly limit classification accuracy. |
| **Random Forest** | Identical test accuracy to kNN (95.61%) with higher ROC AUC (0.9927), demonstrating effective ensemble bagging that smooths out decision tree variance. |
| **Overall Winner for your dataset?** | **Logistic Regression** — Achieved the highest accuracy, F1 score, and Matthews Correlation Coefficient, proving optimal for this linearly separable dataset. |
