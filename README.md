# Machine Learning Model Evaluation Platform

An interactive Streamlit web application that trains, evaluates, and compares multiple Machine Learning classification algorithms on a multi-feature dataset.

## Project Overview
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

1. Clone the repository:
   ```bash
   git clone [https://github.com/AditiDas1928/VS_ML_Assignment_2.git](https://github.com/AditiDas1928/VS_ML_Assignment_2.git)

   ## Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9883 | 0.9988 | 0.9817 | 1.0000 | 0.9907 | 0.9752 |
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| kNN | 0.9649 | 0.9976 | 0.9550 | 0.9907 | 0.9725 | 0.9253 |
| Naive Bayes | 0.9298 | 0.9901 | 0.9279 | 0.9626 | 0.9450 | 0.8494 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Demonstrates strong linear decision boundaries with an accuracy of 98.83% and perfect recall (1.0000), indicating minimal false negatives on the test set. |
| **Decision Tree** | Achieves a perfect score across all metrics (1.0000), successfully isolating feature conditions but carrying a high risk of overfitting to the evaluation split. |
| **kNN** | Displays reliable performance (96.49% accuracy, 0.9253 MCC), benefiting from normalized feature scaling across nearest neighbor distance computations. |
| **Naive Bayes** | Records the lowest overall performance (92.98% accuracy, 0.8494 MCC) due to its core assumption of feature independence, which is violated by correlated features in this dataset. |
| **Random Forest (Ensemble)** | Achieves flawless metrics (1.0000 across all indicators) by combining multiple decision trees to reduce variance while maintaining maximum predictive power. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — Recommended as the top model because ensemble averaging mitigates individual decision tree overfitting while maintaining 100% classification precision and recall. |