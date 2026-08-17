"""
Breast Cancer Wisconsin (Diagnostic) - Model Training Pipeline (train_model.py)
================================================================================
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

RANDOM_STATE = 42
OUTPUT_DIR = "saved_models"
MODELS_USING_SCALED_INPUT = ["Logistic Regression", "kNN"]


def load_data():
    """Load the sklearn breast cancer dataset as a DataFrame/Series pair."""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


def tune_decision_tree(X_train, y_train, cv):
    """
    Grid search over depth/leaf constraints instead of guessing them.
    A previous run with a hand-picked max_depth=4, min_samples_leaf=3
    scored WORSE in CV than the unconstrained tree, so constraints
    must be validated, not assumed to help.
    """
    param_grid = {
        "max_depth": [3, 4, 5, 6, 8, None],
        "min_samples_leaf": [1, 2, 3, 5],
    }
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"  Decision Tree best params: {grid.best_params_} "
          f"(CV accuracy={grid.best_score_:.4f})")
    return grid.best_estimator_


def tune_random_forest(X_train, y_train, cv):
    """Light grid search for Random Forest; bagging already regularizes it,
    so the search space is smaller and mainly tunes n_estimators/depth."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [5, 8, None],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(f"  Random Forest best params: {grid.best_params_} "
          f"(CV accuracy={grid.best_score_:.4f})")
    return grid.best_estimator_


def build_models(X_train, X_train_scaled, y_train, cv):
    """Construct the final model dict, tuning tree-based models via CV."""
    print("--- Tuning tree-based models via GridSearchCV (10-fold) ---")
    decision_tree = tune_decision_tree(X_train, y_train, cv)
    random_forest = tune_random_forest(X_train, y_train, cv)

    return {
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE, max_iter=10000
        ),
        "Decision Tree": decision_tree,
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes": GaussianNB(),
        "Random Forest": random_forest,
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_proba), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    X, y = load_data()

    # 2. Stratified 80/20 split for the final hold-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Fit scaler ONLY on training data - prevents leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.pkl"))

    # Save the feature schema so inference-time input can be validated
    with open(os.path.join(OUTPUT_DIR, "feature_schema.json"), "w") as f:
        json.dump({"feature_names": list(X.columns)}, f, indent=2)

    # 4. Build models, tuning tree-based ones via 10-fold CV grid search
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    models = build_models(X_train, X_train_scaled, y_train, cv)

    # 5. Train, cross-validate, evaluate, and save each model
    print("\n--- Training, Cross-Validating, and Evaluating ---")
    all_metrics = {}
    for name, model in models.items():
        use_scaled = name in MODELS_USING_SCALED_INPUT
        Xtr = X_train_scaled if use_scaled else X_train
        Xte = X_test_scaled if use_scaled else X_test
        Xcv = X_train_scaled if use_scaled else X_train  # CV on training data only

        model.fit(Xtr, y_train)

        # Cross-validated accuracy (more trustworthy than a single split)
        cv_scores = cross_val_score(model, Xcv, y_train, cv=cv, scoring="accuracy")

        # Single held-out test set metrics (for the final report)
        test_metrics = evaluate(model, Xte, y_test)

        all_metrics[name] = {
            "test_set_metrics": test_metrics,
            "cv_accuracy_mean": round(cv_scores.mean(), 4),
            "cv_accuracy_std": round(cv_scores.std(), 4),
            "scaled_input": use_scaled,
        }

        filename = f"{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, os.path.join(OUTPUT_DIR, filename))
        print(f"  {name:22s} | test_acc={test_metrics['Accuracy']:.4f} "
              f"| cv_acc={cv_scores.mean():.4f} +/- {cv_scores.std():.4f} "
              f"| saved -> {filename}")

    # 6. Save all metrics + run metadata alongside the models
    all_metrics["_metadata"] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test_size": 0.2,
        "random_state": RANDOM_STATE,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # 7. Export the held-out test set for the Streamlit app
    test_data = X_test.copy()
    test_data["target"] = y_test.values
    test_data.to_csv("test_data.csv", index=False)

    print(f"\nSaved 'test_data.csv' ({len(test_data)} samples)")
    print(f"All models, scaler, schema, and metrics.json saved to '{OUTPUT_DIR}/'")

    print("\n--- Final Test-Set Comparison ---")
    summary = pd.DataFrame({
        name: all_metrics[name]["test_set_metrics"]
        for name in models
    }).T
    print(summary.to_string())


if __name__ == "__main__":
    main()