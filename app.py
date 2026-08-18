import streamlit as st
import pandas as pd
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Classification Dashboard", layout="wide")
st.title("Machine Learning Classification Evaluation Platform")

MODEL_DIR = "model/saved_models"
MODELS_USING_SCALED_INPUT = ["Logistic Regression", "kNN"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
}


@st.cache_resource
def load_artifacts():
    """
    Load the scaler, feature schema, and every trained model produced by
    train_model.py. Cached so files are read from disk once per session,
    not on every dropdown change.
    """
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_schema.json")) as f:
        schema = json.load(f)

    models = {
        name: joblib.load(os.path.join(MODEL_DIR, fname))
        for name, fname in MODEL_FILES.items()
    }

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    train_metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            train_metrics = json.load(f)

    return scaler, schema["feature_names"], models, train_metrics


try:
    scaler, feature_names, models, train_metrics = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Could not find saved model artifacts in '{MODEL_DIR}/'. "
        f"Run train_model.py first to generate them.\n\nMissing: {e}"
    )
    st.stop()

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Test Dataset Preview")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("Dataset missing target column!")
    else:
        X = df.drop(columns=["target"])
        y = df["target"]

        # Validate the uploaded file has the exact columns the model expects,
        # in the right order - protects against silently wrong predictions.
        missing_cols = set(feature_names) - set(X.columns)
        extra_cols = set(X.columns) - set(feature_names)
        if missing_cols:
            st.error(f"Uploaded file is missing expected columns: {sorted(missing_cols)}")
            st.stop()
        if extra_cols:
            st.warning(f"Ignoring unexpected columns not used by the model: {sorted(extra_cols)}")
        X = X[feature_names]  # enforce correct column order

        st.sidebar.header("Model Selection")
        selected_model = st.sidebar.selectbox(
            "Choose Estimator",
            list(models.keys())
        )

        clf = models[selected_model]
        use_scaled = selected_model in MODELS_USING_SCALED_INPUT

        # INFERENCE ONLY - no .fit() here. The model was already trained by
        # train_pipeline.py; we're only scoring it on newly uploaded data.
        X_input = scaler.transform(X) if use_scaled else X
        preds = clf.predict(X_input)
        probs = clf.predict_proba(X_input)[:, 1]

        st.subheader(f"Evaluation Metrics: {selected_model}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{accuracy_score(y, preds):.4f}")
        c2.metric("AUC", f"{roc_auc_score(y, probs):.4f}")
        c3.metric("Precision", f"{precision_score(y, preds):.4f}")
        c4.metric("Recall", f"{recall_score(y, preds):.4f}")
        c5.metric("F1 Score", f"{f1_score(y, preds):.4f}")
        c6.metric("MCC", f"{matthews_corrcoef(y, preds):.4f}")

        if train_metrics and selected_model in train_metrics:
            tm = train_metrics[selected_model]
            st.caption(
                f"Training-time 10-fold CV accuracy for this model: "
                f"{tm['cv_accuracy_mean']:.4f} ± {tm['cv_accuracy_std']:.4f} "
                "(reference point - large gaps from the metrics above may signal "
                "the uploaded data differs from the original distribution)."
            )

        col_left, col_right = st.columns(2)
        with col_left:
            st.write("### Confusion Matrix")
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(confusion_matrix(y, preds), annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        with col_right:
            st.write("### Detailed Report")
            st.dataframe(pd.DataFrame(classification_report(y, preds, output_dict=True)).T)
else:
    st.info("Upload `test_data.csv` via the sidebar to execute evaluations.")
