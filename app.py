import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Classification Dashboard", layout="wide")
st.title("Machine Learning Classification Evaluation Platform")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Test Dataset Preview")
    st.dataframe(df.head())

    if 'target' not in df.columns:
        st.error("Dataset missing target column!")
    else:
        X = df.drop(columns=['target'])
        y = df['target']

        st.sidebar.header("Model Selection")
        selected_model = st.sidebar.selectbox(
            "Choose Estimator",
            ["Logistic Regression", "Decision Tree", "kNN", "Naive Bayes", "Random Forest"]
        )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "kNN": KNeighborsClassifier(n_neighbors=5),
            "Naive Bayes": GaussianNB(),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
        }

        clf = models[selected_model]
        if selected_model in ["Logistic Regression", "kNN", "Naive Bayes"]:
            clf.fit(X_scaled, y)
            preds = clf.predict(X_scaled)
            probs = clf.predict_proba(X_scaled)[:, 1]
        else:
            clf.fit(X, y)
            preds = clf.predict(X)
            probs = clf.predict_proba(X)[:, 1]

        st.subheader(f"Evaluation Metrics: {selected_model}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{accuracy_score(y, preds):.4f}")
        c2.metric("AUC", f"{roc_auc_score(y, probs):.4f}")
        c3.metric("Precision", f"{precision_score(y, preds):.4f}")
        c4.metric("Recall", f"{recall_score(y, preds):.4f}")
        c5.metric("F1 Score", f"{f1_score(y, preds):.4f}")
        c6.metric("MCC", f"{matthews_corrcoef(y, preds):.4f}")

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