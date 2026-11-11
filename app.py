import hashlib
import json
import os
import pickle
import sqlite3
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Energy Theft Detection",
    page_icon="⚡",
    layout="wide",
)


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background: linear-gradient(
        135deg,
        #071426 0%,
        #102747 55%,
        #163b63 100%
    );
    color: #F8FAFC;
}

/* Main content area */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #09182B,
        #102A48
    );
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 18px;
    border-radius: 14px;
}

/* Headings */
h1, h2, h3 {
    color: #FFFFFF !important;
}

/* Input fields */
.stNumberInput input,
.stTextInput input {
    background-color: #F8FAFC;
    color: #102033;
    border-radius: 8px;
}


/* Plotly chart containers */
[data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 14px;
    padding: 8px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 20px;
    font-weight: 600;
}

/* Buttons on hover */
.stButton > button:hover {
    background: #0EA5E9;
    color: white;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# These names must match the columns used by train_model.py.
FEATURES = [
    "avg_usage",
    "peak_usage",
    "off_peak_usage",
    "variance",
]

DB_PATH = "predictions.db"


model = None

try:
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
    else:
        st.error("Model file not found. Run train_model.py first.")

except Exception as error:
    st.error(f"Unable to load model: {error}")


# ==========================================
# 5. LOAD MODEL METRICS
# ==========================================

metrics = None

try:
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)

except Exception as error:
    st.warning(f"Unable to load model metrics: {error}")


# ==========================================
# 6. DATABASE FUNCTIONS
# ==========================================

def init_db():
    """Create the prediction history table if it does not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                avg REAL,
                peak REAL,
                off_peak REAL,
                variance REAL,
                prediction INTEGER,
                theft_score REAL
            )
            """
        )


def save_prediction(avg, peak, off_peak, variance, prediction, theft_score):
    """Save one individual prediction to SQLite."""
    timestamp = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO prediction_history (
                timestamp,
                avg,
                peak,
                off_peak,
                variance,
                prediction,
                theft_score
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                avg,
                peak,
                off_peak,
                variance,
                prediction,
                theft_score,
            ),
        )


def save_batch_predictions(data):
    """Save CSV prediction results to SQLite."""
    timestamp = datetime.now().isoformat(timespec="seconds")
    rows = []

    for _, row in data.iterrows():
        rows.append(
            (
                timestamp,
                float(row["avg_usage"]),
                float(row["peak_usage"]),
                float(row["off_peak_usage"]),
                float(row["variance"]),
                int(row["Prediction"]),
                float(row["Theft Score (%)"]),
            )
        )

    if rows:
        with sqlite3.connect(DB_PATH) as conn:
            conn.executemany(
                """
                INSERT INTO prediction_history (
                    timestamp,
                    avg,
                    peak,
                    off_peak,
                    variance,
                    prediction,
                    theft_score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )


def get_history():
    """Return the latest 100 prediction records."""
    with sqlite3.connect(DB_PATH) as conn:
        history = pd.read_sql_query(
            """
            SELECT *
            FROM prediction_history
            ORDER BY id DESC
            LIMIT 100
            """,
            conn,
        )

    return history


init_db()


# ==========================================
# 7. HELPER FUNCTIONS
# ==========================================

def get_theft_score(features):
    """Return the model probability for class 1 (theft)."""
    if not hasattr(model, "predict_proba"):
        return 0.0

    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)

    if 1 not in classes:
        return 0.0

    theft_index = classes.index(1)
    return float(probabilities[theft_index])


def generate_alert(risk_score, avg, peak, off_peak, variance):
    reasons = []

    if avg > 0 and peak > avg * 2:
        reasons.append("Unusually high peak usage")

    if variance > 100:
        reasons.append("High consumption variance")

    if avg > 0 and off_peak < avg * 0.5:
        reasons.append("Low off-peak usage")

    if risk_score > 70:
        message = "High estimated theft score"
    elif risk_score > 40:
        message = "Moderate estimated theft score"
    else:
        message = "Low estimated theft score"

    return message, reasons


# ==========================================
# 8. SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Dashboard Settings")

st.sidebar.markdown(
    "Monitor electricity usage and detect "
    "potential theft using Machine Learning."
)

st.sidebar.markdown("---")
st.sidebar.write("Model: Random Forest")
st.sidebar.write(
    "Status: Active" if model is not None else "Status: Model unavailable"
)


# ==========================================
# 9. MAIN TITLE
# ==========================================

st.title("⚡ Energy Theft Detection Dashboard")
st.markdown("### Smart Grid Monitoring System")

st.write(
    "An ML-powered application for identifying "
    "potentially suspicious electricity consumption."
)


# ==========================================
# 10. MODEL PERFORMANCE
# ==========================================

st.markdown("---")
st.subheader("📊 Model Performance")

if metrics is not None:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    col4.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")
else:
    st.info("Train the model to generate evaluation metrics.")


# ==========================================
# 11. CONFUSION MATRIX
# ==========================================

st.markdown("---")
st.subheader("Model Evaluation - Confusion Matrix")

if os.path.exists("confusion_matrix.png"):
    st.image(
        "confusion_matrix.png",
        caption="Random Forest Confusion Matrix",
        use_container_width=True,
    )


# ==========================================
# 11A. RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

st.markdown("---")
st.subheader("Which features influence the model?")

if model is not None and hasattr(model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#2563EB", "#06B6D4", "#A78BFA"],
        title="Random Forest Feature Importance"
    )
    fig_importance.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#F8FAFC",
        coloraxis_showscale=False,
        xaxis_title="Relative importance",
        yaxis_title="",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig_importance, use_container_width=True)
    st.caption(
        "Feature importance describes how the fitted Random Forest used "
        "features across its trees. It does not show causation."
    )
else:
    st.info("Feature importance is unavailable for the currently loaded model.")


# ==========================================
# 12. SMART METER INPUT
# ==========================================

st.markdown("---")
st.subheader("📥 Enter Smart Meter Data")

col1, col2 = st.columns(2)

with col1:
    avg = st.number_input(
        "Average Usage",
        min_value=0.0,
        value=0.0,
    )

    peak = st.number_input(
        "Peak Usage",
        min_value=0.0,
        value=0.0,
    )

with col2:
    off_peak = st.number_input(
        "Off-Peak Usage",
        min_value=0.0,
        value=0.0,
    )

    variance = st.number_input(
        "Variance",
        min_value=0.0,
        value=0.0,
    )


# ==========================================
# 13. INPUT VISUALIZATION
# ==========================================

st.markdown("---")
st.subheader("Consumption Feature Comparison")

if st.button("Show Usage Visualization"):
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        ["Average", "Peak", "Off-Peak"],
        [avg, peak, off_peak],
    )

    ax.set_ylabel("Usage")
    ax.set_title("Smart Meter Consumption Values")

    st.pyplot(fig)
    plt.close(fig)

    st.metric("Recorded Variance", f"{variance:.2f}")


# ==========================================
# 14. INDIVIDUAL PREDICTION
# ==========================================

st.markdown("---")
st.subheader("🔍 Predict Electricity Theft")

if st.button("Predict Theft", key="predict_btn"):
    if model is None:
        st.error("Model is unavailable. Train the model first.")
    else:
        features = pd.DataFrame(
            [[avg, peak, off_peak, variance]],
            columns=FEATURES,
        )

        prediction = int(model.predict(features)[0])
        theft_score = get_theft_score(features)

        save_prediction(
            avg,
            peak,
            off_peak,
            variance,
            prediction,
            theft_score * 100,
        )

        st.markdown("### Prediction Result")

        if prediction == 1:
            st.error("⚠️ Potential Electricity Theft Detected")
        else:
            st.success("✅ Normal Usage Predicted")

        st.metric(
            "Estimated Theft Score",
            f"{theft_score * 100:.2f}%",
        )

        message, reasons = generate_alert(
            theft_score * 100,
            avg,
            peak,
            off_peak,
            variance,
        )

        st.info(message)

        if reasons:
            st.write("Possible consumption indicators:")
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.write("No predefined consumption indicators were triggered.")

        st.caption(
            "Predictions are model estimates and should "
            "not be treated as confirmed theft."
        )


        # Optional SHAP explanation for this individual prediction.
        st.markdown("#### Why did the model make this prediction?")
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_result = explainer.shap_values(features)

            # SHAP versions return binary-class values in different shapes.
            if isinstance(shap_result, list):
                class_index = list(model.classes_).index(1)
                class_shap = shap_result[class_index][0]
            else:
                shap_array = shap_result.values if hasattr(shap_result, "values") else shap_result
                if len(shap_array.shape) == 3:
                    class_index = list(model.classes_).index(1)
                    class_shap = shap_array[0, :, class_index]
                else:
                    class_shap = shap_array[0]

            explanation_df = pd.DataFrame({
                "Feature": FEATURES,
                "SHAP contribution": class_shap
            })
            explanation_df = explanation_df.sort_values(
                "SHAP contribution", ascending=True
            )

            fig_shap = px.bar(
                explanation_df,
                x="SHAP contribution",
                y="Feature",
                orientation="h",
                color="SHAP contribution",
                color_continuous_scale=["#3B82F6", "#64748B", "#EF4444"],
                title="Feature contributions toward the theft class"
            )
            fig_shap.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#F8FAFC",
                coloraxis_showscale=False,
                xaxis_title="SHAP value (direction and magnitude)",
                yaxis_title="",
                margin=dict(l=10, r=10, t=55, b=10),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
            st.caption(
                "Positive SHAP values push the model output toward the theft "
                "class; negative values push it away. This explains model "
                "behaviour, not the real-world cause of a meter reading."
            )
        except ImportError:
            st.info(
                "Install SHAP to see per-prediction explanations: "
                "`pip install shap`"
            )
        except Exception as shap_error:
            st.warning(f"SHAP explanation could not be generated: {shap_error}")


# ==========================================
# 15. CSV BULK PREDICTION
# ==========================================

st.markdown("---")
st.subheader("📂 Bulk Prediction Using CSV")

uploaded_file = st.file_uploader(
    "Upload Smart Meter CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.write("Uploaded Data Preview")
        st.dataframe(data.head(), use_container_width=True)

        if model is None:
            st.error("Model is unavailable. Train the model first.")

        elif data.empty:
            st.warning("Uploaded CSV is empty.")

        elif not all(col in data.columns for col in FEATURES):
            st.error(
                "CSV must contain these columns: "
                + ", ".join(FEATURES)
            )

        else:
            for col in FEATURES:
                data[col] = pd.to_numeric(data[col], errors="coerce")

            if data[FEATURES].isna().any().any():
                st.error(
                    "CSV contains missing or invalid feature values. "
                    "Please correct them and upload the file again."
                )
            else:
                predictions = model.predict(data[FEATURES])
                probabilities = model.predict_proba(data[FEATURES])

                model_classes = list(model.classes_)

                if 1 not in model_classes:
                    st.error("The trained model does not contain theft class 1.")
                else:
                    theft_index = model_classes.index(1)
                    theft_scores = probabilities[:, theft_index] * 100

                    data["Prediction"] = predictions
                    data["Prediction Label"] = [
                        "Theft Detected" if prediction == 1
                        else "Normal Usage"
                        for prediction in predictions
                    ]
                    data["Theft Score (%)"] = theft_scores

                    st.subheader("Prediction Results")
                    st.dataframe(data, use_container_width=True)

                    st.markdown("#### 🟢 Normal readings vs 🔴 suspected theft")
                    scatter_df = data.copy()
                    scatter_df["Prediction Label"] = scatter_df["Prediction Label"].astype(str)

                    fig_scatter = px.scatter(
                        scatter_df,
                        x="avg_usage",
                        y="peak_usage",
                        color="Prediction Label",
                        color_discrete_map={
                            "Normal Usage": "#22C55E",
                            "Theft Detected": "#EF4444"
                        },
                        hover_data={
                            "off_peak_usage": True,
                            "variance": True,
                            "Theft Score (%)": ":.2f",
                            "Prediction Label": True
                        },
                        title="Smart meter readings by model prediction",
                        labels={
                            "avg_usage": "Average usage",
                            "peak_usage": "Peak usage"
                        }
                    )
                    fig_scatter.update_traces(
                        marker=dict(size=11, opacity=0.82, line=dict(width=0.6, color="#E2E8F0"))
                    )
                    fig_scatter.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#F8FAFC",
                        legend_title_text="Model prediction",
                        margin=dict(l=10, r=10, t=55, b=10),
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    st.caption(
                        "Each dot represents one uploaded meter record. Colours show "
                        "the model's predicted class, not a confirmed theft event."
                    )


                    # Save each uploaded file only once per Streamlit session.
                    file_signature = hashlib.sha256(
                        uploaded_file.getvalue()
                    ).hexdigest()

                    if (
                        st.session_state.get("saved_csv_signature")
                        != file_signature
                    ):
                        save_batch_predictions(data)
                        st.session_state["saved_csv_signature"] = file_signature

                    total = len(data)
                    theft_cases = int((data["Prediction"] == 1).sum())
                    normal_cases = total - theft_cases

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Total Records", total)
                    col2.metric("Potential Theft Cases", theft_cases)
                    col3.metric("Normal Cases", normal_cases)

                    theft_percentage = (
                        (theft_cases / total) * 100 if total > 0 else 0
                    )

                    st.metric(
                        "Potential Theft Percentage",
                        f"{theft_percentage:.2f}%",
                    )

                    st.download_button(
                        "Download Prediction Results",
                        data=data.to_csv(index=False),
                        file_name="energy_theft_predictions.csv",
                        mime="text/csv",
                    )

    except Exception as error:
        st.error(f"CSV processing error: {error}")


# ==========================================
# 16. PREDICTION HISTORY
# ==========================================

st.markdown("---")
st.subheader("🗂️ Prediction History")

history = get_history()

if not history.empty:
    st.dataframe(history, use_container_width=True)

    st.download_button(
        "Download Prediction History",
        data=history.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv",
    )
else:
    st.info("No prediction history available yet. Make your first prediction!")


# ==========================================
# 17. FOOTER
# ==========================================

st.markdown("---")
st.caption(
    "Energy Theft Detection System | "
    "Python | Machine Learning | Streamlit"
)
