import base64
import hashlib
import json
import os
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st

APP_NAME = "Energy Theft Detection System"
APP_SUBTITLE = "Smart Grid Monitoring & Machine Learning-Based Detection"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚡",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

HERO_IMAGE_CANDIDATES = [
    "energy_theft_hero.jpg",
    "Energy theft.jpg",
    "Energy_theft.jpg",
]


@st.cache_data(show_spinner=False)
def load_hero_image():
    """Return the hero image as a Base64 data URI, or None if unavailable.

    Embedding the image in the CSS avoids relative-path problems on
    Streamlit Community Cloud.
    """
    for file_name in HERO_IMAGE_CANDIDATES:
        image_path = BASE_DIR / file_name
        try:
            if image_path.is_file():
                encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
                return f"data:image/jpeg;base64,{encoded}"
        except OSError:
            continue
    return None


HERO_IMAGE = load_hero_image()
HERO_IMAGE_LAYER = f'url("{HERO_IMAGE}")' if HERO_IMAGE else "none"

APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --navy-900: #06101f;
    --navy-800: #0a1d35;
    --navy-700: #102b4b;
    --cyan: #22d3ee;
    --blue: #2563eb;
    --green: #22c55e;
    --red: #ef4444;
    --text: #f8fafc;
    --text-muted: #b6c8de;
    --glass: rgba(255, 255, 255, 0.07);
    --glass-border: rgba(148, 197, 255, 0.18);
}

.stApp {
    background-color: var(--navy-900);
    background-image:
        linear-gradient(rgba(56, 189, 248, 0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56, 189, 248, 0.025) 1px, transparent 1px),
        radial-gradient(circle at 82% 8%, rgba(6, 182, 212, 0.16), transparent 28%),
        radial-gradient(circle at 8% 90%, rgba(37, 99, 235, 0.14), transparent 30%),
        linear-gradient(135deg, #06101f 0%, #0a1d35 52%, #102b4b 100%);
    background-size: 42px 42px, 42px 42px, auto, auto, auto;
    background-attachment: fixed;
    color: var(--text);
}

.stApp,
.stApp p,
.stApp label,
.stApp li,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp button,
.stApp input,
.stApp [data-testid="stMetricValue"],
.stApp [data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, -apple-system, sans-serif;
}

.stApp p,
.stApp li,
.stApp label,
.stApp [data-testid="stWidgetLabel"] p {
    color: var(--text);
}

.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p {
    color: var(--text-muted);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}

h1, h2, h3, h4 {
    color: #FFFFFF !important;
    letter-spacing: -0.01em;
}

h2, h3 {
    margin-top: 0.4rem;
}

hr {
    border-color: rgba(148, 197, 255, 0.16) !important;
    margin: 1.6rem 0 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #09182B, #102A48);
    border-right: 1px solid rgba(34, 211, 238, 0.14);
}

[data-testid="stSidebar"] hr {
    margin: 1rem 0 !important;
}

.sidebar-brand {
    font-size: 1.1rem;
    font-weight: 700;
    line-height: 1.3;
    color: #FFFFFF;
    padding-bottom: 0.25rem;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 500;
    background: var(--glass);
    border: 1px solid var(--glass-border);
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.status-dot.ok {
    background: var(--green);
}

.status-dot.bad {
    background: var(--red);
}

[data-testid="stMetric"] {
    background: var(--glass);
    border: 1px solid var(--glass-border);
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.18);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
}

[data-testid="stMetricLabel"] p {
    color: var(--text-muted) !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF;
    font-weight: 600;
}

.stNumberInput input,
.stTextInput input {
    background-color: #F8FAFC;
    color: #102033;
    border-radius: 8px;
}

.stNumberInput button {
    color: #102033;
}

[data-testid="stPlotlyChart"],
[data-testid="stImage"] img,
[data-testid="stDataFrame"] {
    border-radius: 14px;
}

[data-testid="stPlotlyChart"] {
    background: rgba(255, 255, 255, 0.035);
    border: 1px solid rgba(148, 163, 184, 0.16);
    padding: 8px;
}

[data-testid="stImage"] img {
    border: 1px solid rgba(148, 163, 184, 0.16);
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(148, 163, 184, 0.16);
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

.stButton > button,
.stDownloadButton > button {
    background: linear-gradient(90deg, #2563EB, #06B6D4);
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #0EA5E9;
    color: white;
    border: none;
}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible {
    outline: 2px solid var(--cyan);
    outline-offset: 2px;
}

[data-testid="stFileUploader"] section {
    background: var(--glass);
    border: 1px dashed rgba(34, 211, 238, 0.45);
    border-radius: 14px;
}

.energy-hero {
    position: relative;
    min-height: 260px;
    margin-bottom: 1.5rem;
    padding: 28px;
    display: flex;
    align-items: center;
    border: 1px solid rgba(34, 211, 238, 0.5);
    border-radius: 20px;
    background-image:
        linear-gradient(90deg, rgba(3, 14, 31, 0.92) 0%, rgba(3, 14, 31, 0.6) 55%, rgba(3, 14, 31, 0.3) 100%),
        __HERO_IMAGE__,
        linear-gradient(135deg, #06101f, #102b4b);
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3), inset 0 0 0 1px rgba(255, 255, 255, 0.04);
    overflow: hidden;
}

.energy-hero-panel {
    max-width: 640px;
    padding: 22px 26px;
    border-radius: 16px;
    background: rgba(6, 16, 31, 0.5);
    border: 1px solid rgba(148, 197, 255, 0.2);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
}

.energy-hero-title {
    margin: 0;
    color: #FFFFFF;
    font-size: clamp(1.7rem, 4.2vw, 2.7rem);
    font-weight: 700;
    line-height: 1.15;
    letter-spacing: -0.02em;
}

.energy-hero-subtitle {
    margin: 10px 0 0 0;
    color: var(--cyan);
    font-size: clamp(0.95rem, 2vw, 1.15rem);
    font-weight: 500;
}

.energy-hero-copy {
    margin: 10px 0 0 0;
    color: #d7eafe;
    font-size: 1rem;
    line-height: 1.55;
}

@media (max-width: 640px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .energy-hero {
        min-height: 200px;
        padding: 14px;
        border-radius: 16px;
        background-image:
            linear-gradient(180deg, rgba(3, 14, 31, 0.85) 0%, rgba(3, 14, 31, 0.6) 100%),
            __HERO_IMAGE__,
            linear-gradient(135deg, #06101f, #102b4b);
    }

    .energy-hero-panel {
        padding: 16px 18px;
    }

    .energy-hero-copy {
        font-size: 0.92rem;
    }
}
</style>
"""

st.markdown(
    APP_CSS.replace("__HERO_IMAGE__", HERO_IMAGE_LAYER),
    unsafe_allow_html=True,
)

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


metrics = None

try:
    if os.path.exists("model_metrics.json"):
        with open("model_metrics.json", "r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)

except Exception as error:
    st.warning(f"Unable to load model metrics: {error}")


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


st.sidebar.markdown(
    f'<div class="sidebar-brand">⚡ {APP_NAME}</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    "Monitor electricity usage and detect "
    "potential theft using Machine Learning."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dashboard Settings")
st.sidebar.write("Model: Random Forest")

if model is not None:
    status_html = (
        '<span class="status-pill">'
        '<span class="status-dot ok"></span>Status: Active</span>'
    )
else:
    status_html = (
        '<span class="status-pill">'
        '<span class="status-dot bad"></span>Status: Model unavailable</span>'
    )

st.sidebar.markdown(status_html, unsafe_allow_html=True)


st.markdown(
    '<div class="energy-hero"><div class="energy-hero-panel">'
    f'<div class="energy-hero-title">{APP_NAME}</div>'
    f'<div class="energy-hero-subtitle">{APP_SUBTITLE.replace("&", "&amp;")}</div>'
    '<p class="energy-hero-copy">An ML-powered application for identifying '
    "potentially suspicious electricity consumption.</p>"
    "</div></div>",
    unsafe_allow_html=True,
)


st.subheader("Model Performance")

if metrics is not None:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
    col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
    col4.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")
else:
    st.info("Train the model to generate evaluation metrics.")


st.markdown("---")
st.subheader("Model Evaluation - Confusion Matrix")

if os.path.exists("confusion_matrix.png"):
    st.image(
        "confusion_matrix.png",
        caption="Random Forest Confusion Matrix",
        width="stretch",
    )


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
    st.plotly_chart(fig_importance, width="stretch")
    st.caption(
        "Feature importance describes how the fitted Random Forest used "
        "features across its trees. It does not show causation."
    )
else:
    st.info("Feature importance is unavailable for the currently loaded model.")


st.markdown("---")
st.subheader("Enter Smart Meter Data")

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


st.markdown("---")
st.subheader("Consumption Feature Comparison")

if st.button("Show Usage Visualization"):
    chart_background = "#0b1f38"

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(chart_background)
    ax.set_facecolor(chart_background)

    ax.bar(
        ["Average", "Peak", "Off-Peak"],
        [avg, peak, off_peak],
        color=["#2563EB", "#06B6D4", "#38BDF8"],
        width=0.55,
    )

    ax.set_ylabel("Usage", color="#F8FAFC")
    ax.set_title("Smart Meter Consumption Values", color="#F8FAFC")
    ax.tick_params(colors="#D7EAFE")
    ax.grid(axis="y", color="#94A3B8", alpha=0.18)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#334155")

    st.pyplot(fig)
    plt.close(fig)

    st.metric("Recorded Variance", f"{variance:.2f}")


st.markdown("---")
st.subheader("Predict Electricity Theft")

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

        st.markdown("#### Why did the model make this prediction?")
        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_result = explainer.shap_values(features)

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
            st.plotly_chart(fig_shap, width="stretch")
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


st.markdown("---")
st.subheader("Bulk Prediction Using CSV")

uploaded_file = st.file_uploader(
    "Upload Smart Meter CSV",
    type=["csv"],
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.write("Uploaded Data Preview")
        st.dataframe(data.head(), width="stretch")

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
                    st.dataframe(data, width="stretch")

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
                    st.plotly_chart(fig_scatter, width="stretch")
                    st.caption(
                        "Each dot represents one uploaded meter record. Colours show "
                        "the model's predicted class, not a confirmed theft event."
                    )

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


st.markdown("---")
st.subheader("Prediction History")

history = get_history()

if not history.empty:
    st.dataframe(history, width="stretch")

    st.download_button(
        "Download Prediction History",
        data=history.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv",
    )
else:
    st.info("No prediction history available yet. Make your first prediction!")


st.markdown("---")
st.caption(
    f"{APP_NAME} | "
    "Python | Machine Learning | Streamlit"
)