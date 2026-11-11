
import base64
import hashlib
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


APP_NAME = "Energy Theft Detection System"
APP_SUBTITLE = "Smart Grid Monitoring & Machine Learning-Based Detection"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚡",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent

try:
    API_URL = st.secrets.get(
        "API_URL",
        "http://127.0.0.1:8001"
    ).rstrip("/")
except Exception:
    API_URL = "http://127.0.0.1:8001"

FEATURES = [
    "avg_usage",
    "peak_usage",
    "off_peak_usage",
    "variance",
]

MAX_BATCH_RECORDS = 10000

HERO_IMAGE_CANDIDATES = [
    "energy_theft_hero.jpg",
    "Energy theft.jpg",
    "Energy_theft.jpg",
]


@st.cache_data(show_spinner=False)
def load_hero_image():
    for file_name in HERO_IMAGE_CANDIDATES:
        image_path = BASE_DIR / file_name

        try:
            if image_path.is_file():
                encoded = base64.b64encode(
                    image_path.read_bytes()
                ).decode("ascii")

                return f"data:image/jpeg;base64,{encoded}"

        except OSError:
            continue

    return None


HERO_IMAGE = load_hero_image()

HERO_IMAGE_LAYER = (
    f'url("{HERO_IMAGE}")'
    if HERO_IMAGE
    else "none"
)


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


MODEL_PATH = BASE_DIR / "model.pkl"
METRICS_PATH = BASE_DIR / "model_metrics.json"
CONFUSION_MATRIX_PATH = BASE_DIR / "confusion_matrix.png"

model = None

try:
    if MODEL_PATH.is_file():
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)
    else:
        st.warning(
            "Local model.pkl was not found. "
            "Predictions can still work if the FastAPI backend has the model."
        )

except Exception as error:
    st.warning(f"Unable to load local model: {error}")


metrics = None

try:
    if METRICS_PATH.is_file():
        with open(METRICS_PATH, "r", encoding="utf-8") as metrics_file:
            metrics = json.load(metrics_file)

except Exception as error:
    st.warning(f"Unable to load model metrics: {error}")


def check_backend_status():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        response.raise_for_status()
        return True

    except requests.RequestException:
        return False


def show_api_error(error, action_name="API request"):
    if isinstance(error, requests.HTTPError):
        try:
            detail = error.response.json().get(
                "detail",
                error.response.text,
            )
        except Exception:
            detail = str(error)

        st.error(f"{action_name} failed: {detail}")

    else:
        st.error(
            f"{action_name} failed. "
            f"Please check whether FastAPI is running at {API_URL}. "
            f"Details: {error}"
        )


def get_history():
    try:
        response = requests.get(
            f"{API_URL}/history",
            timeout=15,
        )

        response.raise_for_status()

        history_records = response.json().get(
            "history",
            [],
        )

        return pd.DataFrame(history_records)

    except requests.RequestException as error:
        show_api_error(error, "Loading prediction history")
        return pd.DataFrame()

    except (ValueError, KeyError) as error:
        st.error(f"Invalid history response from backend: {error}")
        return pd.DataFrame()


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
st.sidebar.write("Backend: FastAPI")

backend_active = check_backend_status()

if backend_active:
    backend_status_html = (
        '<span class="status-pill">'
        '<span class="status-dot ok"></span>'
        'API: Connected'
        '</span>'
    )
else:
    backend_status_html = (
        '<span class="status-pill">'
        '<span class="status-dot bad"></span>'
        'API: Disconnected'
        '</span>'
    )

st.sidebar.markdown(
    backend_status_html,
    unsafe_allow_html=True,
)

if model is not None:
    model_status_html = (
        '<span class="status-pill">'
        '<span class="status-dot ok"></span>'
        'Local model: Loaded'
        '</span>'
    )
else:
    model_status_html = (
        '<span class="status-pill">'
        '<span class="status-dot bad"></span>'
        'Local model: Unavailable'
        '</span>'
    )

st.sidebar.markdown(
    model_status_html,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="energy-hero"><div class="energy-hero-panel">'
    f'<div class="energy-hero-title">{APP_NAME}</div>'
    f'<div class="energy-hero-subtitle">'
    f'{APP_SUBTITLE.replace("&", "&amp;")}'
    '</div>'
    '<p class="energy-hero-copy">An ML-powered application for identifying '
    'potentially suspicious electricity consumption.</p>'
    '</div></div>',
    unsafe_allow_html=True,
)


st.subheader("Model Performance")

if metrics is not None:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Accuracy",
        f"{metrics['accuracy'] * 100:.2f}%",
    )

    col2.metric(
        "Precision",
        f"{metrics['precision'] * 100:.2f}%",
    )

    col3.metric(
        "Recall",
        f"{metrics['recall'] * 100:.2f}%",
    )

    col4.metric(
        "F1 Score",
        f"{metrics['f1_score'] * 100:.2f}%",
    )

else:
    st.info("Train the model to generate evaluation metrics.")


st.markdown("---")
st.subheader("Model Evaluation - Confusion Matrix")

if CONFUSION_MATRIX_PATH.is_file():
    st.image(
        str(CONFUSION_MATRIX_PATH),
        caption="Random Forest Confusion Matrix",
        width="stretch",
    )
else:
    st.info("Confusion matrix image was not found.")


st.markdown("---")
st.subheader("Which features influence the model?")

if model is not None and hasattr(model, "feature_importances_"):

    importance_df = pd.DataFrame({
        "Feature": FEATURES,
        "Importance": model.feature_importances_,
    }).sort_values(
        "Importance",
        ascending=True,
    )

    fig_importance = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=[
            "#2563EB",
            "#06B6D4",
            "#A78BFA",
        ],
        title="Random Forest Feature Importance",
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

    st.plotly_chart(
        fig_importance,
        width="stretch",
    )

    st.caption(
        "Feature importance describes how the fitted Random Forest used "
        "features across its trees. It does not show causation."
    )

else:
    st.info(
        "Feature importance is unavailable for the currently loaded model."
    )


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
        color=[
            "#2563EB",
            "#06B6D4",
            "#38BDF8",
        ],
        width=0.55,
    )

    ax.set_ylabel(
        "Usage",
        color="#F8FAFC",
    )

    ax.set_title(
        "Smart Meter Consumption Values",
        color="#F8FAFC",
    )

    ax.tick_params(colors="#D7EAFE")

    ax.grid(
        axis="y",
        color="#94A3B8",
        alpha=0.18,
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_color("#334155")

    st.pyplot(fig)
    plt.close(fig)

    st.metric(
        "Recorded Variance",
        f"{variance:.2f}",
    )


st.markdown("---")
st.subheader("Predict Electricity Theft")

if st.button("Predict Theft", key="predict_btn"):

    payload = {
        "avg_usage": avg,
        "peak_usage": peak,
        "off_peak_usage": off_peak,
        "variance": variance,
    }

    try:
        with st.spinner("Analyzing smart meter data..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=30,
            )

            response.raise_for_status()
            result = response.json()

        prediction = result["prediction"]
        theft_score = float(result["theft_score"])

        features = pd.DataFrame(
            [[avg, peak, off_peak, variance]],
            columns=FEATURES,
        )

        st.markdown("### Prediction Result")

        if prediction == 1:
            st.error("⚠️ Potential Electricity Theft Detected")
        else:
            st.success("✅ Normal Usage Predicted")

        st.metric(
            "Estimated Theft Score",
            f"{theft_score:.2f}%",
        )

        message, reasons = generate_alert(
            theft_score,
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
            st.write(
                "No predefined consumption indicators were triggered."
            )

        st.caption(
            "Predictions are model estimates and should "
            "not be treated as confirmed theft."
        )

        st.markdown("#### Why did the model make this prediction?")

        if model is not None:
            try:
                import shap

                explainer = shap.TreeExplainer(model)
                shap_result = explainer.shap_values(features)

                if isinstance(shap_result, list):
                    class_index = list(model.classes_).index(1)
                    class_shap = shap_result[class_index][0]

                else:
                    shap_array = (
                        shap_result.values
                        if hasattr(shap_result, "values")
                        else shap_result
                    )

                    if len(shap_array.shape) == 3:
                        class_index = list(model.classes_).index(1)
                        class_shap = shap_array[0, :, class_index]
                    else:
                        class_shap = shap_array[0]

                explanation_df = pd.DataFrame({
                    "Feature": FEATURES,
                    "SHAP contribution": class_shap,
                })

                explanation_df = explanation_df.sort_values(
                    "SHAP contribution",
                    ascending=True,
                )

                fig_shap = px.bar(
                    explanation_df,
                    x="SHAP contribution",
                    y="Feature",
                    orientation="h",
                    color="SHAP contribution",
                    color_continuous_scale=[
                        "#3B82F6",
                        "#64748B",
                        "#EF4444",
                    ],
                    title="Feature contributions toward the theft class",
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

                st.plotly_chart(
                    fig_shap,
                    width="stretch",
                )

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
                st.warning(
                    f"SHAP explanation could not be generated: {shap_error}"
                )

        else:
            st.info(
                "The backend returned the prediction, but the local model "
                "is unavailable for SHAP explanations."
            )

    except requests.RequestException as error:
        show_api_error(error, "Individual prediction")

    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Invalid prediction response from backend: {error}")


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
        st.dataframe(
            data.head(),
            width="stretch",
        )

        if data.empty:
            st.warning("Uploaded CSV is empty.")

        elif not all(col in data.columns for col in FEATURES):
            st.error(
                "CSV must contain these columns: "
                + ", ".join(FEATURES)
            )

        else:
            for col in FEATURES:
                data[col] = pd.to_numeric(
                    data[col],
                    errors="coerce",
                )

            if data[FEATURES].isna().any().any():
                st.error(
                    "CSV contains missing or invalid feature values. "
                    "Please correct them and upload the file again."
                )

            elif (data[FEATURES] < 0).any().any():
                st.error(
                    "CSV contains negative feature values. "
                    "All usage and variance values must be non-negative."
                )

            elif len(data) > MAX_BATCH_RECORDS:
                st.error(
                    f"CSV contains {len(data)} records. "
                    f"The maximum supported batch size is "
                    f"{MAX_BATCH_RECORDS} records."
                )

            else:
                file_signature = hashlib.sha256(
                    uploaded_file.getvalue()
                ).hexdigest()

                if (
                    st.session_state.get("csv_result_signature")
                    != file_signature
                ):

                    records = data[FEATURES].to_dict(
                        orient="records",
                    )

                    with st.spinner("Processing bulk predictions..."):
                        response = requests.post(
                            f"{API_URL}/predict/batch",
                            json={"records": records},
                            timeout=120,
                        )

                        response.raise_for_status()
                        result = response.json()

                    results = result["results"]

                    if len(results) != len(data):
                        st.error(
                            "The backend returned an unexpected number "
                            "of prediction results."
                        )

                    else:
                        data["Prediction"] = [
                            item["prediction"]
                            for item in results
                        ]

                        data["Prediction Label"] = [
                            item["prediction_label"]
                            for item in results
                        ]

                        data["Theft Score (%)"] = [
                            item["theft_score"]
                            for item in results
                        ]

                        st.session_state["csv_result_signature"] = (
                            file_signature
                        )

                        st.session_state["csv_result_data"] = (
                            data.copy()
                        )

                else:
                    data = st.session_state[
                        "csv_result_data"
                    ].copy()

                if (
                    "Prediction" in data.columns
                    and "Theft Score (%)" in data.columns
                ):

                    st.subheader("Prediction Results")

                    st.dataframe(
                        data,
                        width="stretch",
                    )

                    st.markdown(
                        "#### 🟢 Normal readings vs 🔴 suspected theft"
                    )

                    scatter_df = data.copy()

                    fig_scatter = px.scatter(
                        scatter_df,
                        x="avg_usage",
                        y="peak_usage",
                        color="Prediction Label",
                        color_discrete_map={
                            "Normal Usage": "#22C55E",
                            "Theft Detected": "#EF4444",
                        },
                        hover_data={
                            "off_peak_usage": True,
                            "variance": True,
                            "Theft Score (%)": ":.2f",
                            "Prediction Label": True,
                        },
                        title="Smart meter readings by model prediction",
                        labels={
                            "avg_usage": "Average usage",
                            "peak_usage": "Peak usage",
                        },
                    )

                    fig_scatter.update_traces(
                        marker=dict(
                            size=11,
                            opacity=0.82,
                            line=dict(
                                width=0.6,
                                color="#E2E8F0",
                            ),
                        ),
                    )

                    fig_scatter.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#F8FAFC",
                        legend_title_text="Model prediction",
                        margin=dict(l=10, r=10, t=55, b=10),
                    )

                    st.plotly_chart(
                        fig_scatter,
                        width="stretch",
                    )

                    st.caption(
                        "Each dot represents one uploaded meter record. "
                        "Colours show the model's predicted class, "
                        "not a confirmed theft event."
                    )

                    total = len(data)

                    theft_cases = int(
                        (data["Prediction"] == 1).sum()
                    )

                    normal_cases = total - theft_cases

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Total Records",
                        total,
                    )

                    col2.metric(
                        "Potential Theft Cases",
                        theft_cases,
                    )

                    col3.metric(
                        "Normal Cases",
                        normal_cases,
                    )

                    theft_percentage = (
                        (theft_cases / total) * 100
                        if total > 0
                        else 0
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

    except requests.RequestException as error:
        show_api_error(error, "CSV batch prediction")

    except (ValueError, KeyError, TypeError) as error:
        st.error(f"Invalid batch prediction response: {error}")

    except Exception as error:
        st.error(f"CSV processing error: {error}")


st.markdown("---")
st.subheader("Prediction History")

history = get_history()

if not history.empty:

    st.dataframe(
        history,
        width="stretch",
    )

    st.download_button(
        "Download Prediction History",
        data=history.to_csv(index=False),
        file_name="prediction_history.csv",
        mime="text/csv",
    )

else:
    st.info(
        "No prediction history available yet. "
        "Make your first prediction!"
    )


st.markdown("---")

st.caption(
    f"{APP_NAME} | "
    "Python | Machine Learning | FastAPI | Streamlit"
)
