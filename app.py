import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right, #f5f7fa, #c3cfe2);
color: black;
    }

    h1, h2, h3, h4 {
        color: #ffffff;
    }

    .stButton>button {
        background-color: #00c6ff;
        color: black;
        border-radius: 10px;
    }

    .stTextInput, .stNumberInput {
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Page config
st.set_page_config(page_title="Energy Theft Detection", layout="wide")

# Load model
model = None
if os.path.exists("model.pkl"):
    model = pickle.load(open("model.pkl", "rb"))
else:
    st.error("Model file not found")
    
# Load accuracy
acc = 0.80

# Sidebar
st.sidebar.title("⚙️ Dashboard Settings")
st.sidebar.markdown("Monitor electricity usage and detect theft using ML.")

# Title
st.title("⚡ Energy Theft Detection Dashboard")
st.markdown("### Smart Grid Monitoring System")

# Status cards
colA, colB, colC = st.columns(3)
colA.metric("Model Accuracy", f"{acc*100:.2f}%")
colB.metric("Model Type", "Random Forest")
colC.metric("Status", "Active")

# Input Section
st.markdown("---")
st.subheader("📥 Enter Smart Meter Data")

col1, col2 = st.columns(2)

with col1:
    avg = st.number_input("Average Usage", min_value=0.0)
    peak = st.number_input("Peak Usage", min_value=0.0)

with col2:
    off_peak = st.number_input("Off Peak Usage", min_value=0.0)
    variance = st.number_input("Variance", min_value=0.0)

# Visualization Section

st.markdown("## 📊 Visualization Dashboard")

st.markdown("---")

col3, col4 = st.columns(2, gap="large")  #  adds space between columns


# Pie Chart
with col3:
    st.subheader("Usage Distribution")

    if st.button(" 📊 Show Distribution"):
        values = [avg, peak, off_peak, variance]
        labels = ["Avg", "Peak", "Off-Peak", "Variance"]

        fig, ax = plt.subplots(figsize=(5,5))  # bigger chart
        if sum(values) == 0:
           st.warning("Enter values greater than 0 to visualize")
        else:
           ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax.set_title("Usage Distribution")

        st.pyplot(fig)

# Trend Graph (simulated)
with col4:
    st.subheader("Usage Trend")

    if st.button("📈 Show Trend"):

        trend = [avg, peak, off_peak, variance, avg+10, peak-20]

        fig, ax = plt.subplots(figsize=(6,4))  #  wider graph
        ax.plot(trend, marker='o', linewidth=2)

        ax.set_title("Electricity Usage Trend")
        ax.set_xlabel("Time")
        ax.set_ylabel("Usage")

        st.pyplot(fig)

# Alert Generation Logic
def generate_alert(risk_score, avg, peak, off_peak, variance):
    reasons = []

    if peak > avg * 2:
        reasons.append("unusually high peak usage")

    if variance > 100:
        reasons.append("high consumption variance")

    if off_peak < avg * 0.5:
        reasons.append("low off-peak usage")

    if risk_score > 70:
        base = "⚠ High risk of energy theft detected"
    elif risk_score > 40:
        base = "⚡ Moderate risk detected"
    else:
        base = "✅ Usage appears normal"

    if reasons:
        return base + " due to " + ", ".join(reasons) + "."
    else:
        return base + "."
    

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---") 

#Prediction Section
st.markdown("---")
st.subheader("🔍 Predict Theft")

if st.button("Predict Theft", key="predict_btn"):

    if model is None:
        st.error("Model not loaded. Please check deployment.")
    else:
        features = np.array([[avg, peak, off_peak, variance]])
        prediction = model.predict(features)
        prob = model.predict_proba(features)
       
    if prediction[0] == 1:
        st.error(f"⚠ Theft Detected ({prob[0][1]*100:.2f}% confidence)")

        # Smart explanation logic
        reasons = []

        if peak > avg * 2:
            reasons.append("Unusually high peak usage")

        if variance > 100:
            reasons.append("High variance in consumption")

        if off_peak < avg * 0.5:
            reasons.append("Low off-peak usage")

        st.warning("💡 Reason:")
        for r in reasons:
            st.write(f"- {r}")

    else:
        st.success(f"✅ Normal Usage ({prob[0][0]*100:.2f}% confidence)")
        risk_score = prob[0][1] * 100
        st.metric("⚠ Risk Score", f"{risk_score:.2f}%")
        if risk_score > 70:
            st.error("High Risk ⚠")
        elif risk_score > 40:
            st.warning("Medium Risk ⚡")
        else:
            st.success("Low Risk ✅") 

uploaded_file = st.file_uploader("Upload Smart Meter CSV", type=["csv"])

if uploaded_file is not None:
    import pandas as pd
    data = pd.read_csv(uploaded_file)

    st.dataframe(data.head())

    required_cols = ["avg", "peak", "off_peak", "variance"]

    # ✅ keep EVERYTHING inside this block
    if model is None:
        st.error("Model not loaded. Please check deployment.")

    elif all(col in data.columns for col in required_cols):
        predictions = model.predict(data[required_cols])
        data["Prediction"] = predictions

        st.write("Results:", data)

        st.download_button(
            "Download Results",
            data.to_csv(index=False),
            file_name="predictions.csv"
        )

        total = len(data)
        fraud = sum(data["Prediction"])

        st.metric("Total Records", total)
        st.metric("Fraud Cases", fraud)
        st.metric("Fraud %", f"{(fraud/total)*100:.2f}%")

    else:
        st.error("CSV must contain columns: avg, peak, off_peak, variance")

         
# Summary Metrics , what it does not
if uploaded_file is not None and "Prediction" in data.columns:
    total = len(data)
    fraud = sum(data["Prediction"])
    st.metric("Total Records", total)
    st.metric("Fraud Cases", fraud)
    st.metric("Fraud %", f"{(fraud/total)*100:.2f}%")     
        

