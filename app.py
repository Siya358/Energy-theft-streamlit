import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
st.set_page_config(page_title="Energy Theft Detection", layout="centered")

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("⚡ Energy Theft Detection System")
st.markdown("### ⚡ Smart Grid Monitoring System")

st.write("Enter smart meter details:")

avg = st.number_input("Average Usage")
peak = st.number_input("Peak Usage")
off_peak = st.number_input("Off Peak Usage")
variance = st.number_input("Variance")


if st.button("Show Usage Graph"):
    values = [avg, peak, off_peak, variance]
    labels = ["Avg", "Peak", "Off-Peak", "Variance"]

    plt.figure()
    plt.bar(labels, values)
    st.pyplot(plt)

if st.button("Predict"):
    features = np.array([[avg, peak, off_peak, variance]])
    
    prediction = model.predict(features)
    prob = model.predict_proba(features)

    if prediction[0] == 1:
        st.error(f"⚠ Theft Detected ({prob[0][1]*100:.2f}% confidence)")
    else:
        st.success(f"✅ Normal Usage ({prob[0][0]*100:.2f}% confidence)")
