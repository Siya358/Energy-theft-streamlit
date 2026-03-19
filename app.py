import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("⚡ Energy Theft Detection System")

st.write("Enter smart meter details:")

avg = st.number_input("Average Usage")
peak = st.number_input("Peak Usage")
off_peak = st.number_input("Off Peak Usage")
variance = st.number_input("Variance")

if st.button("Predict"):
    features = np.array([[avg, peak, off_peak, variance]])
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠ Theft Detected")
    else:
        st.success("✅ Normal Usage")
