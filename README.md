## Energy Theft Detection using Machine Learning

##  Overview
This project detects electricity theft in smart grids using machine learning.  
It analyzes smart meter data and predicts whether the usage is **normal or suspicious**.

---

## Features
-  Detects energy theft using ML model
-  Displays prediction with confidence score
-  Shows usage graph visualization
-  Interactive web app using Streamlit
-  Real-time prediction

---

##  Machine Learning
- Model Used: Random Forest Classifier
- Accuracy: XX%  ← (replace with your actual accuracy)

---

##  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

##  Screenshots
<img width="1280" height="800" alt="Screenshot 2026-04-21 at 5 09 57 PM" src="https://github.com/user-attachments/assets/ee7772ab-b700-4220-b0a9-b7da52da0ef0" />


---
---

##  Project Structure
energy_theft_streamlit/
│
├── app.py
├── train_model.py
├── model.pkl
├── smart_meter_data.csv
├── requirements.txt
└── README.md

---

##  How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
