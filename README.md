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
- Accuracy: 80% 

---

##  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Matplotlib

---

##  Screenshots
<img width="2130" height="1186" alt="ETD image" src="https://github.com/user-attachments/assets/b91d752b-4389-4af3-9a4c-f87e02d17b67" />



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
