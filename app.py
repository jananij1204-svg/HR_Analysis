import streamlit as st
import pickle
import numpy as np

# -------------------------------
# Load Model
# -------------------------------
model_path = "H.pkl"   # CHANGE your model filename here

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("HR Salary Category Prediction App")

# -------------------------------
# User Input Fields
# -------------------------------
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation", 0.0, 1.0, 0.5)
number_project = st.number_input("Number of Projects", 1, 20, 5)
average_montly_hours = st.number_input("Average Monthly Hours", 50, 400, 200)
time_spend_company = st.number_input("Years Spent in Company", 1, 20, 3)
work_accident = st.selectbox("Had Work Accident?", ["No", "Yes"])
promotion_last_5years = st.selectbox("Promotion in Last 5 Years?", ["No", "Yes"])

department = st.selectbox(
    "Department",
    [
        "RandD", "accounting", "hr", "management", "marketing",
        "product_mng", "sales", "support", "technical"
    ]
)

salary = st.selectbox("Current Salary Category", ["low", "medium", "high"])

# -------------------------------
# One-Hot Encoding (Manual)
# -------------------------------
dept_options = [
    "RandD", "accounting", "hr", "management", "marketing",
    "product_mng", "sales", "support", "technical"
]

dept_encoded = [1 if department == d else 0 for d in dept_options]

salary_low  = 1 if salary == "low" else 0
salary_medium = 1 if salary == "medium" else 0
# salary_high is automatically captured when both are 0

work_accident = 1 if work_accident == "Yes" else 0
promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0

# -------------------------------
# Final Input Feature Order (18)
# -------------------------------
features = np.array([
    satisfaction_level,
    last_evaluation,
    number_project,
    average_montly_hours,
    time_spend_company,
    work_accident,
    promotion_last_5years,
    dept_encoded[0],  # Department_RandD
    dept_encoded[1],  # accounting
    dept_encoded[2],  # hr
    dept_encoded[3],  # management
    dept_encoded[4],  # marketing
    dept_encoded[5],  # product_mng
    dept_encoded[6],  # sales
    dept_encoded[7],  # support
    dept_encoded[8],  # technical
    salary_low,
    salary_medium
]).reshape(1, -1)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Salary Category"):
    pred = model.predict(features)[0]   # returns string (low/medium/high)

    st.success(f"Predicted Salary Category: **{pred}**")