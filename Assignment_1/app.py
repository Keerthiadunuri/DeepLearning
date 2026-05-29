# =========================================
# AI-POWERED EMPLOYEE ATTRITION PREDICTION
# STREAMLIT APPLICATION
# =========================================

# Import Libraries
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# =========================================
# LOAD MODEL & SCALER
# =========================================

model = joblib.load("logistic_regression.pkl")

scaler = joblib.load("scaler.pkl")

# =========================================
# APPLICATION TITLE
# =========================================

st.title("📊 AI-Powered Employee Attrition Prediction System")

st.markdown(
    """
    Predict employee attrition risk using Machine Learning.
    """
)

# =========================================
# SIDEBAR
# =========================================

st.sidebar.header("Employee Details")

# =========================================
# USER INPUTS
# =========================================

Age = st.sidebar.slider(
    "Age",
    18,
    60,
    30
)

DailyRate = st.sidebar.slider(
    "Daily Rate",
    100,
    1500,
    800
)

DistanceFromHome = st.sidebar.slider(
    "Distance From Home",
    1,
    30,
    10
)

Education = st.sidebar.selectbox(
    "Education",
    [1, 2, 3, 4, 5]
)

EnvironmentSatisfaction = st.sidebar.selectbox(
    "Environment Satisfaction",
    [1, 2, 3, 4]
)

HourlyRate = st.sidebar.slider(
    "Hourly Rate",
    30,
    100,
    60
)

JobInvolvement = st.sidebar.selectbox(
    "Job Involvement",
    [1, 2, 3, 4]
)

JobLevel = st.sidebar.selectbox(
    "Job Level",
    [1, 2, 3, 4, 5]
)

JobSatisfaction = st.sidebar.selectbox(
    "Job Satisfaction",
    [1, 2, 3, 4]
)

MonthlyIncome = st.sidebar.slider(
    "Monthly Income",
    1000,
    20000,
    5000
)

MonthlyRate = st.sidebar.slider(
    "Monthly Rate",
    2000,
    30000,
    15000
)

NumCompaniesWorked = st.sidebar.slider(
    "Number of Companies Worked",
    0,
    10,
    2
)

PercentSalaryHike = st.sidebar.slider(
    "Percent Salary Hike",
    10,
    30,
    15
)

PerformanceRating = st.sidebar.selectbox(
    "Performance Rating",
    [1, 2, 3, 4]
)

RelationshipSatisfaction = st.sidebar.selectbox(
    "Relationship Satisfaction",
    [1, 2, 3, 4]
)

StockOptionLevel = st.sidebar.selectbox(
    "Stock Option Level",
    [0, 1, 2, 3]
)

TotalWorkingYears = st.sidebar.slider(
    "Total Working Years",
    0,
    40,
    10
)

TrainingTimesLastYear = st.sidebar.slider(
    "Training Times Last Year",
    0,
    10,
    3
)

WorkLifeBalance = st.sidebar.selectbox(
    "Work Life Balance",
    [1, 2, 3, 4]
)

YearsAtCompany = st.sidebar.slider(
    "Years At Company",
    0,
    40,
    5
)

YearsInCurrentRole = st.sidebar.slider(
    "Years In Current Role",
    0,
    20,
    3
)

YearsSinceLastPromotion = st.sidebar.slider(
    "Years Since Last Promotion",
    0,
    15,
    1
)

YearsWithCurrManager = st.sidebar.slider(
    "Years With Current Manager",
    0,
    20,
    3
)

# =========================================
# ENCODED CATEGORICAL FEATURES
# =========================================

BusinessTravel = st.sidebar.selectbox(
    "Business Travel",
    [0, 1, 2]
)

Department = st.sidebar.selectbox(
    "Department",
    [0, 1, 2]
)

EducationField = st.sidebar.selectbox(
    "Education Field",
    [0, 1, 2, 3, 4, 5]
)

Gender = st.sidebar.selectbox(
    "Gender",
    [0, 1]
)

JobRole = st.sidebar.selectbox(
    "Job Role",
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
)

MaritalStatus = st.sidebar.selectbox(
    "Marital Status",
    [0, 1, 2]
)

OverTime = st.sidebar.selectbox(
    "OverTime",
    [0, 1]
)

# =========================================
# CREATE INPUT DATAFRAME
# =========================================

input_data = pd.DataFrame({
    'Age': [Age],
    'BusinessTravel': [BusinessTravel],
    'DailyRate': [DailyRate],
    'Department': [Department],
    'DistanceFromHome': [DistanceFromHome],
    'Education': [Education],
    'EducationField': [EducationField],
    'EnvironmentSatisfaction': [EnvironmentSatisfaction],
    'Gender': [Gender],
    'HourlyRate': [HourlyRate],
    'JobInvolvement': [JobInvolvement],
    'JobLevel': [JobLevel],
    'JobRole': [JobRole],
    'JobSatisfaction': [JobSatisfaction],
    'MaritalStatus': [MaritalStatus],
    'MonthlyIncome': [MonthlyIncome],
    'MonthlyRate': [MonthlyRate],
    'NumCompaniesWorked': [NumCompaniesWorked],
    'OverTime': [OverTime],
    'PercentSalaryHike': [PercentSalaryHike],
    'PerformanceRating': [PerformanceRating],
    'RelationshipSatisfaction': [RelationshipSatisfaction],
    'StockOptionLevel': [StockOptionLevel],
    'TotalWorkingYears': [TotalWorkingYears],
    'TrainingTimesLastYear': [TrainingTimesLastYear],
    'WorkLifeBalance': [WorkLifeBalance],
    'YearsAtCompany': [YearsAtCompany],
    'YearsInCurrentRole': [YearsInCurrentRole],
    'YearsSinceLastPromotion': [YearsSinceLastPromotion],
    'YearsWithCurrManager': [YearsWithCurrManager]
})

# =========================================
# DISPLAY INPUT DATA
# =========================================

st.subheader("Employee Input Data")

st.dataframe(input_data)

# =========================================
# SCALE INPUT DATA
# =========================================

input_scaled = scaler.transform(input_data)

# =========================================
# PREDICTION BUTTON
# =========================================

if st.button("Predict Attrition"):

    prediction = model.predict(input_scaled)

    probability = model.predict_proba(input_scaled)

    attrition_prob = probability[0][1] * 100

    # =====================================
    # RISK LEVEL
    # =====================================

    if attrition_prob > 70:
        risk_level = "HIGH RISK"

    elif attrition_prob > 40:
        risk_level = "MEDIUM RISK"

    else:
        risk_level = "LOW RISK"

    # =====================================
    # DISPLAY RESULTS
    # =====================================

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Employee Likely to Leave")

    else:
        st.success("✅ Employee Likely to Stay")

    st.metric(
        "Attrition Probability",
        f"{attrition_prob:.2f}%"
    )

    st.metric(
        "Risk Level",
        risk_level
    )

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown(
    "Developed using Streamlit & Machine Learning"
)