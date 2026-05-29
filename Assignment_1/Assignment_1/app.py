import streamlit as st
import pandas as pd
import joblib

model = joblib.load("random_forest.pkl")

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊"
)

st.title("AI-Powered Employee Attrition Prediction System")

st.write("Enter Employee Details")

age = st.number_input(
    "Age",
    min_value=18,
    max_value=60,
    value=35
)

daily_rate = st.number_input(
    "Daily Rate",
    value=1000
)

distance_from_home = st.number_input(
    "Distance From Home",
    value=5
)

education = st.selectbox(
    "Education",
    [1,2,3,4,5]
)

environment_satisfaction = st.selectbox(
    "Environment Satisfaction",
    [1,2,3,4]
)

job_involvement = st.selectbox(
    "Job Involvement",
    [1,2,3,4]
)

job_level = st.selectbox(
    "Job Level",
    [1,2,3,4,5]
)

job_satisfaction = st.selectbox(
    "Job Satisfaction",
    [1,2,3,4]
)

monthly_income = st.number_input(
    "Monthly Income",
    value=4500
)

num_companies_worked = st.number_input(
    "Number Of Companies Worked",
    value=2
)

percent_salary_hike = st.number_input(
    "Percent Salary Hike",
    value=15
)

performance_rating = st.selectbox(
    "Performance Rating",
    [3,4]
)

relationship_satisfaction = st.selectbox(
    "Relationship Satisfaction",
    [1,2,3,4]
)

stock_option_level = st.selectbox(
    "Stock Option Level",
    [0,1,2,3]
)

total_working_years = st.number_input(
    "Total Working Years",
    value=10
)

training_times_last_year = st.number_input(
    "Training Times Last Year",
    value=2
)

work_life_balance = st.selectbox(
    "Work Life Balance",
    [1,2,3,4]
)

years_at_company = st.number_input(
    "Years At Company",
    value=5
)

years_in_current_role = st.number_input(
    "Years In Current Role",
    value=3
)

years_since_last_promotion = st.number_input(
    "Years Since Last Promotion",
    value=1
)

years_with_curr_manager = st.number_input(
    "Years With Current Manager",
    value=3
)

if st.button("Predict Attrition"):

    features = pd.DataFrame([[
        age,
        0,
        daily_rate,
        1,
        distance_from_home,
        education,
        1,
        1,
        1000,
        environment_satisfaction,
        1,
        80,
        job_involvement,
        job_level,
        1,
        job_satisfaction,
        1,
        monthly_income,
        20000,
        num_companies_worked,
        0,
        percent_salary_hike,
        performance_rating,
        relationship_satisfaction,
        80,
        stock_option_level,
        total_working_years,
        training_times_last_year,
        work_life_balance,
        years_at_company,
        years_in_current_role,
        years_since_last_promotion,
        years_with_curr_manager,
        1
    ]])

    prediction = model.predict(features)[0]

    probability = model.predict_proba(features)[0][1] * 100

    if probability >= 75:
        risk = "HIGH"
    elif probability >= 40:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    st.subheader("Prediction Result")

    st.write(
        "Attrition Probability:",
        round(probability,2),
        "%"
    )

    st.write(
        "Risk Level:",
        risk
    )

    if prediction == 1:
        st.error(
            "Employee likely to leave."
        )
    else:
        st.success(
            "Employee likely to stay."
        )