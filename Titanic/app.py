import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================
model = tf.keras.models.load_model("titanic_ann_model.h5")

# =====================================================
# HEADER
# =====================================================

st.title("🚢 Titanic Survival Prediction System")

st.subheader(
    "Deep Learning Based Passenger Survival Prediction"
)

st.markdown("---")

# =====================================================
# PROJECT DESCRIPTION
# =====================================================

st.header("Project Description")

st.write("""
This application predicts whether a passenger
would survive during the Titanic disaster
using an Artificial Neural Network (ANN).
""")

st.markdown("---")

# =====================================================
# INPUT SECTION
# =====================================================

st.header("Passenger Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        value=120.0
    )

# =====================================================
# NORMALIZATION
# =====================================================

def preprocess(pclass, age, fare):

    pclass = (pclass - 1) / (3 - 1)
    age = age / 80
    fare = fare / 600

    return np.array([[pclass, age, fare]])

# =====================================================
# PREDICT BUTTON
# =====================================================

if st.button("Predict Survival"):

    # Preprocess Input
    input_data = preprocess(
        pclass,
        age,
        fare
    )

    # Prediction
    prediction = model.predict(input_data)

    probability = prediction[0][0]

    # =================================================
    # PREDICTION LOGIC
    # =================================================

    if probability > 0.5:
        result = "Survived ✅"
    else:
        result = "Not Survived ❌"

    # =================================================
    # OUTPUT SECTION
    # =================================================

    st.markdown("---")

    st.header("Prediction Output")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prediction",
            result
        )

    with col2:
        st.metric(
            "Survival Probability",
            f"{probability:.2f}"
        )

    with col3:
        st.metric(
            "Confidence Score",
            f"{probability*100:.2f}%"
        )

    # =================================================
    # VISUALIZATION
    # =================================================

    st.markdown("---")

    st.header("Prediction Visualization")

    survive = probability
    not_survive = 1 - probability

    labels = ["Survived", "Not Survived"]

    values = [survive, not_survive]

    fig, ax = plt.subplots()

    ax.pie(
        values,
        labels=labels,
        autopct="%1.1f%%"
    )

    st.pyplot(fig)