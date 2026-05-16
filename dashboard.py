import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load trained models

regression_model = joblib.load(
    "models/covid_regression_model.pkl"
)

classification_model = joblib.load(
    "models/covid_classification_model.pkl"
)

# Load processed dataset

df = pd.read_csv(
    "data/processed/uganda_covid_processed.csv"
)

# Risk labels

risk_labels = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Critical"
}

# Dashboard title

st.title("COVID-19 Prediction Dashboard")

st.write(
    """
    Predict future COVID-19 cases and classify
    outbreak risk levels using Machine Learning.
    """
)

# Sidebar inputs

st.sidebar.header("Input Features")

previous_day_cases = st.sidebar.number_input(
    "Previous Day Cases",
    min_value=0.0,
    value=100.0
)

rolling_avg_cases = st.sidebar.number_input(
    "7-Day Rolling Average",
    min_value=0.0,
    value=120.0
)

growth_rate = st.sidebar.number_input(
    "Growth Rate",
    value=0.2
)

# Prediction button

if st.sidebar.button("Predict"):

    # Prepare features

    features = np.array([[
        previous_day_cases,
        rolling_avg_cases,
        growth_rate
    ]])

    # Regression prediction

    predicted_cases = (
        regression_model.predict(features)[0]
    )

    # Classification prediction

    risk_prediction = (
        classification_model.predict(features)[0]
    )

    risk_name = risk_labels[
        risk_prediction
    ]

    # Display results

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Predicted Future Cases",
            round(predicted_cases, 2)
        )

    with col2:

        st.metric(
            "Risk Level",
            risk_name
        )

# COVID trend graph

st.subheader("COVID-19 Daily Cases")

fig, ax = plt.subplots(figsize=(12,5))

ax.plot(
    df["date"],
    df["new_confirmed"]
)

ax.set_title(
    "Daily COVID-19 Cases in Uganda"
)

ax.set_xlabel("Date")

ax.set_ylabel("Cases")

plt.xticks(rotation=45)

st.pyplot(fig)

# Risk category graph

st.subheader("Risk Category Distribution")

risk_counts = (
    df["risk_level"]
    .value_counts()
    .sort_index()
)

risk_names = [
    risk_labels[i]
    for i in risk_counts.index
]

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.bar(
    risk_names,
    risk_counts.values
)

ax2.set_title(
    "COVID-19 Risk Categories"
)

ax2.set_xlabel("Risk Level")

ax2.set_ylabel("Frequency")

st.pyplot(fig2)

# Show dataset

if st.checkbox("Show Processed Dataset"):

    st.dataframe(
        df.head(50)
    )

# Footer

st.write("---")

st.write(
    "Machine Learning Deployment using Streamlit, Docker, and Kubernetes"
)