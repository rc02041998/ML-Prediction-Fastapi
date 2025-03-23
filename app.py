import streamlit as st
import requests
import pandas as pd

BASE_URL = "http://127.0.0.1:8000"  # FastAPI base URL

st.title("ML Model Training & Prediction")

# Upload Dataset & Train Model
st.header("Upload Dataset & Train Model")
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
model_choice = st.selectbox("Select Model", ["random_forest", "logistic_regression"])

if st.button("Train Model"):
    if uploaded_file is not None:
        files = {"file": uploaded_file.getvalue()}
        data = {"model_name": model_choice}
        response = requests.post(f"{BASE_URL}/train/", files=files, data=data)
        if response.status_code == 200:
            st.success(f"Model trained successfully! Metrics: {response.json()['metrics']}")
        else:
            st.error("Failed to train model.")
    else:
        st.warning("Please upload a dataset.")

# Prediction Section
st.header("Make Predictions")
input_values = st.text_input("Enter feature values (comma-separated)")

if st.button("Predict"):
    if input_values:
        try:
            input_list = [float(x) for x in input_values.split(",")]
            response = requests.post(f"{BASE_URL}/predict/", json={"features": input_list})
            if response.status_code == 200:
                st.success(f"Prediction: {response.json()['prediction']}")
            else:
                st.error("Failed to make prediction.")
        except ValueError:
            st.error("Please enter valid numerical values.")
    else:
        st.warning("Please enter feature values.")
