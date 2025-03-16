import streamlit as st
import requests
import pandas as pd
import io

st.title("Vomitoxin Contamination Prediction")

backend_url = "http://127.0.0.1:5000"  # Change if backend runs elsewhere

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    if st.button("Train Model"):
        with st.spinner("Training the model..."):
            files = {'file': uploaded_file.getvalue()}
            response = requests.post(f"{backend_url}/train", files=files)
            if response.status_code == 200:
                st.success("Model training completed successfully!")
            else:
                st.error(response.json()["error"])

    if st.button("Predict Contamination"):
        with st.spinner("Making predictions..."):
            files = {'file': uploaded_file.getvalue()}
            response = requests.post(f"{backend_url}/predict", files=files)
            if response.status_code == 200:
                predictions = response.json()["predictions"]
                df["predicted_vomitoxin"] = predictions
                st.write("Predictions:")
                st.write(df[["vomitoxin_ppb", "predicted_vomitoxin"]])
            else:
                st.error(response.json()["error"])