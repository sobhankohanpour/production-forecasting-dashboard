import streamlit as st


home, upload_dataset, preprocessing_tab, train_ml_model = st.tabs([
    "🏠 Home", 
    "📁 Select/Upload Dataset", 
    "🛠️ Preprocessing", 
    "🤖 Train ML Model"
])

with home:
    st.title("Welcome to Well Production Forecasting Dashboard 🛢️📈")
    st.write(
        "Welcome to the Well Production Forecasting Dashboard — " \
        "your smart companion for data-driven petroleum engineering."
        )
    st.write(
        "This platform leverages advanced machine learning techniques " \
        "to predict oil and gas production rates with high accuracy."
        )
    st.write(
        "Users can upload their own well datasets, allowing the system to " \
        "train custom models tailored to their field conditions and generate precise, scenario-specific forecasts."
        )
    st.write(
        "Whether you're optimizing field development, monitoring reservoir performance, or " \
        "planning future operations, this dashboard provides actionable insights, intuitive " \
        "visualizations, and AI-powered predictions designed for real-world petroleum engineering workflows."
        )
with upload_dataset:
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    uploaded_file
with preprocessing_tab:
    st.write("Data preprocessing...")
with train_ml_model:
    st.write("Training the ML model...")
