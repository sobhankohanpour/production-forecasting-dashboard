import os
import pandas as pd
import streamlit as st

save_dir = os.path.join("data") 

# Ensure the folder exists
os.makedirs(save_dir, exist_ok=True)

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
    if st.toggle("I want to upload my dataset."):
        number_of_user_dataset = st.slider(
            "Number of dataset(s)", min_value=1, max_value=50, value=1
        )
        st.write(f"You must upload {number_of_user_dataset} dataset(s).")

        uploaded_files = []
        for i in range(number_of_user_dataset):
            uploaded_file = st.file_uploader(
                f"Upload dataset #{i+1}", type=["xlsx"], key=f"dataset_{i}"
            )
            if uploaded_file:
                uploaded_file.seek(0)  # reset file pointer
                df_uploaded_file = pd.read_excel(uploaded_file)
                st.dataframe(df_uploaded_file)

                # Save with original name
                save_path = os.path.join(save_dir, uploaded_file.name)
                df_uploaded_file.to_excel(save_path, index=False)
                st.success(f"Dataset #{i+1} saved.")

with preprocessing_tab:
    st.write("Data preprocessing...")

with train_ml_model:
    st.write("Training the ML model...")
