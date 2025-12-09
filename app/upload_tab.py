import pandas as pd
import streamlit as st

# ───────────────────────────────
# Local Imports — Strings
# ───────────────────────────────
from strings.strings import (
    DESC_CUMULATIVE_OIL_PRODUCTION_2020,
    DESC_MCF_GAS_PRODUCTION_BY_COUNTY,
    DESC_MONTHLY_OIL_PRODUCTION_BY_COUNTY,
    DESC_NATURAL_GAS_PRODUCTION,
    NAME_CUMULATIVE_OIL_PRODUCTION_2020,
    NAME_MCF_GAS_PRODUCTION_BY_COUNTY,
    NAME_MONTHLY_OIL_PRODUCTION_BY_COUNTY,
    NAME_NATURAL_GAS_PRODUCTION,
    UPLOAD_DATASET_INFO,
)

def run_upload_tab():
    st.info(UPLOAD_DATASET_INFO)

    choice = st.radio(
        "Choose one option:",
        ["I want to upload my dataset.", "I want to select from the real-world datasets."]
    )

    st.session_state['upload_choice'] = choice

    if choice == "I want to upload my dataset.":
        st.write("====================================================")
        st.info("You can upload only `.xlsx` files.")
        number_of_user_dataset = 1
        uploaded_file = st.file_uploader("Upload your dataset", type=["xlsx"])
        if uploaded_file:
            df_uploaded_file = pd.read_excel(uploaded_file)
            st.session_state['uploaded_dataset'] = df_uploaded_file
            st.session_state['uploaded_filename'] = uploaded_file.name
            st.dataframe(df_uploaded_file)


    elif choice == "I want to select from the real-world datasets.":
        st.write("====================================================")
        DATASET_NAMES = [
            NAME_NATURAL_GAS_PRODUCTION,
            NAME_CUMULATIVE_OIL_PRODUCTION_2020,
            NAME_MONTHLY_OIL_PRODUCTION_BY_COUNTY,
            NAME_MCF_GAS_PRODUCTION_BY_COUNTY
        ]

        selected_dataset = st.radio("Select a dataset:", DATASET_NAMES)

        dataset_descriptions = {
            NAME_NATURAL_GAS_PRODUCTION: DESC_NATURAL_GAS_PRODUCTION,
            NAME_CUMULATIVE_OIL_PRODUCTION_2020: DESC_CUMULATIVE_OIL_PRODUCTION_2020,
            NAME_MONTHLY_OIL_PRODUCTION_BY_COUNTY: DESC_MONTHLY_OIL_PRODUCTION_BY_COUNTY,
            NAME_MCF_GAS_PRODUCTION_BY_COUNTY: DESC_MCF_GAS_PRODUCTION_BY_COUNTY,
        }
        
        st.session_state['selected_dataset'] = selected_dataset


        with st.expander("Dataset Description"):
            st.write(dataset_descriptions[selected_dataset])