import os
import sys

import pandas as pd
import streamlit as st

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.plots import scatterplot

save_dir = os.path.join("data") 

# Ensure the folder exists
os.makedirs(save_dir, exist_ok=True)

home, upload_dataset, data_eng_tab, train_ml_model, prediction_tab = st.tabs([
    "🏠 Home", 
    "📁 Select/Upload Dataset", 
    "🛠️ Data Engineering", 
    "🤖 Train ML Model",
    "🔮 Prediction"
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
    
dataset_1 = False
dataset_2 = False
dataset_3 = False
dataset_4 = False

with upload_dataset:
    st.info(
        """
        You can select from the default datasets included in the project,
        or upload your own custom datasets to use with the models.
        """
    )

    choice = st.radio(
        "Choose one option:",
        ["I want to upload my dataset.", "I want to select from the real-world datasets."]
    )

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
        dataset_names = [
            "North Dakota Natural Gas Production",
            "North Dakota Cumulative Oil Production by Formation Through 2020",
            "North Dakota Historical Monthly Oil Production by County",
            "North Dakota Historical MCF Gas Produced by County"
        ]

        selected_dataset = st.radio("Select a dataset:", dataset_names)

        dataset_descriptions = {
            "North Dakota Natural Gas Production": """
            This dataset comes from North Dakota and contains real-world data. 
            It includes monthly values for gas produced, sold, and flared, along with 
            additional measurements specific to the Bakken formation. Because the data reflects
            actual field-level reporting, it is suitable for analytics, forecasting, and operational studies.
            """,
            "North Dakota Cumulative Oil Production by Formation Through 2020": """
            This dataset presents cumulative oil production in North Dakota by geological formation through December 2020. 
            Each row represents a specific formation, reporting the total oil produced (in barrels), the percentage contribution of 
            that formation to the overall production, and the number of wells associated with it. The dataset covers major formations 
            such as Bakken, Three Forks, Madison, Red River, and others, as well as minor formations, providing a comprehensive overview of 
            North Dakota’s oil production landscape. It is structured to facilitate comparative analysis across formations, evaluation of 
            production contributions, and assessment of well counts relative to output over time.
            """,
            "North Dakota Historical Monthly Oil Production by County": """
            This dataset provides historical monthly oil production data in North Dakota, 
            broken down by county, excluding confidential wells. The data spans from April 1951 to August 2025. 
            Each row corresponds to a specific month, while each column represents a county, reporting the number of 
            barrels of oil produced during that period. 
            Counties included are Adams, Billings, Bottineau, Bowman, Burke, Divide, Dunn, 
            Golden Valley, Hettinger, McHenry, McKenzie, McLean, Mercer, Mountrail, Renville, Slope, 
            Stark, Ward, and Williams. The dataset is structured to facilitate temporal analysis, 
            county-level comparisons, and trend assessment of oil production across North Dakota over time.
            """,
            "North Dakota Historical MCF Gas Produced by County": """
            This dataset provides monthly numerical data for multiple North Dakota counties, 
            including Adams, Billings, Bottineau, Bowman, Burke, Divide, Dunn, Golden Valley, Hettinger, 
            McHenry, McKenzie, McLean, Mercer, Mountrail, Renville, Slope, Stark, Ward, and Williams, 
            spanning from January 1990 to August 2025. Each row corresponds to a specific month, 
            while each column represents a county, reporting the recorded values for that period. 
            The dataset captures quantitative metrics for each county, which could represent population, 
            production, or another county-level measure. 
            It is structured to facilitate temporal analysis, regional comparisons, and trend observation across counties over time.
            """
        }

        st.markdown(f"""
        <div style="
            color:black;
            padding:15px;
            border-radius:8px;
            font-size:14px;
            line-height:1.5;
            max-height:350px;
            overflow-y:auto;
        ">
            {dataset_descriptions[selected_dataset]}
        </div>
        """, unsafe_allow_html=True)


def load_default_dataset(name):
    if name == "North Dakota Natural Gas Production":
        return pd.read_excel("data/ND_gas_1990_to_present.xlsx")
    elif name == "North Dakota Cumulative Oil Production by Formation Through 2020":
        return pd.read_excel("data/ND_cumulative_formation_2020.xlsx")
    elif name == "North Dakota Historical Monthly Oil Production by County":
        return pd.read_excel("data/ND_historical_barrels_of_oil_produced_by_county.xlsx")
    elif name == "North Dakota Historical MCF Gas Produced by County":
        return pd.read_excel("data/ND_historical_MCF_gas_produced_by_county.xlsx")
    else:
        return pd.DataFrame()


with data_eng_tab:
    st.info("Here, you can visualize and process your selected dataset before training your model.")

    if choice == "I want to upload my dataset." and 'uploaded_dataset' in st.session_state:
        st.write(f"You selected your uploaded dataset: {st.session_state['uploaded_filename']}")
        df_to_use = st.session_state['uploaded_dataset']

    elif choice == "I want to select from the real-world datasets.":
        st.write(f"You selected: {selected_dataset}")
        df_to_use = load_default_dataset(selected_dataset)

    else:
        st.write("No dataset selected yet.")
        df_to_use = pd.DataFrame() 

    if not df_to_use.empty:
        st.write("Statistical description of your dataset:")
        st.dataframe(df_to_use.describe())
        if st.toggle("Show Scatter Plot"):
            st.markdown("### Configure your scatter plot")

            df_columns = df_to_use.columns.tolist()

            # Group the axis selectors side by side
            col1, col2 = st.columns(2)
            with col1:
                x = st.radio("Select X‑axis column:", df_columns, index=0)
            with col2:
                y = st.radio("Select Y‑axis column:", df_columns, index=1)

            st.info(f"Plotting **{y}** vs **{x}**")

            scatterplot(df_to_use, x=x, y=y)


with train_ml_model:
    st.info(
        "Here, you can train machine learning models on the selected dataset, " \
        "configure the hyperparameters, and monitor the training process to achieve " \
        "the best performance."
        )

with prediction_tab:
    st.info("Here, you can predict based on the trained model.")
