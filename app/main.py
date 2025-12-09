# ───────────────────────────────
# Standard Library
# ───────────────────────────────
import os
import sys

# Add the project root to sys.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

# ───────────────────────────────
# Third-Party Libraries
# ───────────────────────────────
import pandas as pd
import streamlit as st

# ───────────────────────────────
# Local Imports — Plots
# ───────────────────────────────
from src.plots import (
    barplot,
    boxplot,
    catplot,
    distplot,
    ecdfplot,
    histplot,
    kdeplot,
    lineplot,
    pointplot,
    rugplot,
    scatterplot,
    stripplot,
    swarmplot,
    violinplot,
)

from strings.strings import (
    PLATFORM_DESCRIPTION,
    UPLOAD_INSTRUCTION,
    WELCOME_MESSAGE,
    WORKFLOW_SUPPORT_MESSAGE,
)

# ───────────────────────────────
# Local Imports — App Tabs
# ───────────────────────────────
from app.prediction_tab import run_prediction_tab
from app.train_tab import render_train_tab
from app.upload_tab import run_upload_tab

save_dir = os.path.join("data") 

# Ensure the folder exists
os.makedirs(save_dir, exist_ok=True)

dataset_1 = False
dataset_2 = False
dataset_3 = False
dataset_4 = False

home, upload_dataset, data_eng_tab, train_ml_model, prediction_tab = st.tabs([
    "🏠 Home", 
    "📁 Select/Upload Dataset", 
    "🛠️ Data Engineering", 
    "🤖 Train ML Model",
    "🔮 Prediction"
])


with home:
    st.title("Welcome to Well Production Forecasting Dashboard 🛢️📈")
    st.write(WELCOME_MESSAGE)
    st.write(PLATFORM_DESCRIPTION)
    st.write(UPLOAD_INSTRUCTION)
    st.write(WORKFLOW_SUPPORT_MESSAGE)


with upload_dataset:
    run_upload_tab()


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
    
    selected_dataset = st.session_state.get("selected_dataset", None)

    choice = st.session_state.get("upload_choice", None)

    # Dataset selection
    if choice == "I want to upload my dataset." and 'uploaded_dataset' in st.session_state:
        st.success(f"✅ You selected your uploaded dataset: {st.session_state['uploaded_filename']}")
        df_to_use = st.session_state['uploaded_dataset']

    elif choice == "I want to select from the real-world datasets.":
        st.success(f"🌍 You selected: {selected_dataset}")
        df_to_use = load_default_dataset(selected_dataset)

    else:
        st.warning("⚠️ No dataset selected yet.")
        df_to_use = pd.DataFrame()

    # If dataset is available
    if not df_to_use.empty:
        st.subheader("📊 Dataset Overview")
        st.dataframe(df_to_use.describe())

        # Master toggle for visualization
        if st.toggle("📈 Enable Data Visualization"):
            st.markdown("### 🎨 Choose Plot Group")

            # Group selector
            plot_group = st.selectbox(
                "Select a group of plots:",
                ["Distribution Plots", "Categorical Plots", "Relational Plots"]
            )

            df_columns = df_to_use.columns.tolist()

            # --- Distribution Plots ---
            if plot_group == "Distribution Plots":
                with st.expander("Distribution Plot"):
                    x = st.radio("Select column:", df_columns, index=0, key="dist_x")
                    st.info(f"Plotting distribution of **{x}**")
                    distplot(df_to_use, x=x)

                with st.expander("Histogram"):
                    x = st.radio("Select column:", df_columns, index=0, key="hist_x")
                    bins = st.slider("Number of bins", 5, 100, 10, key="hist_bins")
                    st.info(f"Plotting histogram of **{x}** with {bins} bins")
                    histplot(df_to_use, x=x, bins=bins)

                with st.expander("KDE Plot"):
                    x = st.radio("Select column:", df_columns, index=0, key="kde_x")
                    st.info(f"Plotting KDE of **{x}**")
                    kdeplot(df_to_use, x=x)

                with st.expander("ECDF Plot"):
                    x = st.radio("Select column:", df_columns, index=0, key="ecdf_x")
                    st.info(f"Plotting ECDF of **{x}**")
                    ecdfplot(df_to_use, x=x)

                with st.expander("Rug Plot"):
                    x = st.radio("Select column:", df_columns, index=0, key="rug_x")
                    st.info(f"Plotting rug plot of **{x}**")
                    rugplot(df_to_use, x=x)

            # --- Categorical Plots ---
            elif plot_group == "Categorical Plots":
                with st.expander("Cat Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="cat_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="cat_y")
                    st.info(f"Plotting categorical **{y}** vs **{x}**")
                    catplot(df_to_use, x=x, y=y, kind="box")

                with st.expander("Strip Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="strip_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="strip_y")
                    st.info(f"Plotting strip **{y}** vs **{x}**")
                    stripplot(df_to_use, x=x, y=y)

                with st.expander("Swarm Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="swarm_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="swarm_y")
                    st.info(f"Plotting swarm **{y}** vs **{x}**")
                    swarmplot(df_to_use, x=x, y=y)

                with st.expander("Box Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="box_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="box_y")
                    st.info(f"Plotting box **{y}** vs **{x}**")
                    boxplot(df_to_use, x=x, y=y)

                with st.expander("Violin Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="violin_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="violin_y")
                    st.info(f"Plotting violin **{y}** vs **{x}**")
                    violinplot(df_to_use, x=x, y=y)

                with st.expander("Point Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="point_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="point_y")
                    st.info(f"Plotting point **{y}** vs **{x}**")
                    pointplot(df_to_use, x=x, y=y)

                with st.expander("Bar Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="bar_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="bar_y")
                    st.info(f"Plotting bar **{y}** vs **{x}**")
                    barplot(df_to_use, x=x, y=y)

            # --- Relational Plots ---
            elif plot_group == "Relational Plots":
                with st.expander("Scatter Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="scatter_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="scatter_y")
                    st.info(f"Plotting **{y}** vs **{x}**")
                    scatterplot(df_to_use, x=x, y=y)

                with st.expander("Line Plot"):
                    col1, col2 = st.columns(2)
                    with col1:
                        x = st.radio("Select X‑axis column:", df_columns, index=0, key="line_x")
                    with col2:
                        y = st.radio("Select Y‑axis column:", df_columns, index=1, key="line_y")
                    st.info(f"Plotting **{y}** vs **{x}**")
                    lineplot(df_to_use, x=x, y=y)

        # Master toggle for 'Data Preprocessing'
        if st.toggle("🔧 Enable Data Preprocessing"):
            
            st.subheader("🧹 Data Preprocessing Tools")

            # ==========================================================
            # 1) CLEAN MISSING VALUES
            # ==========================================================
            with st.expander("🧼 Clean Missing Values"):
                missing_count = df_to_use.isna().sum().sum()
                st.write(f"Missing values detected: **{missing_count}**")

                strategy = st.radio(
                    "Choose strategy:",
                    ["None", "Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill Custom Value"],
                    index=0
                )

                if strategy == "Drop Rows":
                    df_to_use.dropna(inplace=True)
                    st.success("✔ Rows with missing values removed.")

                elif strategy == "Drop Columns":
                    df_to_use.dropna(axis=1, inplace=True)
                    st.success("✔ Columns containing missing values removed.")

                elif strategy == "Fill with Mean":
                    df_to_use.fillna(df_to_use.mean(numeric_only=True), inplace=True)
                    st.success("✔ Missing numerical values filled using column means.")

                elif strategy == "Fill with Median":
                    df_to_use.fillna(df_to_use.median(numeric_only=True), inplace=True)
                    st.success("✔ Missing numerical values filled using medians.")

                elif strategy == "Fill with Mode":
                    df_to_use.fillna(df_to_use.mode().iloc[0], inplace=True)
                    st.success("✔ Missing values filled using mode.")

                elif strategy == "Fill Custom Value":
                    custom = st.text_input("Enter value to fill missing cells:")
                    if custom:
                        df_to_use.fillna(custom, inplace=True)
                        st.success(f"✔ Missing values replaced with **{custom}**")


            # ==========================================================
            # 2) REMOVE DUPLICATES
            # ==========================================================
            with st.expander("🗃️ Remove Duplicates"):
                duplicates = df_to_use.duplicated().sum()
                st.write(f"Duplicate rows detected: **{duplicates}**")

                if st.button("Remove Duplicates Now"):
                    df_to_use.drop_duplicates(inplace=True)
                    st.success("✔ Duplicate records removed successfully.")


            # ==========================================================
            # 3) HANDLE OUTLIERS (IQR Method)
            # ==========================================================
            with st.expander("🔍 Handle Outliers"):
                num_cols = df_to_use.select_dtypes(include=["int", "float"]).columns.tolist()

                if len(num_cols) == 0:
                    st.warning("⚠ No numeric columns available for outlier detection.")
                else:
                    col = st.selectbox("Select column to evaluate:", num_cols)

                    if st.button(f"Apply IQR Outlier Filtering on `{col}`"):
                        Q1 = df_to_use[col].quantile(0.25)
                        Q3 = df_to_use[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

                        before = len(df_to_use)
                        df_to_use = df_to_use[(df_to_use[col] >= lower) & (df_to_use[col] <= upper)]
                        removed = before - len(df_to_use)

                        st.success(f"✔ Outliers removed from **{col}** | Rows dropped: **{removed}**")


            # ==========================================================
            # 4) DATA NORMALIZATION / SCALING
            # ==========================================================
            with st.expander("⚖️ Data Normalization"):
                scale_method = st.radio(
                    "Choose scaling method:",
                    ["None", "Min-Max Scaling (0→1)", "Standard Scaling (Z-score)"]
                )

                num_cols = df_to_use.select_dtypes(include=["int", "float"]).columns.tolist()

                if scale_method != "None" and len(num_cols) > 0:

                    from sklearn.preprocessing import MinMaxScaler, StandardScaler

                    if scale_method == "Min-Max Scaling (0→1)":
                        scaler = MinMaxScaler()
                        df_to_use[num_cols] = scaler.fit_transform(df_to_use[num_cols])
                        st.success("✔ Feature scaling completed (Min-Max).")

                    elif scale_method == "Standard Scaling (Z-score)":
                        scaler = StandardScaler()
                        df_to_use[num_cols] = scaler.fit_transform(df_to_use[num_cols])
                        st.success("✔ Standard normalization applied (mean=0, std=1).")

                elif scale_method != "None":
                    st.warning("⚠ No numeric features available for scaling.")

            st.success("✨ Preprocessing applied! You may now proceed to model training.")


with train_ml_model:
    render_train_tab(df_to_use)

with prediction_tab:
    run_prediction_tab()
