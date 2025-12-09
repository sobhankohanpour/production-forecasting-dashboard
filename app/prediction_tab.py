
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def run_prediction_tab():

    st.title("🔮 Model Prediction")
    st.info("Use your trained model to make predictions and generate forecasts.")

    # ============================================
    # 1) Check if trained model exists
    # ============================================
    if (
        "trained_model" not in st.session_state
        or "trained_features" not in st.session_state
        or "trained_target" not in st.session_state
    ):
        st.error("❌ No trained model found. Please train a model in the previous tab first.")
        st.stop()

    model = st.session_state["trained_model"]
    trained_features = st.session_state["trained_features"]
    trained_target = st.session_state["trained_target"]

    st.success(f"📌 Model will predict the target column: **{trained_target}**")

    # ============================================
    # 2) Upload dataset for prediction
    # ============================================
    st.subheader("📁 Upload Dataset for Prediction")
    pred_file = st.file_uploader("Upload dataset (.xlsx)", type=["xlsx"])

    if pred_file:
        df_pred = pd.read_excel(pred_file)
        st.write("### 👀 Preview of Uploaded Data")
        st.dataframe(df_pred.head())
    else:
        st.warning("Please upload a dataset to perform predictions.")
        st.stop()

    # ============================================
    # 3) Ensure required columns exist
    # ============================================
    missing_cols = set(trained_features) - set(df_pred.columns)
    if missing_cols:
        st.error(f"❌ Missing required feature columns: {missing_cols}")
        st.stop()

    # ============================================
    # 4) Preprocess for prediction (same as training)
    # ============================================
    st.subheader("⚙ Auto Preprocessing for Prediction")

    df_input = df_pred[trained_features].copy()

    from sklearn.preprocessing import LabelEncoder

    for col in df_input.columns:

        # Try: datetime → timestamp
        try:
            df_input[col] = pd.to_datetime(df_input[col], errors="raise")
            df_input[col] = df_input[col].astype("int64") // 10**9
            continue
        except:
            pass

        # Encode non-numeric
        if df_input[col].dtype == "object":
            le = LabelEncoder()
            df_input[col] = le.fit_transform(df_input[col].astype(str))

    # Safety: convert any remaining datetime64
    for col in df_input.columns:
        if str(df_input[col].dtype).startswith("datetime64"):
            df_input[col] = df_input[col].astype("int64") // 10**9

    # ============================================
    # 5) Predict
    # ============================================
    if st.button("🔮 Run Prediction"):
        try:
            preds = model.predict(df_input)

            df_result = df_pred.copy()
            df_result[trained_target] = preds  # add predicted column

            st.success("🎉 Prediction Completed Successfully!")
            st.write("### 📊 Prediction Output")
            st.dataframe(df_result)

            # Download
            output_path = "prediction_results.xlsx"
            df_result.to_excel(output_path, index=False)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="📥 Download Predictions",
                    data=f,
                    file_name="prediction_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
