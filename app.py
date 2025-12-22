import streamlit as st
import os
import pandas as pd

from etl.extract.reader import read_csv_safe
from etl.profile.profiler import profile_dataframe
from etl.profile.serializer import ensure_json_serializable
from etl.llm.planner import generate_plan
from etl.executor.tool_executor import execute_plan
from etl.validate.validator import validate_transformation


# --------------------------------------------------
# App setup
# --------------------------------------------------

st.set_page_config(page_title="Data Cleaning Agent", layout="centered")
st.title("🧹 Data Cleaning Agent")
st.write("Upload a CSV, review the cleaning plan, and approve execution.")

UPLOAD_DIR = "data/uploads"
OUTPUT_DIR = "data/outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# File upload
# --------------------------------------------------

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    input_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    output_path = os.path.join(OUTPUT_DIR, f"cleaned_{uploaded_file.name}")

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("CSV uploaded successfully.")

    # --------------------------------------------------
    # STEP A: Read + Profile
    # --------------------------------------------------
    df_raw, _ = read_csv_safe(input_path)
    profile = ensure_json_serializable(profile_dataframe(df_raw))

    st.subheader("📊 Dataset Profile (Summary)")
    st.json(profile["dataset"])

    # --------------------------------------------------
    # STEP B: Generate Plan
    # --------------------------------------------------
    with st.spinner("Generating cleaning plan using LLM..."):
        plan = generate_plan(profile)

    st.subheader("🧠 Proposed Cleaning Plan")
    st.json(plan)

    # --------------------------------------------------
    # STEP C: User approval
    # --------------------------------------------------
    approve = st.checkbox("I approve this plan and want to run it")

    if approve:
        st.warning("⚠️ This will modify the dataset")

        if st.button("Run Cleaning"):
            try:
                df_clean = execute_plan(df_raw, plan)
                validate_transformation(df_raw, df_clean)

                df_clean.to_csv(output_path, index=False)

                st.success("Cleaning completed successfully 🎉")

                st.subheader("⬇ Download Cleaned CSV")
                st.download_button(
                    label="Download cleaned CSV",
                    data=df_clean.to_csv(index=False),
                    file_name=os.path.basename(output_path),
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Execution failed: {e}")

    else:
        st.info("Approve the plan to enable execution.")
