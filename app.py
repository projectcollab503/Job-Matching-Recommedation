
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
job_titles = model.classes_

# Load job info from CSV
try:
    job_info_df = pd.read_csv("job_info.csv")
except FileNotFoundError:
    st.error("🚫 'job_info.csv' not found. Please add it to your project folder.")
    st.stop()

# Streamlit UI
st.title("Udaan – Job Recommender")
st.markdown("🔎 Enter your skills, interests, or qualifications to get top job matches with full details.")

user_input = st.text_input("💬 Describe your background:")

if st.button("🎯 Recommend Jobs"):
    if not user_input.strip():
        st.warning("Please enter your input.")
    else:
        transformed = vectorizer.transform([user_input])
        probabilities = model.predict_proba(transformed)[0]
        top_5_indices = np.argsort(probabilities)[::-1][:5]
        top_5_jobs = job_titles[top_5_indices]

        # Prepare job detail table
        data = []
        for job in top_5_jobs:
            row = job_info_df[job_info_df["Job Title"] == job]

            if not row.empty:
                info = {
                    "Location": row.iloc[0]["Location"],
                    "Salary Range": row.iloc[0]["Salary Range"],
                    "Work Type": row.iloc[0]["Work Type"],
                    "Contact": row.iloc[0]["Contact"],
                    "Responsibilities": row.iloc[0]["Responsibilities"],
                    "Company Profile": row.iloc[0]["Company Profile"]
                }
            else:
                info = {
                    "Location": "N/A",
                    "Salary Range": "N/A",
                    "Work Type": "N/A",
                    "Contact": "N/A",
                    "Responsibilities": "N/A",
                    "Company Profile": "N/A"
                }

            data.append({
                "Job Title": job,
                **info
            })

        df = pd.DataFrame(data)
        st.subheader("📋 Top 5 Job Recommendations")
        st.dataframe(df, use_container_width=True)

        st.caption(f"Compared against {len(job_titles)} job roles. Update 'job_info.csv' to expand info coverage.")
