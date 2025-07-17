
import joblib
import streamlit as st
import numpy as np
import pandas as pd

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
job_titles = model.classes_

# Job details for common predicted titles
job_info = {
    "Data Analyst": {
        "Location": "Mumbai, India",
        "Salary Range": "₹6L–₹12L",
        "Work Type": "Hybrid",
        "Contact": "hr@dataanalystcorp.com",
        "Responsibilities": "Analyze business data and create dashboards.",
        "Company Profile": "Analytics and reporting company."
    },
    "Data Engineer": {
        "Location": "Bangalore, India",
        "Salary Range": "₹8L–₹16L",
        "Work Type": "Full-time",
        "Contact": "jobs@dataengineerhub.com",
        "Responsibilities": "Build and maintain data pipelines.",
        "Company Profile": "Cloud and big data firm."
    },
    "Business Analyst": {
        "Location": "Gurgaon, India",
        "Salary Range": "₹7L–₹14L",
        "Work Type": "Hybrid",
        "Contact": "careers@bizanalytica.in",
        "Responsibilities": "Bridge business needs with technical teams.",
        "Company Profile": "Top-tier consulting firm."
    },
    "Research Analyst": {
        "Location": "Delhi, India",
        "Salary Range": "₹5L–₹9L",
        "Work Type": "Full-time",
        "Contact": "research@analytixlabs.in",
        "Responsibilities": "Perform market and academic research.",
        "Company Profile": "Data-driven research organization."
    },
    "Data Scientist": {
        "Location": "Bangalore, India",
        "Salary Range": "₹10L–₹20L",
        "Work Type": "Full-time",
        "Contact": "hr@datasci.com",
        "Responsibilities": "Analyze data, build ML models, report insights.",
        "Company Profile": "Top-tier AI & analytics company."
    }
}

# Streamlit UI
st.title("Udaan – Job Recommender")
st.markdown("🔎 Enter your skills, interests, or qualifications to get top job matches with full details.")

user_input = st.text_input("💬 Describe your background:")

if st.button("🎯 Recommend Jobs"):
    if not user_input.strip():
        st.warning("Please enter some input.")
    else:
        # Transform input and predict
        transformed = vectorizer.transform([user_input])
        probabilities = model.predict_proba(transformed)[0]
        top_5_indices = np.argsort(probabilities)[::-1][:5]
        top_5_jobs = job_titles[top_5_indices]

        # Prepare job details
        data = []
        for job in top_5_jobs:
            info = job_info.get(job, {
                "Location": "N/A",
                "Salary Range": "N/A",
                "Work Type": "N/A",
                "Contact": "N/A",
                "Responsibilities": "N/A",
                "Company Profile": "N/A"
            })
            data.append({
                "Job Title": job,
                "Location": info["Location"],
                "Salary Range": info["Salary Range"],
                "Work Type": info["Work Type"],
                "Contact": info["Contact"],
                "Responsibilities": info["Responsibilities"],
                "Company Profile": info["Company Profile"]
            })

        # Display results
        df = pd.DataFrame(data)
        st.subheader("📋 Top 5 Job Recommendations")
        st.dataframe(df, use_container_width=True)

        st.caption(f"Compared against {len(job_titles)} possible job roles.")
