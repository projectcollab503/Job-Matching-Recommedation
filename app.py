import joblib
import streamlit as st
import numpy as np

# Load the trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
job_titles = model.classes_

# Streamlit UI
st.title("Udaan – Job Recommender")
st.write("Enter your skills, interests, or qualifications and get top job recommendations!")

# User input
user_input = st.text_input("💬 Describe your background:")

if st.button("🔍 Recommend Jobs"):
    if not user_input.strip():
        st.warning("Please enter some input.")
    else:
        # Transform input using the loaded vectorizer
        transformed = vectorizer.transform([user_input])
        probabilities = model.predict_proba(transformed)[0]

        # Get top 5 job titles
        top_5_indices = np.argsort(probabilities)[::-1][:5]
        top_5_jobs = job_titles[top_5_indices]
        top_5_probs = probabilities[top_5_indices]

        # Display top 5 jobs
        st.subheader("🎯 Top 5 Recommended Jobs:")
        for i, (job, prob) in enumerate(zip(top_5_jobs, top_5_probs), 1):
            st.write(f"{i}. **{job}** – Confidence: {prob:.2%}")
