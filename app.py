import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

st.title("Job Recommender AI Assistant 💼")

user_input = st.text_area("Enter your skills or experience")

if st.button("Recommend Job"):
    if user_input:
        transformed = vectorizer.transform([user_input])
        prediction = model.predict(transformed)
        st.success(f"Recommended Job Title: **{prediction[0]}**")
    else:
        st.warning("Please enter your skills to get a recommendation.")
