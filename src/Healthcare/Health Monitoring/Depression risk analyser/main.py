# main.py

import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
@st.cache_resource
def load_model_and_vectorizer():
    model = joblib.load("assets/depression_model.pkl")
    vectorizer = joblib.load("assets/tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

# App UI
st.set_page_config(page_title="Depression Risk Analyzer", layout="centered")
st.title("🧠 Depression Risk Analyzer")
st.write("Analyze written text to detect possible signs of **suicidal intent** using NLP and machine learning.")

user_input = st.text_area("💬 Enter your message below:", height=250)

if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]

        if prediction.lower() == "suicide":
            st.error("🔴 High Risk Detected: Suicidal intent present.")
        else:
            st.success("🟢 Low Risk: No suicidal intent detected.")

st.markdown("---")
st.markdown("📌 *This tool is for educational purposes only and not a substitute for professional help.*")
