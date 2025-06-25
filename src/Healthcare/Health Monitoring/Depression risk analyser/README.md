 Depression Risk Analyzer
A machine learning–based system that detects signs of moderate to severe depression based on user-generated text inputs — such as journal entries, social media posts, or questionnaire responses.

The system uses Natural Language Processing (NLP) with TF-IDF feature extraction and a Logistic Regression classifier to distinguish between suicidal and non-suicidal text.

Run this program using-  streamlit run main.py

These are the images for the working ui-
[!ui screenshot](assets/1.jpeg)
[!ui screenshot](assets/2.jpeg)

📌 Project Goals
🧠 Predict whether a user is at risk of moderate to severe depression

📝 Accept natural text input (journal-style, Reddit-like, or PHQ‑9 style)

🧪 Support early detection of suicidal intent using AI

🧱 Serve as a foundation for future mental health screening tools

🧠 Model Details
Feature	Description
Model Type	Logistic Regression
Feature Extraction	TF-IDF (Top 10,000 features)
Dataset Size	15,000+ cleaned Reddit posts
Labels	suicide, non-suicide
Evaluation Metrics	Precision, Recall, F1-score

🚀 Features
📖 Text-based input to detect mental health risks

✅ Clean, minimal Streamlit interface

💡 Intelligent predictions in real-time

💾 Model and vectorizer stored with joblib

🔐 Local inference — no sensitive data leaves the machine