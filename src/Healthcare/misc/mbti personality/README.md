 MBTI Personality Type Classifier
A simple NLP-based classifier that predicts a user's Myers–Briggs Type Indicator (MBTI) personality type based on their text posts using TF-IDF and Logistic Regression.

📌 Overview
This project uses the MBTI 16 Personality Dataset from Kaggle to build a multi-class text classification model. The model predicts one of the 16 MBTI types (e.g., INFP, ESTJ, etc.) from a user's written posts.

[!ui screenshot](assets/image.png)

🧾 Dataset
Source: Kaggle - MBTI Personality Type Dataset

Columns:

type: One of the 16 personality types.

posts: A long string of text containing various posts written by the user.

🛠️ Project Structure
bash
Copy
Edit
mbti personality/
├── data/
│   └── mbti.csv                   # Dataset file
├── models/
│   └── model.pkl                    # Trained model + vectorizer + label encoder
├── preprocess.py                    # Loads and preprocesses text data
├── train.py                         # Trains and saves the model
├── predict.py                       # Predicts MBTI type from new posts
├── requirements.txt                 # Python package requirements
└── README.md                        # Project documentation
🧪 Model
Vectorizer: TF-IDF (5000 max features, English stop words removed)

Classifier: Logistic Regression (multi-class)