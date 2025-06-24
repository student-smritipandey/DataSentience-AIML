# train_model.py

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load dataset
print("📥 Loading dataset...")
df = pd.read_csv("data/cleaned_suicide_dataset.csv").dropna()
print(f"✅ Loaded {len(df)} entries.")

# TF-IDF vectorization
print("🔠 Vectorizing text using TF-IDF (max 10,000 features)...")
tfidf = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=10000)
X = tfidf.fit_transform(tqdm(df['text'], desc="🚀 Vectorizing"))
y = df['label']

# Split dataset
print("🧪 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("🧠 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
print("\n📊 Model performance:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
print("\n💾 Saving model and vectorizer to assets/")
joblib.dump(model, "assets/depression_model.pkl")
joblib.dump(tfidf, "assets/tfidf_vectorizer.pkl")
print("✅ Files saved successfully.")
