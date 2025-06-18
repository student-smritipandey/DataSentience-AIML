import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not present
nltk.download('stopwords')

# Load dataset
print("📥 Loading dataset...")
df = pd.read_csv("data/cleaned_suicide_dataset.csv")

# Drop any missing values just in case
df = df.dropna()

print(f"✅ Loaded {len(df)} entries.")

# TF-IDF Vectorizer with limit for speed
print("🔠 Vectorizing text using TF-IDF (max 10,000 features)...")
tfidf = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=10000)

X = tfidf.fit_transform(tqdm(df['text'], desc="🚀 Vectorizing"))
y = df['label']

# Train-test split
print("🧪 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("🧠 Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
print("📊 Model performance:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# --- Test with Custom Input ---
print("\n🧪 Test on Custom Input")

# Example texts
custom_texts = [
    "I feel so alone and hopeless, nothing matters anymore.",  # likely 'suicide'
    "I had a great day at college and feel motivated."          # likely 'non-suicide'
]

# Transform the custom inputs using the trained TF-IDF vectorizer
X_custom = tfidf.transform(custom_texts)

# Predict
custom_preds = model.predict(X_custom)

# Show results
for text, label in zip(custom_texts, custom_preds):
    print(f"\nText: {text}\n→ Predicted Label: {label}")
