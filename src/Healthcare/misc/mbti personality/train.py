import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from preprocess import load_data, preprocess

# Load and prepare
texts, labels = load_data("data/mbti.csv")
le = LabelEncoder()
y = le.fit_transform(labels)

# TF-IDF
X, vectorizer = preprocess(texts, fit=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save
os.makedirs("models", exist_ok=True)
joblib.dump((clf, vectorizer, le), "models/model.pkl")
print("✅ Model saved to models/model.pkl")
