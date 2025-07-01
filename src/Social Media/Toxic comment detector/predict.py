import joblib

# Load model
clf, vectorizer = joblib.load("models/model.pkl")

# Sample comment
sample = "You are a complete idiot and a disgrace."

# Transform
X = vectorizer.transform([sample])
y_pred = clf.predict(X)[0]

# Labels
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Output
print(f"\n📝 Comment: {sample}")
print("⚠️ Toxicity Prediction:")
for label, value in zip(labels, y_pred):
    print(f"{label}: {'✅' if value else '❌'}")
