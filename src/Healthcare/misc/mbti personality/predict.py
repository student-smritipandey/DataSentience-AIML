import joblib

# Load model
clf, vectorizer, le = joblib.load("models/model.pkl")

# Sample posts
samples = [
    "I love staying in and reflecting on ideas. I'm not much into parties or socializing.",
    "I enjoy spontaneous adventures and meeting lots of new people!"
]

for post in samples:
    X = vectorizer.transform([post])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    label = le.inverse_transform([pred])[0]
    print(f"\n📝 Text: {post[:60]}...")
    print(f"🧠 Predicted Type: {label}")
    print(f"📊 Top 3 Confidence: {sorted(zip(le.classes_, proba), key=lambda x: -x[1])[:3]}")
