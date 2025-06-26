import pandas as pd
import joblib
from preprocess import preprocess_data

# Load model and training columns
clf, expected_cols = joblib.load("models/model.pkl")

# 1️⃣ Sample Input - likely NO stroke
sample_input_0 = {
    'id': [101],
    'gender': ['Female'],
    'age': [40],
    'hypertension': [0],
    'heart_disease': [0],
    'ever_married': ['Yes'],
    'work_type': ['Private'],
    'Residence_type': ['Urban'],
    'avg_glucose_level': [95.6],
    'bmi': [24.3],
    'smoking_status': ['never smoked'],
    'stroke': [0]  # dummy label
}

# 2️⃣ Sample Input - likely YES stroke
sample_input_1 = {
    'id': [102],
    'gender': ['Male'],
    'age': [85],
    'hypertension': [1],
    'heart_disease': [1],
    'ever_married': ['Yes'],
    'work_type': ['Self-employed'],
    'Residence_type': ['Rural'],
    'avg_glucose_level': [300.0],
    'bmi': [45.0],
    'smoking_status': ['formerly smoked'],
    'stroke': [0]  # dummy label
}

# 🔍 Prediction Function
def predict_from_input(sample):
    df = pd.DataFrame(sample)
    X, _ = preprocess_data(df)

    # Align features with training columns
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_cols]

    # Predict class and probability
    prediction = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]

    print(f"🔎 Prediction for ID {sample['id'][0]}:")
    print(f"🧠 Stroke Risk: {'Yes' if prediction == 1 else 'No'}")
    print(f"📊 Probability → Stroke=0: {proba[0]:.2f}, Stroke=1: {proba[1]:.2f}")
    print("-----------------------------------------------------------")

# Run predictions
print("\n🔍 Prediction for Sample 1 (likely no stroke):")
predict_from_input(sample_input_0)

print("\n🔍 Prediction for Sample 2 (likely stroke):")
predict_from_input(sample_input_1)
