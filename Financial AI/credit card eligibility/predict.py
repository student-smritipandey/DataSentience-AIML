import pandas as pd
import pickle

def predict_eligibility(user_input: dict):
    # Load trained model
    model = pickle.load(open("model/credit_model.pkl", "rb"))

    # Convert input to DataFrame
    df = pd.DataFrame([user_input])

    # One-hot encode categorical fields
    df = pd.get_dummies(df)

    # Align all columns
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    df = df[model.feature_names_in_]  # reorder correctly

    # ✅ Ensure it's a DataFrame with valid column names
    print("\n🧪 Final columns passed to model:", df.columns.tolist())

    # Predict
    prediction = model.predict(df)[0]
    return "✅ Eligible" if prediction == 1 else "❌ Not Eligible"


# Sample
if __name__ == "__main__":
    user_input = {
        "Gender": 1,
        "Own_car": 1,
        "Own_property": 0,
        "Work_phone": 0,
        "Phone": 0,
        "Email": 0,
        "Unemployed": 0,
        "Num_children": 0,
        "Num_family": 2,
        "Account_length": 35,
        "Total_income": 112500.0,
        "Age": 32.296351,
        "Years_employed": 5.842694,
        "Income_type": "Working",
        "Education_type": "Higher education",
        "Family_status": "Married",
        "Housing_type": "House / apartment",
        "Occupation_type": "Laborers"
    }

    result = predict_eligibility(user_input)
    print("Prediction:", result)

