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

    # Predict
    prediction = model.predict(df)[0]
    return "✅ Eligible" if prediction == 1 else "❌ Not Eligible"

test_profiles = [
    {
        "Gender": 1,
        "Own_car": 1,
        "Own_property": 0,
        "Work_phone": 1,
        "Phone": 1,
        "Email": 1,
        "Unemployed": 0,
        "Num_children": 0,
        "Num_family": 2,
        "Account_length": 36,
        "Total_income": 500000,
        "Age": 30,
        "Years_employed": 6,
        "Income_type": "Working",
        "Education_type": "Higher education",
        "Family_status": "Married",
        "Housing_type": "House / apartment",
        "Occupation_type": "Accountants"
    },
    {
        "Gender": 0,
        "Own_car": 0,
        "Own_property": 1,
        "Work_phone": 0,
        "Phone": 1,
        "Email": 0,
        "Unemployed": 0,
        "Num_children": 0,
        "Num_family": 2,
        "Account_length": 31,
        "Total_income": 157500.0,
        "Age": 27.463945187101718,
        "Years_employed": 4.021985393266117,
        "Income_type": "Working",
        "Education_type": "Secondary / secondary special",
        "Family_status": "Married",
        "Housing_type": "House / apartment",
        "Occupation_type": "Laborers"
    }

   

]
if __name__ == "__main__":
    for i, profile in enumerate(test_profiles, 1):
        result = predict_eligibility(profile)
        print(f"Prediction {i}: {result}")
