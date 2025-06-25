# predict.py

import pandas as pd
import joblib
from preprocess import preprocess_data

def get_user_input():
    try:
        data = {}
        data['Gender'] = input("Gender (Male/Female): ")
        data['Married'] = input("Married (Yes/No): ")
        data['Dependents'] = input("Dependents (0/1/2/3+): ")
        data['Education'] = input("Education (Graduate/Not Graduate): ")
        data['Self_Employed'] = input("Self_Employed (Yes/No): ")
        data['ApplicantIncome'] = float(input("Applicant Income: "))
        data['CoapplicantIncome'] = float(input("Coapplicant Income: "))
        data['LoanAmount'] = float(input("Loan Amount (in thousands): "))
        data['Loan_Amount_Term'] = float(input("Loan Amount Term (in days): "))
        data['Credit_History'] = float(input("Credit History (0 or 1): "))
        data['Property_Area'] = input("Property Area (Urban/Semiurban/Rural): ")
        return pd.DataFrame([data])
    except Exception as e:
        print("❌ Error taking input:", e)
        return None

def predict_loan():
    try:
        print("\n📥 Enter Applicant Info Below:")
        input_df = get_user_input()

        if input_df is None:
            return

        print("\n📦 Preprocessing Input...")
        input_df_processed = preprocess_data(input_df, for_training=False)

        print("📤 Loading Model...")
        model = joblib.load("models/model.pkl")

        print("🔍 Making Prediction...")
        prediction = model.predict(input_df_processed)[0]

        print("\n✅ Loan Status Prediction:", "Approved ✅" if prediction == 1 else "Rejected ❌")

    except Exception as e:
        print("❌ Prediction Error:", e)

if __name__ == "__main__":
    predict_loan()
