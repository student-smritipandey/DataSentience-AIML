import joblib
import pandas as pd

def predict_next_price(open_, high, low, volume):
    model = joblib.load("../model/eth_model.pkl")
    input_df = pd.DataFrame([[open_, high, low, volume]], columns=["Open", "High", "Low", "Volume"])
    prediction = model.predict(input_df)[0]
    print(prediction)
    return prediction
