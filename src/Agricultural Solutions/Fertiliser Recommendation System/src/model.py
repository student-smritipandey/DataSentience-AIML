# src/model.py

import joblib
import pandas as pd
from utils import soil_encoder, crop_encoder, decode_fertilizer

# Load trained model
model = joblib.load('../saved_model/fertilizer_model.pkl')

def predict_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous):
    soil_type_encoded = soil_encoder.transform([soil_type])[0]
    crop_type_encoded = crop_encoder.transform([crop_type])[0]

    input_data = pd.DataFrame([[
        temperature, humidity, moisture,
        soil_type_encoded, crop_type_encoded,
        nitrogen, potassium, phosphorous
    ]], columns=[
        'Temparature', 'Humidity', 'Moisture',
        'Soil Type', 'Crop Type',
        'Nitrogen', 'Potassium', 'Phosphorous'
    ])

    pred = model.predict(input_data)[0]
    return decode_fertilizer(pred)
