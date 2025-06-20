# predict.py

import joblib

# Load saved model and label encoder
model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

# Sample input (format: N, P, K, temperature, humidity, ph, rainfall)
sample_input = [[90, 42, 43, 20.8, 82.0, 6.5, 202.9]]

# Predict
prediction = model.predict(sample_input)
predicted_crop = le.inverse_transform(prediction)[0]

print("Predicted Crop:", predicted_crop)
