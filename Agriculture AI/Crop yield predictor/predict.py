# Predict yield for a new input
import joblib
def predict_yield(area, item, rainfall, pesticide, temperature):
    model = joblib.load('yield_predictor_model.pkl')
    le_area = joblib.load('area_encoder.pkl')
    le_item = joblib.load('item_encoder.pkl')

    area_encoded = le_area.transform([area])[0]
    item_encoded = le_item.transform([item])[0]

    input_data = [[area_encoded, item_encoded, rainfall, pesticide, temperature]]
    prediction = model.predict(input_data)
    return prediction[0]

# Example
yield_pred = predict_yield("Albania", "Potatoes", 1485, 121, 16.06)
print(f"📦 Predicted Yield: {yield_pred:.2f} hg/ha")
