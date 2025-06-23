# src/predict.py

from model import predict_fertilizer

def main():
    print("\n🌾 Fertilizer Recommendation System 🌿")
    print("Enter the following details:\n")

    try:
        temparature = float(input("Temparature (°C): "))  # spelling must match the model
        humidity = float(input("Humidity (%): "))
        moisture = float(input("Moisture (%): "))

        soil_type = input("Soil Type (e.g., Sandy, Loamy, Black, Red, Clayey): ").strip().capitalize()
        crop_type = input("Crop Type (e.g., Wheat, Cotton, Maize, Paddy): ").strip().capitalize()

        nitrogen = int(input("Nitrogen (N): "))
        potassium = int(input("Potassium (K): "))
        phosphorous = int(input("Phosphorous (P): "))

        print("\n🔍 Predicting best fertilizer...\n")
        result = predict_fertilizer(
            temparature, humidity, moisture,
            soil_type, crop_type,
            nitrogen, potassium, phosphorous
        )

        print(f"✅ Recommended Fertilizer: {result}\n")

    except Exception as e:
        print(f"[ERROR] {e}")
        print("Make sure the inputs are valid and match training data format.")

if __name__ == "__main__":
    main()
