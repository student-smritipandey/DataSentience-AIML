from utils.preprocessing import load_and_preprocess_data
from models.arima_model import train_arima_model, forecast_arima
from models.lstm_model import prepare_lstm_data, train_lstm

import numpy as np
import matplotlib.pyplot as plt

def run_arima():
    print("\n--- Running ARIMA ---")
    df = load_and_preprocess_data("data/cleaned_exchange_rates.csv", currency="Indian Rupee")
    model = train_arima_model(df['Exchange_Rate'])
    prediction = forecast_arima(model, steps=10)
    print("ARIMA Prediction:\n", prediction)

def run_lstm():
    print("\n--- Running LSTM ---")
    df = load_and_preprocess_data("data/cleaned_exchange_rates.csv", currency="Indian Rupee")
    X, y, scaler = prepare_lstm_data(df['Exchange_Rate'])
    model = train_lstm(X, y)

    # Predict on training data
    y_pred_scaled = model.predict(X)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_true = scaler.inverse_transform(y.reshape(-1, 1))

    # Print predicted vs actual values
    print("\nLSTM Prediction (last 10 samples):")
    for actual, predicted in zip(y_true[-10:], y_pred[-10:]):
        print(f"Actual: {actual[0]:.2f}, Predicted: {predicted[0]:.2f}")

    # Optional: plot
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("LSTM: Actual vs Predicted Exchange Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_arima()
    run_lstm()
