from utils.preprocessing import load_and_preprocess_data
from models.arima_model import train_arima_model, forecast_arima
from models.lstm_model import prepare_lstm_data, train_lstm

def run_arima():
    df = load_and_preprocess_data("data/cleaned_exchange_rates.csv", currency="Indian Rupee")
    model = train_arima_model(df['Exchange_Rate'])
    prediction = forecast_arima(model, steps=10)
    print("ARIMA Prediction:\n", prediction)

def run_lstm():
    df = load_and_preprocess_data("data/cleaned_exchange_rates.csv", currency="Indian Rupee")
    X, y, scaler = prepare_lstm_data(df['Exchange_Rate'])
    model = train_lstm(X, y)

if __name__ == "__main__":
    run_arima()
    run_lstm()
