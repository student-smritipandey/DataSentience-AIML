import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.preprocessing import load_and_preprocess_data
from models.arima_model import train_arima_model, forecast_arima
from models.lstm_model import prepare_lstm_data, train_lstm

st.set_page_config(page_title="💱 Currency Forecasting (ARIMA & LSTM)", layout="wide")
st.title("💱 Currency Exchange Rate Forecasting")
st.markdown("This app uses **ARIMA** and **LSTM** models to forecast exchange rates for a selected currency.")

# Currency selection
currency = st.selectbox("Choose Currency", ["Indian Rupee", "US Dollar", "Euro", "British Pound"])

# Load and preprocess data
df = load_and_preprocess_data("data/cleaned_exchange_rates.csv", currency=currency)

# Model selection
model_option = st.radio("Choose Forecasting Model", ["ARIMA", "LSTM"])

# Forecast button
if st.button("📈 Forecast"):
    if model_option == "ARIMA":
        st.subheader("📊 ARIMA Forecast")
        model = train_arima_model(df['Exchange_Rate'])
        prediction = forecast_arima(model, steps=10)
        
        st.write("Next 10 Days Prediction:")
        st.dataframe(prediction.rename("Forecast").reset_index())

        fig, ax = plt.subplots()
        df['Exchange_Rate'].plot(ax=ax, label="Historical", color="blue")
        prediction.plot(ax=ax, label="Forecast", color="red")
        plt.title("ARIMA Forecast")
        plt.legend()
        st.pyplot(fig)

    elif model_option == "LSTM":
        st.subheader("🧠 LSTM Forecast")
        X, y, scaler = prepare_lstm_data(df['Exchange_Rate'])
        model = train_lstm(X, y)

        y_pred_scaled = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_true = scaler.inverse_transform(y.reshape(-1, 1))

        st.write("Last 10 Predictions vs Actual:")
        last_10 = [(float(a[0]), float(p[0])) for a, p in zip(y_true[-10:], y_pred[-10:])]
        st.table(pd.DataFrame(last_10, columns=["Actual", "Predicted"]))

        fig, ax = plt.subplots()
        ax.plot(y_true, label="Actual", color="green")
        ax.plot(y_pred, label="Predicted", color="orange")
        plt.title("LSTM: Actual vs Predicted Exchange Rate")
        plt.legend()
        st.pyplot(fig)
