from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

def train_arima_model(df, order=(5,1,0)):
    model = ARIMA(df, order=order)
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit

def forecast_arima(model_fit, steps=30):
    forecast = model_fit.forecast(steps=steps)
    return forecast
