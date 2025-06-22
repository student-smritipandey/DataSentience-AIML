import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

def prepare_lstm_data(df, time_steps=30):
    # Drop missing values
    df = df.dropna()

    # Reshape and scale
    data = df.values.reshape(-1, 1)
    
    # Check for constant values
    if np.all(data == data[0]):
        raise ValueError("Data has no variation — LSTM cannot train on flat values.")

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Debug logs (optional)
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"Sample y values: {y[:5]}")
    print(f"X range: {X.min():.4f} to {X.max():.4f}")

    return X, y, scaler

def train_lstm(X, y):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32)
    return model
