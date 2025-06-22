import pandas as pd

def load_and_preprocess_data(path, currency='Indian Rupee'):
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df.sort_values('Date', inplace=True)
    df = df[['Date', currency]].rename(columns={currency: "Exchange_Rate"})
    df['Exchange_Rate'] = df['Exchange_Rate'].ffill()
    df.set_index('Date', inplace=True)
    return df
