# scripts/data_loader.py
import os
import yfinance as yf
import pandas as pd

def fetch_eth_data(start="2018-01-01", end="2024-12-31"):
    df = yf.download("ETH-USD", start=start, end=end, auto_adjust=True)
    print("Fetched Data:")
    print(df.head())  # Print top rows
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = fetch_eth_data()

    if df.empty:
        print("⚠️ No data fetched. DataFrame is empty!")
    else:
        os.makedirs("../data", exist_ok=True)
        df.to_csv("../data/eth_data.csv")
        print("✅ Data saved to ../data/eth_data.csv")
