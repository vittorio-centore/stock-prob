# src/data_fetch.py

import os
import pandas as pd
import requests
from config import API_KEY

def fetch_data(symbol: str, save_path: str = 'data/stock_data.csv'):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Failed to fetch data for {symbol}: {data}")

    records = []
    for date, metrics in data["Time Series (Daily)"].items():
        records.append({
            "Date": date,
            "Open": float(metrics["1. open"]),
            "High": float(metrics["2. high"]),
            "Low": float(metrics["3. low"]),
            "Close": float(metrics["4. close"]),
        })
    
    df = pd.DataFrame(records)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    fetch_data("AAPL", save_path="data/aapl_stock.csv")
