import os
import requests
import pandas as pd
from config import API_KEY

def fetch_data(symbol, save_path):
    """
    Fetches daily (non-adjusted) historical stock data for `symbol` from Alpha Vantage
    via TIME_SERIES_DAILY, and saves to CSV in `save_path`.
    """
    print(f"Fetching data for {symbol} from Alpha Vantage (TIME_SERIES_DAILY)...")

    function = "TIME_SERIES_DAILY"
    url = (
        f"https://www.alphavantage.co/query"
        f"?function={function}"
        f"&symbol={symbol}"
        f"&outputsize=full"
        f"&apikey={API_KEY}"
    )
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" not in data:
        raise ValueError(f"Failed to fetch data for {symbol}: {data}")

    records = []
    for date_str, metrics in data["Time Series (Daily)"].items():
        records.append({
            "Date": pd.to_datetime(date_str),
            "Open": float(metrics["1. open"]),
            "High": float(metrics["2. high"]),
            "Low": float(metrics["3. low"]),
            "Close": float(metrics["4. close"]),
            "Volume": float(metrics["5. volume"])
        })
    
    df = pd.DataFrame(records).sort_values("Date")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Data for {symbol} saved to {save_path} (rows={len(df)})")

if __name__ == "__main__":
    # Example usage:
    fetch_data("AAPL", "data/AAPL_daily.csv")
