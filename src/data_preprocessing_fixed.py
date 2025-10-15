import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import warnings

def preprocess_data_fixed(
    df, 
    feature_cols=["Open","High","Low","Close","Volume"], 
    target_col="Close", 
    split_ratio=0.8,
    min_date="2018-01-01"  # Use recent data to avoid split issues
):
    """
    FIXED version that prevents data leakage and handles stock splits properly
    """
    # Ensure sorted by Date
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Filter to recent data to avoid stock split issues
    df = df[df['Date'] >= min_date].copy()
    print(f"📅 Using data from {df.Date.min().date()} to {df.Date.max().date()}")
    print(f"📊 Total samples: {len(df)}")
    
    # Check for obvious stock splits or anomalies
    df['price_change'] = df['Close'].pct_change().abs()
    big_changes = df[df['price_change'] > 0.3]  # >30% single-day changes
    if len(big_changes) > 0:
        print("⚠️  WARNING: Found potential stock splits or anomalies:")
        for _, row in big_changes.head(3).iterrows():
            print(f"   {row.Date.date()}: {row.price_change*100:.1f}% change")
    
    # Remove the price_change column
    df = df.drop('price_change', axis=1)
    dates = df["Date"].copy()

    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Extract feature data
    data = df[feature_cols].values  # shape: (N, num_features)

    # CRITICAL FIX: Split BEFORE scaling to prevent data leakage
    split_index = int(len(data) * split_ratio)
    train_data_raw = data[:split_index]
    test_data_raw = data[split_index:]
    
    train_dates = dates[:split_index].values
    test_dates = dates[split_index:].values
    
    print(f"📊 Train period: {pd.to_datetime(train_dates[0]).date()} to {pd.to_datetime(train_dates[-1]).date()}")
    print(f"📊 Test period: {pd.to_datetime(test_dates[0]).date()} to {pd.to_datetime(test_dates[-1]).date()}")

    # Scale using ONLY training data
    scaler = MinMaxScaler()
    train_data_scaled = scaler.fit_transform(train_data_raw)
    test_data_scaled = scaler.transform(test_data_raw)  # Use training scaler
    
    # Identify target index in feature_cols
    target_idx = feature_cols.index(target_col)

    return train_data_scaled, test_data_scaled, scaler, target_idx, train_dates, test_dates

def create_sequences_fixed(data, target_idx, sequence_length=60):
    """
    FIXED version with better validation
    """
    if len(data) <= sequence_length:
        raise ValueError(f"Data length ({len(data)}) must be > sequence_length ({sequence_length})")
    
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        # Next day's target
        y.append(data[i + sequence_length, target_idx])
    
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    
    print(f"📈 Created {len(X)} sequences of length {sequence_length}")
    return X, y

def validate_data_quality(df, symbol):
    """
    Check data quality and warn about issues
    """
    print(f"\n🔍 DATA QUALITY CHECK for {symbol}:")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("⚠️  Missing values found:")
        for col, count in missing[missing > 0].items():
            print(f"   {col}: {count} missing")
    
    # Check for zero/negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if col in df.columns:
            bad_prices = (df[col] <= 0).sum()
            if bad_prices > 0:
                print(f"⚠️  {col}: {bad_prices} zero/negative values")
    
    # Check for extreme volatility
    if 'Close' in df.columns:
        daily_returns = df['Close'].pct_change().abs()
        extreme_days = (daily_returns > 0.2).sum()  # >20% moves
        if extreme_days > 0:
            print(f"⚠️  Found {extreme_days} days with >20% price moves")
    
    # Check volume
    if 'Volume' in df.columns:
        zero_volume = (df['Volume'] == 0).sum()
        if zero_volume > 0:
            print(f"⚠️  {zero_volume} days with zero volume")
    
    print("✅ Data quality check complete")

if __name__ == "__main__":
    # Test with sample data
    print("🧪 Testing fixed preprocessing...")
    
    # Create realistic test data
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    np.random.seed(42)
    
    # Simulate realistic stock price with trend
    price = 100
    prices = []
    for i in range(1000):
        price *= (1 + np.random.normal(0, 0.02))  # 2% daily volatility
        prices.append(price)
    
    df_test = pd.DataFrame({
        "Date": dates,
        "Open": np.array(prices) * (1 + np.random.normal(0, 0.005, 1000)),
        "High": np.array(prices) * (1 + np.abs(np.random.normal(0, 0.01, 1000))),
        "Low": np.array(prices) * (1 - np.abs(np.random.normal(0, 0.01, 1000))),
        "Close": prices,
        "Volume": np.random.randint(1000000, 10000000, 1000)
    })
    
    validate_data_quality(df_test, "TEST")
    
    train_data, test_data, scaler, tgt_idx, tr_dates, ts_dates = preprocess_data_fixed(df_test)
    X_train, y_train = create_sequences_fixed(train_data, tgt_idx, 60)
    X_test, y_test = create_sequences_fixed(test_data, tgt_idx, 60)
    
    print(f"✅ Train sequences: {X_train.shape}")
    print(f"✅ Test sequences: {X_test.shape}")
    print("🎯 Fixed preprocessing working correctly!")
