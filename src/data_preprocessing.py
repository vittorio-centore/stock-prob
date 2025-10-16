import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(
    df, 
    feature_cols=["Open","High","Low","Close","Volume"], 
    target_col="Close", 
    split_ratio=0.8,
    recent_weight=True  # Give more weight to recent price patterns
):
    """
    1) Sorts the DataFrame by Date ascending.
    2) Extracts `feature_cols`.
    3) Scales them to [0,1] with MinMaxScaler.
    4) Splits into train_data and test_data by `split_ratio`.
    5) Returns (train_data, test_data, scaler, target_idx, train_dates, test_dates).
    """
    # Ensure sorted by Date
    df = df.sort_values("Date").reset_index(drop=True)
    # Remember the date column for referencing
    dates = df["Date"].copy()

    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    # Extract feature data
    data = df[feature_cols].values  # shape: (N, num_features)

    # Scale using ALL data to ensure consistent scaling across train/test
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    print(f"📊 Scaling based on ALL data ({len(data)} days)")
    close_idx = feature_cols.index('Close') if 'Close' in feature_cols else 0
    print(f"   Price range: ${data[:, close_idx].min():.2f} - ${data[:, close_idx].max():.2f}")

    # Identify target index in feature_cols
    target_idx = feature_cols.index(target_col)

    # Train/test split
    split_index = int(len(data_scaled) * split_ratio)
    train_data = data_scaled[:split_index]
    test_data  = data_scaled[split_index:]

    train_dates = dates[:split_index].values
    test_dates  = dates[split_index:].values

    return train_data, test_data, scaler, target_idx, train_dates, test_dates

def create_sequences(data, target_idx, sequence_length=60):
    """
    data shape: (N, num_features).
    Returns:
      X: (samples, sequence_length, num_features)
      y: (samples, 1) -> the target at the next step
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        # Next day’s target
        y.append(data[i + sequence_length, target_idx])
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    return X, y

if __name__ == "__main__":
    # Quick test
    df_sample = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=100, freq="D"),
        "Open": np.random.rand(100)*100,
        "High": np.random.rand(100)*100,
        "Low": np.random.rand(100)*100,
        "Close": np.random.rand(100)*100,
        "Volume": np.random.randint(100000,500000,size=100)
    })
    train_data, test_data, scaler, tgt_idx, tr_dates, ts_dates = preprocess_data(df_sample)
    X_train, y_train = create_sequences(train_data, tgt_idx, 3)
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
