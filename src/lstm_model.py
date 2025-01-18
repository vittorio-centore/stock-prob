import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout

from data_fetch import fetch_data
from data_preprocessing import preprocess_data, create_sequences

def build_model(input_shape):
    """
    Builds a simple 2-layer LSTM with dropout, compiled with MSE loss.
    input_shape = (sequence_length, num_features).
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

if __name__ == "__main__":
    # 1) Symbol
    symbol = input("Enter the stock symbol (e.g. AAPL, GOOG, MSFT): ").upper()

    # 2) Fetch data if not present
    csv_path = f"data/{symbol}_daily.csv"
    if not os.path.exists(csv_path):
        fetch_data(symbol, csv_path)

    # 3) Load CSV
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # 4) Preprocess (just raw daily data)
    feature_cols = ["Open","High","Low","Close","Volume"]
    target_col   = "Close"
    train_data, test_data, scaler, target_idx, train_dates, test_dates = preprocess_data(
        df, feature_cols=feature_cols, target_col=target_col, split_ratio=0.8
    )

    # 5) Create sequences
    sequence_length = 60
    X_train, y_train = create_sequences(train_data, target_idx, sequence_length)
    X_test,  y_test  = create_sequences(test_data,  target_idx, sequence_length)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test:", X_test.shape, "y_test:", y_test.shape)

    # 6) Build or load model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{symbol}_lstm_daily.h5"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = load_model(model_path)
    else:
        print("No saved model found. Building and training a new one...")
        model = build_model((X_train.shape[1], X_train.shape[2]))
        model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        model.save(model_path)
        print(f"Model saved at {model_path}")

    # 7) Predict on the test set
    predictions = model.predict(X_test)  # shape (samples,1)

    # 8) Invert predictions properly
    inv_preds = []
    for i in range(len(predictions)):
        # i-th sample => day (i + sequence_length) in test_data
        scaled_row = test_data[i + sequence_length].copy()
        scaled_row[target_idx] = predictions[i, 0]
        unscaled = scaler.inverse_transform([scaled_row])[0]
        inv_preds.append(unscaled[target_idx])
    inv_preds = np.array(inv_preds)

    # Also invert y_test
    inv_y = []
    for i in range(len(y_test)):
        row_copy = test_data[i + sequence_length].copy()
        row_copy[target_idx] = y_test[i, 0]
        unscaled = scaler.inverse_transform([row_copy])[0]
        inv_y.append(unscaled[target_idx])
    inv_y = np.array(inv_y)

    # 9) Align test dates
    test_dates_adj = test_dates[sequence_length:]

    # 10) Plot the last year
    last_year = pd.Timestamp.now() - pd.Timedelta(days=365)
    idx_start = 0
    for i, d in enumerate(test_dates_adj):
        if pd.to_datetime(d) >= last_year:
            idx_start = i
            break

    show_dates = test_dates_adj[idx_start:]
    show_preds = inv_preds[idx_start:]
    show_true  = inv_y[idx_start:]

    plt.figure(figsize=(10,6))
    plt.plot(show_dates, show_true, label="Actual Close", color="blue")
    plt.plot(show_dates, show_preds, label="Predicted Close", color="red")

    # ------------------------------------------------------------------
    # 11) Predict ONLY the 4th day in the future (not days 1,2,3,4).
    #     We'll do the rolling approach but only plot/print the final day
    # ------------------------------------------------------------------
    future_days = 4
    last_seq = X_test[-1:].copy()  # shape (1, 60, num_features)
    # We'll reuse the final scaled row of test_data for baseline O/H/L/Volume
    last_real_scaled_row = test_data[-1].copy()
    
    # Make repeated predictions but only keep the 4th
    final_future_scaled_value = None
    for day_idx in range(1, future_days+1):
        next_pred = model.predict(last_seq)  # shape (1,1)
        # Overwrite the target in a new row
        new_day = last_real_scaled_row.copy()
        new_day[target_idx] = next_pred[0][0]
        # Slide window
        last_seq = np.concatenate(
            [last_seq[:,1:,:], new_day.reshape(1,1,-1)],
            axis=1
        )
        last_real_scaled_row = new_day

        if day_idx == future_days:
            # This is the 4th day
            final_future_scaled_value = next_pred[0][0]

    # Invert just that final future day
    row_copy = last_real_scaled_row.copy()
    row_copy[target_idx] = final_future_scaled_value
    final_unscaled = scaler.inverse_transform([row_copy])[0]
    future_close = final_unscaled[target_idx]

    # Let's define that 4th future day date as test_dates_adj[-1] + 4 days
    last_test_date = pd.to_datetime(test_dates_adj[-1])
    future_day_date = last_test_date + pd.Timedelta(days=future_days)

    # Plot only the single day as a green dot
    plt.scatter([future_day_date], [future_close], color="green", label="4th Day Prediction")
    plt.text(future_day_date, future_close, f"{future_close:.2f}", color="green", fontsize=9, ha="left")

    plt.title(f"Daily Stock Price Prediction for {symbol}\n(Last 1 Year + Only the 4th Future Week)")
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    plt.show()

    # Print only that final 4th day prediction
    print(f"The 4th future day prediction: {future_day_date.date()} => ${future_close:.2f}")
