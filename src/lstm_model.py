import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_preprocessing import preprocess_data  # Function to preprocess the data
from data_fetch import fetch_data  # Function to fetch stock data

# Function to create sequences for LSTM input
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Create input sequences of length `sequence_length`
        X.append(data[i:i + sequence_length])
        # The corresponding target value is the next point after the sequence
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

# Function to build the LSTM model
def build_model(input_shape):
    model = Sequential([
        # First LSTM layer, with `return_sequences=True` to output sequences
        LSTM(50, return_sequences=True, input_shape=input_shape),
        # Dropout to randomly deactivate neurons, preventing overfitting
        Dropout(0.2),
        # Second LSTM layer
        LSTM(50),
        # Dropout again for regularization
        Dropout(0.2),
        # Dense layer to produce the final prediction
        Dense(1)
    ])
    # Compile the model using the Adam optimizer and mean squared error loss
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    # Ask the user for the stock symbol
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, GOOG, MSFT): ").upper()

    # Define the file path to save or load the stock data
    save_path = f"data/{stock_symbol}_stock.csv"
    
    # Fetch stock data if it does not already exist
    if not os.path.exists(save_path):
        fetch_data(stock_symbol, save_path=save_path)
    # Load the stock data into a Pandas DataFrame
    df = pd.read_csv(save_path)

    # Preprocess the data: normalize and split into training and testing sets
    train_data, test_data, scaler = preprocess_data(df)

    # Define the sequence length for the LSTM input
    sequence_length = 50
    # Create sequences for training and testing
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    # Define the file path to save or load the trained LSTM model
    model_path = f"models/{stock_symbol}_lstm_model.h5"
    
    # Check if a pre-trained model exists
    if os.path.exists(model_path):
        # Load the pre-trained model
        model = load_model(model_path)
    else:
        # Build a new LSTM model
        model = build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        # Train the model on the training data
        model.fit(
            X_train, y_train,
            epochs=20,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        # Save the trained model to disk
        os.makedirs("models", exist_ok=True)
        model.save(model_path)

    # Make predictions on the test data
    predictions = model.predict(X_test)
    # Denormalize predictions to get actual stock prices
    predictions = scaler.inverse_transform(predictions)
    # Denormalize the true test values
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Visualize the predictions for the last 365 days
    num_days_to_display = 365
    # Extract the last 365 days of predicted prices
    last_365_predictions = predictions[-num_days_to_display:]
    # Extract the last 365 days of actual prices
    last_365_actual = y_test_original[-num_days_to_display:]
    # Extract the corresponding dates for the last 365 days
    last_365_dates = df['Date'][-num_days_to_display:]

    # Plot the actual vs. predicted stock prices for the last 365 days
    plt.figure(figsize=(10, 6))
    plt.plot(last_365_dates, last_365_actual, label="Actual Prices", color="blue")
    plt.plot(last_365_dates, last_365_predictions, label="Predicted Prices", color="red")
    plt.title(f"Stock Price Prediction for the Last {num_days_to_display} Days ({stock_symbol})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    # Format the x-axis to avoid overlapping dates
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()

    # Show the plot
    plt.show()

    # Predict the stock price for the next day using the last sequence in the test set
    last_sequence = X_test[-1].reshape(1, sequence_length, 1)
    next_day_prediction = model.predict(last_sequence)
    # Denormalize the prediction to get the actual price
    next_day_prediction = scaler.inverse_transform(next_day_prediction)
    print(f"Predicted price for the next day ({stock_symbol}): ${next_day_prediction[0][0]:.2f}")
