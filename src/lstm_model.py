import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from data_preprocessing import preprocess_data
import matplotlib.pyplot as plt
import pandas as pd

def create_sequences(data, sequence_length):
    """
    Converts the data into sequences for LSTM input.

    Args:
        data (array): Scaled data array (training or test set).
        sequence_length (int): Number of time steps to include in each sequence.

    Returns:
        tuple: Input sequences (X) and corresponding outputs (y).
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i+sequence_length])  # Sequence of `sequence_length` steps
        y.append(data[i+sequence_length])   # Next value to predict
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Step 1: Load and preprocess data
    df = pd.read_csv("data/aapl_stock.csv")
    train_data, test_data, scaler = preprocess_data(df)

    # Step 2: Prepare sequences for LSTM
    sequence_length = 50
    X_train, y_train = create_sequences(train_data, sequence_length)
    X_test, y_test = create_sequences(test_data, sequence_length)

    # Step 3: Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)  # Output layer for predicting the next value
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Step 4: Train the model
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Step 5: Evaluate and visualize predictions
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)  # Reverse normalization
    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_original, label='Actual Prices', color='blue')
    plt.plot(predictions, label='Predicted Prices', color='red')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
