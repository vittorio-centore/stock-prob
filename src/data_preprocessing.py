from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_and_explore_data(file_path: str):
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Display the first few rows
    print("First 5 rows of the dataset:")
    print(df.head())
    
    # Display basic statistics
    print("\nDataset statistics:")
    print(df.describe())

    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
    # Plot stock prices
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['Close'], label='Close Price')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Stock Price Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    load_and_explore_data("data/aapl_stock.csv")

def preprocess_data(df, feature='Close', split_ratio=0.8):
    # Sort by date (if not already sorted)
    df = df.sort_values('Date')
    
    # Extract the feature of interest
    data = df[feature].values.reshape(-1, 1)

    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Split into training and test sets
    split_index = int(len(data_scaled) * split_ratio)
    train_data = data_scaled[:split_index]
    test_data = data_scaled[split_index:]

    return train_data, test_data, scaler

if __name__ == "__main__":
    df = pd.read_csv("data/aapl_stock.csv")
    train_data, test_data, scaler = preprocess_data(df)
    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")
