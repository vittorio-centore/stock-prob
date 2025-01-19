# Stock Probability Predictor 📈

A Python-based project that uses **LSTM (Long Short-Term Memory)** neural networks to predict stock prices. The app fetches historical stock data, preprocesses it for time-series forecasting, trains an LSTM model, and provides insights by visualizing predictions and future forecasts.

---

## Features
1. **Data Fetching**: 
   - Automatically retrieves stock data using the **Alpha Vantage API**.
   - Stores data in a structured CSV format.

2. **Data Preprocessing**: 
   - Normalizes data using **MinMaxScaler** for better LSTM performance.
   - Transforms data into sequential inputs for time-series forecasting.

3. **LSTM Model**:
   - Two LSTM layers with dropout for enhanced prediction accuracy.
   - Trains on historical stock data and validates on unseen data.

4. **Visualization**:
   - Plots actual vs. predicted stock prices.
   - Displays a **4-day forecast** as a single point on the graph for clarity.

---

## 🛠️Technologies Used
- **Programming Language**: Python
- **Libraries and Frameworks**:
  - **TensorFlow/Keras**: For building and training the LSTM model.
  - **Pandas**: For data preprocessing and manipulation.
  - **Matplotlib**: For data visualization.
  - **NumPy**: For numerical operations.
  - **scikit-learn**: For scaling data with `MinMaxScaler`.
- **API**: [Alpha Vantage](https://www.alphavantage.co/) for fetching stock market data.

---

## Project Structure
```plaintext
stock-prob/
├── data/                    # Folder to store CSV stock data
├── models/                  # Folder to save trained models
├── data_fetch.py            # Fetches stock data from Alpha Vantage
├── data_preprocessing.py    # Prepares data for LSTM model
├── main.py                  # Main script to train, test, and predict
├── requirements.txt         # Required libraries and dependencies
└── README.md                # Project documentation
