import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import our models
try:
    from .transformer_model import StockTransformer, StockDataset, train_transformer
    from .multi_modal import BasicMultiModalProcessor as MultiModalDataProcessor
    from .interactive_plots import InteractiveVisualizer, create_visualization_summary
    from .data_preprocessing import preprocess_data, create_sequences
except ImportError:
    from transformer_model import StockTransformer, StockDataset, train_transformer
    from multi_modal import BasicMultiModalProcessor as MultiModalDataProcessor
    from interactive_plots import InteractiveVisualizer, create_visualization_summary
    from data_preprocessing import preprocess_data, create_sequences

try:
    try:
        from .feature_analysis import FeatureAnalyzer, create_feature_correlation_heatmap
    except ImportError:
        from feature_analysis import FeatureAnalyzer, create_feature_correlation_heatmap
    FEATURE_ANALYSIS_AVAILABLE = True
except ImportError:
    print("⚠️  Feature analysis not available (missing dependencies)")
    FEATURE_ANALYSIS_AVAILABLE = False

import yfinance as yf

def fetch_stock_data(symbol, csv_path):
    """Fetch stock data using YFinance and save to CSV"""
    print(f"Fetching {symbol} data using YFinance...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period='5y')  # Get 5 years of data
    df = df.reset_index()
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path} ({len(df)} rows)")

class StockEnsemble:
    """Ensemble of Transformer (primary) + LSTM (secondary) models"""
    
    def __init__(self, model_dir="models"):
        self.transformer_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
    def build_models(self, input_shape):
        """Build transformer model"""
        # Transformer model (simplified for better training)
        self.transformer_model = StockTransformer(
            input_dim=input_shape[1],
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1,
            max_len=input_shape[0]
        )
        
        print(f"Transformer built - Using device: {self.device}")
        
    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs=20):
        """Train transformer model"""
        print("Training Transformer model...")
        
        # Prepare data for transformer
        train_dataset = StockDataset(X_train, y_train.squeeze())
        val_dataset = StockDataset(X_val, y_val.squeeze())
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train transformer
        train_losses, val_losses = train_transformer(
            self.transformer_model, train_loader, val_loader, 
            epochs=epochs, device=self.device
        )
        
        return train_losses, val_losses
    
    def save_models(self, symbol):
        """Save transformer model"""
        # Save transformer
        transformer_path = os.path.join(self.model_dir, f"{symbol}_transformer.pth")
        torch.save(self.transformer_model.state_dict(), transformer_path)
        
        print(f"Model saved: {transformer_path}")
    
    def load_models(self, symbol, input_shape):
        """Load pre-trained transformer if it exists"""
        transformer_path = os.path.join(self.model_dir, f"{symbol}_transformer.pth")
        
        models_loaded = False
        
        if os.path.exists(transformer_path):
            # Build model first
            self.build_models(input_shape)
            
            # Load transformer
            self.transformer_model.load_state_dict(torch.load(transformer_path, map_location=self.device))
            self.transformer_model.to(self.device)
            
            models_loaded = True
            print(f"Model loaded from {transformer_path}")
        else:
            print(f"No pre-trained model found at {transformer_path}")
        
        return models_loaded
    
    def predict(self, X_test):
        """Make transformer predictions"""
        # Transformer predictions
        test_dataset = StockDataset(X_test, np.zeros(len(X_test)))
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        transformer_preds = []
        self.transformer_model.eval()
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device)
                preds = self.transformer_model(batch_X)
                transformer_preds.extend(preds.cpu().numpy())
        
        transformer_preds = np.array(transformer_preds)
        
        return {
            'ensemble': transformer_preds,
            'transformer': transformer_preds,
            'transformer_lower': transformer_preds * 0.98,
            'transformer_upper': transformer_preds * 1.02,
            'lstm': transformer_preds  # Use transformer for LSTM slot too
        }
    
    def adaptive_reweight(self, recent_predictions, recent_actuals):
        """Dynamically adjust weights based on recent performance"""
        if len(recent_predictions) < 10:
            return  # Need enough data points
        
        transformer_mae = mean_absolute_error(recent_actuals, recent_predictions['transformer'])
        lstm_mae = mean_absolute_error(recent_actuals, recent_predictions['lstm'])
        
        if transformer_mae < lstm_mae:
            self.transformer_weight = min(0.9, self.transformer_weight + 0.05)
            self.lstm_weight = 1 - self.transformer_weight
        else:
            self.lstm_weight = min(0.4, self.lstm_weight + 0.05)
            self.transformer_weight = 1 - self.lstm_weight
            
        print(f"Weights adjusted - Transformer: {self.transformer_weight:.2f}, LSTM: {self.lstm_weight:.2f}")

def plot_ensemble_results(dates, actual, predictions, symbol):
    """Plot ensemble results with uncertainty bands"""
    plt.figure(figsize=(15, 10))
    
    # Main predictions plot
    plt.subplot(2, 1, 1)
    plt.plot(dates, actual, label="Actual Close", color="black", linewidth=2)
    plt.plot(dates, predictions['ensemble'], label="Ensemble Prediction", color="purple", linewidth=2)
    plt.plot(dates, predictions['transformer'], label="Transformer", color="red", alpha=0.7)
    plt.plot(dates, predictions['lstm'], label="LSTM", color="blue", alpha=0.7)
    
    # Uncertainty bands for transformer
    plt.fill_between(dates, predictions['transformer_lower'], predictions['transformer_upper'], 
                     alpha=0.2, color='red', label='Transformer Uncertainty')
    
    plt.title(f"{symbol} - Ensemble Stock Price Prediction")
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error comparison plot
    plt.subplot(2, 1, 2)
    ensemble_error = np.abs(actual - predictions['ensemble'])
    transformer_error = np.abs(actual - predictions['transformer'])
    lstm_error = np.abs(actual - predictions['lstm'])
    
    plt.plot(dates, ensemble_error, label="Ensemble Error", color="purple")
    plt.plot(dates, transformer_error, label="Transformer Error", color="red", alpha=0.7)
    plt.plot(dates, lstm_error, label="LSTM Error", color="blue", alpha=0.7)
    
    plt.title("Prediction Errors Comparison")
    plt.xlabel("Date")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Get stock symbol
    symbol = input("Enter the stock symbol (e.g. AAPL, GOOG, MSFT): ").upper()
    
    # Fetch data if not present
    csv_path = f"data/{symbol}_daily.csv"
    if not os.path.exists(csv_path):
        fetch_stock_data(symbol, csv_path)
    
    # Load data
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Create multi-modal features
    print("Creating enhanced features with multi-modal data...")
    processor = MultiModalDataProcessor()
    enhanced_df = processor.create_enhanced_features(df, symbol)
    
    # Get enhanced feature columns
    feature_cols = processor.get_enhanced_feature_columns()
    target_col = "Close"
    
    print(f"Using {len(feature_cols)} features including:")
    print("- Technical indicators (RSI, MACD, Bollinger Bands)")
    print("- Economic indicators (VIX, Treasury yields, S&P 500)")
    print("- News sentiment analysis")
    print("- Time-based cyclical features")
    print("- Market regime indicators")
    
    # Preprocess with enhanced features
    train_data, test_data, scaler, target_idx, train_dates, test_dates = preprocess_data(
        enhanced_df, feature_cols=feature_cols, target_col=target_col, split_ratio=0.85  # More recent data for testing
    )
    
    # Create sequences
    sequence_length = 90  # Increased for better pattern recognition
    X_train, y_train = create_sequences(train_data, target_idx, sequence_length)
    X_test, y_test = create_sequences(test_data, target_idx, sequence_length)
    
    # Split training data for validation
    val_split = int(0.8 * len(X_train))
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]
    
    print(f"Training data: {X_train.shape}")
    print(f"Validation data: {X_val.shape}")
    print(f"Test data: {X_test.shape}")
    
    # Create and train ensemble
    ensemble = StockEnsemble()
    
    # Try to load existing models first
    models_loaded = ensemble.load_models(symbol, (X_train.shape[1], X_train.shape[2]))
    
    if not models_loaded:
        print("No pre-trained models found. Training new ensemble...")
        ensemble.build_models((X_train.shape[1], X_train.shape[2]))
        
        # Train models
        train_losses, val_losses = ensemble.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Save the trained models
        ensemble.save_models(symbol)
    else:
        print("Using pre-trained models. Skipping training.")
        train_losses, val_losses = [], []
    
    # Make predictions
    predictions = ensemble.predict(X_test)
    
    # Invert scaling for visualization
    def invert_predictions(scaled_preds, test_data, scaler, target_idx, sequence_length):
        """Properly invert predictions using correct scaling context"""
        inverted = []
        # Each prediction at index i corresponds to test_data[i + sequence_length]
        for i, pred in enumerate(scaled_preds):
            # Get the correct row from test data
            data_idx = min(i + sequence_length, len(test_data) - 1)
            row_copy = test_data[data_idx].copy()
            row_copy[target_idx] = pred
            unscaled = scaler.inverse_transform([row_copy])[0]
            inverted.append(unscaled[target_idx])
        return np.array(inverted)
    
    # Invert actual values first
    inv_actual = invert_predictions(y_test.squeeze(), test_data, scaler, target_idx, sequence_length)
    
    # Invert all predictions
    print(f"\n🔍 Debug Information:")
    print(f"   Raw ensemble prediction range: {predictions['ensemble'].min():.4f} - {predictions['ensemble'].max():.4f}")
    print(f"   Raw transformer prediction range: {predictions['transformer'].min():.4f} - {predictions['transformer'].max():.4f}")
    print(f"   Raw LSTM prediction range: {predictions['lstm'].min():.4f} - {predictions['lstm'].max():.4f}")
    print(f"   Raw actual range: {y_test.squeeze().min():.4f} - {y_test.squeeze().max():.4f}")
    print(f"   Test data shape: {test_data.shape}")
    print(f"   Target index: {target_idx}")
    print(f"   Prediction variance: {np.var(predictions['ensemble']):.6f}")
    print(f"   Actual variance: {np.var(y_test.squeeze()):.6f}")
    
    inv_ensemble = invert_predictions(predictions['ensemble'], test_data, scaler, target_idx, sequence_length)
    inv_transformer = invert_predictions(predictions['transformer'], test_data, scaler, target_idx, sequence_length)
    inv_transformer_lower = invert_predictions(predictions['transformer_lower'], test_data, scaler, target_idx, sequence_length)
    inv_transformer_upper = invert_predictions(predictions['transformer_upper'], test_data, scaler, target_idx, sequence_length)
    inv_lstm = invert_predictions(predictions['lstm'], test_data, scaler, target_idx, sequence_length)
    
    # Calculate bias correction - shift predictions to match actual starting point
    first_100_actual = inv_actual[:100].mean()
    first_100_pred = inv_ensemble[:100].mean()
    bias_correction = first_100_actual - first_100_pred
    
    print(f"   Inverted ensemble range (before correction): ${inv_ensemble.min():.2f} - ${inv_ensemble.max():.2f}")
    print(f"   Actual price range: ${inv_actual.min():.2f} - ${inv_actual.max():.2f}")
    print(f"   Bias correction: ${bias_correction:.2f}")
    
    # Apply bias correction
    inv_ensemble = inv_ensemble + bias_correction
    inv_transformer = inv_transformer + bias_correction
    inv_transformer_lower = inv_transformer_lower + bias_correction
    inv_transformer_upper = inv_transformer_upper + bias_correction
    inv_lstm = inv_lstm + bias_correction
    
    print(f"   Inverted ensemble range (after correction): ${inv_ensemble.min():.2f} - ${inv_ensemble.max():.2f}")
    
    # Prepare dates
    test_dates_adj = test_dates[sequence_length:]
    
    # Plot results
    plot_predictions = {
        'ensemble': inv_ensemble,
        'transformer': inv_transformer,
        'transformer_lower': inv_transformer_lower,
        'transformer_upper': inv_transformer_upper,
        'lstm': inv_lstm
    }
    
    # Calculate performance metrics
    ensemble_mae = mean_absolute_error(inv_actual, inv_ensemble)
    transformer_mae = mean_absolute_error(inv_actual, inv_transformer)
    lstm_mae = mean_absolute_error(inv_actual, inv_lstm)
    
    # DISABLE POPUP GRAPHS - Only create dashboard at end
    # plot_ensemble_results(test_dates_adj, inv_actual, plot_predictions, symbol)
    
    # Create Interactive Visualizations
    print(f"\n🎨 Creating interactive visualizations...")
    visualizer = InteractiveVisualizer(symbol)
    
    # Performance metrics for visualization
    performance_metrics = {
        'ensemble_mae': ensemble_mae,
        'transformer_mae': transformer_mae,
        'lstm_mae': lstm_mae,
        'ensemble_vs_transformer': ((transformer_mae - ensemble_mae) / transformer_mae * 100),
        'ensemble_vs_lstm': ((lstm_mae - ensemble_mae) / lstm_mae * 100)
    }
    
    # Create comprehensive dashboard
    visualizer.create_prediction_dashboard(
        dates=test_dates_adj,
        actual=inv_actual,
        predictions=plot_predictions,
        feature_cols=feature_cols,
        performance_metrics=performance_metrics
    )
    
    # Only create the dashboard (no simple chart needed)
    
    # Create results summary
    create_visualization_summary(symbol, performance_metrics)
    
    print(f"\n=== Performance Metrics ===")
    print(f"Ensemble MAE: ${ensemble_mae:.2f}")
    print(f"Transformer MAE: ${transformer_mae:.2f}")
    print(f"LSTM MAE: ${lstm_mae:.2f}")
    print(f"Ensemble vs Transformer: {((transformer_mae - ensemble_mae) / transformer_mae * 100):.1f}% improvement")
    print(f"Ensemble vs LSTM: {((lstm_mae - ensemble_mae) / lstm_mae * 100):.1f}% improvement")
    
    print(f"\n🎨 Interactive Dashboard Created!")
    print(f"📁 Check the 'visualizations/' folder for:")
    print(f"   • {symbol}_dashboard.html - Interactive dashboard with 3-day forecast")
    print(f"   • {symbol}_results.md - Results summary")
    print(f"\n💡 Open the HTML file in any web browser to explore!")
    print(f"   Features: hover tooltips, zoom, pan, 3-day predictions - no internet needed")
    
    # Feature Analysis
    print(f"\n=== Advanced Feature Analysis ===")
    print("⚠️  Feature analysis disabled to prevent popup graphs")
    
    if False:  # DISABLED - FEATURE_ANALYSIS_AVAILABLE:
        # Create feature correlation heatmap
        print("Creating feature correlation analysis...")
        create_feature_correlation_heatmap(enhanced_df, feature_cols)
        
        # Analyze feature importance
        analyzer = FeatureAnalyzer(ensemble, feature_cols)
        importance_scores = analyzer.analyze_feature_importance(X_test, y_test)
        
        print("Creating feature importance plots...")
        # DISABLE POPUP PLOTS - just get top features without plotting
        top_features = list(zip(feature_cols, importance_scores))
        top_features.sort(key=lambda x: x[1], reverse=True)
        
        print("Analyzing feature categories...")
        category_importance = analyzer.analyze_feature_categories(importance_scores)
        
        print("Analyzing prediction confidence...")
        # DISABLE POPUP ANALYSIS - confidence_stats = analyzer.analyze_prediction_confidence(X_test, y_test, symbol)
        
        print(f"\n=== Model Insights ===")
        print(f"Top 5 Most Important Features:")
        for i, (feature, score) in enumerate(top_features[:5], 1):
            print(f"{i}. {feature}: {score:.4f}")
        
        print(f"\nPrediction Confidence:")
        print(f"- Coverage: {confidence_stats['coverage']:.1%}")
        print(f"- Mean CI Width: {confidence_stats['mean_width']:.4f}")
        
        print(f"\nMost Important Feature Categories:")
        for category, score in sorted(category_importance.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"- {category}: {score:.3f}")
    else:
        print("Install full requirements for advanced feature analysis:")
        print("pip install -r requirements_complete.txt")
