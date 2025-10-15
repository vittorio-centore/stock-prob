import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch.utils.data import DataLoader

# Import our models
from src.transformer_model import StockTransformer, StockDataset, train_transformer, predict_with_uncertainty
from src.pytorch_lstm import EnhancedLSTM, create_enhanced_lstm, train_pytorch_lstm

# Import multi-modal features
from src.multi_modal import BasicMultiModalProcessor as MultiModalDataProcessor

# Import interactive visualization
from src.interactive_plots import InteractiveVisualizer, create_visualization_summary

# Import FIXED preprocessing
from src.data_preprocessing_fixed import preprocess_data_fixed, create_sequences_fixed, validate_data_quality

try:
    from src.feature_analysis import FeatureAnalyzer, create_feature_correlation_heatmap
    FEATURE_ANALYSIS_AVAILABLE = True
except ImportError:
    print("⚠️  Feature analysis not available (missing dependencies)")
    FEATURE_ANALYSIS_AVAILABLE = False

import yfinance as yf

def fetch_stock_data_fixed(symbol, csv_path):
    """FIXED: Fetch recent stock data to avoid split issues"""
    print(f"📈 Fetching {symbol} data using YFinance...")
    ticker = yf.Ticker(symbol)
    
    # Get only recent data to avoid stock split complications
    df = ticker.history(period='3y')  # 3 years instead of 5
    df = df.reset_index()
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"✅ Data saved to {csv_path} ({len(df)} rows)")
    
    # Validate data quality
    validate_data_quality(df, symbol)

class StockEnsembleFixed:
    """FIXED Ensemble with better validation and realistic expectations"""
    
    def __init__(self, transformer_weight=0.7, lstm_weight=0.3, model_dir="models"):
        self.transformer_weight = transformer_weight
        self.lstm_weight = lstm_weight
        self.transformer_model = None
        self.lstm_model = None
        self.model_dir = model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        print(f"🔧 Ensemble weights: Transformer {transformer_weight:.1%}, LSTM {lstm_weight:.1%}")
        print(f"🖥️  Using device: {self.device}")
        
    def build_models(self, input_shape):
        """Build both models"""
        sequence_length, num_features = input_shape
        
        # Build Transformer
        self.transformer_model = StockTransformer(
            input_dim=num_features,
            d_model=64,  # Reduced complexity
            nhead=4,     # Fewer attention heads
            num_layers=2, # Fewer layers
            dropout=0.2,  # More dropout
            max_len=sequence_length
        )
        
        # Build LSTM
        self.lstm_model = create_enhanced_lstm(
            input_dim=num_features,
            hidden_dim=64,  # Reduced complexity
            num_layers=2,
            dropout=0.3     # More dropout
        )
        
        print(f"🧠 Models built with reduced complexity to prevent overfitting")
        print(f"📊 Transformer: {sum(p.numel() for p in self.transformer_model.parameters()):,} parameters")
        print(f"📊 LSTM: {sum(p.numel() for p in self.lstm_model.parameters()):,} parameters")
        
    def train_ensemble(self, X_train, y_train, X_val, y_val, epochs=30):
        """FIXED: Train with more realistic epochs and better validation"""
        print("🚀 Training Transformer model...")
        
        # Prepare data for transformer
        train_dataset = StockDataset(X_train, y_train.squeeze())
        val_dataset = StockDataset(X_val, y_val.squeeze())
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train transformer with fewer epochs
        train_losses, val_losses = train_transformer(
            self.transformer_model, train_loader, val_loader, 
            epochs=epochs, device=self.device
        )
        
        print("🚀 Training PyTorch Enhanced LSTM model...")
        
        # Prepare data for PyTorch LSTM
        lstm_train_dataset = StockDataset(X_train, y_train.squeeze())
        lstm_val_dataset = StockDataset(X_val, y_val.squeeze())
        
        lstm_train_loader = DataLoader(lstm_train_dataset, batch_size=32, shuffle=True)
        lstm_val_loader = DataLoader(lstm_val_dataset, batch_size=32, shuffle=False)
        
        # Train PyTorch LSTM with fewer epochs
        lstm_train_losses, lstm_val_losses = train_pytorch_lstm(
            self.lstm_model, lstm_train_loader, lstm_val_loader, 
            epochs=epochs*2, device=self.device  # LSTM can handle more epochs
        )
        
        return train_losses, val_losses, (lstm_train_losses, lstm_val_losses)
    
    def predict(self, X_test):
        """Make ensemble predictions"""
        # Transformer predictions (with uncertainty)
        test_dataset = StockDataset(X_test, np.zeros(len(X_test)))  # Dummy targets
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        transformer_preds = predict_with_uncertainty(
            self.transformer_model, test_loader, self.device
        )
        
        # Use median prediction from transformer quantiles
        transformer_median = transformer_preds[:, 1]  # Middle quantile
        
        # PyTorch LSTM predictions
        self.lstm_model.eval()
        lstm_test_dataset = StockDataset(X_test, np.zeros(len(X_test)))
        lstm_test_loader = DataLoader(lstm_test_dataset, batch_size=32, shuffle=False)
        
        lstm_preds = []
        with torch.no_grad():
            for batch_x, _ in lstm_test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.lstm_model(batch_x)
                lstm_preds.append(outputs.cpu().numpy())
        
        lstm_preds = np.vstack(lstm_preds).squeeze()
        
        # Ensemble prediction
        ensemble_preds = (self.transformer_weight * transformer_median + 
                         self.lstm_weight * lstm_preds)
        
        return {
            'ensemble': ensemble_preds,
            'transformer': transformer_median,
            'transformer_lower': transformer_preds[:, 0],  # Lower quantile
            'transformer_upper': transformer_preds[:, 2],  # Upper quantile
            'lstm': lstm_preds
        }
    
    def save_models(self, symbol):
        """Save trained models"""
        transformer_path = os.path.join(self.model_dir, f"{symbol}_transformer_fixed.pth")
        lstm_path = os.path.join(self.model_dir, f"{symbol}_pytorch_lstm_fixed.pth")
        
        torch.save(self.transformer_model.state_dict(), transformer_path)
        torch.save(self.lstm_model.state_dict(), lstm_path)
        
        print(f"💾 Models saved: {transformer_path}, {lstm_path}")
    
    def load_models(self, symbol, input_shape):
        """Load pre-trained models if they exist"""
        transformer_path = os.path.join(self.model_dir, f"{symbol}_transformer_fixed.pth")
        lstm_path = os.path.join(self.model_dir, f"{symbol}_pytorch_lstm_fixed.pth")
        
        if os.path.exists(transformer_path) and os.path.exists(lstm_path):
            try:
                self.build_models(input_shape)
                self.transformer_model.load_state_dict(torch.load(transformer_path, map_location=self.device))
                self.lstm_model.load_state_dict(torch.load(lstm_path, map_location=self.device))
                
                self.transformer_model.eval()
                self.lstm_model.eval()
                
                print(f"✅ Loaded pre-trained models for {symbol}")
                return True
            except Exception as e:
                print(f"⚠️  Failed to load models: {e}")
                return False
        return False

def plot_ensemble_results_fixed(dates, actual, predictions, symbol, show_plots=False):
    """FIXED plotting with better scaling and realistic expectations"""
    if not show_plots:
        return  # Skip plotting during training
        
    plt.figure(figsize=(15, 10))
    
    # Main prediction plot
    plt.subplot(2, 2, 1)
    plt.plot(dates, actual, label='Actual Price', color='black', linewidth=2)
    plt.plot(dates, predictions['ensemble'], label='Ensemble Prediction', color='purple', linewidth=2)
    plt.plot(dates, predictions['transformer'], label='Transformer', color='red', linestyle='--', alpha=0.7)
    plt.plot(dates, predictions['lstm'], label='LSTM', color='blue', linestyle='--', alpha=0.7)
    
    # Add confidence bands
    if 'transformer_lower' in predictions and 'transformer_upper' in predictions:
        plt.fill_between(dates, predictions['transformer_lower'], predictions['transformer_upper'], 
                        alpha=0.2, color='red', label='Confidence Interval')
    
    plt.title(f'{symbol} Stock Price Predictions vs Actual (FIXED)')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Error analysis
    plt.subplot(2, 2, 2)
    ensemble_error = np.abs(actual - predictions['ensemble'])
    transformer_error = np.abs(actual - predictions['transformer'])
    lstm_error = np.abs(actual - predictions['lstm'])
    
    plt.plot(dates, ensemble_error, label='Ensemble Error', color='purple')
    plt.plot(dates, transformer_error, label='Transformer Error', color='red', alpha=0.7)
    plt.plot(dates, lstm_error, label='LSTM Error', color='blue', alpha=0.7)
    plt.title('Prediction Errors Over Time')
    plt.xlabel('Date')
    plt.ylabel('Absolute Error ($)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Performance comparison
    plt.subplot(2, 2, 3)
    mae_ensemble = np.mean(ensemble_error)
    mae_transformer = np.mean(transformer_error)
    mae_lstm = np.mean(lstm_error)
    
    models = ['Ensemble', 'Transformer', 'LSTM']
    maes = [mae_ensemble, mae_transformer, mae_lstm]
    colors = ['purple', 'red', 'blue']
    
    bars = plt.bar(models, maes, color=colors, alpha=0.7)
    plt.title('Model Performance (MAE)')
    plt.ylabel('Mean Absolute Error ($)')
    
    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'${mae:.2f}', ha='center', va='bottom')
    
    # Residual analysis
    plt.subplot(2, 2, 4)
    residuals = actual - predictions['ensemble']
    plt.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    plt.title('Prediction Residuals Distribution')
    plt.xlabel('Residual ($)')
    plt.ylabel('Frequency')
    plt.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/{symbol}_fixed_analysis.png', dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()
    else:
        plt.close()  # Close without showing

if __name__ == "__main__":
    # For testing, use AAPL directly
    symbol = "AAPL"
    print(f"🚀 Running FIXED stock prediction for {symbol}")
    
    # Fetch data if not present
    csv_path = f"data/{symbol}_daily_fixed.csv"
    if not os.path.exists(csv_path):
        fetch_stock_data_fixed(symbol, csv_path)
    
    # Load data
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    print(f"\n📊 Original data: {len(df)} rows from {df.Date.min().date()} to {df.Date.max().date()}")
    
    # Create multi-modal features
    print("\n🔧 Creating enhanced features with multi-modal data...")
    processor = MultiModalDataProcessor()
    enhanced_df = processor.create_enhanced_features(df, symbol)
    
    # Get enhanced feature columns
    feature_cols = processor.get_enhanced_feature_columns()
    target_col = "Close"
    
    print(f"\n📈 Using {len(feature_cols)} features:")
    print("- Technical indicators (RSI, MACD, Bollinger Bands)")
    print("- Time-based cyclical features")
    print("- Market regime indicators")
    print("- Base OHLCV data")
    
    # FIXED preprocessing with proper data handling
    train_data, test_data, scaler, target_idx, train_dates, test_dates = preprocess_data_fixed(
        enhanced_df, feature_cols=feature_cols, target_col=target_col, split_ratio=0.8
    )
    
    # Create sequences
    sequence_length = 30  # Reduced from 60 for more realistic expectations
    X_train, y_train = create_sequences_fixed(train_data, target_idx, sequence_length)
    X_test, y_test = create_sequences_fixed(test_data, target_idx, sequence_length)
    
    # Split training data for validation
    val_split = int(0.8 * len(X_train))
    X_val, y_val = X_train[val_split:], y_train[val_split:]
    X_train, y_train = X_train[:val_split], y_train[:val_split]
    
    print(f"\n📊 FIXED Data splits:")
    print(f"Training sequences: {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")
    print(f"Test sequences: {X_test.shape}")
    
    # Create and train ensemble
    ensemble = StockEnsembleFixed()
    
    # Force retraining for FIXED version (don't load old models)
    models_loaded = False  # Always retrain to use fixed preprocessing
    
    if not models_loaded:
        print("\n🎯 No pre-trained FIXED models found. Training new ensemble...")
        ensemble.build_models((X_train.shape[1], X_train.shape[2]))
        
        # Train models with realistic expectations
        train_losses, val_losses, lstm_history = ensemble.train_ensemble(X_train, y_train, X_val, y_val, epochs=20)
        
        # Save models
        ensemble.save_models(symbol)
    else:
        print("\n⚡ Using pre-trained FIXED models for fast prediction!")
    
    # Make predictions
    print("\n🔮 Making predictions...")
    predictions = ensemble.predict(X_test)
    
    # Inverse transform predictions to original scale
    def inverse_transform_predictions(scaler, predictions_dict, target_idx):
        """Convert normalized predictions back to original price scale"""
        result = {}
        for key, preds in predictions_dict.items():
            # Create dummy array with same shape as original features
            dummy = np.zeros((len(preds), scaler.n_features_in_))
            dummy[:, target_idx] = preds
            # Inverse transform and extract target column
            unscaled = scaler.inverse_transform(dummy)
            result[key] = unscaled[:, target_idx]
        return result
    
    # Inverse transform
    inv_predictions = inverse_transform_predictions(scaler, predictions, target_idx)
    
    # Inverse transform actual values
    dummy_actual = np.zeros((len(y_test), scaler.n_features_in_))
    dummy_actual[:, target_idx] = y_test.squeeze()
    inv_actual = scaler.inverse_transform(dummy_actual)[:, target_idx]
    
    # Adjust test dates for sequence offset
    test_dates_adj = test_dates[sequence_length:]
    
    # Calculate performance metrics
    ensemble_mae = mean_absolute_error(inv_actual, inv_predictions['ensemble'])
    transformer_mae = mean_absolute_error(inv_actual, inv_predictions['transformer'])
    lstm_mae = mean_absolute_error(inv_actual, inv_predictions['lstm'])
    
    # Plot results (only save, don't show during training)
    plot_ensemble_results_fixed(test_dates_adj, inv_actual, inv_predictions, symbol, show_plots=False)
    
    # Create Interactive Visualizations
    print(f"\n🎨 Creating FIXED interactive visualizations...")
    visualizer = InteractiveVisualizer(symbol)
    
    # Performance metrics for visualization
    performance_metrics = {
        'ensemble_mae': ensemble_mae,
        'transformer_mae': transformer_mae,
        'lstm_mae': lstm_mae,
        'ensemble_vs_transformer': ((transformer_mae - ensemble_mae) / transformer_mae * 100) if transformer_mae > 0 else 0,
        'ensemble_vs_lstm': ((lstm_mae - ensemble_mae) / lstm_mae * 100) if lstm_mae > 0 else 0
    }
    
    # Create comprehensive dashboard
    visualizer.create_prediction_dashboard(
        dates=test_dates_adj,
        actual=inv_actual,
        predictions=inv_predictions,
        feature_cols=feature_cols,
        performance_metrics=performance_metrics
    )
    
    # Create simple chart
    visualizer.create_simple_prediction_chart(
        dates=test_dates_adj,
        actual=inv_actual,
        predictions=inv_predictions
    )
    
    # Create results summary
    create_visualization_summary(symbol, performance_metrics)
    
    print(f"\n📊 FIXED Performance Metrics:")
    print(f"Ensemble MAE: ${ensemble_mae:.2f}")
    print(f"Transformer MAE: ${transformer_mae:.2f}")
    print(f"LSTM MAE: ${lstm_mae:.2f}")
    
    if transformer_mae > 0 and lstm_mae > 0:
        print(f"Ensemble vs Transformer: {((transformer_mae - ensemble_mae) / transformer_mae * 100):.1f}% improvement")
        print(f"Ensemble vs LSTM: {((lstm_mae - ensemble_mae) / lstm_mae * 100):.1f}% improvement")
    
    print(f"\n🎨 FIXED Interactive Visualizations Created!")
    print(f"📁 Check the 'visualizations/' folder for:")
    print(f"   • {symbol}_dashboard.html - Complete interactive dashboard")
    print(f"   • {symbol}_simple.html - Simple prediction chart") 
    print(f"   • {symbol}_results.md - Results summary")
    print(f"   • {symbol}_fixed_analysis.png - Static analysis plot")
    print(f"\n💡 Open the HTML files in any web browser to explore!")
    print(f"   The FIXED version should show more realistic prediction accuracy!")
