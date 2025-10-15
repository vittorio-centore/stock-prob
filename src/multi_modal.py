import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BasicTechnicalIndicators:
    """Basic technical indicators using only pandas/numpy"""
    
    @staticmethod
    def calculate_rsi(prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral value
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd.fillna(0), macd_signal.fillna(0), macd_histogram.fillna(0)
    
    @staticmethod
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        bb_position = (prices - lower_band) / (upper_band - lower_band)
        return upper_band.fillna(prices), lower_band.fillna(prices), bb_position.fillna(0.5)
    
    @staticmethod
    def calculate_volume_indicators(volume, prices):
        """Calculate volume-based indicators"""
        # On-Balance Volume
        price_change = np.sign(prices.diff()).fillna(0)
        obv = (price_change * volume).cumsum()
        
        # Volume Moving Average
        volume_ma = volume.rolling(window=20).mean()
        volume_ratio = volume / volume_ma
        
        return obv.fillna(0), volume_ratio.fillna(1)

class BasicMultiModalProcessor:
    """Simplified multi-modal processor that works with basic dependencies"""
    
    def __init__(self):
        self.tech_indicators = BasicTechnicalIndicators()
    
    def create_enhanced_features(self, df, symbol):
        """Create enhanced feature set with real data only"""
        # Ensure Date column is datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 1. Technical Indicators (REAL)
        df = self._add_technical_indicators(df)
        
        # 2. Time-based features (REAL)
        df = self._add_time_features(df)
        
        # 3. Market regime indicators (REAL)
        df = self._add_market_regime_features(df)
        
        # Fill any remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def _add_technical_indicators(self, df):
        """Add technical analysis indicators"""
        # RSI
        df['rsi'] = self.tech_indicators.calculate_rsi(df['Close'])
        
        # MACD
        macd, macd_signal, macd_hist = self.tech_indicators.calculate_macd(df['Close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_position = self.tech_indicators.calculate_bollinger_bands(df['Close'])
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_position'] = bb_position
        
        # Volume indicators
        obv, volume_ratio = self.tech_indicators.calculate_volume_indicators(df['Volume'], df['Close'])
        df['obv'] = obv
        df['volume_ratio'] = volume_ratio
        
        # Price momentum
        df['price_change_1d'] = df['Close'].pct_change(1).fillna(0)
        df['price_change_5d'] = df['Close'].pct_change(5).fillna(0)
        df['price_change_20d'] = df['Close'].pct_change(20).fillna(0)
        
        return df
    
    def _add_time_features(self, df):
        """Add time-based cyclical features"""
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['month'] = df['Date'].dt.month
        df['quarter'] = df['Date'].dt.quarter
        
        # Cyclical encoding for better neural network processing
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _add_market_regime_features(self, df):
        """Add market regime detection features"""
        # Calculate 20-day volatility first
        df['volatility_20d'] = df['Close'].rolling(window=20).std().fillna(0)
        
        # Volatility regime
        df['volatility_regime'] = (df['volatility_20d'] > df['volatility_20d'].rolling(window=60).mean().fillna(df['volatility_20d'])).astype(int)
        
        # Trend strength
        df['sma_20'] = df['Close'].rolling(window=20).mean().fillna(df['Close'])
        df['sma_50'] = df['Close'].rolling(window=50).mean().fillna(df['Close'])
        df['trend_strength'] = ((df['sma_20'] - df['sma_50']) / df['Close']).fillna(0)
        
        return df
    
    def get_enhanced_feature_columns(self):
        """Return list of all enhanced feature columns (REAL DATA ONLY)"""
        base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        technical_features = [
            'rsi', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'obv', 'volume_ratio',
            'price_change_1d', 'price_change_5d', 'price_change_20d'
        ]
        
        time_features = ['day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
        
        regime_features = ['volatility_regime', 'trend_strength']
        
        return base_features + technical_features + time_features + regime_features

# For backward compatibility
MultiModalDataProcessor = BasicMultiModalProcessor

if __name__ == "__main__":
    print("✅ Enhanced Technical Features Ready!")
    print("Features included:")
    print("- Technical indicators (RSI, MACD, Bollinger Bands)")
    print("- Time-based cyclical features")
    print("- Market regime indicators")
    print("- Price momentum analysis")
    print("\nAll features use REAL data - no simulated/fake data!")
    print("Total: 21 features from original 5 OHLCV")
