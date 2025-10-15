# 🚀 Advanced Stock Predictor with Transformer + LSTM Ensemble

## 📋 Overview
Sophisticated stock price prediction system combining PyTorch Transformer and Enhanced LSTM models with comprehensive technical analysis and interactive visualizations.

## 🎯 Quick Start

### 1. Setup Environment
```bash
python -m venv stock_env
source stock_env/bin/activate  # On macOS/Linux
# stock_env\Scripts\activate  # On Windows
pip install --upgrade pip
```

### 2. Install All Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the System
```bash
python src/ensemble_model.py
```

## 📊 **COMPLETE TECHNICAL SPECIFICATION**

### 🔬 **Data Source & Collection**
- **Primary Source**: Yahoo Finance (YFinance API)
- **Data Type**: Daily OHLCV (Open, High, Low, Close, Volume)
- **Symbols Supported**: Any publicly traded stock (AAPL, GOOGL, MSFT, NVDA, etc.)
- **Historical Range**: Automatically fetches maximum available history
- **Data Quality**: Real-time, no simulated or synthetic data

### 📈 **Data Splitting Strategy**
```
🗓️ CHRONOLOGICAL SPLIT (Time-Series Aware):
├── Training Data:   64% (oldest data)
├── Validation Data: 16% (middle period) 
└── Test Data:       20% (most recent data)

Example with 1000 trading days:
├── Training:   Days 1-640   (for learning patterns)
├── Validation: Days 641-800 (for model selection during training)
└── Test:       Days 801-1000 (for final evaluation)

🔄 SEQUENCE CONFIGURATION:
- Sequence Length: 60 trading days lookback
- Prediction Target: Next day's closing price
- Input Shape: (batch_size, 60, 21)
```

### 🧠 **AI Architecture Deep Dive**

#### **🎯 Ensemble Strategy**
- **Primary Model**: PyTorch Transformer (75% weight)
- **Secondary Model**: PyTorch Enhanced LSTM (25% weight)
- **Combination Method**: Adaptive weighted ensemble
- **Weight Adjustment**: Dynamic based on recent performance

#### **🔮 Transformer Model (Primary Predictor)**
```python
Architecture:
├── Input Projection: Linear(21 → 128)
├── Positional Encoding: Sinusoidal (max_len=100)
├── Transformer Encoder: 4 layers
│   ├── Multi-Head Attention: 8 heads
│   ├── Feed-Forward: 128 → 512 → 128
│   ├── Dropout: 0.1
│   └── Activation: GELU
├── Output Projection: Linear(128 → 64)
└── Quantile Heads: 3 outputs (10%, 50%, 90% quantiles)

Training Configuration:
├── Loss Function: Quantile Loss (for uncertainty)
├── Optimizer: AdamW (lr=0.001, weight_decay=0.01)
├── Scheduler: ReduceLROnPlateau
├── Epochs: 50 (default)
├── Batch Size: 32
├── Early Stopping: 15 epochs patience
└── Gradient Clipping: max_norm=1.0
```

#### **🔄 Enhanced LSTM Model (Secondary Predictor)**
```python
Architecture:
├── Input Normalization: BatchNorm1d
├── Bidirectional LSTM: 2 layers × 128 hidden units
│   ├── Dropout: 0.3 between layers
│   └── Forget Gate Bias: 1.0 (for better gradient flow)
├── Batch Normalization: After LSTM
├── Dense Layers: 256 → 128 → 64 → 1
│   ├── Activation: ELU (better than ReLU for negative values)
│   ├── Dropout: 0.4
│   └── Residual Connections: Skip connections for deep layers
└── Weight Initialization: Xavier/Glorot uniform

Training Configuration:
├── Loss Function: HuberLoss (delta=0.1, robust to outliers)
├── Optimizer: AdamW (lr=0.001, weight_decay=0.01)
├── Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)
├── Epochs: 100 (with early stopping)
├── Batch Size: 32
├── Early Stopping: 20 epochs patience
├── Gradient Clipping: max_norm=1.0
└── Best Model Restoration: Saves best validation performance
```

### 🛠️ **Feature Engineering (21 Enhanced Features)**

#### **📊 Technical Indicators (9 features)**
```python
1. RSI (14-period): Relative Strength Index
2. MACD Line: 12-EMA - 26-EMA  
3. MACD Signal: 9-EMA of MACD
4. MACD Histogram: MACD - Signal
5. Bollinger Position: (Price - Lower) / (Upper - Lower)
6. On-Balance Volume: Cumulative volume flow
7. Volume Ratio: Current / 20-day average volume
8. Price Change 1D: (Close - Previous Close) / Previous Close
9. Price Change 5D: 5-day momentum
```

#### **⏰ Time-Based Features (4 features)**
```python
10. Day of Week (Sin): sin(2π × day / 7)
11. Day of Week (Cos): cos(2π × day / 7)  
12. Month (Sin): sin(2π × month / 12)
13. Month (Cos): cos(2π × month / 12)
```

#### **📈 Market Regime Indicators (2 features)**
```python
14. Volatility Regime: Rolling 20-day standard deviation
15. Trend Strength: Directional movement indicator
```

#### **📋 Base OHLCV Features (5 features)**
```python
16. Open Price (normalized)
17. High Price (normalized)
18. Low Price (normalized) 
19. Close Price (normalized)
20. Volume (normalized)
```

#### **💪 Price Momentum (1 feature)**
```python
21. Multi-timeframe Momentum: Combined 1D, 5D, 20D momentum
```

### 🔧 **Technology Stack & Why We Chose Each**

#### **🔥 PyTorch Framework**
**Why PyTorch over TensorFlow?**
- ✅ **Dynamic Computation Graphs**: Better for research and experimentation
- ✅ **Pythonic Design**: More intuitive debugging and development
- ✅ **Superior Transformer Support**: Native attention mechanisms
- ✅ **Better Memory Management**: For large sequence models
- ✅ **Academic Standard**: Latest research uses PyTorch
- ✅ **Unified Framework**: Both models in same ecosystem

#### **🎯 Why Transformer as Primary Model?**
```python
Advantages for Stock Prediction:
├── Self-Attention Mechanism: Captures long-range dependencies
├── Parallel Processing: Faster training than RNNs
├── No Vanishing Gradients: Better for long sequences
├── Multi-Head Attention: Focuses on different aspects simultaneously
├── Positional Encoding: Understands time relationships
└── Uncertainty Quantification: Built-in confidence intervals
```

#### **🔄 Why LSTM as Secondary Model?**
```python
Advantages for Time Series:
├── Sequential Processing: Natural for time series data
├── Memory Cells: Remembers important past information
├── Bidirectional: Sees both past and future context
├── Robust to Noise: Good for volatile financial data
├── Complementary: Different approach than Transformer
└── Proven Track Record: Established success in finance
```

#### **⚖️ Why Ensemble Both Models?**
```python
Ensemble Benefits:
├── Reduced Overfitting: Multiple perspectives
├── Better Generalization: Combines different strengths
├── Uncertainty Reduction: More robust predictions
├── Risk Mitigation: If one model fails, other compensates
└── Performance Boost: Often 5-15% improvement over single models
```

### 📊 **Data Preprocessing Pipeline**

#### **🔄 Step-by-Step Process**
```python
1. Data Collection:
   └── YFinance API → Raw OHLCV data

2. Data Cleaning:
   ├── Sort by date (chronological order)
   ├── Remove missing values (forward/backward fill)
   └── Validate data integrity

3. Feature Engineering:
   ├── Calculate technical indicators (TA-Lib)
   ├── Add time-based features (cyclical encoding)
   ├── Compute market regime indicators
   └── Create momentum features

4. Normalization:
   ├── MinMaxScaler (0 to 1 range)
   ├── Fit on training data only
   └── Transform all splits consistently

5. Sequence Creation:
   ├── Create 60-day lookback windows
   ├── Target: Next day closing price
   └── Sliding window approach

6. Data Loading:
   ├── PyTorch DataLoader
   ├── Batch size: 32
   └── Shuffle training, preserve order for validation/test
```

### 🎨 **Visualization & Analysis**

#### **📊 Interactive Plotly Visualizations**
```python
Generated Files:
├── {SYMBOL}_dashboard.html: Complete 6-panel dashboard
│   ├── Main prediction chart with confidence bands
│   ├── Prediction error analysis over time
│   ├── Model performance comparison (bar charts)
│   ├── Uncertainty coverage analysis
│   ├── Feature importance ranking (top 10)
│   └── Prediction accuracy distribution
├── {SYMBOL}_simple.html: Clean prediction chart
└── {SYMBOL}_results.md: Performance summary text

Features:
├── Interactive: Hover tooltips, zoom, pan
├── Standalone: No internet required
├── Professional: Publication-ready styling
├── Mobile-friendly: Responsive design
└── Shareable: Send HTML files to anyone
```

#### **📈 Performance Metrics**
```python
Evaluation Metrics:
├── Mean Absolute Error (MAE): Average prediction error in $
├── Root Mean Square Error (RMSE): Penalizes large errors
├── Mean Absolute Percentage Error (MAPE): Relative error %
├── Directional Accuracy: % of correct up/down predictions
├── Confidence Coverage: % of actuals within prediction bands
└── Ensemble Improvement: % better than individual models
```

### 🏗️ **Project Structure**
```
stocks/
├── requirements.txt          # All 60+ dependencies with versions
├── README.md                # This comprehensive guide
├── src/
│   ├── ensemble_model.py    # 🎯 Main system (431 lines)
│   ├── transformer_model.py # PyTorch Transformer (177 lines)
│   ├── pytorch_lstm.py      # Enhanced LSTM (252 lines)
│   ├── multi_modal.py       # Feature engineering (162 lines)
│   ├── interactive_plots.py # Plotly visualizations (365 lines)
│   ├── feature_analysis.py  # Feature importance (188 lines)
│   └── data_preprocessing.py # Data pipeline (75 lines)
├── data/                    # Stock data cache (auto-created)
├── models/                  # Trained model weights (auto-created)
└── visualizations/          # Interactive charts (auto-created)
```

### ⚙️ **Training Process Details**

#### **🔄 Complete Training Pipeline**
```python
1. Data Preparation:
   ├── Fetch stock data (YFinance)
   ├── Engineer 21 features
   ├── Split: 64% train, 16% val, 20% test
   └── Create sequences (60-day windows)

2. Model Initialization:
   ├── Transformer: 4 layers, 8 heads, 128 d_model
   ├── LSTM: 2 bidirectional layers, 128 hidden
   └── Load pre-trained weights if available

3. Training Loop:
   ├── Transformer Training (50 epochs):
   │   ├── Quantile loss for uncertainty
   │   ├── AdamW optimizer
   │   ├── Early stopping (15 epochs patience)
   │   └── Save best validation model
   └── LSTM Training (100 epochs):
       ├── Huber loss (robust to outliers)
       ├── Cosine annealing scheduler
       ├── Early stopping (20 epochs patience)
       └── Gradient clipping for stability

4. Ensemble Creation:
   ├── Load best individual models
   ├── Set weights: 75% Transformer, 25% LSTM
   ├── Adaptive weight adjustment based on performance
   └── Save complete ensemble

5. Final Evaluation:
   ├── Predict on test set (never seen during training)
   ├── Calculate performance metrics
   ├── Generate interactive visualizations
   └── Create results summary
```

### 🔍 **Model Validation Strategy**

#### **🎯 Why Our Approach Works**
```python
Time Series Validation:
├── No Random Shuffling: Preserves temporal order
├── Chronological Split: Realistic trading scenario
├── Walk-Forward Validation: Simulates real-world deployment
├── Out-of-Sample Testing: True unseen data evaluation
└── Multiple Metrics: Comprehensive performance assessment

Overfitting Prevention:
├── Early Stopping: Prevents memorization
├── Dropout Regularization: Random neuron deactivation
├── Weight Decay: L2 regularization penalty
├── Batch Normalization: Stable training dynamics
└── Validation Monitoring: Real-time performance tracking
```

### 🚀 **Performance Optimizations**

#### **⚡ Speed & Memory Optimizations**
```python
Training Optimizations:
├── Gradient Clipping: Prevents exploding gradients
├── Mixed Precision: Faster training (if GPU available)
├── Batch Processing: Efficient parallel computation
├── Model Checkpointing: Resume interrupted training
└── Early Stopping: Avoid unnecessary computation

Memory Management:
├── Sequence Chunking: Process data in manageable pieces
├── Gradient Accumulation: Simulate larger batch sizes
├── Model Pruning: Remove unnecessary parameters
└── Efficient Data Loading: Minimize RAM usage
```

### 📋 **System Requirements**
```python
Minimum Requirements:
├── Python: 3.9+
├── RAM: 4GB (8GB+ recommended)
├── Storage: 2GB free space
├── Internet: For initial data download
└── OS: Windows 10+, macOS 10.14+, Ubuntu 18.04+

Recommended Setup:
├── RAM: 16GB+ 
├── CPU: 8+ cores
├── GPU: CUDA-compatible (optional, auto-detected)
└── SSD: For faster data loading
```

### 🔧 **Installation & Troubleshooting**

#### **📦 Dependencies (60+ packages)**
```bash
# Core ML Stack
torch>=2.8.0, pandas, numpy, scikit-learn

# Financial Data
yfinance, pandas-datareader, alpha-vantage

# Technical Analysis  
TA-Lib (may need system installation)

# Visualization
plotly, matplotlib, seaborn

# Advanced ML
xgboost, lightgbm, optuna

# Web & API
streamlit, fastapi, dash

# Full list in requirements.txt
```

#### **🛠️ Common Issues & Solutions**
```bash
# TA-Lib Installation Issues:
# macOS:
brew install ta-lib
pip install TA-Lib

# Ubuntu:
sudo apt-get install libta-lib-dev
pip install TA-Lib

# Windows:
# Download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

# Memory Issues:
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Import Hanging:
# First PyTorch import may be slow (normal)
# Subsequent runs will be faster
```

### 🎯 **Usage Examples**

#### **🚀 Basic Usage**
```python
# Run with default settings (AAPL)
python src/ensemble_model.py

# The system will:
# 1. Download AAPL data automatically
# 2. Train Transformer + LSTM ensemble  
# 3. Generate predictions with uncertainty
# 4. Create interactive HTML visualizations
# 5. Display performance metrics
```

#### **📊 Output Files**
```python
After running, check these folders:

visualizations/
├── AAPL_dashboard.html    # Complete interactive dashboard
├── AAPL_simple.html       # Clean prediction chart  
└── AAPL_results.md        # Performance summary

models/
├── AAPL_transformer.pth   # Trained Transformer weights
└── AAPL_lstm.pth          # Trained LSTM weights

data/
└── AAPL_enhanced.csv      # Processed data with all features
```

## 🎉 **Ready to Predict the Future!**

This system combines cutting-edge AI research with practical financial engineering to create a robust, interpretable, and high-performance stock prediction tool.

**Key Strengths:**
- ✅ **State-of-the-art AI**: Transformer + LSTM ensemble
- ✅ **Comprehensive Features**: 21 engineered features from basic OHLCV
- ✅ **Uncertainty Quantification**: Know when to trust predictions
- ✅ **Interactive Visualizations**: Professional charts and analysis
- ✅ **Pure PyTorch**: Modern, unified framework
- ✅ **Real Data Only**: No fake or simulated features
- ✅ **Production Ready**: Model persistence, error handling, logging

**Perfect for:** Research, backtesting, educational purposes, and algorithm development.

🚀 **Start predicting now:** `python src/ensemble_model.py` 📈🤖
