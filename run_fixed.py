#!/usr/bin/env python3
"""
Simple script to run the FIXED stock predictor without popup graphs
"""
import os
import sys

# Add src to path
sys.path.append('src')

# Set matplotlib to non-interactive backend to prevent popups
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Now import and run the fixed ensemble
from src.ensemble_model_fixed import *

if __name__ == "__main__":
    print("🚀 Running FIXED Stock Predictor (No Popups)")
    print("=" * 50)
    
    # The main code from ensemble_model_fixed.py will run automatically
    # when we import it, but let's be explicit
    
    symbol = "AAPL"
    print(f"📊 Analyzing {symbol} with FIXED methodology...")
    print("✅ No popup graphs will appear")
    print("✅ All plots saved to visualizations/ folder")
    print("✅ Interactive dashboard will be created")
    print("\n🔄 Starting analysis...")
