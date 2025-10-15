import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class FeatureAnalyzer:
    """Analyze feature importance and model interpretability"""
    
    def __init__(self, ensemble_model, feature_columns):
        self.ensemble = ensemble_model
        self.feature_columns = feature_columns
        
    def analyze_feature_importance(self, X_test, y_test, n_samples=100):
        """Analyze feature importance using permutation importance"""
        print("Analyzing feature importance...")
        
        # Get baseline predictions
        baseline_preds = self.ensemble.predict(X_test[:n_samples])
        baseline_mae = mean_absolute_error(y_test[:n_samples], baseline_preds['ensemble'])
        
        importance_scores = {}
        
        for i, feature_name in enumerate(self.feature_columns):
            # Create permuted version of the data
            X_permuted = X_test[:n_samples].copy()
            
            # Permute the i-th feature across all samples
            feature_values = X_permuted[:, :, i].copy()
            np.random.shuffle(feature_values.flatten())
            X_permuted[:, :, i] = feature_values.reshape(X_permuted[:, :, i].shape)
            
            # Get predictions with permuted feature
            permuted_preds = self.ensemble.predict(X_permuted)
            permuted_mae = mean_absolute_error(y_test[:n_samples], permuted_preds['ensemble'])
            
            # Importance is the increase in error when feature is permuted
            importance = permuted_mae - baseline_mae
            importance_scores[feature_name] = importance
            
        return importance_scores
    
    def plot_feature_importance(self, importance_scores, top_n=15, symbol="STOCK"):
        """Plot feature importance"""
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Take top N features
        top_features = sorted_features[:top_n]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        features, scores = zip(*top_features)
        colors = ['red' if score > 0 else 'green' for score in scores]
        
        bars = plt.barh(range(len(features)), scores, color=colors, alpha=0.7)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Impact on Prediction Error (MAE increase when permuted)')
        plt.title(f'Top {top_n} Feature Importance Analysis\n(Red = Harmful when removed, Green = Helpful when removed)')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(score + (0.001 if score >= 0 else -0.001), i, f'{score:.4f}', 
                    ha='left' if score >= 0 else 'right', va='center')
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{symbol}_confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Don't show, just save
        
        return top_features
    
    def analyze_feature_categories(self, importance_scores):
        """Analyze importance by feature categories"""
        categories = {
            'Price Data': ['Open', 'High', 'Low', 'Close', 'Volume'],
            'Technical Indicators': ['rsi', 'macd', 'macd_signal', 'macd_histogram', 'bb_position', 'obv', 'volume_ratio'],
            'Price Momentum': ['price_change_1d', 'price_change_5d', 'price_change_20d'],
            'Economic Indicators': ['vix', 'treasury_10y', 'sp500', 'dollar_index'],
            'News Sentiment': ['news_sentiment', 'news_volume', 'sentiment_ma_3d', 'sentiment_ma_7d'],
            'Time Features': ['day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'],
            'Market Regime': ['volatility_regime', 'trend_strength']
        }
        
        category_importance = {}
        for category, features in categories.items():
            category_score = sum(abs(importance_scores.get(feature, 0)) for feature in features if feature in importance_scores)
            category_importance[category] = category_score
        
        # Plot category importance
        plt.figure(figsize=(10, 6))
        categories_sorted = sorted(category_importance.items(), key=lambda x: x[1], reverse=True)
        
        cats, scores = zip(*categories_sorted)
        bars = plt.bar(range(len(cats)), scores, color='skyblue', alpha=0.7)
        plt.xticks(range(len(cats)), cats, rotation=45, ha='right')
        plt.ylabel('Total Importance Score')
        plt.title('Feature Category Importance Analysis')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{symbol}_confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Don't show, just save
        
        return category_importance
    
    def analyze_prediction_confidence(self, X_test, y_test, symbol="STOCK", n_samples=200):
        """Analyze prediction confidence using transformer uncertainty"""
        predictions = self.ensemble.predict(X_test[:n_samples])
        
        # Calculate confidence intervals
        transformer_lower = predictions['transformer_lower']
        transformer_upper = predictions['transformer_upper']
        transformer_pred = predictions['transformer']
        
        confidence_width = transformer_upper - transformer_lower
        actual_values = y_test[:n_samples].flatten()
        
        # Check if actual values fall within confidence intervals
        within_ci = ((actual_values >= transformer_lower) & (actual_values <= transformer_upper))
        coverage = np.mean(within_ci)
        
        # Plot confidence analysis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Predictions with confidence intervals
        indices = range(len(transformer_pred))
        ax1.plot(indices, actual_values, label='Actual', color='black', linewidth=2)
        ax1.plot(indices, transformer_pred, label='Transformer Prediction', color='red', alpha=0.8)
        ax1.fill_between(indices, transformer_lower, transformer_upper, 
                        alpha=0.3, color='red', label='Confidence Interval')
        ax1.set_title(f'Prediction Confidence Analysis (Coverage: {coverage:.1%})')
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Scaled Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence width distribution
        ax2.hist(confidence_width, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(np.mean(confidence_width), color='red', linestyle='--', 
                   label=f'Mean Width: {np.mean(confidence_width):.4f}')
        ax2.set_title('Distribution of Confidence Interval Widths')
        ax2.set_xlabel('Confidence Interval Width')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'visualizations/{symbol}_confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()  # Don't show, just save
        
        return {
            'coverage': coverage,
            'mean_width': np.mean(confidence_width),
            'median_width': np.median(confidence_width)
        }

def create_feature_correlation_heatmap(enhanced_df, feature_cols):
    """Create correlation heatmap of features"""
    # Calculate correlation matrix
    correlation_matrix = enhanced_df[feature_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(15, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()  # Don't show, just save
    
    return correlation_matrix

if __name__ == "__main__":
    print("Feature Analysis Tools Ready!")
    print("Use these tools to understand:")
    print("1. Which features the model finds most important")
    print("2. How confident the model is in its predictions") 
    print("3. Feature correlations and relationships")
    print("4. Category-wise feature importance")
