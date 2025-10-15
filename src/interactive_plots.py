import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import os

class InteractiveVisualizer:
    """Create interactive Plotly visualizations for stock prediction results"""
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.output_dir = "visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def create_prediction_dashboard(self, dates, actual, predictions, feature_cols, performance_metrics):
        """Create a comprehensive interactive dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Stock Price Predictions vs Actual',
                'Prediction Errors Over Time', 
                'Model Performance Comparison',
                'Uncertainty Bands (Transformer)',
                'Feature Importance (Top 10)',
                'Prediction Accuracy Distribution'
            ],
            specs=[
                [{"colspan": 2}, None],  # Full width for main chart
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "histogram"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.1
        )
        
        # 1. Main prediction chart (full width)
        self._add_main_predictions(fig, dates, actual, predictions, row=1, col=1)
        
        # 2. Prediction errors
        self._add_error_analysis(fig, dates, actual, predictions, row=2, col=1)
        
        # 3. Model comparison
        self._add_model_comparison(fig, performance_metrics, row=2, col=2)
        
        # 4. Uncertainty bands (if available)
        if 'transformer_lower' in predictions and 'transformer_upper' in predictions:
            self._add_uncertainty_bands(fig, dates, actual, predictions, row=3, col=1)
        
        # 5. Feature importance (mock data for now - will be real when feature analysis works)
        self._add_feature_importance(fig, feature_cols, row=3, col=2)
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"{self.symbol} Stock Prediction Dashboard",
            title_x=0.5,
            title_font_size=24,
            showlegend=True,
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=12)
        )
        
        # Save as HTML
        html_file = os.path.join(self.output_dir, f"{self.symbol}_dashboard.html")
        fig.write_html(html_file)
        print(f"📊 Interactive dashboard saved: {html_file}")
        
        return fig
    
    def _add_main_predictions(self, fig, dates, actual, predictions, row, col):
        """Add main prediction vs actual chart"""
        
        # Actual prices
        fig.add_trace(
            go.Scatter(
                x=dates, y=actual,
                mode='lines',
                name='Actual Price',
                line=dict(color='black', width=2),
                hovertemplate='<b>Actual</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Ensemble predictions
        fig.add_trace(
            go.Scatter(
                x=dates, y=predictions['ensemble'],
                mode='lines',
                name='Ensemble Prediction',
                line=dict(color='purple', width=2),
                hovertemplate='<b>Ensemble</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Transformer predictions
        fig.add_trace(
            go.Scatter(
                x=dates, y=predictions['transformer'],
                mode='lines',
                name='Transformer',
                line=dict(color='red', width=1.5, dash='dot'),
                hovertemplate='<b>Transformer</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # LSTM predictions
        fig.add_trace(
            go.Scatter(
                x=dates, y=predictions['lstm'],
                mode='lines',
                name='LSTM',
                line=dict(color='blue', width=1.5, dash='dash'),
                hovertemplate='<b>LSTM</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add uncertainty bands if available
        if 'transformer_lower' in predictions and 'transformer_upper' in predictions:
            fig.add_trace(
                go.Scatter(
                    x=dates, y=predictions['transformer_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            
            fig.add_trace(
                go.Scatter(
                    x=dates, y=predictions['transformer_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.2)',
                    name='Confidence Interval',
                    hovertemplate='<b>Confidence Band</b><br>Date: %{x}<br>Upper: $%{y:.2f}<extra></extra>'
                ),
                row=row, col=col
            )
    
    def _add_error_analysis(self, fig, dates, actual, predictions, row, col):
        """Add prediction error analysis"""
        
        ensemble_error = np.abs(actual - predictions['ensemble'])
        transformer_error = np.abs(actual - predictions['transformer'])
        lstm_error = np.abs(actual - predictions['lstm'])
        
        fig.add_trace(
            go.Scatter(
                x=dates, y=ensemble_error,
                mode='lines',
                name='Ensemble Error',
                line=dict(color='purple', width=2),
                hovertemplate='<b>Ensemble Error</b><br>Date: %{x}<br>Error: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates, y=transformer_error,
                mode='lines',
                name='Transformer Error',
                line=dict(color='red', width=1.5),
                hovertemplate='<b>Transformer Error</b><br>Date: %{x}<br>Error: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates, y=lstm_error,
                mode='lines',
                name='LSTM Error',
                line=dict(color='blue', width=1.5),
                hovertemplate='<b>LSTM Error</b><br>Date: %{x}<br>Error: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    def _add_model_comparison(self, fig, performance_metrics, row, col):
        """Add model performance comparison"""
        
        models = ['Ensemble', 'Transformer', 'LSTM']
        mae_values = [
            performance_metrics.get('ensemble_mae', 0),
            performance_metrics.get('transformer_mae', 0),
            performance_metrics.get('lstm_mae', 0)
        ]
        
        colors = ['purple', 'red', 'blue']
        
        fig.add_trace(
            go.Bar(
                x=models, y=mae_values,
                name='MAE Comparison',
                marker_color=colors,
                hovertemplate='<b>%{x}</b><br>MAE: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    def _add_uncertainty_bands(self, fig, dates, actual, predictions, row, col):
        """Add detailed uncertainty analysis"""
        
        # Calculate coverage (how often actual falls within bands)
        within_bands = (
            (actual >= predictions['transformer_lower']) & 
            (actual <= predictions['transformer_upper'])
        )
        coverage = np.mean(within_bands) * 100
        
        # Plot coverage over time (rolling window)
        window_size = 20
        rolling_coverage = pd.Series(within_bands.astype(int)).rolling(window_size).mean() * 100
        
        fig.add_trace(
            go.Scatter(
                x=dates, y=rolling_coverage,
                mode='lines',
                name=f'Coverage (avg: {coverage:.1f}%)',
                line=dict(color='green', width=2),
                hovertemplate='<b>Coverage</b><br>Date: %{x}<br>Coverage: %{y:.1f}%<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add horizontal line at 80% (target coverage)
        fig.add_hline(y=80, line_dash="dash", line_color="gray", 
                     annotation_text="Target: 80%", row=row, col=col)
    
    def _add_feature_importance(self, fig, feature_cols, row, col):
        """Add feature importance (simulated for now)"""
        
        # Simulate feature importance (in real implementation, this would come from feature analysis)
        np.random.seed(42)  # Consistent results
        importance_scores = np.random.exponential(0.1, len(feature_cols))
        
        # Sort features by importance
        feature_importance = list(zip(feature_cols, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 10
        top_features = feature_importance[:10]
        features, scores = zip(*top_features)
        
        fig.add_trace(
            go.Bar(
                y=list(features)[::-1],  # Reverse for better display
                x=list(scores)[::-1],
                orientation='h',
                name='Feature Importance',
                marker_color='lightblue',
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    def create_simple_prediction_chart(self, dates, actual, predictions):
        """Create a simple, focused prediction chart"""
        
        fig = go.Figure()
        
        # Actual prices
        fig.add_trace(go.Scatter(
            x=dates, y=actual,
            mode='lines',
            name='Actual Price',
            line=dict(color='black', width=3),
            hovertemplate='<b>Actual</b><br>%{x}<br>$%{y:.2f}<extra></extra>'
        ))
        
        # Ensemble predictions
        fig.add_trace(go.Scatter(
            x=dates, y=predictions['ensemble'],
            mode='lines',
            name='AI Prediction',
            line=dict(color='purple', width=3),
            hovertemplate='<b>AI Prediction</b><br>%{x}<br>$%{y:.2f}<extra></extra>'
        ))
        
        # Confidence bands
        if 'transformer_lower' in predictions and 'transformer_upper' in predictions:
            fig.add_trace(go.Scatter(
                x=dates, y=predictions['transformer_upper'],
                mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=dates, y=predictions['transformer_lower'],
                mode='lines', line=dict(width=0),
                fill='tonexty', fillcolor='rgba(128,0,128,0.2)',
                name='Confidence Zone',
                hovertemplate='<b>Confidence Zone</b><br>%{x}<br>$%{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title=f'{self.symbol} Stock Price Prediction',
            title_x=0.5,
            title_font_size=20,
            xaxis_title='Date',
            yaxis_title='Stock Price ($)',
            template='plotly_white',
            height=600,
            hovermode='x unified'
        )
        
        # Save as HTML
        html_file = os.path.join(self.output_dir, f"{self.symbol}_simple.html")
        fig.write_html(html_file)
        print(f"📈 Simple chart saved: {html_file}")
        
        return fig

def create_visualization_summary(symbol, performance_metrics):
    """Create a text summary of results"""
    
    summary = f"""
# {symbol} Stock Prediction Results

## Model Performance
- **Ensemble MAE**: ${performance_metrics.get('ensemble_mae', 0):.2f}
- **Transformer MAE**: ${performance_metrics.get('transformer_mae', 0):.2f}  
- **LSTM MAE**: ${performance_metrics.get('lstm_mae', 0):.2f}

## Improvements
- **Ensemble vs Transformer**: {performance_metrics.get('ensemble_vs_transformer', 0):.1f}% better
- **Ensemble vs LSTM**: {performance_metrics.get('ensemble_vs_lstm', 0):.1f}% better

## Files Generated
- `{symbol}_dashboard.html` - Complete interactive dashboard
- `{symbol}_simple.html` - Simple prediction chart

## How to View
1. Open the HTML files in any web browser
2. Interactive features: hover, zoom, pan
3. No internet connection required
"""
    
    with open(f"visualizations/{symbol}_results.md", "w") as f:
        f.write(summary)
    
    print(f"📄 Results summary saved: visualizations/{symbol}_results.md")

if __name__ == "__main__":
    print("🎨 Interactive Plotly Visualizer Ready!")
    print("Features:")
    print("- Interactive charts with hover tooltips")
    print("- Zoom and pan functionality") 
    print("- Standalone HTML files (no server needed)")
    print("- Professional styling")
    print("- Comprehensive dashboard + simple chart")
    print("- Uncertainty visualization")
    print("- Model comparison charts")
