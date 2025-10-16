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
        """Create a focused dashboard with main predictions and model analysis"""
        
        # Create simple 2-chart layout
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                'Stock Price Predictions vs Actual (with 3-Day Forecast)',
                'Model Performance Analysis'
            ],
            vertical_spacing=0.15,
            row_heights=[0.75, 0.25]  # Main chart bigger
        )
        
        # 1. Main prediction chart with 3-day forecast
        self._add_main_predictions_with_forecast(fig, dates, actual, predictions, row=1, col=1)
        
        # 2. Model performance analysis
        self._add_detailed_model_analysis(fig, actual, predictions, performance_metrics, row=2, col=1)
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=f"{self.symbol} Stock Prediction Dashboard",
            title_x=0.5,
            title_font_size=24,
            showlegend=True,
            template="plotly_white",
            font=dict(family="Arial, sans-serif", size=14)
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

    def _add_main_predictions_with_forecast(self, fig, dates, actual, predictions, row, col):
        """Add main prediction chart with 3-day future forecast"""
        
        # Convert dates to pandas datetime for easier manipulation
        dates_pd = pd.to_datetime(dates)
        
        # Create 3-day future dates
        last_date = dates_pd[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=3, freq='D')
        
        # Generate 3-day predictions based on recent trend
        recent_trend = np.mean(np.diff(predictions['ensemble'][-10:]))  # Last 10 days trend
        last_price = predictions['ensemble'][-1]
        
        # Project future based on trend
        future_predictions = []
        for i in range(3):
            future_pred = last_price + recent_trend * (i+1)
            future_predictions.append(future_pred)
        
        # Combine current and future dates/predictions
        all_dates = np.concatenate([dates_pd, future_dates])
        ensemble_with_future = np.concatenate([predictions['ensemble'], future_predictions])
        
        # Actual prices (only historical)
        fig.add_trace(
            go.Scatter(
                x=dates_pd,
                y=actual,
                mode='lines',
                name='Actual Price',
                line=dict(color='black', width=3),
                hovertemplate='<b>Actual</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Ensemble predictions (historical + future)
        fig.add_trace(
            go.Scatter(
                x=all_dates,
                y=ensemble_with_future,
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
                x=dates_pd,
                y=predictions['transformer'],
                mode='lines',
                name='Transformer',
                line=dict(color='red', width=2, dash='dot'),
                hovertemplate='<b>Transformer</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # LSTM predictions
        fig.add_trace(
            go.Scatter(
                x=dates_pd,
                y=predictions['lstm'],
                mode='lines',
                name='LSTM',
                line=dict(color='blue', width=2, dash='dash'),
                hovertemplate='<b>LSTM</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Confidence bands (if available)
        if 'transformer_lower' in predictions and 'transformer_upper' in predictions:
            fig.add_trace(
                go.Scatter(
                    x=dates_pd,
                    y=predictions['transformer_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=dates_pd,
                    y=predictions['transformer_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255,0,0,0.1)',
                    name='Confidence Interval',
                    hovertemplate='<b>Confidence Band</b><br>Upper: $%{y:.2f}<extra></extra>'
                ),
                row=row, col=col
            )
        
        # Add forecast period indicator
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines+markers',
                name='3-Day Forecast',
                line=dict(color='orange', width=3, dash='dash'),
                marker=dict(size=8, color='orange'),
                hovertemplate='<b>3-Day Forecast</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Add vertical line to separate historical from forecast
        fig.add_shape(
            type="line",
            x0=dates_pd[-1], x1=dates_pd[-1],
            y0=0, y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dash"),
            row=row, col=col
        )
        
        # Add annotation for forecast
        fig.add_annotation(
            x=dates_pd[-1],
            y=0.9,
            yref="paper",
            text="3-Day Forecast →",
            showarrow=False,
            font=dict(color="gray", size=12),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=row, col=col)
        fig.update_yaxes(title_text="Stock Price ($)", row=row, col=col)
    
    def _add_detailed_model_analysis(self, fig, actual, predictions, performance_metrics, row, col):
        """Add detailed model performance analysis"""
        
        # Calculate errors for each model
        ensemble_error = np.abs(actual - predictions['ensemble'])
        transformer_error = np.abs(actual - predictions['transformer'])
        lstm_error = np.abs(actual - predictions['lstm'])
        
        # Model names and their MAE values
        models = ['Ensemble', 'Transformer', 'LSTM']
        mae_values = [
            np.mean(ensemble_error),
            np.mean(transformer_error), 
            np.mean(lstm_error)
        ]
        
        # Color coding: green for best, red for worst
        colors = ['purple', 'red', 'blue']
        
        # Create bar chart
        fig.add_trace(
            go.Bar(
                x=models,
                y=mae_values,
                name='Mean Absolute Error',
                marker_color=colors,
                text=[f'${mae:.2f}' for mae in mae_values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>MAE: $%{y:.2f}<br>Lower is better<extra></extra>'
            ),
            row=row, col=col
        )
        
        # Update axes
        fig.update_xaxes(title_text="Model", row=row, col=col)
        fig.update_yaxes(title_text="Mean Absolute Error ($)", row=row, col=col)

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
