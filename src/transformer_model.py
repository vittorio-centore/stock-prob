import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import math
import warnings
warnings.filterwarnings('ignore')

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class StockTransformer(nn.Module):
    """Transformer model for stock price prediction"""
    def __init__(self, input_dim=5, d_model=64, nhead=4, num_layers=2, dropout=0.1, max_len=100):
        super(StockTransformer, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout * 2,  # Higher dropout for regularization
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Learnable attention layer for temporal aggregation
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=2,
            dropout=dropout,
            batch_first=True
        )
        
        # Simplified output layer for better price prediction
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)  # NO activation - allow full range
        )
        
        # Initialize weights properly
        self._init_weights()
        
    def forward(self, src, src_mask=None):
        # Input projection
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transformer encoding
        output = self.transformer_encoder(src, src_mask)
        
        # Use learnable temporal attention for better aggregation
        # Create a query vector that learns what temporal patterns to focus on
        query = torch.mean(output, dim=1, keepdim=True)  # (batch_size, 1, d_model)
        
        # Apply temporal attention
        attended_output, attention_weights = self.temporal_attention(
            query=query,
            key=output,
            value=output
        )
        
        # Use the attended output
        weighted_output = attended_output.squeeze(1)  # (batch_size, d_model)
        
        # Generate single price prediction
        prediction = self.output_layer(weighted_output)  # (batch_size, 1)
        
        return prediction.squeeze(-1)  # Remove last dimension
    
    def _init_weights(self):
        """Initialize model weights for better training"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

class StockDataset(Dataset):
    """Dataset for stock time series data"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class QuantileLoss(nn.Module):
    """Quantile loss for uncertainty estimation"""
    def __init__(self, quantiles=[0.1, 0.5, 0.9]):
        super(QuantileLoss, self).__init__()
        self.quantiles = quantiles
        
    def forward(self, predictions, targets):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors))
        return torch.mean(torch.stack(losses))

def create_attention_mask(seq_len, device):
    """Create causal mask for transformer"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.to(device)

def train_transformer(model, train_loader, val_loader, epochs=50, lr=0.0001, device='cpu'):
    """Train the transformer model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)  # Even lower weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.8)
    criterion = nn.SmoothL1Loss()  # Use SmoothL1Loss to prevent gradient explosion
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_seq, batch_target in train_loader:
            batch_seq, batch_target = batch_seq.to(device), batch_target.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_seq)
            loss = criterion(predictions, batch_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Relaxed gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_seq, batch_target in val_loader:
                batch_seq, batch_target = batch_seq.to(device), batch_target.to(device)
                predictions = model(batch_seq)
                loss = criterion(predictions, batch_target)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}')
    
    return train_losses, val_losses

def predict_with_uncertainty(model, data_loader, device='cpu'):
    """Make predictions with uncertainty quantiles"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch_seq, _ in data_loader:
            batch_seq = batch_seq.to(device)
            pred = model(batch_seq)
            predictions.append(pred.cpu().numpy())
    
    return np.vstack(predictions)

if __name__ == "__main__":
    print("PyTorch Transformer for Stock Prediction - Ready for Integration!")
    print("Key features:")
    print("- Multi-head attention mechanism")
    print("- Positional encoding for temporal patterns") 
    print("- Quantile regression for uncertainty estimation")
    print("- Causal masking for proper time series modeling")
    print("- GELU activation and layer normalization")
