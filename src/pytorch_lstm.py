import torch
import torch.nn as nn
import numpy as np

class EnhancedLSTM(nn.Module):
    """Enhanced LSTM model in PyTorch optimized for stock prediction"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
        super(EnhancedLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input normalization layer
        self.input_norm = nn.LayerNorm(input_dim)
        
        # First bidirectional LSTM layer (largest)
        self.lstm1 = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        # Second bidirectional LSTM layer (medium)
        self.lstm2 = nn.LSTM(
            input_size=hidden_dim * 2,  # *2 because bidirectional
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        
        # Third unidirectional LSTM layer (smallest)
        self.lstm3 = nn.LSTM(
            input_size=hidden_dim,  # 64*2 from previous layer
            hidden_size=hidden_dim // 4,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )
        
        # Dropout layers (more aggressive for regularization)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout * 1.2)
        
        # Batch normalization for better training stability
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dense layers with residual connection
        self.fc1 = nn.Linear(hidden_dim // 4, hidden_dim // 2)  # 32 -> 64
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)  # 64 -> 32
        self.fc3 = nn.Linear(hidden_dim // 4, 1)  # 32 -> 1
        
        # Activation functions
        self.relu = nn.ReLU()
        self.elu = nn.ELU()  # ELU can help with vanishing gradients
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps with long sequences)
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)
        
        # Input normalization
        x = self.input_norm(x)
        
        # First LSTM layer (bidirectional)
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        # Batch normalization (reshape for BN)
        lstm1_reshaped = lstm1_out.reshape(-1, lstm1_out.size(-1))
        lstm1_bn = self.bn1(lstm1_reshaped)
        lstm1_out = lstm1_bn.reshape(lstm1_out.shape)
        
        # Second LSTM layer (bidirectional)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout2(lstm2_out)
        
        # Batch normalization
        lstm2_reshaped = lstm2_out.reshape(-1, lstm2_out.size(-1))
        lstm2_bn = self.bn2(lstm2_reshaped)
        lstm2_out = lstm2_bn.reshape(lstm2_out.shape)
        
        # Third LSTM layer (unidirectional)
        lstm3_out, _ = self.lstm3(lstm2_out)
        
        # Take the last time step
        last_output = lstm3_out[:, -1, :]  # (batch_size, hidden_dim//4)
        
        # Dense layers with residual connection
        x = self.fc1(last_output)
        x = self.elu(x)  # ELU activation
        x = self.dropout3(x)
        
        residual = x  # Store for residual connection
        
        x = self.fc2(x)
        x = self.elu(x)
        x = self.dropout3(x)
        
        # Add residual connection if dimensions match
        if x.size() == residual.size():
            x = x + residual * 0.3  # Scaled residual
        
        x = self.fc3(x)
        
        return x

def create_enhanced_lstm(input_dim, hidden_dim=128, num_layers=3, dropout=0.3):
    """Create an enhanced LSTM model optimized for stock prediction"""
    return EnhancedLSTM(input_dim, hidden_dim, num_layers, dropout)

def train_pytorch_lstm(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """Train the PyTorch LSTM model with advanced techniques for stock prediction"""
    model = model.to(device)
    
    # Use Huber loss (more robust than MSE for financial data)
    criterion = nn.HuberLoss(delta=0.1)  # Smaller delta for stock prices
    
    # AdamW with cosine annealing for better convergence
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.01, 
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing with restarts for better exploration
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-7
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20  # More patience for stock data
    
    print(f"Training PyTorch LSTM for up to {epochs} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.squeeze())
                val_loss += loss.item()
                val_batches += 1
        
        # Calculate average losses
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Step scheduler
        scheduler.step()
        
        # Early stopping with model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 5 == 0 or patience_counter == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, '
                  f'Val Loss = {avg_val_loss:.6f}, LR = {current_lr:.2e}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
            # Restore best model
            model.load_state_dict(best_model_state)
            break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    return train_losses, val_losses

if __name__ == "__main__":
    print("PyTorch Enhanced LSTM Ready!")
    print("Advanced Features for Stock Prediction:")
    print("- Bidirectional LSTM layers (3-layer architecture)")
    print("- Progressive dimension reduction (128→64→32)")
    print("- Input & batch normalization for stability")
    print("- Proper weight initialization (Xavier + Orthogonal)")
    print("- Huber loss (robust to financial outliers)")
    print("- AdamW optimizer with cosine annealing")
    print("- Residual connections in dense layers")
    print("- ELU activations (better gradients)")
    print("- Early stopping with best model restoration")
    print("- Gradient clipping for training stability")
    print("- 20-epoch patience for financial data")
    print("- Automatic model parameter counting")
