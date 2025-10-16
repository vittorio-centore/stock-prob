import torch
import torch.nn as nn
import numpy as np

class EnhancedLSTM(nn.Module):
    """Simplified LSTM model matching transformer complexity"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super(EnhancedLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Simple LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias
                n = param.size(0)
                start, end = n // 4, n // 2
                param.data[start:end].fill_(1.)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Input normalization
        x = self.input_norm(x)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last time step
        last_output = lstm_out[:, -1, :]
        
        # Dense layers
        x = self.dropout(last_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        output = self.fc2(x)
        
        return output.squeeze(-1)

def create_enhanced_lstm(input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
    """Create a simplified LSTM model"""
    return EnhancedLSTM(input_dim, hidden_dim, num_layers, dropout)

def train_pytorch_lstm(model, train_loader, val_loader, epochs=50, lr=0.0001, device='cpu'):
    """Train the LSTM model"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)
    criterion = nn.SmoothL1Loss()
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
