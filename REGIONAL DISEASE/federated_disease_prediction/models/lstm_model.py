"""
LSTM-based Models for Disease Outbreak Prediction

This module implements Long Short-Term Memory (LSTM) networks for
time-series forecasting of disease outbreaks.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any


class LSTMModel(nn.Module):
    """
    LSTM model for disease outbreak prediction.
    
    Predicts outbreak risk based on sequential health and environmental data.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = False,
        attention: bool = False
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden state
            num_layers: Number of LSTM layers
            output_size: Number of output features (1 for binary classification)
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
            attention: Whether to use attention mechanism
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attention = attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Calculate output dimension
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        
        # Attention mechanism
        if attention:
            self.attention_layer = nn.Sequential(
                nn.Linear(lstm_output_size, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def attention_weights(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Compute attention weights over time steps.
        
        Args:
            lstm_output: Output from LSTM (batch, seq_len, hidden_size)
            
        Returns:
            Attention weights (batch, seq_len, 1)
        """
        weights = self.attention_layer(lstm_output)
        weights = torch.softmax(weights, dim=1)
        return weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            Output tensor (batch, output_size)
        """
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.attention:
            # Apply attention
            attn_weights = self.attention_weights(lstm_out)
            context = torch.sum(lstm_out * attn_weights, dim=1)
        else:
            # Use last hidden state
            if self.bidirectional:
                # Concatenate forward and backward hidden states
                context = torch.cat((hidden[-2], hidden[-1]), dim=1)
            else:
                context = hidden[-1]
        
        # Fully connected layers
        output = self.fc(context)
        
        return output
    
    def get_attention_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights
        """
        if not self.attention:
            raise ValueError("Model was not initialized with attention")
        
        self.eval()
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            attn_weights = self.attention_weights(lstm_out)
        
        return attn_weights


class BiLSTMModel(LSTMModel):
    """Bidirectional LSTM model."""
    
    def __init__(self, *args, **kwargs):
        kwargs['bidirectional'] = True
        super().__init__(*args, **kwargs)


class StackedLSTM(nn.Module):
    """
    Deep stacked LSTM with residual connections.
    
    Suitable for capturing complex temporal patterns in disease data.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int] = [128, 128, 64],
        output_size: int = 1,
        dropout: float = 0.2,
        use_residual: bool = True
    ):
        super(StackedLSTM, self).__init__()
        
        self.use_residual = use_residual and len(hidden_sizes) > 1
        
        # Build LSTM layers
        self.lstm_layers = nn.ModuleList()
        
        for i, hidden_size in enumerate(hidden_sizes):
            input_dim = input_size if i == 0 else hidden_sizes[i-1]
            
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0  # No dropout for single layer
                )
            )
        
        # Projection layers for residual connections
        if self.use_residual:
            self.projections = nn.ModuleList()
            for i in range(len(hidden_sizes) - 1):
                if hidden_sizes[i] != hidden_sizes[i+1]:
                    self.projections.append(
                        nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
                    )
                else:
                    self.projections.append(None)
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_sizes[-1], hidden_sizes[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_sizes[-1] // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        current = x
        
        for i, lstm in enumerate(self.lstm_layers):
            lstm_out, _ = lstm(current)
            
            # Take last time step
            current = lstm_out[:, -1, :]
            
            # Residual connection
            if self.use_residual and i > 0:
                if self.projections[i-1] is not None:
                    residual = self.projections[i-1](prev_output)
                else:
                    residual = prev_output
                
                if residual.shape == current.shape:
                    current = current + residual
            
            prev_output = current
            current = self.dropout(current)
        
        output = self.fc(current)
        return output


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for anomaly detection in disease data.
    
    Can detect unusual patterns that might indicate outbreak onset.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        latent_size: int = 32,
        dropout: float = 0.2
    ):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.encoder_fc = nn.Linear(hidden_size, latent_size)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_size, hidden_size)
        
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        _, (hidden, _) = self.encoder(x)
        latent = self.encoder_fc(hidden[-1])
        return latent
    
    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode latent representation to sequence."""
        hidden = self.decoder_fc(latent)
        hidden = hidden.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
        
        # Initialize input for decoder
        batch_size = latent.size(0)
        decoder_input = torch.zeros(batch_size, seq_len, self.decoder.hidden_size)
        decoder_input = decoder_input.to(latent.device)
        
        output, _ = self.decoder(decoder_input, (hidden, torch.zeros_like(hidden)))
        return output
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through autoencoder.
        
        Returns:
            Tuple of (reconstructed sequence, latent representation)
        """
        latent = self.encode(x)
        reconstructed = self.decode(latent, x.size(1))
        return reconstructed, latent
    
    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error for anomaly detection."""
        reconstructed, _ = self.forward(x)
        error = torch.mean((x - reconstructed) ** 2, dim=(1, 2))
        return error


class TemporalConvLSTM(nn.Module):
    """
    Temporal Convolutional LSTM.
    
    Combines temporal convolution for local feature extraction
    with LSTM for long-term dependencies.
    """
    
    def __init__(
        self,
        input_size: int,
        num_filters: List[int] = [64, 128],
        kernel_sizes: List[int] = [3, 3],
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(TemporalConvLSTM, self).__init__()
        
        # Temporal convolutional layers
        conv_layers = []
        in_channels = input_size
        
        for out_channels, kernel_size in zip(num_filters, kernel_sizes):
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (batch, seq_len, input_size)
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # Back to (batch, seq_len, channels)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use last hidden state
        output = self.fc(hidden[-1])
        
        return output
