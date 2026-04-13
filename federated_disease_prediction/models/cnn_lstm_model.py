"""
CNN-LSTM Hybrid Model for Disease Outbreak Prediction

This module implements a hybrid CNN-LSTM architecture that combines
spatial feature extraction (CNN) with temporal modeling (LSTM) for
spatio-temporal disease outbreak forecasting.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM hybrid model for spatio-temporal disease prediction.
    
    Architecture:
    1. CNN layers for spatial feature extraction from multi-variate time series
    2. LSTM layers for temporal modeling
    3. Fully connected layers for prediction
    """
    
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        cnn_filters: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [3, 3, 3],
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.3,
        use_attention: bool = True,
        pooling: str = 'adaptive'
    ):
        """
        Initialize CNN-LSTM model.
        
        Args:
            input_size: Number of input features
            sequence_length: Length of input sequences
            cnn_filters: List of filter sizes for CNN layers
            kernel_sizes: List of kernel sizes for CNN layers
            lstm_hidden: Hidden size for LSTM
            lstm_layers: Number of LSTM layers
            output_size: Number of output features
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
            pooling: Pooling type ('max', 'avg', 'adaptive', or None)
        """
        super(CNNLSTMModel, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.use_attention = use_attention
        
        # CNN layers for spatial feature extraction
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        
        in_channels = input_size
        for i, (out_channels, kernel_size) in enumerate(zip(cnn_filters, kernel_sizes)):
            # Convolutional layer
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )
            )
            
            # Batch normalization
            self.bn_layers.append(nn.BatchNorm1d(out_channels))
            
            # Pooling
            if pooling == 'max':
                self.pool_layers.append(nn.MaxPool1d(2))
            elif pooling == 'avg':
                self.pool_layers.append(nn.AvgPool1d(2))
            elif pooling == 'adaptive':
                self.pool_layers.append(nn.AdaptiveAvgPool1d(sequence_length // (2 ** (i + 1))))
            else:
                self.pool_layers.append(nn.Identity())
            
            in_channels = out_channels
        
        # Calculate CNN output size
        self.cnn_output_channels = cnn_filters[-1]
        
        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(lstm_hidden, lstm_hidden // 2),
                nn.Tanh(),
                nn.Linear(lstm_hidden // 2, 1)
            )
        
        # Fully connected layers
        fc_input_size = lstm_hidden
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, lstm_hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 4, output_size)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor (batch_size, output_size)
        """
        # x shape: (batch, seq_len, input_size)
        batch_size = x.size(0)
        
        # Transpose for CNN: (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
            x = conv(x)
            x = bn(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = pool(x)
        
        # Transpose back for LSTM: (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # LSTM temporal modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        
        if self.use_attention:
            # Apply attention
            attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, lstm_hidden)
        else:
            # Use last hidden state
            context = hidden[-1]  # (batch, lstm_hidden)
        
        # Fully connected layers
        output = self.fc(context)
        
        return output
    
    def get_feature_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract feature maps from CNN layers for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps from each CNN layer
        """
        feature_maps = []
        
        x = x.transpose(1, 2)
        
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x)
            x = torch.relu(x)
            feature_maps.append(x.detach())
        
        return feature_maps
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor
            
        Returns:
            Attention weights
        """
        if not self.use_attention:
            raise ValueError("Model was not initialized with attention")
        
        self.eval()
        with torch.no_grad():
            # CNN feature extraction
            x_cnn = x.transpose(1, 2)
            for conv, bn, pool in zip(self.conv_layers, self.bn_layers, self.pool_layers):
                x_cnn = conv(x_cnn)
                x_cnn = bn(x_cnn)
                x_cnn = torch.relu(x_cnn)
                x_cnn = pool(x_cnn)
            
            x_cnn = x_cnn.transpose(1, 2)
            
            # LSTM
            lstm_out, _ = self.lstm(x_cnn)
            
            # Attention
            attn_weights = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_weights, dim=1)
        
        return attn_weights


class MultiScaleCNNLSTM(nn.Module):
    """
    Multi-scale CNN-LSTM with parallel convolutions at different scales.
    
    Captures patterns at multiple temporal resolutions.
    """
    
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        scales: List[int] = [3, 5, 7],
        base_filters: int = 64,
        lstm_hidden: int = 128,
        output_size: int = 1,
        dropout: float = 0.3
    ):
        super(MultiScaleCNNLSTM, self).__init__()
        
        # Multi-scale convolutional branches
        self.branches = nn.ModuleList()
        
        for kernel_size in scales:
            branch = nn.Sequential(
                nn.Conv1d(input_size, base_filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(),
                nn.Conv1d(base_filters, base_filters, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(base_filters),
                nn.ReLU(),
            )
            self.branches.append(branch)
        
        # Concatenated features from all branches
        self.feature_fusion = nn.Conv1d(
            base_filters * len(scales),
            base_filters * len(scales),
            1
        )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=base_filters * len(scales),
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Output
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_size)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale feature extraction."""
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        
        # Multi-scale feature extraction
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)
        
        # Concatenate features from all branches
        fused = torch.cat(branch_outputs, dim=1)
        fused = self.feature_fusion(fused)
        fused = torch.relu(fused)
        fused = self.dropout(fused)
        
        # Transpose for LSTM
        fused = fused.transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(fused)
        
        # Output
        output = self.fc(hidden[-1])
        
        return output


class ResidualCNNLSTM(nn.Module):
    """
    Residual CNN-LSTM with skip connections.
    
    Helps with gradient flow in deep networks.
    """
    
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        num_blocks: int = 3,
        base_filters: int = 64,
        lstm_hidden: int = 128,
        output_size: int = 1,
        dropout: float = 0.3
    ):
        super(ResidualCNNLSTM, self).__init__()
        
        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv1d(input_size, base_filters, 7, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU()
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.residual_blocks.append(
                ResidualBlock(base_filters, base_filters, dropout)
            )
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=base_filters,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )
        
        # Output
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections."""
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Transpose for LSTM
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Output
        output = self.fc(hidden[-1])
        
        return output


class ResidualBlock(nn.Module):
    """Residual block for CNN."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.3
    ):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Projection if dimensions don't match
        if in_channels != out_channels:
            self.projection = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.projection = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual
        if self.projection is not None:
            residual = self.projection(residual)
        
        out = out + residual
        out = torch.relu(out)
        
        return out
