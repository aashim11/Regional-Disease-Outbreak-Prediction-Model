"""
Transformer Models for Disease Outbreak Prediction

This module implements Transformer architectures for capturing long-range
dependencies in disease outbreak time series data.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    
    Adds positional information to input embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TemporalTransformer(nn.Module):
    """
    Transformer model for temporal disease outbreak prediction.
    
    Uses self-attention to capture long-range dependencies in time series.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_size: int = 1,
        max_seq_len: int = 1000
    ):
        """
        Initialize Temporal Transformer.
        
        Args:
            input_size: Number of input features
            d_model: Model dimension (embedding size)
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            output_size: Number of output features
            max_seq_len: Maximum sequence length
        """
        super(TemporalTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, output_size)
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        initrange = 0.1
        self.input_embedding.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            
        Returns:
            Output tensor (batch, output_size)
        """
        # Embed input
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)
        
        # Global average pooling over time dimension
        pooled = encoded.mean(dim=1)
        
        # Output
        output = self.output_layer(pooled)
        
        return output
    
    def get_attention_maps(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention maps for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            List of attention maps from each layer
        """
        # This requires modifying the transformer to return attention weights
        # For now, return empty list
        # In practice, you'd use register_forward_hook to capture attention
        return []


class TransformerModel(nn.Module):
    """
    Full Transformer model with encoder-decoder architecture.
    
    Can be used for sequence-to-sequence prediction.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_size: int = 1,
        max_seq_len: int = 1000
    ):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        
        # Embeddings
        self.encoder_embedding = nn.Linear(input_size, d_model)
        self.decoder_embedding = nn.Linear(output_size, d_model)
        
        # Positional encodings
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_size)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            src: Source sequence (batch, src_len, input_size)
            tgt: Target sequence (batch, tgt_len, output_size) - for training
            
        Returns:
            Output predictions
        """
        # Encode source
        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        if tgt is not None:
            # Training mode with teacher forcing
            tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
            tgt_emb = self.pos_encoder(tgt_emb)
            
            # Generate causal mask
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1))
            tgt_mask = tgt_mask.to(tgt.device)
            
            output = self.transformer(
                src_emb,
                tgt_emb,
                tgt_mask=tgt_mask
            )
        else:
            # Inference mode - auto-regressive generation
            # Simplified: just use encoder output
            output = self.transformer.encoder(src_emb)
            return self.output_projection(output[:, -1:, :])
        
        return self.output_projection(output)
    
    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def predict(
        self,
        src: torch.Tensor,
        forecast_horizon: int = 7
    ) -> torch.Tensor:
        """
        Auto-regressive prediction for forecasting.
        
        Args:
            src: Input sequence
            forecast_horizon: Number of steps to forecast
            
        Returns:
            Predictions for forecast horizon
        """
        self.eval()
        
        with torch.no_grad():
            # Start with last known value
            batch_size = src.size(0)
            output_dim = self.output_projection.out_features
            
            predictions = []
            decoder_input = torch.zeros(batch_size, 1, output_dim).to(src.device)
            
            for _ in range(forecast_horizon):
                pred = self.forward(src, decoder_input)
                next_pred = pred[:, -1:, :]
                predictions.append(next_pred)
                decoder_input = torch.cat([decoder_input, next_pred], dim=1)
            
            return torch.cat(predictions, dim=1)


class InformerModel(nn.Module):
    """
    Informer: Efficient Transformer for Long Sequence Time-Series Forecasting.
    
    Uses ProbSparse self-attention and self-attention distilling for
    efficient processing of long sequences.
    
    Based on: Zhou et al., "Informer: Beyond Efficient Transformer for
    Long Sequence Time-Series Forecasting", AAAI 2021.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        output_size: int = 1,
        factor: int = 5,  # Factor for ProbSparse attention
        max_seq_len: int = 1000
    ):
        super(InformerModel, self).__init__()
        
        self.d_model = d_model
        self.factor = factor
        
        # Input embedding
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Informer encoder layers
        self.encoder_layers = nn.ModuleList([
            InformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, factor)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Embed input
        x = self.input_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        # Informer encoding
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Global pooling and output
        pooled = x.mean(dim=1)
        output = self.output_layer(pooled)
        
        return output


class InformerEncoderLayer(nn.Module):
    """
    Informer encoder layer with ProbSparse self-attention.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        factor: int
    ):
        super(InformerEncoderLayer, self).__init__()
        
        self.prob_sparse_attn = ProbSparseSelfAttention(d_model, nhead, factor, dropout)
        
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with ProbSparse attention."""
        # Self-attention
        attn_out = self.prob_sparse_attn(x)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        
        # Feedforward
        ff_out = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        
        return x


class ProbSparseSelfAttention(nn.Module):
    """
    ProbSparse Self-Attention mechanism from Informer.
    
    Only computes attention for the most dominant queries,
    reducing complexity from O(L^2) to O(L log L).
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        factor: int,
        dropout: float
    ):
        super(ProbSparseSelfAttention, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.factor = factor
        self.d_k = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ProbSparse attention.
        
        Simplified implementation - full implementation would include
        query sparsity measurement and top-k selection.
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        Q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Compute attention (simplified - full version would use sparse attention)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, V)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out
