"""
Graph Neural Network Models for Disease Outbreak Prediction

This module implements GNN architectures for modeling spatial spread
of diseases across geographic regions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any
import numpy as np


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer.
    
    Basic GCN layer that aggregates information from neighboring nodes.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (batch, num_nodes, num_nodes)
            
        Returns:
            Updated node features (batch, num_nodes, out_features)
        """
        # Linear transformation
        support = torch.matmul(x, self.weight)
        
        # Graph convolution: aggregate neighbor features
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GNNModel(nn.Module):
    """
    Graph Neural Network for regional disease outbreak prediction.
    
    Models disease spread across geographic regions using graph structure.
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_features: List[int] = [128, 128],
        out_features: int = 1,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        """
        Initialize GNN model.
        
        Args:
            num_nodes: Number of regions (nodes) in the graph
            in_features: Number of input features per node
            hidden_features: List of hidden layer sizes
            out_features: Number of output features per node
            dropout: Dropout rate
            use_attention: Whether to use graph attention
        """
        super(GNNModel, self).__init__()
        
        self.num_nodes = num_nodes
        self.use_attention = use_attention
        
        # Graph convolutional layers
        self.conv_layers = nn.ModuleList()
        
        layer_sizes = [in_features] + hidden_features
        for i in range(len(layer_sizes) - 1):
            if use_attention:
                self.conv_layers.append(
                    GraphAttentionLayer(
                        layer_sizes[i],
                        layer_sizes[i + 1],
                        dropout=dropout
                    )
                )
            else:
                self.conv_layers.append(
                    GraphConvLayer(layer_sizes[i], layer_sizes[i + 1])
                )
        
        # Temporal modeling (LSTM for each node)
        self.temporal_encoder = nn.LSTM(
            input_size=hidden_features[-1],
            hidden_size=hidden_features[-1] // 2,
            num_layers=1,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_features[-1] // 2, hidden_features[-1] // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features[-1] // 4, out_features)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features (batch, num_nodes, seq_len, in_features)
            adj: Adjacency matrix (batch, num_nodes, num_nodes) or (num_nodes, num_nodes)
            
        Returns:
            Predictions (batch, num_nodes, out_features)
        """
        batch_size, num_nodes, seq_len, in_features = x.shape
        
        # Process each time step through GNN
        gnn_outputs = []
        
        for t in range(seq_len):
            node_features = x[:, :, t, :]  # (batch, num_nodes, in_features)
            
            # Apply GNN layers
            for conv in self.conv_layers:
                if self.use_attention:
                    node_features = F.elu(conv(node_features, adj))
                else:
                    node_features = F.relu(conv(node_features, adj))
                node_features = F.dropout(node_features, p=0.3, training=self.training)
            
            gnn_outputs.append(node_features)
        
        # Stack temporal GNN outputs
        gnn_sequence = torch.stack(gnn_outputs, dim=2)  # (batch, num_nodes, seq_len, hidden)
        
        # Reshape for LSTM: (batch * num_nodes, seq_len, hidden)
        gnn_sequence = gnn_sequence.view(batch_size * num_nodes, seq_len, -1)
        
        # Temporal encoding
        lstm_out, (hidden, _) = self.temporal_encoder(gnn_sequence)
        
        # Use last hidden state
        temporal_features = hidden[-1]  # (batch * num_nodes, hidden//2)
        
        # Output prediction
        output = self.output_layer(temporal_features)
        
        # Reshape back: (batch, num_nodes, out_features)
        output = output.view(batch_size, num_nodes, -1)
        
        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).
    
    Uses attention mechanisms to weight the importance of neighbor nodes.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.3,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Learnable parameters
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention.
        
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (batch, num_nodes, num_nodes)
            
        Returns:
            Updated node features (batch, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = x.shape
        
        # Linear transformation
        Wh = torch.matmul(x, self.W)  # (batch, num_nodes, out_features)
        
        # Compute attention coefficients
        # Repeat Wh for all nodes to compute pairwise attention
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # (batch, num_nodes, num_nodes, out_features)
        Wh2 = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)  # (batch, num_nodes, num_nodes, out_features)
        
        # Concatenate and compute attention scores
        e = torch.cat([Wh1, Wh2], dim=-1)  # (batch, num_nodes, num_nodes, 2*out_features)
        e = self.leakyrelu(torch.matmul(e, self.a).squeeze(-1))  # (batch, num_nodes, num_nodes)
        
        # Mask attention scores based on adjacency
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # (batch, num_nodes, out_features)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class SpatioTemporalGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network.
    
    Combines spatial graph convolutions with temporal convolutions
    for spatio-temporal disease spread modeling.
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        hidden_features: int = 64,
        num_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.3,
        output_size: int = 1
    ):
        super(SpatioTemporalGNN, self).__init__()
        
        self.num_nodes = num_nodes
        
        # Temporal convolution layers
        self.temporal_convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = in_features if i == 0 else hidden_features
            self.temporal_convs.append(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=hidden_features,
                    kernel_size=(1, kernel_size),
                    padding=(0, kernel_size // 2)
                )
            )
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList()
        for i in range(num_layers):
            self.graph_convs.append(
                GraphConvLayer(hidden_features, hidden_features)
            )
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(hidden_features) for _ in range(num_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, output_size)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (batch, num_nodes, seq_len, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Predictions (batch, num_nodes, output_size)
        """
        batch_size, num_nodes, seq_len, in_features = x.shape
        
        # Reshape for temporal convolution: (batch, in_features, num_nodes, seq_len)
        x = x.permute(0, 3, 1, 2)
        
        # Spatio-temporal layers
        for temp_conv, graph_conv, bn in zip(
            self.temporal_convs,
            self.graph_convs,
            self.batch_norms
        ):
            # Temporal convolution
            x = temp_conv(x)
            x = F.relu(x)
            
            # Reshape for graph convolution: (batch, hidden, num_nodes, seq_len)
            # Process each time step
            x_graph = []
            for t in range(x.size(-1)):
                x_t = x[:, :, :, t]  # (batch, hidden, num_nodes)
                x_t = x_t.permute(0, 2, 1)  # (batch, num_nodes, hidden)
                
                # Graph convolution
                x_t = F.relu(graph_conv(x_t, adj))
                x_graph.append(x_t)
            
            # Stack back
            x = torch.stack(x_graph, dim=-1)  # (batch, num_nodes, hidden, seq_len)
            x = x.permute(0, 2, 1, 3)  # (batch, hidden, num_nodes, seq_len)
            
            # Batch normalization
            x = bn(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # Global average pooling over time
        x = x.mean(dim=-1)  # (batch, hidden, num_nodes)
        
        # Transpose for output layer
        x = x.permute(0, 2, 1)  # (batch, num_nodes, hidden)
        
        # Output predictions for each node
        output = self.output_layer(x)  # (batch, num_nodes, output_size)
        
        return output


class DiffusionConvLayer(nn.Module):
    """
    Diffusion Convolution Layer for capturing multi-hop spatial dependencies.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        k_hop: int = 2
    ):
        super(DiffusionConvLayer, self).__init__()
        
        self.k_hop = k_hop
        
        # Learnable weights for each hop
        self.weights = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(in_features, out_features))
            for _ in range(k_hop)
        ])
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with diffusion convolution.
        
        Args:
            x: Node features (batch, num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Updated features (batch, num_nodes, out_features)
        """
        # Normalize adjacency
        degree = torch.sum(adj, dim=-1, keepdim=True)
        degree = torch.clamp(degree, min=1.0)  # Avoid division by zero
        adj_norm = adj / degree
        
        output = torch.zeros(
            x.size(0), x.size(1), self.weights[0].size(1),
            device=x.device
        )
        
        # Multi-hop diffusion
        adj_power = torch.eye(adj.size(0), device=adj.device)
        for k in range(self.k_hop):
            adj_power = torch.matmul(adj_power, adj_norm)
            diffused = torch.matmul(adj_power, x)
            output += torch.matmul(diffused, self.weights[k])
        
        return output


def create_region_graph(
    region_coordinates: List[Tuple[float, float]],
    connectivity_radius: float = 100.0
) -> torch.Tensor:
    """
    Create adjacency matrix for regions based on geographic proximity.
    
    Args:
        region_coordinates: List of (latitude, longitude) tuples
        connectivity_radius: Maximum distance for edge creation (in km)
        
    Returns:
        Adjacency matrix
    """
    num_regions = len(region_coordinates)
    adj = torch.zeros(num_regions, num_regions)
    
    for i in range(num_regions):
        for j in range(i + 1, num_regions):
            # Calculate distance (simplified - use Haversine for accuracy)
            lat1, lon1 = region_coordinates[i]
            lat2, lon2 = region_coordinates[j]
            
            # Euclidean distance approximation (for small distances)
            distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
            
            if distance <= connectivity_radius:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    
    # Add self-loops
    adj = adj + torch.eye(num_regions)
    
    return adj
