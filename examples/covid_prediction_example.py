"""
COVID-19 Wave Prediction - Case Study

This example demonstrates using federated learning with Graph Neural Networks
to predict COVID-19 waves using mobility and clinical data.
"""

import numpy as np
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from federated_disease_prediction.core.server import FederatedServer, ServerConfig
from federated_disease_prediction.core.client import FederatedClient, ClientConfig
from federated_disease_prediction.models.gnn_model import SpatioTemporalGNN
from federated_disease_prediction.data.synthetic_data import generate_covid_data
from federated_disease_prediction.data.preprocessing import DataPreprocessor
from federated_disease_prediction.utils.metrics import MetricsCalculator


def prepare_covid_data():
    """Prepare COVID-19 outbreak data."""
    print("Generating synthetic COVID-19 outbreak data...")
    data, adj_matrix = generate_covid_data(num_regions=50, num_days=365)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Regions: {data['region_id'].nunique()}")
    
    return data, adj_matrix


def create_regional_datasets(data, adj_matrix, feature_cols, target_col, seq_length=14):
    """
    Create datasets organized by regions for GNN training.
    
    Args:
        data: Full dataset
        adj_matrix: Region adjacency matrix
        feature_cols: Feature columns
        target_col: Target column
        seq_length: Sequence length
        
    Returns:
        Tuple of (node_features, targets, adj_matrix)
    """
    num_regions = data['region_id'].nunique()
    num_days = len(data) // num_regions
    num_features = len(feature_cols)
    
    # Organize data by region and time
    node_features = np.zeros((num_regions, num_days, num_features))
    targets = np.zeros((num_regions, num_days))
    
    for region_id in range(num_regions):
        region_data = data[data['region_id'] == region_id].sort_values('date')
        node_features[region_id] = region_data[feature_cols].values
        targets[region_id] = region_data[target_col].values
    
    # Create sequences
    sequences = []
    sequence_targets = []
    
    for t in range(num_days - seq_length):
        seq = node_features[:, t:t+seq_length, :]  # (num_regions, seq_len, features)
        target = targets[:, t+seq_length]  # (num_regions,)
        sequences.append(seq)
        sequence_targets.append(target)
    
    sequences = np.array(sequences)  # (num_samples, num_regions, seq_len, features)
    sequence_targets = np.array(sequence_targets)  # (num_samples, num_regions)
    
    # Split
    n = len(sequences)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_data = (
        torch.FloatTensor(sequences[:train_end]),
        torch.FloatTensor(sequence_targets[:train_end])
    )
    val_data = (
        torch.FloatTensor(sequences[train_end:val_end]),
        torch.FloatTensor(sequence_targets[train_end:val_end])
    )
    test_data = (
        torch.FloatTensor(sequences[val_end:]),
        torch.FloatTensor(sequence_targets[val_end:])
    )
    
    adj_tensor = torch.FloatTensor(adj_matrix)
    
    return train_data, val_data, test_data, adj_tensor


def create_gnn_federated_system():
    """Create federated learning system with GNN."""
    print("\nSetting up GNN-based federated learning system...")
    
    # Generate data
    data, adj_matrix = prepare_covid_data()
    
    # Define features
    feature_cols = [
        'new_cases', 'hospitalizations', 'deaths', 'positive_rate',
        'temperature', 'humidity', 'mobility_index', 'travel_volume'
    ]
    target_col = 'target'
    
    # Preprocess
    preprocessor = DataPreprocessor(
        missing_strategy='interpolation',
        outlier_method='iqr',
        scaler_type='robust'
    )
    processed_data = preprocessor.fit_transform(data, feature_cols + [target_col])
    
    # Create regional datasets
    train_data, val_data, test_data, adj_tensor = create_regional_datasets(
        processed_data, adj_matrix, feature_cols, target_col, seq_length=14
    )
    
    num_regions = adj_matrix.shape[0]
    
    # Create GNN model
    model = SpatioTemporalGNN(
        num_nodes=num_regions,
        in_features=len(feature_cols),
        hidden_features=64,
        num_layers=2,
        output_size=1,
        dropout=0.3
    )
    
    # Create server
    server_config = ServerConfig(
        num_rounds=30,
        clients_per_round=3,
        aggregation_strategy='fedavg'
    )
    
    server = FederatedServer(server_config, model, test_data=test_data)
    
    # Create clients (each client represents a group of regions)
    num_clients = 5
    regions_per_client = num_regions // num_clients
    
    for client_id in range(num_clients):
        # Get regions for this client
        start_region = client_id * regions_per_client
        end_region = start_region + regions_per_client
        
        # Extract data for client's regions
        X_train_client = train_data[0][:, start_region:end_region, :, :]
        y_train_client = train_data[1][:, start_region:end_region]
        
        X_val_client = val_data[0][:, start_region:end_region, :, :]
        y_val_client = val_data[1][:, start_region:end_region]
        
        # Create client model
        client_model = SpatioTemporalGNN(
            num_nodes=regions_per_client,
            in_features=len(feature_cols),
            hidden_features=64,
            num_layers=2,
            output_size=1,
            dropout=0.3
        )
        
        client_config = ClientConfig(
            client_id=f'region_group_{client_id}',
            local_epochs=2,
            batch_size=16,
            learning_rate=0.001
        )
        
        client = FederatedClient(
            config=client_config,
            model=client_model,
            train_data=(X_train_client, y_train_client),
            val_data=(X_val_client, y_val_client)
        )
        
        server.register_client(client)
    
    print(f"Created {num_clients} regional group clients")
    
    return server, test_data, adj_tensor


def evaluate_gnn_model(server, test_data, adj_matrix):
    """Evaluate GNN model on test data."""
    print("\nEvaluating GNN model...")
    
    X_test, y_test = test_data
    model = server.get_global_model()
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i in range(len(X_test)):
            x = X_test[i:i+1]  # (1, num_regions, seq_len, features)
            y = y_test[i]  # (num_regions,)
            
            # Forward pass
            pred = model(x, adj_matrix)
            pred = torch.sigmoid(pred)
            
            all_preds.append(pred.squeeze().numpy())
            all_targets.append(y.numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_targets = np.array(all_targets).flatten()
    
    # Convert to binary
    binary_preds = (all_preds > 0.5).astype(int)
    
    # Calculate metrics
    metrics = MetricsCalculator.calculate_classification_metrics(
        all_targets, binary_preds, all_preds
    )
    
    print("\nGNN Evaluation Results:")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1 Score:  {metrics.f1_score:.4f}")
    print(f"  ROC AUC:   {metrics.roc_auc:.4f}")
    
    return metrics


def main():
    """Main execution."""
    print("="*60)
    print("COVID-19 Wave Prediction - GNN Federated Learning Case Study")
    print("="*60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create system
    server, test_data, adj_matrix = create_gnn_federated_system()
    
    # Train
    print("\nStarting training...")
    metrics_history = server.train()
    
    # Evaluate
    metrics = evaluate_gnn_model(server, test_data, adj_matrix)
    
    # Summary
    summary = server.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save
    server.save_checkpoint('covid_gnn_checkpoint.pt')
    print("\nSaved checkpoint: covid_gnn_checkpoint.pt")
    
    print("\n" + "="*60)
    print("COVID-19 Case Study Complete!")
    print("="*60)
    
    return server, metrics


if __name__ == "__main__":
    server, metrics = main()
