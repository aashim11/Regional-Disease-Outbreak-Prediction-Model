"""
Dengue Outbreak Prediction - Case Study

This example demonstrates using the federated learning system
to predict dengue outbreaks using weather and clinical data.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from federated_disease_prediction.core.server import FederatedServer, ServerConfig
from federated_disease_prediction.core.client import FederatedClient, ClientConfig
from federated_disease_prediction.core.aggregator import FedAvg
from federated_disease_prediction.models.cnn_lstm_model import CNNLSTMModel
from federated_disease_prediction.data.synthetic_data import generate_dengue_data
from federated_disease_prediction.data.preprocessing import DataPreprocessor, FeatureEngineer
from federated_disease_prediction.utils.metrics import MetricsCalculator
from federated_disease_prediction.visualization.plots import (
    plot_time_series,
    plot_outbreak_heatmap,
    plot_federated_learning_progress
)


def prepare_dengue_data():
    """
    Prepare dengue outbreak data.
    
    Returns:
        Tuple of (data, adjacency_matrix)
    """
    print("Generating synthetic dengue outbreak data...")
    data, adj_matrix = generate_dengue_data(num_regions=20, num_days=730)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Regions: {data['region_id'].nunique()}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    return data, adj_matrix


def preprocess_data(data, feature_cols, target_col):
    """
    Preprocess data for modeling.
    
    Args:
        data: Raw data DataFrame
        feature_cols: Feature column names
        target_col: Target column name
        
    Returns:
        Preprocessed data
    """
    print("Preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        missing_strategy='interpolation',
        outlier_method='iqr',
        scaler_type='standard'
    )
    
    # Fit and transform
    processed_data = preprocessor.fit_transform(data, feature_cols + [target_col])
    
    return processed_data, preprocessor


def create_client_data(data, region_id, feature_cols, target_col, seq_length=30):
    """
    Create data for a single client (region).
    
    Args:
        data: Full dataset
        region_id: Region identifier
        feature_cols: Feature columns
        target_col: Target column
        seq_length: Sequence length
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Filter data for region
    region_data = data[data['region_id'] == region_id].sort_values('date')
    
    # Extract features and target
    X = region_data[feature_cols].values
    y = region_data[target_col].values
    
    # Create sequences
    sequences = []
    targets = []
    
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i + seq_length])
        targets.append(y[i + seq_length])
    
    sequences = np.array(sequences)
    targets = np.array(targets)
    
    # Split into train/val/test
    n = len(sequences)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    X_train = torch.FloatTensor(sequences[:train_end])
    y_train = torch.FloatTensor(targets[:train_end]).unsqueeze(1)
    
    X_val = torch.FloatTensor(sequences[train_end:val_end])
    y_val = torch.FloatTensor(targets[train_end:val_end]).unsqueeze(1)
    
    X_test = torch.FloatTensor(sequences[val_end:])
    y_test = torch.FloatTensor(targets[val_end:]).unsqueeze(1)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def create_federated_system(num_clients=10):
    """
    Create federated learning system for dengue prediction.
    
    Args:
        num_clients: Number of clients (regions)
        
    Returns:
        Tuple of (server, clients, test_data)
    """
    print(f"\nSetting up federated learning system with {num_clients} clients...")
    
    # Generate data
    data, adj_matrix = prepare_dengue_data()
    
    # Define features
    feature_cols = [
        'new_cases', 'hospitalizations', 'positive_rate',
        'temperature', 'humidity', 'rainfall',
        'mobility_index', 'population_density'
    ]
    target_col = 'target'
    
    # Preprocess
    processed_data, preprocessor = preprocess_data(data, feature_cols, target_col)
    
    # Create model
    model = CNNLSTMModel(
        input_size=len(feature_cols),
        sequence_length=30,
        cnn_filters=[64, 128],
        lstm_hidden=128,
        output_size=1,
        dropout=0.3
    )
    
    # Create server
    server_config = ServerConfig(
        num_rounds=20,
        clients_per_round=5,
        aggregation_strategy='fedavg',
        eval_interval=5
    )
    
    # Create test data (from all regions)
    all_test_data = []
    for region_id in range(min(num_clients, 20)):
        _, _, test_data = create_client_data(
            processed_data, region_id, feature_cols, target_col
        )
        all_test_data.append(test_data)
    
    # Combine test data
    X_test = torch.cat([d[0] for d in all_test_data], dim=0)
    y_test = torch.cat([d[1] for d in all_test_data], dim=0)
    
    server = FederatedServer(server_config, model, test_data=(X_test, y_test))
    
    # Create clients
    clients = []
    for region_id in range(num_clients):
        train_data, val_data, _ = create_client_data(
            processed_data, region_id, feature_cols, target_col
        )
        
        client_config = ClientConfig(
            client_id=f'region_{region_id}',
            local_epochs=3,
            batch_size=32,
            learning_rate=0.001,
            dp_enabled=True,
            dp_epsilon=1.0
        )
        
        # Create new model instance for each client
        client_model = CNNLSTMModel(
            input_size=len(feature_cols),
            sequence_length=30,
            cnn_filters=[64, 128],
            lstm_hidden=128,
            output_size=1,
            dropout=0.3
        )
        
        client = FederatedClient(
            config=client_config,
            model=client_model,
            train_data=train_data,
            val_data=val_data
        )
        
        clients.append(client)
        server.register_client(client)
    
    print(f"Created {len(clients)} clients")
    
    return server, clients, (X_test, y_test)


def run_federated_training(server):
    """
    Run federated training.
    
    Args:
        server: Federated server
        
    Returns:
        Training metrics
    """
    print("\n" + "="*60)
    print("Starting Federated Training")
    print("="*60)
    
    metrics = server.train()
    
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    
    return metrics


def evaluate_model(server, test_data):
    """
    Evaluate trained model.
    
    Args:
        server: Federated server
        test_data: Test data tuple
        
    Returns:
        Evaluation metrics
    """
    print("\nEvaluating global model...")
    
    X_test, y_test = test_data
    
    # Get predictions
    model = server.get_global_model()
    model.eval()
    
    with torch.no_grad():
        y_pred_logits = model(X_test)
        y_pred_prob = torch.sigmoid(y_pred_logits)
        y_pred = (y_pred_prob > 0.5).float()
    
    # Calculate metrics
    y_true_np = y_test.numpy().flatten()
    y_pred_np = y_pred.numpy().flatten()
    y_prob_np = y_pred_prob.numpy().flatten()
    
    metrics = MetricsCalculator.calculate_classification_metrics(
        y_true_np, y_pred_np, y_prob_np
    )
    
    print("\nEvaluation Results:")
    print(f"  Accuracy:  {metrics.accuracy:.4f}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1 Score:  {metrics.f1_score:.4f}")
    print(f"  ROC AUC:   {metrics.roc_auc:.4f}")
    
    return metrics


def visualize_results(server, metrics_history, data):
    """
    Visualize training results.
    
    Args:
        server: Federated server
        metrics_history: Training metrics history
        data: Dataset
    """
    print("\nGenerating visualizations...")
    
    # Plot training progress
    rounds = [m['round'] for m in metrics_history if 'global_metrics' in m and m['global_metrics']]
    accuracies = [m['global_metrics'].get('accuracy', 0) for m in metrics_history if 'global_metrics' in m and m['global_metrics']]
    
    if rounds and accuracies:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rounds, accuracies, marker='o')
        ax.set_xlabel('Round')
        ax.set_ylabel('Accuracy')
        ax.set_title('Federated Learning Progress - Dengue Prediction')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('dengue_fl_progress.png', dpi=300)
        print("Saved: dengue_fl_progress.png")
    
    # Plot outbreak heatmap
    try:
        fig = plot_outbreak_heatmap(
            data,
            date_col='date',
            region_col='region_id',
            value_col='new_cases',
            title='Dengue Cases by Region and Time'
        )
        fig.savefig('dengue_heatmap.png', dpi=300)
        print("Saved: dengue_heatmap.png")
    except Exception as e:
        print(f"Could not create heatmap: {e}")


def main():
    """Main execution function."""
    print("="*60)
    print("Dengue Outbreak Prediction - Federated Learning Case Study")
    print("="*60)
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create federated system
    server, clients, test_data = create_federated_system(num_clients=10)
    
    # Run training
    metrics_history = run_federated_training(server)
    
    # Evaluate
    metrics = evaluate_model(server, test_data)
    
    # Get training summary
    summary = server.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save checkpoint
    server.save_checkpoint('dengue_model_checkpoint.pt')
    print("\nSaved checkpoint: dengue_model_checkpoint.pt")
    
    print("\n" + "="*60)
    print("Case Study Complete!")
    print("="*60)
    
    return server, metrics


if __name__ == "__main__":
    server, metrics = main()
