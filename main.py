"""
Federated Learning for Regional Disease Outbreak Prediction

Main entry point for running the federated learning system.
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

from federated_disease_prediction.core.server import FederatedServer, ServerConfig
from federated_disease_prediction.core.client import FederatedClient, ClientConfig
from federated_disease_prediction.models.cnn_lstm_model import CNNLSTMModel
from federated_disease_prediction.data.synthetic_data import generate_dengue_data
from federated_disease_prediction.data.preprocessing import DataPreprocessor
from federated_disease_prediction.visualization.dashboard import launch_dashboard


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_federated_system(config: dict):
    """Setup federated learning system based on configuration."""
    print("Setting up federated learning system...")
    
    # Generate synthetic data
    data, adj_matrix = generate_dengue_data(
        num_regions=config['data'].get('num_regions', 20),
        num_days=config['data'].get('num_days', 730)
    )
    
    # Preprocess
    feature_cols = config['data'].get('features', [
        'new_cases', 'hospitalizations', 'temperature', 'humidity'
    ])
    target_col = config['data'].get('target', 'target')
    
    preprocessor = DataPreprocessor(
        missing_strategy=config['data']['preprocessing'].get('fill_missing', 'interpolation'),
        scaler_type='standard'
    )
    
    processed_data = preprocessor.fit_transform(data, feature_cols + [target_col])
    
    # Create model
    model_config = config['model']
    model = CNNLSTMModel(
        input_size=len(feature_cols),
        sequence_length=model_config.get('sequence_length', 30),
        cnn_filters=model_config['cnn'].get('num_filters', [64, 128]),
        lstm_hidden=model_config['lstm'].get('hidden_size', 128),
        output_size=1,
        dropout=model_config['lstm'].get('dropout', 0.2)
    )
    
    # Create server
    fl_config = config['federated_learning']
    server_config = ServerConfig(
        num_rounds=fl_config.get('num_rounds', 100),
        clients_per_round=fl_config.get('clients_per_round', 10),
        aggregation_strategy=fl_config.get('aggregation_strategy', 'fedavg')
    )
    
    server = FederatedServer(server_config, model)
    
    # Create clients
    num_clients = fl_config.get('total_clients', 10)
    
    for i in range(num_clients):
        client_config = ClientConfig(
            client_id=f'client_{i}',
            local_epochs=fl_config.get('local_epochs', 5),
            batch_size=fl_config.get('batch_size', 32),
            learning_rate=fl_config.get('learning_rate', 0.001),
            dp_enabled=config['privacy']['differential_privacy'].get('enabled', False),
            dp_epsilon=config['privacy']['differential_privacy'].get('epsilon', 1.0)
        )
        
        # Create client data (simplified)
        region_data = processed_data[processed_data['region_id'] == i % 20]
        
        if len(region_data) > 100:
            X = torch.FloatTensor(region_data[feature_cols].values[:100])
            y = torch.FloatTensor(region_data[target_col].values[:100]).unsqueeze(1)
            
            client_model = CNNLSTMModel(
                input_size=len(feature_cols),
                sequence_length=30,
                cnn_filters=[64, 128],
                lstm_hidden=128,
                output_size=1,
                dropout=0.2
            )
            
            client = FederatedClient(
                config=client_config,
                model=client_model,
                train_data=(X, y),
                val_data=(X[:20], y[:20])
            )
            
            server.register_client(client)
    
    return server


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Federated Learning for Disease Outbreak Prediction'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'dashboard', 'demo'],
        default='train',
        help='Execution mode'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if args.mode == 'train':
        # Setup and train
        server = setup_federated_system(config)
        
        print("\nStarting federated training...")
        metrics = server.train()
        
        # Save results
        server.save_checkpoint('final_model.pt')
        print("\nTraining complete! Model saved to final_model.pt")
        
    elif args.mode == 'dashboard':
        # Launch dashboard
        print("Launching dashboard...")
        launch_dashboard()
        
    elif args.mode == 'demo':
        # Run demo
        print("Running demo...")
        from examples.dengue_prediction_example import main as dengue_demo
        dengue_demo()


if __name__ == "__main__":
    main()
