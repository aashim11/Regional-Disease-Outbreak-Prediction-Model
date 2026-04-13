"""
Dataset Classes for Disease Outbreak Prediction

This module implements PyTorch Dataset classes for loading and
batching disease outbreak data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable


class DiseaseDataset(Dataset):
    """
    Dataset for disease outbreak prediction.
    
    Handles time series data with multiple features.
    """
    
    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 30,
        transform: Optional[Callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data: Feature array (num_samples, num_features) or (num_samples, seq_len, num_features)
            targets: Target array (num_samples,) or (num_samples, prediction_horizon)
            sequence_length: Length of sequences for time series models
            transform: Optional transform to apply
        """
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Create sequences if data is not already sequenced
        if len(data.shape) == 2:
            self.sequences, self.sequence_targets = self._create_sequences()
        else:
            self.sequences = data
            self.sequence_targets = targets
    
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from flat data."""
        sequences = []
        targets = []
        
        for i in range(len(self.data) - self.sequence_length):
            seq = self.data[i:i + self.sequence_length]
            target = self.targets[i + self.sequence_length - 1]
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        x = torch.FloatTensor(self.sequences[idx])
        y = torch.FloatTensor([self.sequence_targets[idx]])
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


class FederatedDataset:
    """
    Federated dataset that manages data for multiple clients.
    
    Simulates data distributed across multiple healthcare institutions.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        num_clients: int = 10,
        split_by: str = 'region',
        iid: bool = False,
        alpha: float = 0.5  # Dirichlet distribution parameter for non-IID
    ):
        """
        Initialize federated dataset.
        
        Args:
            data: Full dataset
            num_clients: Number of clients
            split_by: Column to split by (if not IID)
            iid: Whether to split data IID
            alpha: Dirichlet concentration parameter (lower = more non-IID)
        """
        self.data = data
        self.num_clients = num_clients
        self.iid = iid
        self.alpha = alpha
        
        self.client_datasets: Dict[str, DiseaseDataset] = {}
        self.client_data_indices: Dict[str, List[int]] = {}
        
        self._split_data(split_by)
    
    def _split_data(self, split_by: str) -> None:
        """Split data among clients."""
        if self.iid:
            # IID split: random shuffle
            indices = np.random.permutation(len(self.data))
            split_sizes = np.array_split(indices, self.num_clients)
        else:
            # Non-IID split using Dirichlet distribution
            if split_by in self.data.columns:
                # Split by geographic region or institution
                unique_regions = self.data[split_by].unique()
                split_sizes = self._dirichlet_split_by_region(split_by, unique_regions)
            else:
                # Label-based non-IID split
                split_sizes = self._dirichlet_split_by_label()
        
        # Create client datasets
        for i, indices in enumerate(split_sizes):
            client_id = f'client_{i}'
            self.client_data_indices[client_id] = indices.tolist()
    
    def _dirichlet_split_by_region(
        self,
        split_by: str,
        regions: np.ndarray
    ) -> List[np.ndarray]:
        """Split data using Dirichlet distribution by region."""
        client_indices = [[] for _ in range(self.num_clients)]
        
        for region in regions:
            region_data = self.data[self.data[split_by] == region]
            region_indices = region_data.index.values
            
            # Sample from Dirichlet
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            proportions = (proportions * len(region_indices)).astype(int)
            
            # Ensure sum matches
            proportions[-1] = len(region_indices) - proportions[:-1].sum()
            
            # Assign to clients
            start = 0
            for i, prop in enumerate(proportions):
                client_indices[i].extend(region_indices[start:start + prop])
                start += prop
        
        return [np.array(indices) for indices in client_indices]
    
    def _dirichlet_split_by_label(self) -> List[np.ndarray]:
        """Split data using Dirichlet distribution by label."""
        # Assume last column is label
        labels = self.data.iloc[:, -1].unique()
        client_indices = [[] for _ in range(self.num_clients)]
        
        for label in labels:
            label_data = self.data[self.data.iloc[:, -1] == label]
            label_indices = label_data.index.values
            
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            proportions = (proportions * len(label_indices)).astype(int)
            proportions[-1] = len(label_indices) - proportions[:-1].sum()
            
            start = 0
            for i, prop in enumerate(proportions):
                client_indices[i].extend(label_indices[start:start + prop])
                start += prop
        
        return [np.array(indices) for indices in client_indices]
    
    def get_client_data(
        self,
        client_id: str,
        feature_columns: List[str],
        target_column: str,
        sequence_length: int = 30
    ) -> DiseaseDataset:
        """
        Get dataset for a specific client.
        
        Args:
            client_id: Client identifier
            feature_columns: List of feature column names
            target_column: Target column name
            sequence_length: Sequence length for time series
            
        Returns:
            DiseaseDataset for the client
        """
        if client_id not in self.client_data_indices:
            raise ValueError(f"Unknown client: {client_id}")
        
        indices = self.client_data_indices[client_id]
        client_df = self.data.iloc[indices]
        
        # Extract features and targets
        X = client_df[feature_columns].values
        y = client_df[target_column].values
        
        return DiseaseDataset(X, y, sequence_length)
    
    def get_client_ids(self) -> List[str]:
        """Get list of client IDs."""
        return list(self.client_data_indices.keys())
    
    def get_data_distribution(self) -> Dict[str, Dict[str, int]]:
        """
        Get distribution of data across clients.
        
        Returns:
            Dictionary with client statistics
        """
        distribution = {}
        
        for client_id, indices in self.client_data_indices.items():
            client_data = self.data.iloc[indices]
            
            distribution[client_id] = {
                'num_samples': len(indices),
                'num_features': len(self.data.columns) - 1,
            }
            
            # Add label distribution if available
            if self.data.columns[-1] in client_data.columns:
                label_col = self.data.columns[-1]
                distribution[client_id]['label_distribution'] = (
                    client_data[label_col].value_counts().to_dict()
                )
        
        return distribution


class MultiModalDataset(Dataset):
    """
    Dataset for multi-modal disease data.
    
    Combines clinical, environmental, and mobility data.
    """
    
    def __init__(
        self,
        clinical_data: np.ndarray,
        environmental_data: np.ndarray,
        mobility_data: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None,
        sequence_length: int = 30
    ):
        """
        Initialize multi-modal dataset.
        
        Args:
            clinical_data: Clinical features
            environmental_data: Environmental features
            mobility_data: Mobility features (optional)
            targets: Target labels
            sequence_length: Sequence length
        """
        self.clinical_data = clinical_data
        self.environmental_data = environmental_data
        self.mobility_data = mobility_data
        self.targets = targets
        self.sequence_length = sequence_length
    
    def __len__(self) -> int:
        return len(self.clinical_data) - self.sequence_length
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Get multi-modal sample."""
        # Extract sequences
        clinical_seq = torch.FloatTensor(
            self.clinical_data[idx:idx + self.sequence_length]
        )
        env_seq = torch.FloatTensor(
            self.environmental_data[idx:idx + self.sequence_length]
        )
        
        data = {
            'clinical': clinical_seq,
            'environmental': env_seq
        }
        
        if self.mobility_data is not None:
            mobility_seq = torch.FloatTensor(
                self.mobility_data[idx:idx + self.sequence_length]
            )
            data['mobility'] = mobility_seq
        
        if self.targets is not None:
            target = torch.FloatTensor([self.targets[idx + self.sequence_length - 1]])
        else:
            target = torch.FloatTensor([0])
        
        return data, target


class GraphDataset(Dataset):
    """
    Dataset for graph-structured disease data.
    
    Represents regions as nodes in a graph with spatial connections.
    """
    
    def __init__(
        self,
        node_features: np.ndarray,
        adjacency_matrix: np.ndarray,
        targets: np.ndarray,
        sequence_length: int = 30
    ):
        """
        Initialize graph dataset.
        
        Args:
            node_features: Node features (num_regions, seq_len, num_features)
            adjacency_matrix: Adjacency matrix (num_regions, num_regions)
            targets: Target labels (num_regions,)
            sequence_length: Sequence length
        """
        self.node_features = node_features
        self.adjacency_matrix = adjacency_matrix
        self.targets = targets
        self.sequence_length = sequence_length
        
        self.num_regions = node_features.shape[0]
    
    def __len__(self) -> int:
        return max(0, self.node_features.shape[1] - self.sequence_length)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get graph sample."""
        # Extract sequence for all regions
        x = torch.FloatTensor(
            self.node_features[:, idx:idx + self.sequence_length, :]
        )
        adj = torch.FloatTensor(self.adjacency_matrix)
        y = torch.FloatTensor([self.targets[:, idx + self.sequence_length - 1]])
        
        return x, adj, y


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Args:
        dataset: Dataset to split
        batch_size: Batch size
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader
