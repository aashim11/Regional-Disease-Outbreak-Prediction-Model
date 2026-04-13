"""
Federated Learning Client Implementation

This module implements the client-side logic for federated learning,
including local training, privacy mechanisms, and communication with the server.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
import time
import logging
from abc import ABC, abstractmethod
import copy

from ..privacy.differential_privacy import DifferentialPrivacy
from ..privacy.secure_aggregation import SecureAggregation


@dataclass
class ClientConfig:
    """Configuration for federated learning client."""
    client_id: str
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    device: str = 'cpu'
    dp_enabled: bool = False
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    secure_agg_enabled: bool = False
    verbose: bool = False


@dataclass
class TrainingResult:
    """Result of local training on client."""
    client_id: str
    model_weights: Dict[str, np.ndarray]
    num_samples: int
    metrics: Dict[str, float]
    round_number: int
    training_time: float
    timestamp: float = field(default_factory=time.time)


class FederatedClient:
    """
    Federated Learning Client
    
    Represents a healthcare institution (hospital, clinic, lab) that
    participates in federated learning while keeping data local.
    """
    
    def __init__(
        self,
        config: ClientConfig,
        model: nn.Module,
        train_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        self.config = config
        self.client_id = config.client_id
        self.model = model.to(config.device)
        self.device = config.device
        
        # Data
        self.train_data = train_data
        self.val_data = val_data
        self.train_loader = None
        self.val_loader = None
        
        if train_data is not None:
            self._create_data_loaders()
        
        # Privacy mechanisms
        self.dp_engine = None
        if config.dp_enabled:
            self.dp_engine = DifferentialPrivacy(
                epsilon=config.dp_epsilon,
                delta=config.dp_delta,
                max_grad_norm=config.dp_max_grad_norm
            )
        
        self.secure_agg = None
        if config.secure_agg_enabled:
            self.secure_agg = SecureAggregation()
        
        # Training state
        self.optimizer = None
        self.criterion = None
        self.current_round = 0
        self.training_history: List[Dict[str, float]] = []
        
        # Logging
        self.logger = logging.getLogger(f"Client-{self.client_id}")
        if config.verbose:
            self.logger.setLevel(logging.DEBUG)
        
        self._setup_training()
    
    def _create_data_loaders(self) -> None:
        """Create PyTorch data loaders from training and validation data."""
        if self.train_data is not None:
            X_train, y_train = self.train_data
            train_dataset = TensorDataset(X_train, y_train)
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
        
        if self.val_data is not None:
            X_val, y_val = self.val_data
            val_dataset = TensorDataset(X_val, y_val)
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
    
    def _setup_training(self) -> None:
        """Setup optimizer and loss function."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        self.criterion = nn.BCEWithLogitsLoss()
    
    def set_global_weights(self, global_weights: Dict[str, np.ndarray]) -> None:
        """
        Update local model with global weights from server.
        
        Args:
            global_weights: Dictionary of model weights from server
        """
        self.model.load_state_dict({
            k: torch.from_numpy(v).to(self.device)
            for k, v in global_weights.items()
        })
    
    def get_weights(self) -> Dict[str, np.ndarray]:
        """
        Get current local model weights.
        
        Returns:
            Dictionary of model weights as numpy arrays
        """
        return {
            k: v.cpu().numpy()
            for k, v in self.model.state_dict().items()
        }
    
    def train_round(
        self,
        global_weights: Optional[Dict[str, np.ndarray]] = None,
        round_number: int = 0
    ) -> TrainingResult:
        """
        Perform one round of local training.
        
        Args:
            global_weights: Global model weights from server
            round_number: Current federated round number
            
        Returns:
            TrainingResult containing model updates and metrics
        """
        start_time = time.time()
        self.current_round = round_number
        
        # Load global weights if provided
        if global_weights is not None:
            self.set_global_weights(global_weights)
        
        if self.train_loader is None:
            raise ValueError("No training data available")
        
        # Training loop
        self.model.train()
        epoch_losses = []
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy if enabled
                if self.dp_engine is not None:
                    self.dp_engine.clip_gradients(self.model)
                
                self.optimizer.step()
                
                # Add noise if differential privacy is enabled
                if self.dp_engine is not None:
                    self.dp_engine.add_noise(self.model)
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            
            if self.config.verbose:
                self.logger.debug(
                    f"Round {round_number}, Epoch {epoch+1}/{self.config.local_epochs}: "
                    f"Loss = {avg_epoch_loss:.4f}"
                )
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        metrics['train_loss'] = np.mean(epoch_losses)
        
        training_time = time.time() - start_time
        
        # Store training history
        self.training_history.append({
            'round': round_number,
            **metrics,
            'time': training_time
        })
        
        # Prepare result
        result = TrainingResult(
            client_id=self.client_id,
            model_weights=self.get_weights(),
            num_samples=len(self.train_loader.dataset),
            metrics=metrics,
            round_number=round_number,
            training_time=training_time
        )
        
        return result
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate evaluation metrics on validation set."""
        if self.val_loader is None:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                preds = torch.sigmoid(outputs) > 0.5
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        # Calculate metrics
        tp = np.sum((all_preds == 1) & (all_labels == 1))
        fp = np.sum((all_preds == 1) & (all_labels == 0))
        tn = np.sum((all_preds == 0) & (all_labels == 0))
        fn = np.sum((all_preds == 0) & (all_labels == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
    
    def evaluate(self, test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Optional test data tuple (X, y)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if test_data is not None:
            X_test, y_test = test_data
            test_dataset = TensorDataset(X_test, y_test)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        elif self.val_loader is not None:
            test_loader = self.val_loader
        else:
            raise ValueError("No test data available")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = torch.sigmoid(outputs)
                preds = probs > 0.5
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        all_probs = np.array(all_probs).flatten()
        
        # Calculate comprehensive metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix, classification_report
        )
        
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
        }
        
        return metrics
    
    def predict(self, X: torch.Tensor) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input tensor
            
        Returns:
            Predicted probabilities
        """
        self.model.eval()
        X = X.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X)
            probs = torch.sigmoid(outputs)
        
        return probs.cpu().numpy()
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about local data (without exposing raw data).
        
        Returns:
            Dictionary of data statistics
        """
        if self.train_data is None:
            return {}
        
        X_train, y_train = self.train_data
        
        # Calculate statistics without exposing individual records
        stats = {
            'num_samples': len(X_train),
            'num_features': X_train.shape[1] if len(X_train.shape) > 1 else 1,
            'positive_ratio': float(torch.mean(y_train)),
            'feature_means': torch.mean(X_train, dim=0).numpy().tolist(),
            'feature_stds': torch.std(X_train, dim=0).numpy().tolist(),
        }
        
        return stats


class HospitalClient(FederatedClient):
    """Specialized client for hospitals with clinical data."""
    
    def __init__(
        self,
        hospital_id: str,
        model: nn.Module,
        clinical_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ):
        config = ClientConfig(
            client_id=f"hospital_{hospital_id}",
            **kwargs
        )
        super().__init__(config, model, clinical_data)
        self.hospital_id = hospital_id
        self.institution_type = "hospital"


class ClinicClient(FederatedClient):
    """Specialized client for clinics."""
    
    def __init__(
        self,
        clinic_id: str,
        model: nn.Module,
        clinical_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ):
        config = ClientConfig(
            client_id=f"clinic_{clinic_id}",
            **kwargs
        )
        super().__init__(config, model, clinical_data)
        self.clinic_id = clinic_id
        self.institution_type = "clinic"


class LabClient(FederatedClient):
    """Specialized client for diagnostic laboratories."""
    
    def __init__(
        self,
        lab_id: str,
        model: nn.Module,
        lab_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs
    ):
        config = ClientConfig(
            client_id=f"lab_{lab_id}",
            **kwargs
        )
        super().__init__(config, model, lab_data)
        self.lab_id = lab_id
        self.institution_type = "laboratory"
