"""
Validation Strategies for Time Series and Federated Learning

This module implements cross-validation strategies suitable for
temporal data and federated learning scenarios.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple, Callable, Iterator
from sklearn.model_selection import KFold, StratifiedKFold
from abc import ABC, abstractmethod


class BaseValidator(ABC):
    """Abstract base class for validators."""
    
    @abstractmethod
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test splits."""
        pass


class TemporalValidator(BaseValidator):
    """
    Time series cross-validation.
    
    Respects temporal ordering to prevent data leakage.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        gap: int = 0
    ):
        """
        Initialize temporal validator.
        
        Args:
            n_splits: Number of splits
            test_size: Size of test set (if None, uses expanding window)
            gap: Gap between train and test sets
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate temporal train/test splits.
        
        Args:
            X: Feature array
            y: Target array (not used, for compatibility)
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        if self.test_size is None:
            # Expanding window
            fold_size = n_samples // (self.n_splits + 1)
            
            for i in range(self.n_splits):
                train_end = (i + 1) * fold_size
                test_start = train_end + self.gap
                test_end = test_start + fold_size
                
                if test_end > n_samples:
                    break
                
                train_indices = np.arange(0, train_end)
                test_indices = np.arange(test_start, test_end)
                
                yield train_indices, test_indices
        else:
            # Sliding window
            for i in range(self.n_splits):
                test_end = n_samples - i * self.test_size
                test_start = test_end - self.test_size
                train_end = test_start - self.gap
                
                if train_end <= 0:
                    break
                
                train_indices = np.arange(0, train_end)
                test_indices = np.arange(test_start, test_end)
                
                yield train_indices, test_indices


class CrossValidator:
    """
    Cross-validation for federated learning models.
    
    Supports various validation strategies.
    """
    
    def __init__(
        self,
        validator_type: str = 'temporal',
        n_splits: int = 5,
        **kwargs
    ):
        """
        Initialize cross-validator.
        
        Args:
            validator_type: Type of validation ('temporal', 'kfold', 'stratified')
            n_splits: Number of splits
            **kwargs: Additional arguments for validator
        """
        self.validator_type = validator_type
        self.n_splits = n_splits
        self.kwargs = kwargs
        
        self.validator = self._create_validator()
    
    def _create_validator(self) -> BaseValidator:
        """Create the appropriate validator."""
        if self.validator_type == 'temporal':
            return TemporalValidator(n_splits=self.n_splits, **self.kwargs)
        elif self.validator_type == 'kfold':
            return KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        elif self.validator_type == 'stratified':
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        else:
            raise ValueError(f"Unknown validator type: {self.validator_type}")
    
    def cross_validate(
        self,
        model: torch.nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        fit_fn: Callable,
        score_fn: Callable,
        device: str = 'cpu'
    ) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Feature array
            y: Target array
            fit_fn: Function to fit model
            score_fn: Function to score model
            device: Device to use
            
        Returns:
            Dictionary with scores for each fold
        """
        scores = {'train': [], 'test': []}
        
        for fold, (train_idx, test_idx) in enumerate(self.validator.split(X, y)):
            print(f"Fold {fold + 1}/{self.n_splits}")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Reset model
            model.apply(self._reset_weights)
            
            # Fit model
            fit_fn(model, X_train, y_train, device)
            
            # Score model
            train_score = score_fn(model, X_train, y_train, device)
            test_score = score_fn(model, X_test, y_test, device)
            
            scores['train'].append(train_score)
            scores['test'].append(test_score)
            
            print(f"  Train score: {train_score:.4f}")
            print(f"  Test score: {test_score:.4f}")
        
        return scores
    
    @staticmethod
    def _reset_weights(m: torch.nn.Module) -> None:
        """Reset model weights."""
        if hasattr(m, 'reset_parameters'):
            m.reset_parameters()


class FederatedValidator:
    """
    Validation for federated learning scenarios.
    
    Validates models on both local client data and global test set.
    """
    
    def __init__(self, test_clients: Optional[List[str]] = None):
        """
        Initialize federated validator.
        
        Args:
            test_clients: List of client IDs to use for testing
        """
        self.test_clients = test_clients or []
    
    def validate_local_models(
        self,
        clients: Dict[str, Any],
        metric_fn: Callable
    ) -> Dict[str, Dict[str, float]]:
        """
        Validate each client's local model on their own data.
        
        Args:
            clients: Dictionary of clients
            metric_fn: Function to calculate metrics
            
        Returns:
            Dictionary of validation results per client
        """
        results = {}
        
        for client_id, client in clients.items():
            if hasattr(client, 'evaluate'):
                metrics = client.evaluate()
                results[client_id] = metrics
            else:
                # Use provided metric function
                results[client_id] = metric_fn(client)
        
        return results
    
    def validate_global_model(
        self,
        global_model: torch.nn.Module,
        test_data: Tuple[torch.Tensor, torch.Tensor],
        metric_fn: Callable
    ) -> Dict[str, float]:
        """
        Validate global model on test data.
        
        Args:
            global_model: Global model
            test_data: Test data tuple (X, y)
            metric_fn: Function to calculate metrics
            
        Returns:
            Dictionary of validation metrics
        """
        X_test, y_test = test_data
        return metric_fn(global_model, X_test, y_test)
    
    def compare_local_vs_global(
        self,
        clients: Dict[str, Any],
        global_model: torch.nn.Module,
        metric_fn: Callable
    ) -> Dict[str, Any]:
        """
        Compare local models with global model.
        
        Args:
            clients: Dictionary of clients
            global_model: Global model
            metric_fn: Function to calculate metrics
            
        Returns:
            Comparison results
        """
        results = {
            'local': {},
            'global': {},
            'improvement': {}
        }
        
        for client_id, client in clients.items():
            # Local model performance
            if hasattr(client, 'evaluate'):
                local_metrics = client.evaluate()
            else:
                local_metrics = metric_fn(client)
            
            results['local'][client_id] = local_metrics
            
            # Global model performance on client data
            if hasattr(client, 'val_data') and client.val_data is not None:
                X_val, y_val = client.val_data
                global_metrics = metric_fn(global_model, X_val, y_val)
                results['global'][client_id] = global_metrics
                
                # Calculate improvement
                if 'accuracy' in local_metrics and 'accuracy' in global_metrics:
                    improvement = global_metrics['accuracy'] - local_metrics['accuracy']
                    results['improvement'][client_id] = improvement
        
        return results


class BootstrapValidator:
    """
    Bootstrap validation for uncertainty estimation.
    """
    
    def __init__(self, n_bootstrap: int = 100, random_state: int = 42):
        """
        Initialize bootstrap validator.
        
        Args:
            n_bootstrap: Number of bootstrap samples
            random_state: Random seed
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        np.random.seed(random_state)
    
    def bootstrap_score(
        self,
        model: torch.nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        score_fn: Callable,
        device: str = 'cpu'
    ) -> Tuple[float, float, Tuple[float, float]]:
        """
        Calculate bootstrap confidence interval for model score.
        
        Args:
            model: Model to validate
            X: Feature array
            y: Target array
            score_fn: Function to calculate score
            device: Device to use
            
        Returns:
            Tuple of (mean_score, std_score, (ci_lower, ci_upper))
        """
        scores = []
        n_samples = len(X)
        
        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Calculate score
            score = score_fn(model, X_boot, y_boot, device)
            scores.append(score)
        
        scores = np.array(scores)
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        ci_lower = np.percentile(scores, 2.5)
        ci_upper = np.percentile(scores, 97.5)
        
        return mean_score, std_score, (ci_lower, ci_upper)


class HoldoutValidator:
    """
    Simple train/validation/test split validator.
    """
    
    def __init__(
        self,
        train_size: float = 0.7,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize holdout validator.
        
        Args:
            train_size: Proportion for training
            val_size: Proportion for validation
            random_state: Random seed
        """
        self.train_size = train_size
        self.val_size = val_size
        self.random_state = random_state
    
    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test sets.
        
        Args:
            X: Feature array
            y: Target array (not used, for compatibility)
            
        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        np.random.seed(self.random_state)
        
        n_samples = len(X)
        indices = np.random.permutation(n_samples)
        
        train_end = int(n_samples * self.train_size)
        val_end = train_end + int(n_samples * self.val_size)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        return train_indices, val_indices, test_indices
