"""
Ensemble Models for Disease Outbreak Prediction

This module implements ensemble methods that combine multiple models
for improved prediction accuracy and robustness.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class EnsembleModel(nn.Module):
    """
    Weighted ensemble of multiple models.
    
    Combines predictions from different architectures to improve
    accuracy and robustness.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        model_weights: Optional[List[float]] = None,
        learn_weights: bool = True
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of models to ensemble
            model_weights: Fixed weights for each model (if not learning)
            learn_weights: Whether to learn ensemble weights
        """
        super(EnsembleModel, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.learn_weights = learn_weights
        
        if learn_weights:
            # Learnable weights (softmax to ensure they sum to 1)
            self.raw_weights = nn.Parameter(torch.ones(self.num_models))
        else:
            # Fixed weights
            if model_weights is None:
                model_weights = [1.0 / self.num_models] * self.num_models
            self.register_buffer('fixed_weights', torch.tensor(model_weights))
    
    def get_weights(self) -> torch.Tensor:
        """Get normalized ensemble weights."""
        if self.learn_weights:
            return torch.softmax(self.raw_weights, dim=0)
        else:
            return self.fixed_weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all models and combine predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted ensemble prediction
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(x))
            predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # (num_models, batch, output)
        
        # Get weights
        weights = self.get_weights()
        
        # Weighted average
        ensemble_pred = torch.sum(
            predictions * weights.view(-1, 1, 1),
            dim=0
        )
        
        return ensemble_pred
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict with uncertainty estimation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean prediction, uncertainty)
        """
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(x))
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty


class StackingEnsemble(nn.Module):
    """
    Stacking ensemble with meta-learner.
    
    Uses a meta-model to combine predictions from base models.
    """
    
    def __init__(
        self,
        base_models: List[nn.Module],
        meta_learner: nn.Module,
        freeze_base: bool = True
    ):
        """
        Initialize stacking ensemble.
        
        Args:
            base_models: List of base models
            meta_learner: Meta-learning model
            freeze_base: Whether to freeze base model weights
        """
        super(StackingEnsemble, self).__init__()
        
        self.base_models = nn.ModuleList(base_models)
        self.meta_learner = meta_learner
        
        # Freeze base models if specified
        if freeze_base:
            for model in self.base_models:
                for param in model.parameters():
                    param.requires_grad = False
    
    def get_base_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get predictions from all base models.
        
        Args:
            x: Input tensor
            
        Returns:
            Stacked predictions from base models
        """
        predictions = []
        
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
            predictions.append(pred)
        
        # Concatenate predictions
        stacked = torch.cat(predictions, dim=-1)
        
        return stacked
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacking ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Final prediction from meta-learner
        """
        # Get base predictions
        base_preds = self.get_base_predictions(x)
        
        # Pass through meta-learner
        output = self.meta_learner(base_preds)
        
        return output


class BootstrapEnsemble(nn.Module):
    """
    Bootstrap ensemble for uncertainty quantification.
    
    Trains multiple models on bootstrap samples of the data.
    """
    
    def __init__(
        self,
        model_class: type,
        model_kwargs: Dict[str, Any],
        num_models: int = 5
    ):
        """
        Initialize bootstrap ensemble.
        
        Args:
            model_class: Model class to instantiate
            model_kwargs: Keyword arguments for model initialization
            num_models: Number of models in ensemble
        """
        super(BootstrapEnsemble, self).__init__()
        
        self.models = nn.ModuleList([
            model_class(**model_kwargs) for _ in range(num_models)
        ])
        self.num_models = num_models
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean prediction, standard deviation)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(x))
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)
        
        return mean, std
    
    def predict_with_confidence(
        self,
        x: torch.Tensor,
        confidence_level: float = 0.95
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with confidence intervals.
        
        Args:
            x: Input tensor
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = torch.sigmoid(model(x))
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (num_models, batch, output)
        
        mean = torch.mean(predictions, dim=0)
        
        # Calculate percentiles for confidence intervals
        alpha = (1 - confidence_level) / 2
        lower = torch.quantile(predictions, alpha, dim=0)
        upper = torch.quantile(predictions, 1 - alpha, dim=0)
        
        return mean, lower, upper


class DynamicEnsemble(nn.Module):
    """
    Dynamic ensemble that selects models based on input.
    
    Uses a gating network to weight models dynamically.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        gating_network: nn.Module
    ):
        """
        Initialize dynamic ensemble.
        
        Args:
            models: List of expert models
            gating_network: Network that outputs weights for each model
        """
        super(DynamicEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.gating_network = gating_network
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dynamic weighting.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (ensemble prediction, gating weights)
        """
        # Get gating weights
        gate_logits = self.gating_network(x)
        gate_weights = torch.softmax(gate_logits, dim=-1)
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=1)  # (batch, num_models, output)
        
        # Weighted combination
        weights = gate_weights.unsqueeze(-1)  # (batch, num_models, 1)
        ensemble_pred = torch.sum(predictions * weights, dim=1)
        
        return ensemble_pred, gate_weights


class DiversityEnsemble(nn.Module):
    """
    Ensemble that promotes diversity among models.
    
    Uses negative correlation learning to encourage diverse predictions.
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        lambda_diversity: float = 0.5
    ):
        """
        Initialize diversity ensemble.
        
        Args:
            models: List of models
            lambda_diversity: Weight for diversity term in loss
        """
        super(DiversityEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.lambda_diversity = lambda_diversity
        self.num_models = len(models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)
        
        # Average prediction
        ensemble_pred = torch.mean(predictions, dim=0)
        
        return ensemble_pred
    
    def diversity_loss(
        self,
        predictions: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diversity-aware loss.
        
        Args:
            predictions: Model predictions (num_models, batch, output)
            target: Ground truth
            
        Returns:
            Diversity loss
        """
        # MSE loss for each model
        mse_loss = torch.mean((predictions - target.unsqueeze(0)) ** 2)
        
        # Diversity term: encourage different errors
        ensemble_pred = torch.mean(predictions, dim=0, keepdim=True)
        individual_errors = predictions - target.unsqueeze(0)
        ensemble_error = ensemble_pred - target.unsqueeze(0)
        
        # Negative correlation
        diversity = torch.mean((individual_errors - ensemble_error) ** 2)
        
        # Combined loss
        total_loss = mse_loss - self.lambda_diversity * diversity
        
        return total_loss


def create_ensemble_from_configs(
    model_configs: List[Dict[str, Any]],
    ensemble_type: str = 'weighted'
) -> nn.Module:
    """
    Factory function to create ensemble from model configurations.
    
    Args:
        model_configs: List of model configuration dictionaries
        ensemble_type: Type of ensemble ('weighted', 'stacking', 'bootstrap')
        
    Returns:
        Ensemble model
    """
    from .lstm_model import LSTMModel
    from .cnn_lstm_model import CNNLSTMModel
    from .transformer_model import TemporalTransformer
    
    models = []
    
    for config in model_configs:
        model_type = config.pop('type', 'lstm')
        
        if model_type == 'lstm':
            model = LSTMModel(**config)
        elif model_type == 'cnn_lstm':
            model = CNNLSTMModel(**config)
        elif model_type == 'transformer':
            model = TemporalTransformer(**config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        models.append(model)
    
    if ensemble_type == 'weighted':
        return EnsembleModel(models, learn_weights=True)
    elif ensemble_type == 'stacking':
        # Create simple meta-learner
        meta_input_size = len(models)  # One prediction per model
        meta_learner = nn.Sequential(
            nn.Linear(meta_input_size, meta_input_size // 2),
            nn.ReLU(),
            nn.Linear(meta_input_size // 2, 1)
        )
        return StackingEnsemble(models, meta_learner)
    elif ensemble_type == 'bootstrap':
        # Use first model's config for bootstrap
        return BootstrapEnsemble(
            model_configs[0].get('model_class', LSTMModel),
            model_configs[0],
            num_models=5
        )
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")
