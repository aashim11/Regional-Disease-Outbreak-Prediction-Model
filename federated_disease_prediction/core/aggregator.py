"""
Federated Learning Aggregation Strategies

This module implements various aggregation algorithms for combining
model updates from multiple clients in a federated learning system.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import copy


@dataclass
class ClientUpdate:
    """Represents an update from a client."""
    client_id: str
    model_weights: Dict[str, np.ndarray]
    num_samples: int
    metrics: Dict[str, float]
    round_number: int
    timestamp: float


class AggregationStrategy(ABC):
    """Abstract base class for federated aggregation strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.round_number = 0
        
    @abstractmethod
    def aggregate(
        self,
        global_weights: Dict[str, np.ndarray],
        client_updates: List[ClientUpdate],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate client updates into new global model weights.
        
        Args:
            global_weights: Current global model weights
            client_updates: List of updates from clients
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Updated global model weights
        """
        pass
    
    def _validate_updates(self, client_updates: List[ClientUpdate]) -> None:
        """Validate that all client updates have compatible weight structures."""
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Check that all updates have the same keys
        first_keys = set(client_updates[0].model_weights.keys())
        for update in client_updates[1:]:
            if set(update.model_weights.keys()) != first_keys:
                raise ValueError("Inconsistent model weight keys across clients")


class FedAvg(AggregationStrategy):
    """
    Federated Averaging (FedAvg) algorithm.
    
    Reference: McMahan et al., "Communication-Efficient Learning of Deep
    Networks from Decentralized Data", AISTATS 2017.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.use_weighted_average = self.config.get('use_weighted_average', True)
    
    def aggregate(
        self,
        global_weights: Dict[str, np.ndarray],
        client_updates: List[ClientUpdate],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate using weighted average based on number of samples.
        
        Formula: w_global = sum(n_k * w_k) / sum(n_k)
        where n_k is the number of samples for client k
        """
        self._validate_updates(client_updates)
        
        # Calculate total number of samples
        total_samples = sum(update.num_samples for update in client_updates)
        
        if total_samples == 0:
            raise ValueError("Total number of samples is zero")
        
        # Initialize aggregated weights
        aggregated_weights = {}
        
        # Get all weight keys from first update
        weight_keys = client_updates[0].model_weights.keys()
        
        for key in weight_keys:
            # Weighted sum of client weights
            weighted_sum = None
            
            for update in client_updates:
                weight = update.model_weights[key]
                
                if self.use_weighted_average:
                    # Weight by number of samples
                    coeff = update.num_samples / total_samples
                else:
                    # Simple average
                    coeff = 1.0 / len(client_updates)
                
                if weighted_sum is None:
                    weighted_sum = coeff * weight
                else:
                    weighted_sum += coeff * weight
            
            aggregated_weights[key] = weighted_sum
        
        self.round_number += 1
        return aggregated_weights


class FedProx(AggregationStrategy):
    """
    Federated Proximal (FedProx) algorithm with regularization.
    
    Adds a proximal term to the local objective to handle heterogeneous data
    and improve convergence with non-IID data.
    
    Reference: Li et al., "Federated Optimization in Heterogeneous Networks",
    MLSys 2020.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.mu = self.config.get('mu', 0.01)  # Proximal term coefficient
    
    def aggregate(
        self,
        global_weights: Dict[str, np.ndarray],
        client_updates: List[ClientUpdate],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate with proximal regularization.
        
        The proximal term encourages local models to stay close to the global model,
        which helps with convergence on heterogeneous data.
        """
        self._validate_updates(client_updates)
        
        total_samples = sum(update.num_samples for update in client_updates)
        
        aggregated_weights = {}
        
        for key in global_weights.keys():
            weighted_sum = None
            
            for update in client_updates:
                weight = update.model_weights[key]
                coeff = update.num_samples / total_samples
                
                if weighted_sum is None:
                    weighted_sum = coeff * weight
                else:
                    weighted_sum += coeff * weight
            
            # Apply proximal regularization
            # w_new = (1 - mu) * w_agg + mu * w_global
            aggregated_weights[key] = (
                (1 - self.mu) * weighted_sum + 
                self.mu * global_weights[key]
            )
        
        self.round_number += 1
        return aggregated_weights


class FedNova(AggregationStrategy):
    """
    Federated Normalized Averaging (FedNova) algorithm.
    
    Normalizes the aggregation by the number of local steps to handle
    clients with different amounts of local computation.
    
    Reference: Wang et al., "Tackling the Objective Inconsistency Problem
    in Heterogeneous Federated Optimization", NeurIPS 2020.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.eta = self.config.get('learning_rate', 0.01)
    
    def aggregate(
        self,
        global_weights: Dict[str, np.ndarray],
        client_updates: List[ClientUpdate],
        local_steps: Optional[List[int]] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate with normalization by local steps.
        
        Args:
            global_weights: Current global model weights
            client_updates: List of updates from clients
            local_steps: Number of local SGD steps for each client
        """
        self._validate_updates(client_updates)
        
        if local_steps is None:
            # Assume equal number of steps if not provided
            local_steps = [1] * len(client_updates)
        
        total_samples = sum(update.num_samples for update in client_updates)
        
        # Calculate normalized coefficients
        tau_eff = sum(
            (update.num_samples / total_samples) * steps
            for update, steps in zip(client_updates, local_steps)
        )
        
        aggregated_weights = {}
        
        for key in global_weights.keys():
            normalized_sum = None
            
            for update, steps in zip(client_updates, local_steps):
                weight = update.model_weights[key]
                
                # Normalize by effective steps
                coeff = (update.num_samples / total_samples) * (tau_eff / steps)
                
                if normalized_sum is None:
                    normalized_sum = coeff * weight
                else:
                    normalized_sum += coeff * weight
            
            aggregated_weights[key] = normalized_sum
        
        self.round_number += 1
        return aggregated_weights


class Scaffold(AggregationStrategy):
    """
    SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.
    
    Uses control variates to correct for client drift in non-IID settings.
    
    Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
    for Federated Learning", ICML 2020.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.global_control_variate: Optional[Dict[str, np.ndarray]] = None
        self.client_control_variates: Dict[str, Dict[str, np.ndarray]] = {}
        self.learning_rate = self.config.get('learning_rate', 0.01)
    
    def aggregate(
        self,
        global_weights: Dict[str, np.ndarray],
        client_updates: List[ClientUpdate],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate using control variates to correct for client drift.
        """
        self._validate_updates(client_updates)
        
        # Initialize global control variate if needed
        if self.global_control_variate is None:
            self.global_control_variate = {
                key: np.zeros_like(value) 
                for key, value in global_weights.items()
            }
        
        total_samples = sum(update.num_samples for update in client_updates)
        
        aggregated_weights = {}
        delta_control_variate = {}
        
        for key in global_weights.keys():
            weighted_sum = None
            control_sum = None
            
            for update in client_updates:
                weight = update.model_weights[key]
                coeff = update.num_samples / total_samples
                
                # Get client control variate
                client_cv = self.client_control_variates.get(
                    update.client_id, {}
                ).get(key, np.zeros_like(weight))
                
                if weighted_sum is None:
                    weighted_sum = coeff * weight
                    control_sum = coeff * client_cv
                else:
                    weighted_sum += coeff * weight
                    control_sum += coeff * client_cv
            
            aggregated_weights[key] = weighted_sum
            
            # Update global control variate
            delta_control_variate[key] = (
                self.global_control_variate[key] - control_sum
            )
            self.global_control_variate[key] -= delta_control_variate[key]
        
        self.round_number += 1
        return aggregated_weights
    
    def get_control_variate_update(
        self, 
        client_id: str
    ) -> Optional[Dict[str, np.ndarray]]:
        """Get control variate update for a specific client."""
        return self.client_control_variates.get(client_id)


class AdaptiveAggregation(AggregationStrategy):
    """
    Adaptive aggregation that dynamically adjusts based on client quality.
    
    Weights clients based on their historical performance and reliability.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client_history: Dict[str, List[float]] = {}
        self.history_window = self.config.get('history_window', 5)
        self.min_weight = self.config.get('min_weight', 0.1)
    
    def aggregate(
        self,
        global_weights: Dict[str, np.ndarray],
        client_updates: List[ClientUpdate],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate with adaptive weighting based on client performance history.
        """
        self._validate_updates(client_updates)
        
        # Update client history with current metrics
        for update in client_updates:
            if update.client_id not in self.client_history:
                self.client_history[update.client_id] = []
            
            # Use accuracy as quality metric (can be customized)
            quality = update.metrics.get('accuracy', 0.5)
            self.client_history[update.client_id].append(quality)
            
            # Keep only recent history
            if len(self.client_history[update.client_id]) > self.history_window:
                self.client_history[update.client_id].pop(0)
        
        # Calculate adaptive weights
        client_weights = {}
        for update in client_updates:
            history = self.client_history.get(update.client_id, [0.5])
            avg_quality = np.mean(history)
            # Combine sample count with quality
            client_weights[update.client_id] = (
                update.num_samples * max(avg_quality, self.min_weight)
            )
        
        total_weight = sum(client_weights.values())
        
        aggregated_weights = {}
        
        for key in global_weights.keys():
            weighted_sum = None
            
            for update in client_updates:
                weight = update.model_weights[key]
                coeff = client_weights[update.client_id] / total_weight
                
                if weighted_sum is None:
                    weighted_sum = coeff * weight
                else:
                    weighted_sum += coeff * weight
            
            aggregated_weights[key] = weighted_sum
        
        self.round_number += 1
        return aggregated_weights


def get_aggregation_strategy(
    strategy_name: str,
    config: Optional[Dict[str, Any]] = None
) -> AggregationStrategy:
    """
    Factory function to create aggregation strategy instances.
    
    Args:
        strategy_name: Name of the aggregation strategy
        config: Configuration dictionary for the strategy
        
    Returns:
        AggregationStrategy instance
    """
    strategies = {
        'fedavg': FedAvg,
        'fedprox': FedProx,
        'fednova': FedNova,
        'scaffold': Scaffold,
        'adaptive': AdaptiveAggregation,
    }
    
    strategy_class = strategies.get(strategy_name.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown aggregation strategy: {strategy_name}")
    
    return strategy_class(config)
