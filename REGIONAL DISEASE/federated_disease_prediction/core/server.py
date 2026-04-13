"""
Federated Learning Server Implementation

This module implements the central server for federated learning,
handling client coordination, model aggregation, and global model updates.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from .aggregator import (
    AggregationStrategy, 
    ClientUpdate, 
    get_aggregation_strategy
)
from .client import FederatedClient, TrainingResult


@dataclass
class ServerConfig:
    """Configuration for federated learning server."""
    num_rounds: int = 100
    clients_per_round: int = 10
    fraction_clients: float = 0.2
    aggregation_strategy: str = 'fedavg'
    aggregation_config: Dict[str, Any] = field(default_factory=dict)
    min_clients: int = 2
    timeout_seconds: float = 300.0
    checkpoint_dir: str = './checkpoints'
    log_interval: int = 1
    eval_interval: int = 5
    device: str = 'cpu'


@dataclass
class ServerState:
    """Current state of the federated learning server."""
    round_number: int = 0
    global_weights: Optional[Dict[str, np.ndarray]] = None
    client_updates: List[ClientUpdate] = field(default_factory=list)
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)


class FederatedServer:
    """
    Federated Learning Server
    
    Coordinates the federated learning process, including:
    - Client selection and management
    - Model aggregation
    - Global model distribution
    - Evaluation and monitoring
    """
    
    def __init__(
        self,
        config: ServerConfig,
        model: nn.Module,
        test_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        self.config = config
        self.model = model.to(config.device)
        self.device = config.device
        self.test_data = test_data
        
        # Initialize aggregation strategy
        self.aggregator = get_aggregation_strategy(
            config.aggregation_strategy,
            config.aggregation_config
        )
        
        # Server state
        self.state = ServerState()
        self.state.global_weights = self._get_model_weights()
        
        # Client management
        self.registered_clients: Dict[str, FederatedClient] = {}
        self.available_clients: Set[str] = set()
        self.client_last_seen: Dict[str, float] = {}
        
        # Threading
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Logging
        self.logger = logging.getLogger("FederatedServer")
        self.logger.setLevel(logging.INFO)
        
        # Callbacks
        self.round_start_callbacks: List[Callable] = []
        self.round_end_callbacks: List[Callable] = []
        self.aggregation_callbacks: List[Callable] = []
    
    def _get_model_weights(self) -> Dict[str, np.ndarray]:
        """Get current global model weights."""
        return {
            k: v.cpu().numpy()
            for k, v in self.model.state_dict().items()
        }
    
    def _set_model_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Set global model weights."""
        self.model.load_state_dict({
            k: torch.from_numpy(v).to(self.device)
            for k, v in weights.items()
        })
    
    def register_client(self, client: FederatedClient) -> None:
        """
        Register a new client with the server.
        
        Args:
            client: FederatedClient instance to register
        """
        with self.lock:
            self.registered_clients[client.client_id] = client
            self.available_clients.add(client.client_id)
            self.client_last_seen[client.client_id] = time.time()
            
            self.logger.info(
                f"Registered client {client.client_id}. "
                f"Total clients: {len(self.registered_clients)}"
            )
    
    def unregister_client(self, client_id: str) -> None:
        """Unregister a client from the server."""
        with self.lock:
            if client_id in self.registered_clients:
                del self.registered_clients[client_id]
                self.available_clients.discard(client_id)
                del self.client_last_seen[client_id]
                self.logger.info(f"Unregistered client {client_id}")
    
    def mark_client_available(self, client_id: str) -> None:
        """Mark a client as available for training."""
        with self.lock:
            if client_id in self.registered_clients:
                self.available_clients.add(client_id)
                self.client_last_seen[client_id] = time.time()
    
    def mark_client_unavailable(self, client_id: str) -> None:
        """Mark a client as unavailable."""
        with self.lock:
            self.available_clients.discard(client_id)
    
    def select_clients(self, num_clients: Optional[int] = None) -> List[str]:
        """
        Select clients for the current round.
        
        Args:
            num_clients: Number of clients to select (default: clients_per_round)
            
        Returns:
            List of selected client IDs
        """
        if num_clients is None:
            num_clients = self.config.clients_per_round
        
        with self.lock:
            available = list(self.available_clients)
        
        if len(available) < self.config.min_clients:
            raise ValueError(
                f"Not enough available clients. "
                f"Need {self.config.min_clients}, have {len(available)}"
            )
        
        # Weighted sampling based on data size
        weights = []
        for client_id in available:
            client = self.registered_clients[client_id]
            stats = client.get_data_statistics()
            weights.append(stats.get('num_samples', 1))
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        num_to_select = min(num_clients, len(available))
        selected = np.random.choice(
            available,
            size=num_to_select,
            replace=False,
            p=weights
        ).tolist()
        
        return selected
    
    def distribute_model(self, client_ids: List[str]) -> None:
        """
        Distribute global model to selected clients.
        
        Args:
            client_ids: List of client IDs to receive the model
        """
        global_weights = self.state.global_weights
        
        for client_id in client_ids:
            if client_id in self.registered_clients:
                client = self.registered_clients[client_id]
                client.set_global_weights(global_weights)
    
    def collect_updates(
        self,
        client_ids: List[str],
        timeout: Optional[float] = None
    ) -> List[ClientUpdate]:
        """
        Collect model updates from clients.
        
        Args:
            client_ids: List of client IDs to collect from
            timeout: Maximum time to wait for updates
            
        Returns:
            List of client updates
        """
        if timeout is None:
            timeout = self.config.timeout_seconds
        
        updates = []
        
        def train_client(client_id: str) -> Optional[ClientUpdate]:
            try:
                client = self.registered_clients[client_id]
                result = client.train_round(
                    global_weights=self.state.global_weights,
                    round_number=self.state.round_number
                )
                
                return ClientUpdate(
                    client_id=result.client_id,
                    model_weights=result.model_weights,
                    num_samples=result.num_samples,
                    metrics=result.metrics,
                    round_number=result.round_number,
                    timestamp=result.timestamp
                )
            except Exception as e:
                self.logger.error(f"Error training client {client_id}: {e}")
                return None
        
        # Execute training in parallel
        futures = {
            self.executor.submit(train_client, cid): cid 
            for cid in client_ids
        }
        
        for future in as_completed(futures, timeout=timeout):
            client_id = futures[future]
            try:
                update = future.result()
                if update is not None:
                    updates.append(update)
                    self.mark_client_available(client_id)
            except Exception as e:
                self.logger.error(f"Failed to get update from {client_id}: {e}")
                self.mark_client_unavailable(client_id)
        
        return updates
    
    def aggregate_updates(self, updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
        """
        Aggregate client updates using the configured strategy.
        
        Args:
            updates: List of client updates
            
        Returns:
            Aggregated global weights
        """
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Execute aggregation
        new_weights = self.aggregator.aggregate(
            self.state.global_weights,
            updates
        )
        
        # Execute callbacks
        for callback in self.aggregation_callbacks:
            callback(self.state.global_weights, updates, new_weights)
        
        return new_weights
    
    def evaluate_global_model(self) -> Dict[str, float]:
        """
        Evaluate the global model on test data.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.test_data is None:
            return {}
        
        self.model.eval()
        X_test, y_test = self.test_data
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_test)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            # Calculate metrics
            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, 
                f1_score, roc_auc_score
            )
            
            y_true = y_test.cpu().numpy().flatten()
            y_pred = preds.cpu().numpy().flatten()
            y_prob = probs.cpu().numpy().flatten()
            
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
            }
        
        return metrics
    
    def run_round(self) -> Dict[str, Any]:
        """
        Execute one round of federated learning.
        
        Returns:
            Dictionary containing round results and metrics
        """
        round_start = time.time()
        
        # Execute round start callbacks
        for callback in self.round_start_callbacks:
            callback(self.state.round_number)
        
        # Select clients
        selected_clients = self.select_clients()
        self.logger.info(
            f"Round {self.state.round_number}: Selected {len(selected_clients)} clients"
        )
        
        # Distribute global model
        self.distribute_model(selected_clients)
        
        # Collect updates
        updates = self.collect_updates(selected_clients)
        
        if len(updates) < self.config.min_clients:
            self.logger.warning(
                f"Insufficient updates received ({len(updates)}). Skipping aggregation."
            )
            return {'round': self.state.round_number, 'status': 'skipped'}
        
        # Aggregate updates
        new_weights = self.aggregate_updates(updates)
        self.state.global_weights = new_weights
        self._set_model_weights(new_weights)
        
        # Evaluate global model
        global_metrics = {}
        if self.state.round_number % self.config.eval_interval == 0:
            global_metrics = self.evaluate_global_model()
            self.logger.info(
                f"Round {self.state.round_number} Global Metrics: {global_metrics}"
            )
        
        # Calculate client metrics summary
        client_metrics = defaultdict(list)
        for update in updates:
            for metric_name, value in update.metrics.items():
                client_metrics[metric_name].append(value)
        
        round_metrics = {
            'round': self.state.round_number,
            'num_clients': len(selected_clients),
            'num_updates': len(updates),
            'round_time': time.time() - round_start,
            'global_metrics': global_metrics,
            'client_metrics': {
                name: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
                for name, values in client_metrics.items()
            }
        }
        
        self.state.metrics_history.append(round_metrics)
        self.state.round_number += 1
        
        # Execute round end callbacks
        for callback in self.round_end_callbacks:
            callback(self.state.round_number - 1, round_metrics)
        
        return round_metrics
    
    def train(self) -> List[Dict[str, Any]]:
        """
        Run the complete federated learning training process.
        
        Returns:
            List of metrics for each round
        """
        self.logger.info(
            f"Starting federated learning for {self.config.num_rounds} rounds"
        )
        
        all_metrics = []
        
        for round_num in range(self.config.num_rounds):
            try:
                metrics = self.run_round()
                all_metrics.append(metrics)
                
                if round_num % self.config.log_interval == 0:
                    self.logger.info(
                        f"Completed round {round_num}/{self.config.num_rounds}"
                    )
                
            except Exception as e:
                self.logger.error(f"Error in round {round_num}: {e}")
                raise
        
        total_time = time.time() - self.state.start_time
        self.logger.info(
            f"Federated learning completed in {total_time:.2f} seconds"
        )
        
        return all_metrics
    
    def get_global_model(self) -> nn.Module:
        """Get the current global model."""
        return self.model
    
    def save_checkpoint(self, filepath: Optional[str] = None) -> None:
        """
        Save server checkpoint.
        
        Args:
            filepath: Path to save checkpoint (default: auto-generated)
        """
        import os
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        if filepath is None:
            filepath = os.path.join(
                self.config.checkpoint_dir,
                f"checkpoint_round_{self.state.round_number}.pt"
            )
        
        checkpoint = {
            'round_number': self.state.round_number,
            'global_weights': self.state.global_weights,
            'model_state_dict': self.model.state_dict(),
            'metrics_history': self.state.metrics_history,
            'config': self.config,
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Saved checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load server checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.state.round_number = checkpoint['round_number']
        self.state.global_weights = checkpoint['global_weights']
        self.state.metrics_history = checkpoint['metrics_history']
        
        self._set_model_weights(self.state.global_weights)
        
        self.logger.info(
            f"Loaded checkpoint from round {self.state.round_number}"
        )
    
    def add_callback(self, event: str, callback: Callable) -> None:
        """
        Add a callback for server events.
        
        Args:
            event: Event type ('round_start', 'round_end', 'aggregation')
            callback: Callback function
        """
        if event == 'round_start':
            self.round_start_callbacks.append(callback)
        elif event == 'round_end':
            self.round_end_callbacks.append(callback)
        elif event == 'aggregation':
            self.aggregation_callbacks.append(callback)
        else:
            raise ValueError(f"Unknown event type: {event}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training progress.
        
        Returns:
            Dictionary containing training summary
        """
        if not self.state.metrics_history:
            return {}
        
        # Extract global metrics over time
        global_accuracies = [
            m['global_metrics'].get('accuracy', 0)
            for m in self.state.metrics_history
            if m.get('global_metrics')
        ]
        
        return {
            'total_rounds': self.state.round_number,
            'total_time': time.time() - self.state.start_time,
            'final_accuracy': global_accuracies[-1] if global_accuracies else 0,
            'best_accuracy': max(global_accuracies) if global_accuracies else 0,
            'num_registered_clients': len(self.registered_clients),
            'num_available_clients': len(self.available_clients),
        }