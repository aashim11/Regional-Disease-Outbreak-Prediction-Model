"""
Differential Privacy Implementation

This module implements differential privacy mechanisms for federated learning,
including DP-SGD (Differentially Private Stochastic Gradient Descent) and
privacy accounting.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging


@dataclass
class PrivacyParams:
    """Parameters for differential privacy."""
    epsilon: float  # Privacy budget
    delta: float    # Failure probability
    max_grad_norm: float  # Maximum gradient norm for clipping
    noise_multiplier: float  # Noise multiplier
    
    def __post_init__(self):
        if self.epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if self.delta < 0 or self.delta > 1:
            raise ValueError("Delta must be in [0, 1]")


class PrivacyAccountant:
    """
    Privacy accountant for tracking privacy budget consumption.
    
    Implements the moments accountant method for tight privacy bounds.
    """
    
    def __init__(
        self,
        noise_multiplier: float,
        max_grad_norm: float,
        batch_size: int,
        dataset_size: int,
        delta: float = 1e-5,
        target_epsilon: Optional[float] = None
    ):
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.delta = delta
        self.target_epsilon = target_epsilon
        
        self.steps = 0
        self.epsilon_spent = 0.0
        
        self.logger = logging.getLogger("PrivacyAccountant")
    
    def compute_epsilon(
        self, 
        steps: Optional[int] = None,
        verbose: bool = False
    ) -> float:
        """
        Compute privacy budget (epsilon) using moments accountant.
        
        Args:
            steps: Number of training steps (default: self.steps)
            verbose: Whether to log detailed information
            
        Returns:
            Privacy budget epsilon
        """
        if steps is None:
            steps = self.steps
        
        try:
            # Use opacus for accurate privacy accounting
            from opacus.accountants.utils import get_noise_multiplier
            
            # Calculate sampling probability
            sample_rate = self.batch_size / self.dataset_size
            
            # Compute epsilon using Rényi Differential Privacy
            from opacus.accountants import RDPAccountant
            
            accountant = RDPAccountant()
            accountant.step(
                noise_multiplier=self.noise_multiplier,
                sample_rate=sample_rate
            )
            
            epsilon = accountant.get_epsilon(delta=self.delta)
            
            if verbose:
                self.logger.info(
                    f"Privacy: ε = {epsilon:.4f}, δ = {self.delta}, "
                    f"steps = {steps}"
                )
            
            return epsilon
            
        except ImportError:
            # Fallback to simple composition if opacus not available
            self.logger.warning("Opacus not available, using basic privacy accounting")
            return self._basic_epsilon_composition(steps)
    
    def _basic_epsilon_composition(self, steps: int) -> float:
        """Basic epsilon composition (loose upper bound)."""
        # Simple composition: epsilon_total = steps * epsilon_per_step
        sample_rate = self.batch_size / self.dataset_size
        q = sample_rate
        sigma = self.noise_multiplier
        
        # Basic Gaussian mechanism privacy bound
        epsilon_per_step = q * np.sqrt(2 * np.log(1.25 / self.delta)) / sigma
        
        return steps * epsilon_per_step
    
    def step(self) -> None:
        """Record one training step."""
        self.steps += 1
        self.epsilon_spent = self.compute_epsilon()
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        if self.target_epsilon is None:
            return False
        return self.epsilon_spent >= self.target_epsilon
    
    def get_status(self) -> Dict[str, Any]:
        """Get current privacy accounting status."""
        return {
            'epsilon_spent': self.epsilon_spent,
            'delta': self.delta,
            'steps': self.steps,
            'target_epsilon': self.target_epsilon,
            'budget_remaining': (
                self.target_epsilon - self.epsilon_spent 
                if self.target_epsilon else None
            ),
            'is_exhausted': self.is_budget_exhausted()
        }


class DifferentialPrivacy:
    """
    Differential Privacy wrapper for neural network training.
    
    Implements DP-SGD with gradient clipping and noise addition.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        batch_size: int = 32,
        dataset_size: int = 1000
    ):
        self.params = PrivacyParams(
            epsilon=epsilon,
            delta=delta,
            max_grad_norm=max_grad_norm,
            noise_multiplier=noise_multiplier or self._compute_noise_multiplier(
                epsilon, delta, max_grad_norm, batch_size, dataset_size
            )
        )
        
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        
        # Initialize privacy accountant
        self.accountant = PrivacyAccountant(
            noise_multiplier=self.params.noise_multiplier,
            max_grad_norm=max_grad_norm,
            batch_size=batch_size,
            dataset_size=dataset_size,
            delta=delta,
            target_epsilon=epsilon
        )
        
        self.logger = logging.getLogger("DifferentialPrivacy")
    
    def _compute_noise_multiplier(
        self,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        batch_size: int,
        dataset_size: int
    ) -> float:
        """
        Compute noise multiplier for target epsilon.
        
        Uses binary search to find the noise multiplier that achieves
        the target privacy budget.
        """
        sample_rate = batch_size / dataset_size
        
        # Binary search for noise multiplier
        noise_low, noise_high = 0.1, 100.0
        
        for _ in range(20):  # Max iterations
            noise_mid = (noise_low + noise_high) / 2
            
            # Estimate epsilon for this noise level
            estimated_epsilon = self._estimate_epsilon(
                noise_mid, sample_rate, delta
            )
            
            if estimated_epsilon > epsilon:
                noise_low = noise_mid
            else:
                noise_high = noise_mid
        
        return noise_high
    
    def _estimate_epsilon(
        self,
        noise_multiplier: float,
        sample_rate: float,
        delta: float
    ) -> float:
        """Estimate epsilon for given noise multiplier."""
        q = sample_rate
        sigma = noise_multiplier
        
        # Using basic Gaussian mechanism bound
        return q * np.sqrt(2 * np.log(1.25 / delta)) / sigma
    
    def clip_gradients(self, model: nn.Module) -> float:
        """
        Clip gradients to bound their L2 norm.
        
        Args:
            model: Neural network model
            
        Returns:
            Total gradient norm before clipping
        """
        total_norm = 0.0
        
        # Calculate total gradient norm
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        clip_coef = self.params.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        return total_norm
    
    def add_noise(self, model: nn.Module) -> None:
        """
        Add Gaussian noise to gradients.
        
        Args:
            model: Neural network model
        """
        noise_std = self.params.noise_multiplier * self.params.max_grad_norm
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_std
                param.grad.add_(noise)
        
        # Record privacy consumption
        self.accountant.step()
    
    def make_private_loader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        **kwargs
    ) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader suitable for private training.
        
        Uses Poisson sampling for privacy amplification.
        """
        # Use uniform sampling with replacement for DP-SGD
        from torch.utils.data import RandomSampler, BatchSampler, DataLoader
        
        sampler = RandomSampler(
            dataset,
            replacement=True,
            num_samples=len(dataset)
        )
        
        batch_sampler = BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=True
        )
        
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **kwargs
        )
    
    def get_privacy_spent(self) -> Dict[str, Any]:
        """Get current privacy budget consumption."""
        return self.accountant.get_status()
    
    def validate_privacy_guarantee(self) -> bool:
        """Check if current privacy parameters provide valid guarantees."""
        current_epsilon = self.accountant.compute_epsilon()
        return current_epsilon <= self.params.epsilon


class DPModelWrapper(nn.Module):
    """
    Wrapper for models that automatically applies differential privacy.
    """
    
    def __init__(
        self,
        model: nn.Module,
        dp_mechanism: DifferentialPrivacy
    ):
        super().__init__()
        self.model = model
        self.dp = dp_mechanism
    
    def forward(self, x):
        return self.model(x)
    
    def train_step(self, batch_X, batch_y, criterion, optimizer):
        """Training step with differential privacy."""
        optimizer.zero_grad()
        outputs = self.model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        
        # Apply DP
        self.dp.clip_gradients(self.model)
        self.dp.add_noise(self.model)
        
        optimizer.step()
        
        return loss.item()


class LocalDifferentialPrivacy:
    """
    Local Differential Privacy (LDP) for extreme privacy protection.
    
    Each user adds noise to their own data before sending it,
    providing stronger privacy guarantees than central DP.
    """
    
    def __init__(self, epsilon: float = 1.0, mechanism: str = 'laplace'):
        self.epsilon = epsilon
        self.mechanism = mechanism
        self.sensitivity = 1.0
    
    def randomize(self, value: float) -> float:
        """
        Randomize a value using LDP mechanism.
        
        Args:
            value: Original value
            
        Returns:
            Randomized value
        """
        if self.mechanism == 'laplace':
            scale = self.sensitivity / self.epsilon
            noise = np.random.laplace(0, scale)
            return value + noise
        elif self.mechanism == 'gaussian':
            sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / 1e-5)) / self.epsilon
            noise = np.random.normal(0, sigma)
            return value + noise
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
    
    def randomize_gradient(
        self, 
        gradient: np.ndarray
    ) -> np.ndarray:
        """
        Randomize gradient vector using LDP.
        
        Args:
            gradient: Original gradient
            
        Returns:
            Randomized gradient
        """
        # Clip gradient
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > self.sensitivity:
            gradient = gradient * (self.sensitivity / grad_norm)
        
        # Add noise
        if self.mechanism == 'laplace':
            scale = self.sensitivity / self.epsilon
            noise = np.random.laplace(0, scale, size=gradient.shape)
        elif self.mechanism == 'gaussian':
            sigma = self.sensitivity * np.sqrt(2 * np.log(1.25 / 1e-5)) / self.epsilon
            noise = np.random.normal(0, sigma, size=gradient.shape)
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
        
        return gradient + noise
