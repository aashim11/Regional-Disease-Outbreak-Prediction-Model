"""
Secure Multi-Party Computation and Secure Aggregation

This module implements secure aggregation protocols for federated learning,
ensuring that the server can only see aggregated updates, not individual
client updates.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod
import secrets
import logging
from collections import defaultdict


@dataclass
class SecureAggConfig:
    """Configuration for secure aggregation."""
    num_clients: int
    threshold: int  # Minimum clients needed for aggregation
    num_bits: int = 32  # Number of bits for quantization
    modulus: Optional[int] = None


class ShamirSecretSharing:
    """
    Shamir's Secret Sharing scheme.
    
    Splits a secret into n shares where any t shares can reconstruct
    the secret, but fewer than t shares reveal no information.
    """
    
    def __init__(self, threshold: int, num_shares: int, prime: Optional[int] = None):
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = prime or (2**31 - 1)  # Large prime
        self.logger = logging.getLogger("ShamirSecretSharing")
    
    def _eval_at_poly(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x."""
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % self.prime
        return result
    
    def split_secret(self, secret: int) -> List[Tuple[int, int]]:
        """
        Split a secret into shares.
        
        Args:
            secret: Secret integer to split
            
        Returns:
            List of (x, y) share pairs
        """
        # Generate random polynomial coefficients
        coefficients = [secret] + [
            secrets.randbelow(self.prime) 
            for _ in range(self.threshold - 1)
        ]
        
        # Generate shares
        shares = []
        for i in range(1, self.num_shares + 1):
            y = self._eval_at_poly(coefficients, i)
            shares.append((i, y))
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> int:
        """
        Reconstruct secret from shares using Lagrange interpolation.
        
        Args:
            shares: List of (x, y) share pairs
            
        Returns:
            Reconstructed secret
        """
        if len(shares) < self.threshold:
            raise ValueError(
                f"Need at least {self.threshold} shares, got {len(shares)}"
            )
        
        secret = 0
        
        for i, (x_i, y_i) in enumerate(shares):
            # Compute Lagrange basis polynomial
            numerator = 1
            denominator = 1
            
            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-x_j)) % self.prime
                    denominator = (denominator * (x_i - x_j)) % self.prime
            
            # Lagrange coefficient
            lagrange_coeff = numerator * pow(denominator, -1, self.prime)
            secret = (secret + y_i * lagrange_coeff) % self.prime
        
        return secret


class SecureAggregation:
    """
    Secure Aggregation Protocol for Federated Learning.
    
    Ensures that the server learns only the aggregated model update
    and nothing about individual client updates.
    
    Based on: Bonawitz et al., "Practical Secure Aggregation for
    Privacy-Preserving Machine Learning", CCS 2017.
    """
    
    def __init__(self, config: Optional[SecureAggConfig] = None):
        self.config = config or SecureAggConfig(
            num_clients=10,
            threshold=5
        )
        self.logger = logging.getLogger("SecureAggregation")
        
        # Pairwise seeds for mask generation
        self.pairwise_seeds: Dict[Tuple[str, str], int] = {}
        
        # Self masks
        self.self_masks: Dict[str, int] = {}
        
        # Quantization parameters
        self.quantization_bits = self.config.num_bits
        self.modulus = self.config.modulus or (2 ** self.quantization_bits)
    
    def generate_keys(self, client_id: str) -> Dict[str, Any]:
        """
        Generate cryptographic keys for a client.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            Dictionary containing public and private keys
        """
        # Generate pairwise seeds with other clients
        # In practice, these would be established via secure key exchange
        private_key = secrets.randbelow(self.modulus)
        
        return {
            'client_id': client_id,
            'private_key': private_key,
            'public_key': None  # Would be derived from private key
        }
    
    def establish_pairwise_seeds(
        self,
        client_id: str,
        peer_ids: List[str]
    ) -> None:
        """
        Establish pairwise seeds with other clients.
        
        In a real implementation, this would use Diffie-Hellman key exchange.
        """
        for peer_id in peer_ids:
            if client_id < peer_id:
                seed = secrets.randbelow(2**128)
                self.pairwise_seeds[(client_id, peer_id)] = seed
                self.pairwise_seeds[(peer_id, client_id)] = seed
    
    def generate_mask(
        self,
        client_id: str,
        shape: Tuple[int, ...],
        round_num: int
    ) -> np.ndarray:
        """
        Generate mask for a client's weights.
        
        The mask is the sum of pairwise masks with other clients
        plus a self-mask.
        """
        mask = np.zeros(shape, dtype=np.int64)
        
        # Add pairwise masks
        for (c1, c2), seed in self.pairwise_seeds.items():
            if c1 == client_id:
                # Generate mask from seed
                np.random.seed(seed + round_num)
                pairwise_mask = np.random.randint(
                    -self.modulus//2, 
                    self.modulus//2, 
                    size=shape
                )
                mask += pairwise_mask
            elif c2 == client_id:
                # Subtract mask (masks cancel out in aggregation)
                np.random.seed(seed + round_num)
                pairwise_mask = np.random.randint(
                    -self.modulus//2, 
                    self.modulus//2, 
                    size=shape
                )
                mask -= pairwise_mask
        
        # Add self-mask (will be revealed after aggregation)
        if client_id not in self.self_masks:
            self.self_masks[client_id] = secrets.randbelow(2**128)
        
        np.random.seed(self.self_masks[client_id] + round_num)
        self_mask = np.random.randint(
            -self.modulus//2, 
            self.modulus//2, 
            size=shape
        )
        mask += self_mask
        
        return mask % self.modulus
    
    def quantize_weights(
        self,
        weights: Dict[str, np.ndarray],
        clipping_bound: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Quantize floating-point weights to integers.
        
        Args:
            weights: Dictionary of weight arrays
            clipping_bound: Maximum absolute value for clipping
            
        Returns:
            Quantized weights
        """
        quantized = {}
        
        for key, weight in weights.items():
            # Clip weights
            clipped = np.clip(weight, -clipping_bound, clipping_bound)
            
            # Scale to integer range
            scale = (self.modulus / 2 - 1) / clipping_bound
            quantized_weight = np.round(clipped * scale).astype(np.int64)
            
            # Ensure positive for modular arithmetic
            quantized_weight = (quantized_weight + self.modulus) % self.modulus
            
            quantized[key] = quantized_weight
        
        return quantized
    
    def dequantize_weights(
        self,
        quantized_weights: Dict[str, np.ndarray],
        clipping_bound: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Convert quantized weights back to floating-point.
        
        Args:
            quantized_weights: Dictionary of quantized weights
            clipping_bound: Clipping bound used for quantization
            
        Returns:
            Dequantized floating-point weights
        """
        weights = {}
        scale = (self.modulus / 2 - 1) / clipping_bound
        
        for key, q_weight in quantized_weights.items():
            # Handle modular arithmetic
            q_weight = q_weight.astype(np.float64)
            q_weight[q_weight >= self.modulus // 2] -= self.modulus
            
            # Scale back
            weights[key] = q_weight / scale
        
        return weights
    
    def mask_weights(
        self,
        client_id: str,
        weights: Dict[str, np.ndarray],
        round_num: int
    ) -> Dict[str, np.ndarray]:
        """
        Apply mask to client's weights.
        
        Args:
            client_id: Client identifier
            weights: Client's model weights
            round_num: Current round number
            
        Returns:
            Masked weights
        """
        # Quantize weights
        quantized = self.quantize_weights(weights)
        
        # Generate and apply mask
        masked = {}
        for key, q_weight in quantized.items():
            mask = self.generate_mask(client_id, q_weight.shape, round_num)
            masked[key] = (q_weight + mask) % self.modulus
        
        return masked
    
    def aggregate_masked_weights(
        self,
        masked_weights: List[Dict[str, np.ndarray]],
        missing_clients: Optional[Set[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate masked weights from multiple clients.
        
        The masks cancel out during aggregation, revealing only the sum.
        
        Args:
            masked_weights: List of masked weight dictionaries
            missing_clients: Set of clients that didn't submit updates
            
        Returns:
            Aggregated weights (still masked if clients are missing)
        """
        if not masked_weights:
            raise ValueError("No weights to aggregate")
        
        # Sum all masked weights
        aggregated = {}
        weight_keys = masked_weights[0].keys()
        
        for key in weight_keys:
            summed = sum(weights[key] for weights in masked_weights)
            aggregated[key] = summed % self.modulus
        
        # If clients are missing, we need to handle their masks
        if missing_clients:
            self.logger.warning(
                f"Missing clients: {missing_clients}. "
                f"Need to handle their masks."
            )
            # In practice, would use secret sharing to recover masks
            # For now, this is a simplified version
        
        return aggregated
    
    def unmask_aggregate(
        self,
        aggregated_masked: Dict[str, np.ndarray],
        surviving_clients: List[str],
        round_num: int
    ) -> Dict[str, np.ndarray]:
        """
        Remove masks from aggregated weights.
        
        Args:
            aggregated_masked: Aggregated masked weights
            surviving_clients: List of clients that participated
            round_num: Current round number
            
        Returns:
            Unmasked aggregated weights
        """
        # Remove self-masks from surviving clients
        unmasked = {}
        
        for key, value in aggregated_masked.items():
            total_mask = np.zeros_like(value, dtype=np.int64)
            
            for client_id in surviving_clients:
                if client_id in self.self_masks:
                    np.random.seed(self.self_masks[client_id] + round_num)
                    mask = np.random.randint(
                        -self.modulus//2, 
                        self.modulus//2, 
                        size=value.shape
                    )
                    total_mask = (total_mask + mask) % self.modulus
            
            # Remove mask
            unmasked_value = (value - total_mask) % self.modulus
            unmasked[key] = unmasked_value
        
        # Dequantize
        return self.dequantize_weights(unmasked)


class SMPCAggregator:
    """
    Secure Multi-Party Computation Aggregator.
    
    Uses SMPC techniques to compute aggregates without revealing
    individual values.
    """
    
    def __init__(self, num_parties: int, threshold: int):
        self.num_parties = num_parties
        self.threshold = threshold
        self.secret_sharing = ShamirSecretSharing(threshold, num_parties)
        self.logger = logging.getLogger("SMPCAggregator")
    
    def share_values(
        self,
        values: Dict[str, np.ndarray],
        client_id: str
    ) -> List[Dict[str, List[Tuple[int, int]]]]:
        """
        Share values using secret sharing.
        
        Args:
            values: Dictionary of values to share
            client_id: Client identifier
            
        Returns:
            List of shares for each party
        """
        shares_per_party = [defaultdict(list) for _ in range(self.num_parties)]
        
        for key, value_array in values.items():
            for val in value_array.flatten():
                # Convert float to fixed-point integer
                fixed_point = int(val * 10000)  # Scale factor
                
                # Create shares
                shares = self.secret_sharing.split_secret(fixed_point)
                
                # Distribute shares
                for party_idx, (x, y) in enumerate(shares):
                    shares_per_party[party_idx][key].append((x, y))
        
        return shares_per_party
    
    def aggregate_shares(
        self,
        all_shares: List[List[Dict[str, List[Tuple[int, int]]]]]
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate shares from multiple clients.
        
        Args:
            all_shares: List of share collections from clients
            
        Returns:
            Aggregated values
        """
        # Sum shares for each party
        summed_shares = [defaultdict(list) for _ in range(self.num_parties)]
        
        for client_shares in all_shares:
            for party_idx, party_shares in enumerate(client_shares):
                for key, shares in party_shares.items():
                    if not summed_shares[party_idx][key]:
                        summed_shares[party_idx][key] = shares
                    else:
                        # Sum shares (modular addition)
                        summed = []
                        for (x1, y1), (x2, y2) in zip(
                            summed_shares[party_idx][key], shares
                        ):
                            assert x1 == x2
                            summed.append((x1, (y1 + y2) % self.secret_sharing.prime))
                        summed_shares[party_idx][key] = summed
        
        # Reconstruct aggregated values
        aggregated = {}
        
        for key in summed_shares[0].keys():
            # Collect shares from threshold parties
            shares_for_key = [
                summed_shares[i][key][:self.threshold] 
                for i in range(self.threshold)
            ]
            
            # Reconstruct each value
            reconstructed = []
            for share_tuple in zip(*shares_for_key):
                secret = self.secret_sharing.reconstruct_secret(list(share_tuple))
                reconstructed.append(secret / 10000)  # Scale back
            
            # Reshape to original shape (assuming 1D for simplicity)
            aggregated[key] = np.array(reconstructed)
        
        return aggregated
