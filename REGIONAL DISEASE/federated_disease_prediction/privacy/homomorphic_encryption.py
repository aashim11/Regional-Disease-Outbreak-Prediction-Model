"""
Homomorphic Encryption for Federated Learning

This module implements homomorphic encryption schemes that allow
computation on encrypted data, enabling the server to aggregate
encrypted model updates without decryption.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging


@dataclass
class HEConfig:
    """Configuration for homomorphic encryption."""
    key_size: int = 2048
    polynomial_degree: int = 4096
    coefficient_modulus: Optional[List[int]] = None
    scale: float = 2**40
    security_level: int = 128


class HomomorphicEncryption(ABC):
    """Abstract base class for homomorphic encryption schemes."""
    
    def __init__(self, config: Optional[HEConfig] = None):
        self.config = config or HEConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate_keys(self) -> Tuple[Any, Any]:
        """Generate public and private keys."""
        pass
    
    @abstractmethod
    def encrypt(self, plaintext: Union[int, float, np.ndarray], public_key: Any) -> Any:
        """Encrypt plaintext."""
        pass
    
    @abstractmethod
    def decrypt(self, ciphertext: Any, private_key: Any) -> Union[int, float, np.ndarray]:
        """Decrypt ciphertext."""
        pass
    
    @abstractmethod
    def add(self, ct1: Any, ct2: Any) -> Any:
        """Homomorphic addition."""
        pass
    
    @abstractmethod
    def multiply(self, ct1: Any, ct2: Any) -> Any:
        """Homomorphic multiplication."""
        pass


class PaillierEncryption(HomomorphicEncryption):
    """
    Paillier Partially Homomorphic Encryption.
    
    Supports homomorphic addition: E(a) * E(b) = E(a + b)
    Suitable for federated learning aggregation where we need to sum
    encrypted gradients.
    """
    
    def __init__(self, config: Optional[HEConfig] = None):
        super().__init__(config)
        self.public_key = None
        self.private_key = None
        self._init_he()
    
    def _init_he(self) -> None:
        """Initialize Paillier cryptosystem."""
        try:
            from phe import paillier
            self.paillier = paillier
            self.logger.info("Paillier encryption initialized")
        except ImportError:
            self.logger.error(
                "phe library not installed. "
                "Install with: pip install phe"
            )
            raise
    
    def generate_keys(self) -> Tuple[Any, Any]:
        """Generate Paillier key pair."""
        public_key, private_key = self.paillier.generate_paillier_keypair(
            n_length=self.config.key_size
        )
        self.public_key = public_key
        self.private_key = private_key
        return public_key, private_key
    
    def encrypt(
        self, 
        plaintext: Union[int, float, np.ndarray], 
        public_key: Optional[Any] = None
    ) -> Any:
        """
        Encrypt plaintext using Paillier.
        
        Args:
            plaintext: Value or array to encrypt
            public_key: Public key (uses stored key if None)
            
        Returns:
            Encrypted value(s)
        """
        pk = public_key or self.public_key
        if pk is None:
            raise ValueError("No public key available")
        
        if isinstance(plaintext, np.ndarray):
            # Encrypt array element-wise
            encrypted = np.array([
                pk.encrypt(float(x)) for x in plaintext.flatten()
            ])
            return encrypted.reshape(plaintext.shape)
        else:
            return pk.encrypt(float(plaintext))
    
    def decrypt(
        self, 
        ciphertext: Any, 
        private_key: Optional[Any] = None
    ) -> Union[int, float, np.ndarray]:
        """
        Decrypt ciphertext using Paillier.
        
        Args:
            ciphertext: Encrypted value(s)
            private_key: Private key (uses stored key if None)
            
        Returns:
            Decrypted value(s)
        """
        sk = private_key or self.private_key
        if sk is None:
            raise ValueError("No private key available")
        
        if isinstance(ciphertext, np.ndarray):
            # Decrypt array element-wise
            decrypted = np.array([
                sk.decrypt(x) for x in ciphertext.flatten()
            ])
            return decrypted.reshape(ciphertext.shape)
        else:
            return sk.decrypt(ciphertext)
    
    def add(self, ct1: Any, ct2: Any) -> Any:
        """
        Homomorphic addition: E(a) * E(b) = E(a + b)
        
        Args:
            ct1: First encrypted value
            ct2: Second encrypted value
            
        Returns:
            Encrypted sum
        """
        if isinstance(ct1, np.ndarray) and isinstance(ct2, np.ndarray):
            return np.array([c1 + c2 for c1, c2 in zip(ct1.flatten(), ct2.flatten())])
        return ct1 + ct2
    
    def multiply(self, ct1: Any, scalar: float) -> Any:
        """
        Homomorphic multiplication by scalar: E(a)^k = E(a * k)
        
        Args:
            ct1: Encrypted value
            scalar: Scalar multiplier
            
        Returns:
            Encrypted product
        """
        if isinstance(ct1, np.ndarray):
            return np.array([c * scalar for c in ct1.flatten()])
        return ct1 * scalar
    
    def aggregate_encrypted_weights(
        self,
        encrypted_weights_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate encrypted model weights.
        
        Args:
            encrypted_weights_list: List of encrypted weight dictionaries
            
        Returns:
            Aggregated encrypted weights
        """
        if not encrypted_weights_list:
            raise ValueError("No encrypted weights to aggregate")
        
        aggregated = {}
        weight_keys = encrypted_weights_list[0].keys()
        
        for key in weight_keys:
            # Sum encrypted values homomorphically
            total = encrypted_weights_list[0][key]
            for enc_weights in encrypted_weights_list[1:]:
                total = self.add(total, enc_weights[key])
            aggregated[key] = total
        
        return aggregated


class CKKSEncryption(HomomorphicEncryption):
    """
    CKKS (Cheon-Kim-Kim-Song) Homomorphic Encryption.
    
    Supports approximate arithmetic on vectors, making it suitable
    for machine learning applications. Supports both addition and
    multiplication.
    
    Note: This is a simplified interface. Real CKKS implementation
    would use libraries like Microsoft SEAL or TenSEAL.
    """
    
    def __init__(self, config: Optional[HEConfig] = None):
        super().__init__(config)
        self.context = None
        self._init_ckks()
    
    def _init_ckks(self) -> None:
        """Initialize CKKS context."""
        try:
            import tenseal as ts
            self.ts = ts
            
            # Create CKKS context
            self.context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.config.polynomial_degree,
                coeff_mod_bit_sizes=self.config.coefficient_modulus or [40, 20, 40]
            )
            self.context.global_scale = self.config.scale
            self.context.generate_galois_keys()
            
            self.logger.info("CKKS encryption initialized")
            
        except ImportError:
            self.logger.warning(
                "tenseal not installed. "
                "Install with: pip install tenseal"
            )
            # Create mock implementation for demonstration
            self.ts = None
    
    def generate_keys(self) -> Tuple[Any, Any]:
        """Generate CKKS key pair (context contains both keys)."""
        if self.context is None:
            raise RuntimeError("CKKS not initialized")
        return self.context, self.context
    
    def encrypt(
        self, 
        plaintext: Union[float, np.ndarray], 
        public_key: Optional[Any] = None
    ) -> Any:
        """Encrypt plaintext using CKKS."""
        if self.ts is None:
            # Mock encryption for demonstration
            return {'mock': True, 'data': plaintext}
        
        if isinstance(plaintext, np.ndarray):
            return self.ts.ckks_vector(self.context, plaintext.tolist())
        else:
            return self.ts.ckks_vector(self.context, [plaintext])
    
    def decrypt(
        self, 
        ciphertext: Any, 
        private_key: Optional[Any] = None
    ) -> np.ndarray:
        """Decrypt ciphertext using CKKS."""
        if self.ts is None:
            # Mock decryption
            return ciphertext.get('data', 0)
        
        return np.array(ciphertext.decrypt())
    
    def add(self, ct1: Any, ct2: Any) -> Any:
        """Homomorphic addition."""
        if self.ts is None:
            return {'mock': True, 'data': ct1.get('data', 0) + ct2.get('data', 0)}
        return ct1 + ct2
    
    def multiply(self, ct1: Any, ct2: Any) -> Any:
        """Homomorphic multiplication."""
        if self.ts is None:
            return {'mock': True, 'data': ct1.get('data', 0) * ct2.get('data', 0)}
        return ct1 * ct2


class EncryptedAggregator:
    """
    Aggregator for encrypted model updates.
    
    Enables the server to aggregate model updates without seeing
    the plaintext values.
    """
    
    def __init__(self, encryption_scheme: str = 'paillier'):
        self.encryption_scheme = encryption_scheme
        self.he = self._create_he()
        self.logger = logging.getLogger("EncryptedAggregator")
    
    def _create_he(self) -> HomomorphicEncryption:
        """Create homomorphic encryption instance."""
        if self.encryption_scheme == 'paillier':
            return PaillierEncryption()
        elif self.encryption_scheme == 'ckks':
            return CKKSEncryption()
        else:
            raise ValueError(f"Unknown scheme: {self.encryption_scheme}")
    
    def setup(self) -> Tuple[Any, Any]:
        """Generate and return keys."""
        return self.he.generate_keys()
    
    def encrypt_weights(
        self,
        weights: Dict[str, np.ndarray],
        public_key: Any
    ) -> Dict[str, Any]:
        """
        Encrypt model weights.
        
        Args:
            weights: Dictionary of weight arrays
            public_key: Public encryption key
            
        Returns:
            Dictionary of encrypted weights
        """
        encrypted = {}
        for key, weight in weights.items():
            # Flatten weight array for encryption
            flat_weight = weight.flatten()
            encrypted[key] = self.he.encrypt(flat_weight, public_key)
        return encrypted
    
    def decrypt_weights(
        self,
        encrypted_weights: Dict[str, Any],
        private_key: Any,
        shapes: Dict[str, Tuple[int, ...]]
    ) -> Dict[str, np.ndarray]:
        """
        Decrypt aggregated weights.
        
        Args:
            encrypted_weights: Dictionary of encrypted weights
            private_key: Private decryption key
            shapes: Original shapes of weight arrays
            
        Returns:
            Dictionary of decrypted weights
        """
        decrypted = {}
        for key, enc_weight in encrypted_weights.items():
            flat_decrypted = self.he.decrypt(enc_weight, private_key)
            decrypted[key] = flat_decrypted.reshape(shapes[key])
        return decrypted
    
    def aggregate(
        self,
        encrypted_updates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate encrypted model updates.
        
        Args:
            encrypted_updates: List of encrypted weight dictionaries
            
        Returns:
            Aggregated encrypted weights
        """
        if not encrypted_updates:
            raise ValueError("No updates to aggregate")
        
        aggregated = {}
        weight_keys = encrypted_updates[0].keys()
        
        for key in weight_keys:
            # Sum encrypted values
            total = encrypted_updates[0][key]
            for enc_update in encrypted_updates[1:]:
                total = self.he.add(total, enc_update[key])
            aggregated[key] = total
        
        return aggregated


def federated_averaging_encrypted(
    encrypted_updates: List[Dict[str, Any]],
    num_samples: List[int],
    he_aggregator: EncryptedAggregator,
    private_key: Any,
    weight_shapes: Dict[str, Tuple[int, ...]]
) -> Dict[str, np.ndarray]:
    """
    Perform federated averaging on encrypted updates.
    
    Args:
        encrypted_updates: List of encrypted model updates
        num_samples: Number of samples for each client
        he_aggregator: Homomorphic encryption aggregator
        private_key: Private key for decryption
        weight_shapes: Shapes of weight arrays
        
    Returns:
        Decrypted aggregated weights
    """
    # Aggregate encrypted updates
    aggregated = he_aggregator.aggregate(encrypted_updates)
    
    # Decrypt
    decrypted = he_aggregator.decrypt_weights(
        aggregated, private_key, weight_shapes
    )
    
    # Compute weighted average (needs to be done after decryption for Paillier)
    total_samples = sum(num_samples)
    averaged = {
        key: weight / total_samples
        for key, weight in decrypted.items()
    }
    
    return averaged
