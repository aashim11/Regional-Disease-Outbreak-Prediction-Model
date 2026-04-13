"""
Federated Learning System for Regional Disease Outbreak Prediction

This package implements a privacy-preserving federated learning system
for predicting disease outbreaks using distributed healthcare data.
"""

__version__ = "1.0.0"
__author__ = "FL-Disease-Prediction Team"

from .core import (
    FederatedServer,
    FederatedClient,
    AggregationStrategy,
)

from .models import (
    LSTMModel,
    CNNLSTMModel,
    TransformerModel,
    GNNModel,
)

from .privacy import (
    DifferentialPrivacy,
    SecureAggregation,
    HomomorphicEncryption,
)

__all__ = [
    "FederatedServer",
    "FederatedClient",
    "AggregationStrategy",
    "LSTMModel",
    "CNNLSTMModel",
    "TransformerModel",
    "GNNModel",
    "DifferentialPrivacy",
    "SecureAggregation",
    "HomomorphicEncryption",
]
