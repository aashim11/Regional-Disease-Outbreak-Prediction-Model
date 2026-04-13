"""Core federated learning components."""

from .server import FederatedServer
from .client import FederatedClient
from .aggregator import AggregationStrategy, FedAvg, FedProx, FedNova
from .communication import CommunicationProtocol, GRPCProtocol, HTTPSProtocol

__all__ = [
    "FederatedServer",
    "FederatedClient",
    "AggregationStrategy",
    "FedAvg",
    "FedProx",
    "FedNova",
    "CommunicationProtocol",
    "GRPCProtocol",
    "HTTPSProtocol",
]
