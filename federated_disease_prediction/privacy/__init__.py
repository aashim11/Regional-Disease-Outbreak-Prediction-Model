"""Privacy-preserving mechanisms for federated learning."""

from .differential_privacy import DifferentialPrivacy, PrivacyAccountant
from .secure_aggregation import SecureAggregation, ShamirSecretSharing
from .homomorphic_encryption import HomomorphicEncryption
from .attacks import MembershipInferenceAttack, ModelInversionAttack

__all__ = [
    "DifferentialPrivacy",
    "PrivacyAccountant",
    "SecureAggregation",
    "ShamirSecretSharing",
    "HomomorphicEncryption",
    "MembershipInferenceAttack",
    "ModelInversionAttack",
]
