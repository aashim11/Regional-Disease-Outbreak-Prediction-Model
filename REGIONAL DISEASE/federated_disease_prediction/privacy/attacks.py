"""
Privacy Attacks and Defenses

This module implements various privacy attacks against federated learning
systems and corresponding defense mechanisms.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging


@dataclass
class AttackResult:
    """Result of a privacy attack."""
    success_rate: float
    precision: float
    recall: float
    details: Dict[str, Any]


class PrivacyAttack(ABC):
    """Abstract base class for privacy attacks."""
    
    def __init__(self, target_model: nn.Module):
        self.target_model = target_model
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def execute(self, **kwargs) -> AttackResult:
        """Execute the attack."""
        pass


class MembershipInferenceAttack(PrivacyAttack):
    """
    Membership Inference Attack (MIA).
    
    Determines whether a specific data point was used in training the model.
    Based on: Shokri et al., "Membership Inference Attacks Against
    Machine Learning Models", IEEE S&P 2017.
    """
    
    def __init__(
        self, 
        target_model: nn.Module,
        shadow_models: Optional[List[nn.Module]] = None
    ):
        super().__init__(target_model)
        self.shadow_models = shadow_models or []
        self.attack_model = None
    
    def train_attack_model(
        self,
        shadow_train_data: List[Tuple[torch.Tensor, int]],
        shadow_test_data: List[Tuple[torch.Tensor, int]]
    ) -> None:
        """
        Train attack model using shadow models.
        
        Args:
            shadow_train_data: Data from shadow models' training sets
            shadow_test_data: Data not in shadow models' training sets
        """
        # Collect prediction vectors and labels
        train_predictions = []
        train_labels = []
        
        for data, label in shadow_train_data:
            pred = self._get_prediction_vector(data)
            train_predictions.append(pred)
            train_labels.append(1)  # Member
        
        for data, label in shadow_test_data:
            pred = self._get_prediction_vector(data)
            train_predictions.append(pred)
            train_labels.append(0)  # Non-member
        
        # Train attack model (simple classifier)
        X = np.array(train_predictions)
        y = np.array(train_labels)
        
        from sklearn.ensemble import RandomForestClassifier
        self.attack_model = RandomForestClassifier(n_estimators=100)
        self.attack_model.fit(X, y)
        
        self.logger.info("Attack model trained")
    
    def _get_prediction_vector(self, data: torch.Tensor) -> np.ndarray:
        """Get prediction vector from target model."""
        self.target_model.eval()
        with torch.no_grad():
            output = self.target_model(data.unsqueeze(0))
            probs = torch.sigmoid(output).cpu().numpy()
        return probs.flatten()
    
    def execute(
        self,
        candidate_data: List[Tuple[torch.Tensor, int]],
        true_membership: Optional[List[int]] = None
    ) -> AttackResult:
        """
        Execute membership inference attack.
        
        Args:
            candidate_data: Data points to test
            true_membership: Ground truth membership (for evaluation)
            
        Returns:
            Attack results
        """
        if self.attack_model is None:
            raise ValueError("Attack model not trained")
        
        predictions = []
        for data, _ in candidate_data:
            pred_vector = self._get_prediction_vector(data)
            predictions.append(pred_vector)
        
        X = np.array(predictions)
        membership_pred = self.attack_model.predict(X)
        membership_proba = self.attack_model.predict_proba(X)[:, 1]
        
        if true_membership is not None:
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            
            accuracy = accuracy_score(true_membership, membership_pred)
            precision = precision_score(true_membership, membership_pred, zero_division=0)
            recall = recall_score(true_membership, membership_pred, zero_division=0)
        else:
            accuracy = precision = recall = 0.0
        
        return AttackResult(
            success_rate=accuracy,
            precision=precision,
            recall=recall,
            details={
                'predictions': membership_pred.tolist(),
                'probabilities': membership_proba.tolist()
            }
        )
    
    def defense_gradient_clipping(
        self,
        model: nn.Module,
        max_norm: float = 1.0
    ) -> None:
        """
        Apply gradient clipping as defense against MIA.
        
        Args:
            model: Model to clip gradients for
            max_norm: Maximum gradient norm
        """
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    def defense_regularization(
        self,
        model: nn.Module,
        regularization_strength: float = 0.01
    ) -> torch.Tensor:
        """
        Compute regularization term to reduce overfitting.
        
        Args:
            model: Model to regularize
            regularization_strength: Strength of regularization
            
        Returns:
            Regularization loss
        """
        l2_reg = torch.tensor(0.0)
        for param in model.parameters():
            l2_reg += torch.norm(param, 2)
        return regularization_strength * l2_reg


class ModelInversionAttack(PrivacyAttack):
    """
    Model Inversion Attack.
    
    Reconstructs training data from model predictions.
    Based on: Fredrikson et al., "Model Inversion Attacks that Exploit
    Confidence Information", ACM CCS 2015.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = 'cpu'
    ):
        super().__init__(target_model)
        self.input_shape = input_shape
        self.device = device
    
    def execute(
        self,
        target_label: int,
        num_iterations: int = 1000,
        learning_rate: float = 0.1,
        regularization: float = 0.01
    ) -> AttackResult:
        """
        Execute model inversion attack.
        
        Args:
            target_label: Label to reconstruct data for
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate for optimization
            regularization: Regularization strength
            
        Returns:
            Attack results with reconstructed data
        """
        # Initialize random input
        reconstructed = torch.randn(
            self.input_shape, 
            requires_grad=True, 
            device=self.device
        )
        
        optimizer = torch.optim.Adam([reconstructed], lr=learning_rate)
        
        self.target_model.eval()
        
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            
            # Get model prediction
            output = self.target_model(reconstructed.unsqueeze(0))
            probs = torch.sigmoid(output)
            
            # Loss: maximize probability of target label
            target = torch.tensor([[target_label]], dtype=torch.float32, device=self.device)
            prediction_loss = nn.BCELoss()(probs, target)
            
            # Regularization to encourage realistic data
            reg_loss = regularization * torch.norm(reconstructed, 2)
            
            total_loss = prediction_loss + reg_loss
            total_loss.backward()
            optimizer.step()
            
            if iteration % 100 == 0:
                self.logger.debug(
                    f"Iteration {iteration}: Loss = {total_loss.item():.4f}"
                )
        
        # Evaluate reconstruction quality
        with torch.no_grad():
            final_output = self.target_model(reconstructed.unsqueeze(0))
            final_prob = torch.sigmoid(final_output).item()
        
        return AttackResult(
            success_rate=final_prob,
            precision=final_prob,
            recall=final_prob,
            details={
                'reconstructed_data': reconstructed.detach().cpu().numpy(),
                'confidence': final_prob,
                'iterations': num_iterations
            }
        )


class GradientInversionAttack(PrivacyAttack):
    """
    Gradient Inversion Attack (Deep Leakage from Gradients).
    
    Reconstructs training data from gradients.
    Based on: Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        input_shape: Tuple[int, ...],
        device: str = 'cpu'
    ):
        super().__init__(target_model)
        self.input_shape = input_shape
        self.device = device
    
    def execute(
        self,
        original_gradients: List[torch.Tensor],
        num_iterations: int = 1000,
        learning_rate: float = 0.1
    ) -> AttackResult:
        """
        Execute gradient inversion attack.
        
        Args:
            original_gradients: Gradients leaked from client
            num_iterations: Number of optimization iterations
            learning_rate: Learning rate
            
        Returns:
            Attack results with reconstructed data
        """
        # Initialize dummy data and label
        dummy_data = torch.randn(
            self.input_shape,
            requires_grad=True,
            device=self.device
        )
        dummy_label = torch.randn(
            (1, 1),
            requires_grad=True,
            device=self.device
        )
        
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=learning_rate)
        
        criterion = nn.BCEWithLogitsLoss()
        
        for iteration in range(num_iterations):
            def closure():
                optimizer.zero_grad()
                
                # Compute gradients on dummy data
                output = self.target_model(dummy_data.unsqueeze(0))
                loss = criterion(output, dummy_label)
                
                dummy_gradients = torch.autograd.grad(
                    loss,
                    self.target_model.parameters(),
                    create_graph=True
                )
                
                # Match gradients
                grad_loss = 0
                for dg, og in zip(dummy_gradients, original_gradients):
                    grad_loss += torch.norm(dg - og, 2)
                
                grad_loss.backward()
                return grad_loss
            
            loss = optimizer.step(closure)
            
            if iteration % 100 == 0:
                self.logger.debug(f"Iteration {iteration}: Loss = {loss.item():.4f}")
        
        return AttackResult(
            success_rate=1.0 / (1.0 + loss.item()),
            precision=0.0,  # Hard to measure without ground truth
            recall=0.0,
            details={
                'reconstructed_data': dummy_data.detach().cpu().numpy(),
                'reconstructed_label': dummy_label.detach().cpu().numpy(),
                'final_loss': loss.item()
            }
        )


class PropertyInferenceAttack(PrivacyAttack):
    """
    Property Inference Attack.
    
    Infers properties of the training data that are not directly
    related to the main learning task.
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        meta_classifier: Optional[Any] = None
    ):
        super().__init__(target_model)
        self.meta_classifier = meta_classifier
    
    def train_meta_classifier(
        self,
        models_with_property: List[nn.Module],
        models_without_property: List[nn.Module]
    ) -> None:
        """
        Train meta-classifier to detect property.
        
        Args:
            models_with_property: Models trained on data with property
            models_without_property: Models trained on data without property
        """
        from sklearn.ensemble import RandomForestClassifier
        
        features = []
        labels = []
        
        # Extract features from models
        for model in models_with_property:
            features.append(self._extract_model_features(model))
            labels.append(1)
        
        for model in models_without_property:
            features.append(self._extract_model_features(model))
            labels.append(0)
        
        X = np.array(features)
        y = np.array(labels)
        
        self.meta_classifier = RandomForestClassifier(n_estimators=100)
        self.meta_classifier.fit(X, y)
        
        self.logger.info("Meta-classifier trained")
    
    def _extract_model_features(self, model: nn.Module) -> np.ndarray:
        """Extract features from model weights."""
        features = []
        for param in model.parameters():
            features.extend([
                torch.mean(param).item(),
                torch.std(param).item(),
                torch.min(param).item(),
                torch.max(param).item(),
            ])
        return np.array(features)
    
    def execute(self, **kwargs) -> AttackResult:
        """Execute property inference attack."""
        if self.meta_classifier is None:
            raise ValueError("Meta-classifier not trained")
        
        features = self._extract_model_features(self.target_model)
        prediction = self.meta_classifier.predict([features])[0]
        probability = self.meta_classifier.predict_proba([features])[0]
        
        return AttackResult(
            success_rate=float(prediction),
            precision=probability[1] if prediction == 1 else probability[0],
            recall=float(prediction),
            details={'probability': probability.tolist()}
        )


class DefenseMechanisms:
    """
    Collection of defense mechanisms against privacy attacks.
    """
    
    @staticmethod
    def gradient_compression(
        gradients: List[torch.Tensor],
        compression_ratio: float = 0.1
    ) -> List[torch.Tensor]:
        """
        Compress gradients by keeping only top-k values.
        
        Args:
            gradients: List of gradient tensors
            compression_ratio: Ratio of gradients to keep
            
        Returns:
            Compressed gradients
        """
        compressed = []
        
        for grad in gradients:
            if grad is None:
                compressed.append(None)
                continue
            
            # Flatten and find threshold
            flat_grad = grad.flatten()
            k = int(len(flat_grad) * compression_ratio)
            threshold = torch.topk(torch.abs(flat_grad), k)[0][-1]
            
            # Zero out small gradients
            mask = torch.abs(grad) >= threshold
            compressed_grad = grad * mask.float()
            
            compressed.append(compressed_grad)
        
        return compressed
    
    @staticmethod
    def gradient_sparsification(
        gradients: List[torch.Tensor],
        sparsity: float = 0.99
    ) -> List[torch.Tensor]:
        """
        Sparsify gradients by random sampling.
        
        Args:
            gradients: List of gradient tensors
            sparsity: Fraction of gradients to zero out
            
        Returns:
            Sparsified gradients
        """
        sparse_gradients = []
        
        for grad in gradients:
            if grad is None:
                sparse_gradients.append(None)
                continue
            
            # Random mask
            mask = torch.rand_like(grad) > sparsity
            sparse_grad = grad * mask.float()
            
            # Scale to preserve magnitude
            scale = 1.0 / (1.0 - sparsity)
            sparse_grad *= scale
            
            sparse_gradients.append(sparse_grad)
        
        return sparse_gradients
    
    @staticmethod
    def add_gradient_noise(
        gradients: List[torch.Tensor],
        noise_multiplier: float = 1.0
    ) -> List[torch.Tensor]:
        """
        Add Gaussian noise to gradients (also used in DP-SGD).
        
        Args:
            gradients: List of gradient tensors
            noise_multiplier: Noise scale
            
        Returns:
            Noisy gradients
        """
        noisy_gradients = []
        
        for grad in gradients:
            if grad is None:
                noisy_gradients.append(None)
                continue
            
            noise = torch.randn_like(grad) * noise_multiplier
            noisy_gradients.append(grad + noise)
        
        return noisy_gradients
    
    @staticmethod
    def activation_clipping(
        activations: torch.Tensor,
        max_value: float = 10.0
    ) -> torch.Tensor:
        """
        Clip activation values to prevent information leakage.
        
        Args:
            activations: Layer activations
            max_value: Maximum activation value
            
        Returns:
            Clipped activations
        """
        return torch.clamp(activations, -max_value, max_value)
