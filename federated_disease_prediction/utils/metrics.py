"""
Evaluation Metrics for Disease Outbreak Prediction

This module implements various evaluation metrics for assessing
model performance in outbreak prediction tasks.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    average_precision_score, precision_recall_curve,
    classification_report, matthews_corrcoef
)
from dataclasses import dataclass


@dataclass
class ClassificationMetrics:
    """Classification metrics container."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    average_precision: float
    matthews_corrcoef: float
    specificity: float
    npv: float  # Negative Predictive Value
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'average_precision': self.average_precision,
            'matthews_corrcoef': self.matthews_corrcoef,
            'specificity': self.specificity,
            'npv': self.npv
        }


@dataclass
class RegressionMetrics:
    """Regression metrics container."""
    mae: float  # Mean Absolute Error
    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mape: float  # Mean Absolute Percentage Error
    r2: float  # R-squared
    explained_variance: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'mae': self.mae,
            'mse': self.mse,
            'rmse': self.rmse,
            'mape': self.mape,
            'r2': self.r2,
            'explained_variance': self.explained_variance
        }


class MetricsCalculator:
    """
    Comprehensive metrics calculator for outbreak prediction.
    
    Handles both classification (outbreak detection) and
    regression (case count forecasting) tasks.
    """
    
    @staticmethod
    def calculate_classification_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> ClassificationMetrics:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            ClassificationMetrics object
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # ROC AUC
        if y_prob is not None and len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
            avg_precision = average_precision_score(y_true, y_prob)
        else:
            roc_auc = 0.5
            avg_precision = 0.5
        
        # Confusion matrix based metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            average_precision=avg_precision,
            matthews_corrcoef=mcc,
            specificity=specificity,
            npv=npv
        )
    
    @staticmethod
    def calculate_regression_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> RegressionMetrics:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RegressionMetrics object
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # MAPE with handling for zero values
        non_zero_mask = y_true != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
        else:
            mape = np.inf
        
        r2 = r2_score(y_true, y_pred)
        
        # Explained variance
        explained_var = 1 - np.var(y_true - y_pred) / np.var(y_true) if np.var(y_true) > 0 else 0
        
        return RegressionMetrics(
            mae=mae,
            mse=mse,
            rmse=rmse,
            mape=mape,
            r2=r2,
            explained_variance=explained_var
        )
    
    @staticmethod
    def calculate_early_detection_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        detection_window: int = 7
    ) -> Dict[str, float]:
        """
        Calculate metrics specific to early outbreak detection.
        
        Args:
            y_true: True outbreak labels (time series)
            y_pred: Predicted outbreak labels
            detection_window: Window size for early detection
            
        Returns:
            Dictionary of early detection metrics
        """
        # Find outbreak start points
        true_outbreaks = np.where(np.diff(y_true) == 1)[0]
        pred_outbreaks = np.where(np.diff(y_pred) == 1)[0]
        
        # Calculate early detection rate
        early_detections = 0
        missed_outbreaks = 0
        
        for true_start in true_outbreaks:
            # Check if any prediction within detection window before outbreak
            window_start = max(0, true_start - detection_window)
            if np.any(y_pred[window_start:true_start] == 1):
                early_detections += 1
            else:
                missed_outbreaks += 1
        
        total_outbreaks = len(true_outbreaks)
        
        early_detection_rate = early_detections / total_outbreaks if total_outbreaks > 0 else 0
        false_alarm_rate = len(pred_outbreaks) / len(y_pred) if len(y_pred) > 0 else 0
        
        return {
            'early_detection_rate': early_detection_rate,
            'missed_outbreaks': missed_outbreaks,
            'total_outbreaks': total_outbreaks,
            'false_alarm_rate': false_alarm_rate,
            'avg_detection_lead_time': detection_window if early_detections > 0 else 0
        }
    
    @staticmethod
    def calculate_spatial_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        region_ids: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculate metrics per region for spatial analysis.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            region_ids: Region identifiers
            
        Returns:
            Dictionary with per-region metrics
        """
        unique_regions = np.unique(region_ids)
        region_metrics = {}
        
        for region in unique_regions:
            mask = region_ids == region
            region_y_true = y_true[mask]
            region_y_pred = y_pred[mask]
            
            if len(np.unique(region_y_true)) > 1:
                region_metrics[f'region_{region}'] = {
                    'accuracy': accuracy_score(region_y_true, region_y_pred),
                    'precision': precision_score(region_y_true, region_y_pred, zero_division=0),
                    'recall': recall_score(region_y_true, region_y_pred, zero_division=0),
                    'f1': f1_score(region_y_true, region_y_pred, zero_division=0),
                    'num_samples': len(region_y_true)
                }
        
        return region_metrics
    
    @staticmethod
    def get_confusion_matrix_components(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, int]:
        """
        Get confusion matrix components.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with TP, FP, TN, FN counts
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        return {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    
    @staticmethod
    def calculate_roc_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate ROC curve data.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Tuple of (fpr, tpr, auc)
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        return fpr, tpr, auc
    
    @staticmethod
    def calculate_precision_recall_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate precision-recall curve data.
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            
        Returns:
            Tuple of (precision, recall, average_precision)
        """
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        return precision, recall, avg_precision
    
    @staticmethod
    def print_classification_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[List[str]] = None
    ) -> str:
        """
        Generate classification report string.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
            
        Returns:
            Classification report string
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            zero_division=0
        )


class FederatedMetricsAggregator:
    """
    Aggregate metrics from multiple federated learning clients.
    """
    
    @staticmethod
    def aggregate_classification_metrics(
        client_metrics: List[ClassificationMetrics],
        client_weights: Optional[List[float]] = None
    ) -> ClassificationMetrics:
        """
        Weighted aggregation of classification metrics.
        
        Args:
            client_metrics: List of metrics from clients
            client_weights: Weights for each client
            
        Returns:
            Aggregated metrics
        """
        if client_weights is None:
            client_weights = [1.0] * len(client_metrics)
        
        total_weight = sum(client_weights)
        
        # Weighted average of metrics
        aggregated = {}
        for metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 
                           'roc_auc', 'average_precision', 'matthews_corrcoef',
                           'specificity', 'npv']:
            values = [getattr(m, metric_name) for m in client_metrics]
            aggregated[metric_name] = sum(w * v for w, v in zip(client_weights, values)) / total_weight
        
        return ClassificationMetrics(**aggregated)
    
    @staticmethod
    def aggregate_regression_metrics(
        client_metrics: List[RegressionMetrics],
        client_weights: Optional[List[float]] = None
    ) -> RegressionMetrics:
        """
        Weighted aggregation of regression metrics.
        
        Args:
            client_metrics: List of metrics from clients
            client_weights: Weights for each client
            
        Returns:
            Aggregated metrics
        """
        if client_weights is None:
            client_weights = [1.0] * len(client_metrics)
        
        total_weight = sum(client_weights)
        
        aggregated = {}
        for metric_name in ['mae', 'mse', 'rmse', 'mape', 'r2', 'explained_variance']:
            values = [getattr(m, metric_name) for m in client_metrics]
            aggregated[metric_name] = sum(w * v for w, v in zip(client_weights, values)) / total_weight
        
        return RegressionMetrics(**aggregated)
