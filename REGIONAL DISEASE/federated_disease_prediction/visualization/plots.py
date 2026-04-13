"""
Plotting Utilities for Disease Outbreak Prediction

This module provides static plotting functions for generating
visualizations of outbreak data and model results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import warnings

# Set default style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_time_series(
    data: pd.DataFrame,
    date_col: str = 'date',
    value_col: str = 'cases',
    region_col: Optional[str] = None,
    predictions: Optional[pd.DataFrame] = None,
    title: str = "Disease Cases Over Time",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot time series with optional predictions.
    
    Args:
        data: DataFrame with time series data
        date_col: Date column name
        value_col: Value column name
        region_col: Region column for multi-region plots
        predictions: Optional predictions DataFrame
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if region_col and region_col in data.columns:
        # Multi-region plot
        regions = data[region_col].unique()
        for region in regions:
            region_data = data[data[region_col] == region]
            ax.plot(
                region_data[date_col],
                region_data[value_col],
                label=f'Region {region}',
                alpha=0.7
            )
        ax.legend()
    else:
        # Single time series
        ax.plot(
            data[date_col],
            data[value_col],
            label='Actual',
            color='blue',
            linewidth=2
        )
    
    # Add predictions
    if predictions is not None:
        ax.plot(
            predictions[date_col],
            predictions['predicted'],
            label='Predicted',
            color='red',
            linestyle='--',
            linewidth=2
        )
        
        # Confidence interval
        if 'lower_bound' in predictions.columns and 'upper_bound' in predictions.columns:
            ax.fill_between(
                predictions[date_col],
                predictions['lower_bound'],
                predictions['upper_bound'],
                alpha=0.2,
                color='red',
                label='Confidence Interval'
            )
    
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_outbreak_heatmap(
    data: pd.DataFrame,
    date_col: str = 'date',
    region_col: str = 'region',
    value_col: str = 'cases',
    title: str = "Outbreak Intensity by Region and Time",
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot heatmap of outbreak intensity.
    
    Args:
        data: DataFrame with outbreak data
        date_col: Date column
        region_col: Region column
        value_col: Value column
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Pivot data for heatmap
    heatmap_data = data.pivot_table(
        values=value_col,
        index=region_col,
        columns=date_col,
        aggfunc='sum'
    )
    
    # Downsample columns if too many
    if heatmap_data.shape[1] > 50:
        step = heatmap_data.shape[1] // 50
        heatmap_data = heatmap_data.iloc[:, ::step]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        heatmap_data,
        cmap='YlOrRd',
        cbar_kws={'label': value_col.replace('_', ' ').title()},
        ax=ax
    )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Region')
    ax.set_title(title)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
    title: str = "Model Performance Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of multiple models.
    
    Args:
        results: Dictionary mapping model names to metric dictionaries
        metrics: List of metrics to compare
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        values = [results[model].get(metric, 0) for metric in metrics]
        ax.bar(x + i * width, values, width, label=model)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.title() for m in metrics])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    figsize: Tuple[int, int] = (14, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history: Dictionary with metric histories
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss plot
    if 'loss' in history:
        axes[0].plot(history['loss'], label='Training Loss')
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Over Time')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    if 'accuracy' in history:
        axes[1].plot(history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in history:
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Over Time')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels or ['Negative', 'Positive'],
        yticklabels=labels or ['Negative', 'Positive'],
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curves(
    results: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    title: str = "ROC Curves",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Args:
        results: Dictionary mapping model names to (fpr, tpr, auc) tuples
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, (fpr, tpr, auc) in results.items():
        ax.plot(
            fpr,
            tpr,
            label=f'{model_name} (AUC = {auc:.3f})',
            linewidth=2
        )
    
    # Random classifier line
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_spatial_distribution(
    data: pd.DataFrame,
    lat_col: str = 'lat',
    lon_col: str = 'lon',
    value_col: str = 'cases',
    title: str = "Spatial Distribution of Cases",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot spatial distribution of cases.
    
    Args:
        data: DataFrame with spatial data
        lat_col: Latitude column
        lon_col: Longitude column
        value_col: Value column for coloring
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        data[lon_col],
        data[lat_col],
        c=data[value_col],
        s=data.get('population', 100) / 10,
        cmap='YlOrRd',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    
    plt.colorbar(scatter, ax=ax, label=value_col.replace('_', ' ').title())
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importance_scores: Importance scores
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    # Sort by importance
    indices = np.argsort(importance_scores)[::-1]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot top 20 features
    top_n = min(20, len(feature_names))
    indices = indices[:top_n]
    
    ax.barh(
        range(top_n),
        importance_scores[indices],
        align='center'
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_federated_learning_progress(
    rounds: List[int],
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    title: str = "Federated Learning Progress",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot federated learning training progress.
    
    Args:
        rounds: List of round numbers
        train_metrics: Training metrics per round
        val_metrics: Validation metrics per round
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    if 'loss' in train_metrics:
        axes[0].plot(rounds, train_metrics['loss'], label='Train Loss', marker='o')
        if 'loss' in val_metrics:
            axes[0].plot(rounds, val_metrics['loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Over Rounds')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    if 'accuracy' in train_metrics:
        axes[1].plot(rounds, train_metrics['accuracy'], label='Train Accuracy', marker='o')
        if 'accuracy' in val_metrics:
            axes[1].plot(rounds, val_metrics['accuracy'], label='Val Accuracy', marker='s')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Over Rounds')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_client_contributions(
    client_data: Dict[str, Dict[str, Any]],
    metric: str = 'accuracy',
    title: str = "Client Contributions",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot contributions from different federated learning clients.
    
    Args:
        client_data: Dictionary with client metrics
        metric: Metric to visualize
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    clients = list(client_data.keys())
    values = [client_data[c].get(metric, 0) for c in clients]
    samples = [client_data[c].get('num_samples', 0) for c in clients]
    
    # Bar plot of metric
    axes[0].bar(clients, values)
    axes[0].set_xlabel('Client')
    axes[0].set_ylabel(metric.title())
    axes[0].set_title(f'{metric.title()} by Client')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Pie chart of data distribution
    axes[1].pie(samples, labels=clients, autopct='%1.1f%%')
    axes[1].set_title('Data Distribution Across Clients')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
