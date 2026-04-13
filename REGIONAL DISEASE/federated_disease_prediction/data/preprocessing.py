"""
Data Preprocessing and Feature Engineering

This module implements data preprocessing pipelines and feature engineering
techniques for disease outbreak prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats
from scipy.interpolate import interp1d
import logging


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for healthcare data.
    
    Handles missing values, outliers, normalization, and data cleaning.
    """
    
    def __init__(
        self,
        missing_strategy: str = 'interpolation',
        outlier_method: str = 'iqr',
        scaler_type: str = 'standard',
        fill_value: Optional[float] = None
    ):
        """
        Initialize preprocessor.
        
        Args:
            missing_strategy: Strategy for handling missing values
            outlier_method: Method for outlier detection/removal
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            fill_value: Value to use for constant imputation
        """
        self.missing_strategy = missing_strategy
        self.outlier_method = outlier_method
        self.scaler_type = scaler_type
        self.fill_value = fill_value
        
        self.scalers: Dict[str, Any] = {}
        self.imputers: Dict[str, Any] = {}
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        
        self.logger = logging.getLogger("DataPreprocessor")
    
    def fit(self, data: pd.DataFrame, feature_columns: List[str]) -> 'DataPreprocessor':
        """
        Fit preprocessor on training data.
        
        Args:
            data: Training DataFrame
            feature_columns: Columns to preprocess
            
        Returns:
            Self
        """
        for col in feature_columns:
            if col not in data.columns:
                continue
            
            col_data = data[col].values
            
            # Calculate statistics
            self.feature_stats[col] = {
                'mean': np.nanmean(col_data),
                'std': np.nanstd(col_data),
                'min': np.nanmin(col_data),
                'max': np.nanmax(col_data),
                'median': np.nanmedian(col_data),
                'q25': np.nanpercentile(col_data, 25),
                'q75': np.nanpercentile(col_data, 75),
            }
            
            # Initialize imputer
            if self.missing_strategy == 'mean':
                self.imputers[col] = SimpleImputer(strategy='mean')
            elif self.missing_strategy == 'median':
                self.imputers[col] = SimpleImputer(strategy='median')
            elif self.missing_strategy == 'knn':
                self.imputers[col] = KNNImputer(n_neighbors=5)
            elif self.missing_strategy == 'constant':
                fill = self.fill_value if self.fill_value is not None else 0
                self.imputers[col] = SimpleImputer(strategy='constant', fill_value=fill)
            
            # Fit imputer if needed
            if col in self.imputers:
                valid_data = col_data[~np.isnan(col_data)].reshape(-1, 1)
                if len(valid_data) > 0:
                    self.imputers[col].fit(valid_data)
            
            # Initialize scaler
            if self.scaler_type == 'standard':
                self.scalers[col] = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scalers[col] = MinMaxScaler()
            elif self.scaler_type == 'robust':
                self.scalers[col] = RobustScaler()
            
            # Fit scaler on clean data
            if col in self.scalers:
                clean_data = self._remove_outliers(col_data)
                clean_data = clean_data[~np.isnan(clean_data)].reshape(-1, 1)
                if len(clean_data) > 0:
                    self.scalers[col].fit(clean_data)
        
        return self
    
    def transform(
        self,
        data: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.
        
        Args:
            data: DataFrame to transform
            feature_columns: Columns to transform
            
        Returns:
            Transformed DataFrame
        """
        result = data.copy()
        
        for col in feature_columns:
            if col not in data.columns:
                continue
            
            values = data[col].values.copy()
            
            # Handle missing values
            values = self._handle_missing(values, col)
            
            # Handle outliers
            values = self._handle_outliers(values, col)
            
            # Scale
            if col in self.scalers:
                values = self.scalers[col].transform(values.reshape(-1, 1)).flatten()
            
            result[col] = values
        
        return result
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        feature_columns: List[str]
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(data, feature_columns).transform(data, feature_columns)
    
    def _handle_missing(
        self,
        values: np.ndarray,
        column: str
    ) -> np.ndarray:
        """Handle missing values."""
        if self.missing_strategy == 'interpolation':
            # Time series interpolation
            nans = np.isnan(values)
            if nans.any():
                valid_idx = np.where(~nans)[0]
                valid_values = values[~nans]
                
                if len(valid_idx) > 1:
                    interp = interp1d(
                        valid_idx,
                        valid_values,
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                    all_idx = np.arange(len(values))
                    values = interp(all_idx)
                elif len(valid_idx) == 1:
                    values[:] = valid_values[0]
        
        elif self.missing_strategy in ['mean', 'median', 'knn', 'constant']:
            if column in self.imputers:
                values = self.imputers[column].transform(
                    values.reshape(-1, 1)
                ).flatten()
        
        elif self.missing_strategy == 'forward_fill':
            values = pd.Series(values).fillna(method='ffill').fillna(method='bfill').values
        
        return values
    
    def _handle_outliers(
        self,
        values: np.ndarray,
        column: str
    ) -> np.ndarray:
        """Detect and handle outliers."""
        if self.outlier_method == 'none':
            return values
        
        stats_dict = self.feature_stats.get(column, {})
        
        if self.outlier_method == 'iqr':
            q25 = stats_dict.get('q25', np.percentile(values[~np.isnan(values)], 25))
            q75 = stats_dict.get('q75', np.percentile(values[~np.isnan(values)], 75))
            iqr = q75 - q25
            lower = q25 - 1.5 * iqr
            upper = q75 + 1.5 * iqr
            
            values = np.clip(values, lower, upper)
        
        elif self.outlier_method == 'zscore':
            mean = stats_dict.get('mean', np.nanmean(values))
            std = stats_dict.get('std', np.nanstd(values))
            if std > 0:
                z_scores = np.abs((values - mean) / std)
                values = np.where(z_scores > 3, mean, values)
        
        elif self.outlier_method == 'winsorize':
            values = stats.mstats.winsorize(values, limits=[0.05, 0.05])
        
        return values
    
    def _remove_outliers(self, values: np.ndarray) -> np.ndarray:
        """Remove outliers for scaler fitting."""
        if self.outlier_method == 'iqr':
            q25, q75 = np.percentile(values[~np.isnan(values)], [25, 75])
            iqr = q75 - q25
            lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
            return values[(values >= lower) & (values <= upper)]
        return values


class FeatureEngineer:
    """
    Feature engineering for disease outbreak prediction.
    
    Creates temporal, statistical, and domain-specific features.
    """
    
    def __init__(
        self,
        lag_features: List[int] = [1, 3, 7, 14],
        rolling_windows: List[int] = [7, 14, 30],
        include_seasonality: bool = True,
        include_trend: bool = True,
        include_interactions: bool = False
    ):
        """
        Initialize feature engineer.
        
        Args:
            lag_features: List of lag periods
            rolling_windows: List of rolling window sizes
            include_seasonality: Whether to add seasonal features
            include_trend: Whether to add trend features
            include_interactions: Whether to add interaction features
        """
        self.lag_features = lag_features
        self.rolling_windows = rolling_windows
        self.include_seasonality = include_seasonality
        self.include_trend = include_trend
        self.include_interactions = include_interactions
        
        self.logger = logging.getLogger("FeatureEngineer")
    
    def create_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        date_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create engineered features.
        
        Args:
            data: Input DataFrame
            target_col: Target variable column
            date_col: Date column name
            
        Returns:
            DataFrame with engineered features
        """
        result = data.copy()
        
        # Ensure data is sorted by date if available
        if date_col and date_col in result.columns:
            result[date_col] = pd.to_datetime(result[date_col])
            result = result.sort_values(date_col)
        
        # Lag features
        for lag in self.lag_features:
            result[f'{target_col}_lag_{lag}'] = result[target_col].shift(lag)
        
        # Rolling statistics
        for window in self.rolling_windows:
            result[f'{target_col}_rolling_mean_{window}'] = (
                result[target_col].rolling(window=window, min_periods=1).mean()
            )
            result[f'{target_col}_rolling_std_{window}'] = (
                result[target_col].rolling(window=window, min_periods=1).std()
            )
            result[f'{target_col}_rolling_max_{window}'] = (
                result[target_col].rolling(window=window, min_periods=1).max()
            )
            result[f'{target_col}_rolling_min_{window}'] = (
                result[target_col].rolling(window=window, min_periods=1).min()
            )
            
            # Rate of change
            result[f'{target_col}_roc_{window}'] = (
                result[target_col].diff(window) / result[target_col].shift(window)
            )
        
        # Exponential moving average
        result[f'{target_col}_ema_7'] = result[target_col].ewm(span=7).mean()
        result[f'{target_col}_ema_14'] = result[target_col].ewm(span=14).mean()
        
        # Seasonality features
        if self.include_seasonality and date_col and date_col in result.columns:
            result = self._add_seasonality_features(result, date_col)
        
        # Trend features
        if self.include_trend:
            result = self._add_trend_features(result, target_col)
        
        # Interaction features
        if self.include_interactions:
            result = self._add_interaction_features(result, target_col)
        
        return result
    
    def _add_seasonality_features(
        self,
        data: pd.DataFrame,
        date_col: str
    ) -> pd.DataFrame:
        """Add seasonality-related features."""
        result = data.copy()
        
        # Extract datetime components
        result['day_of_week'] = result[date_col].dt.dayofweek
        result['month'] = result[date_col].dt.month
        result['day_of_year'] = result[date_col].dt.dayofyear
        result['week_of_year'] = result[date_col].dt.isocalendar().week
        result['quarter'] = result[date_col].dt.quarter
        
        # Cyclical encoding for periodic features
        result['month_sin'] = np.sin(2 * np.pi * result['month'] / 12)
        result['month_cos'] = np.cos(2 * np.pi * result['month'] / 12)
        result['day_of_week_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_week_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        result['day_of_year_sin'] = np.sin(2 * np.pi * result['day_of_year'] / 365)
        result['day_of_year_cos'] = np.cos(2 * np.pi * result['day_of_year'] / 365)
        
        return result
    
    def _add_trend_features(
        self,
        data: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """Add trend-related features."""
        result = data.copy()
        
        # Differencing
        result[f'{target_col}_diff_1'] = result[target_col].diff(1)
        result[f'{target_col}_diff_7'] = result[target_col].diff(7)
        
        # Percentage change
        result[f'{target_col}_pct_change'] = result[target_col].pct_change()
        
        # Cumulative sum
        result[f'{target_col}_cumsum'] = result[target_col].cumsum()
        
        return result
    
    def _add_interaction_features(
        self,
        data: pd.DataFrame,
        target_col: str
    ) -> pd.DataFrame:
        """Add interaction features between variables."""
        result = data.copy()
        
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        
        # Add interaction terms for key features
        for i, col1 in enumerate(numeric_cols[:5]):  # Limit to first 5
            for col2 in numeric_cols[i+1:6]:
                if col1 != target_col and col2 != target_col:
                    result[f'{col1}_x_{col2}'] = result[col1] * result[col2]
        
        return result
    
    def create_sequences(
        self,
        data: np.ndarray,
        sequence_length: int,
        prediction_horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.
        
        Args:
            data: Input array
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps to predict
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - prediction_horizon + 1):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length:i + sequence_length + prediction_horizon])
        
        return np.array(X), np.array(y)


class DataSplitter:
    """
    Split data for federated learning while preserving temporal structure.
    """
    
    def __init__(self, test_size: float = 0.2, val_size: float = 0.1):
        self.test_size = test_size
        self.val_size = val_size
    
    def temporal_split(
        self,
        data: pd.DataFrame,
        date_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (preserves time order).
        
        Args:
            data: Input DataFrame
            date_col: Date column
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        data = data.sort_values(date_col)
        
        n = len(data)
        test_idx = int(n * (1 - self.test_size))
        val_idx = int(n * (1 - self.test_size - self.val_size))
        
        train = data.iloc[:val_idx]
        val = data.iloc[val_idx:test_idx]
        test = data.iloc[test_idx:]
        
        return train, val, test
    
    def federated_split(
        self,
        data: pd.DataFrame,
        num_clients: int,
        split_by: str = 'region'
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data for federated learning by region or institution.
        
        Args:
            data: Input DataFrame
            num_clients: Number of clients
            split_by: Column to split by
            
        Returns:
            Dictionary mapping client IDs to DataFrames
        """
        if split_by not in data.columns:
            # Random split if split_by column doesn't exist
            indices = np.array_split(np.arange(len(data)), num_clients)
            return {
                f'client_{i}': data.iloc[idx] 
                for i, idx in enumerate(indices)
            }
        
        # Split by unique values in split_by column
        unique_values = data[split_by].unique()
        
        client_data = {}
        for i, value in enumerate(unique_values[:num_clients]):
            client_data[f'client_{i}'] = data[data[split_by] == value].copy()
        
        return client_data
