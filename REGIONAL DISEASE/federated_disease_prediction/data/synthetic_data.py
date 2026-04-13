"""
Synthetic Data Generation for Disease Outbreak Prediction

This module generates realistic synthetic data for testing and
demonstration purposes.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta


class SyntheticDataGenerator:
    """
    Generator for synthetic disease outbreak data.
    
    Creates realistic time series data with seasonal patterns,
    spatial correlations, and outbreak events.
    """
    
    def __init__(
        self,
        num_regions: int = 10,
        num_days: int = 365 * 2,  # 2 years
        base_infection_rate: float = 0.05,
        seed: int = 42
    ):
        """
        Initialize data generator.
        
        Args:
            num_regions: Number of geographic regions
            num_days: Number of days to generate
            base_infection_rate: Base infection rate
            seed: Random seed
        """
        self.num_regions = num_regions
        self.num_days = num_days
        self.base_infection_rate = base_infection_rate
        
        np.random.seed(seed)
        
        # Generate region coordinates
        self.region_coords = self._generate_region_coordinates()
        
        # Generate adjacency matrix based on distance
        self.adjacency_matrix = self._generate_adjacency_matrix()
    
    def _generate_region_coordinates(self) -> List[Tuple[float, float]]:
        """Generate random coordinates for regions."""
        # Generate coordinates in a grid-like pattern
        coords = []
        grid_size = int(np.ceil(np.sqrt(self.num_regions)))
        
        for i in range(self.num_regions):
            row = i // grid_size
            col = i % grid_size
            
            # Add some randomness to grid positions
            lat = 20 + row * 5 + np.random.normal(0, 1)
            lon = -100 + col * 5 + np.random.normal(0, 1)
            
            coords.append((lat, lon))
        
        return coords
    
    def _generate_adjacency_matrix(self) -> np.ndarray:
        """Generate adjacency matrix based on geographic proximity."""
        adj = np.zeros((self.num_regions, self.num_regions))
        
        for i in range(self.num_regions):
            for j in range(i + 1, self.num_regions):
                lat1, lon1 = self.region_coords[i]
                lat2, lon2 = self.region_coords[j]
                
                # Simple Euclidean distance (for demonstration)
                distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
                
                # Connect nearby regions
                if distance < 10:
                    weight = 1.0 / (1.0 + distance)
                    adj[i, j] = weight
                    adj[j, i] = weight
        
        # Add self-loops
        np.fill_diagonal(adj, 1.0)
        
        return adj
    
    def generate_clinical_data(self) -> pd.DataFrame:
        """
        Generate synthetic clinical data.
        
        Returns:
            DataFrame with clinical features
        """
        dates = pd.date_range(
            start=datetime(2022, 1, 1),
            periods=self.num_days,
            freq='D'
        )
        
        data = []
        
        for region_id in range(self.num_regions):
            # Generate base time series with seasonality
            t = np.arange(self.num_days)
            
            # Seasonal pattern (higher in winter)
            seasonal = 0.3 * np.sin(2 * np.pi * t / 365 - np.pi/2) + 0.5
            
            # Weekly pattern (lower on weekends)
            weekly = 0.1 * np.sin(2 * np.pi * t / 7)
            
            # Trend
            trend = 0.001 * t
            
            # Random noise
            noise = np.random.normal(0, 0.1, self.num_days)
            
            # Outbreak events (random spikes)
            outbreaks = np.zeros(self.num_days)
            max_start = max(1, self.num_days - 30)
            num_outbreaks = min(np.random.randint(1, 3), max_start)
            outbreak_starts = np.random.choice(
                max_start,
                size=num_outbreaks,
                replace=False
            )
            
            for start in outbreak_starts:
                duration = np.random.randint(14, 30)
                intensity = np.random.uniform(0.5, 1.0)
                outbreak_pattern = intensity * np.exp(-0.1 * np.arange(duration))
                outbreaks[start:start + duration] = outbreak_pattern[:min(duration, self.num_days - start)]
            
            # Combine components
            cases = self.base_infection_rate * (
                seasonal + weekly + trend + noise + outbreaks
            )
            cases = np.maximum(cases, 0)  # Ensure non-negative
            
            # Generate other clinical features
            hospitalizations = cases * np.random.uniform(0.1, 0.3, self.num_days)
            deaths = cases * np.random.uniform(0.01, 0.05, self.num_days)
            tests_conducted = cases * np.random.uniform(10, 20, self.num_days)
            positive_rate = cases / (tests_conducted + 1)
            
            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'region_id': region_id,
                    'region_lat': self.region_coords[region_id][0],
                    'region_lon': self.region_coords[region_id][1],
                    'new_cases': cases[i],
                    'hospitalizations': hospitalizations[i],
                    'deaths': deaths[i],
                    'tests_conducted': tests_conducted[i],
                    'positive_rate': positive_rate[i],
                    'is_outbreak': outbreaks[i] > 0.3
                })
        
        return pd.DataFrame(data)
    
    def generate_environmental_data(self) -> pd.DataFrame:
        """
        Generate synthetic environmental data.
        
        Returns:
            DataFrame with environmental features
        """
        dates = pd.date_range(
            start=datetime(2022, 1, 1),
            periods=self.num_days,
            freq='D'
        )
        
        data = []
        
        for region_id in range(self.num_regions):
            t = np.arange(self.num_days)
            
            # Temperature with seasonal variation
            base_temp = 20 + 10 * np.sin(2 * np.pi * t / 365 - np.pi/2)
            temperature = base_temp + np.random.normal(0, 3, self.num_days)
            
            # Humidity (inverse correlation with temperature)
            base_humidity = 60 - 20 * np.sin(2 * np.pi * t / 365 - np.pi/2)
            humidity = base_humidity + np.random.normal(0, 5, self.num_days)
            humidity = np.clip(humidity, 0, 100)
            
            # Rainfall (random events)
            rainfall = np.random.exponential(2, self.num_days)
            rainfall[rainfall > 50] = 0  # Most days no rain
            
            # Air quality index
            aqi = 50 + 30 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 10, self.num_days)
            aqi = np.clip(aqi, 0, 300)
            
            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'region_id': region_id,
                    'temperature': temperature[i],
                    'humidity': humidity[i],
                    'rainfall': rainfall[i],
                    'air_quality_index': aqi[i]
                })
        
        return pd.DataFrame(data)
    
    def generate_mobility_data(self) -> pd.DataFrame:
        """
        Generate synthetic mobility data.
        
        Returns:
            DataFrame with mobility features
        """
        dates = pd.date_range(
            start=datetime(2022, 1, 1),
            periods=self.num_days,
            freq='D'
        )
        
        data = []
        
        for region_id in range(self.num_regions):
            t = np.arange(self.num_days)
            
            # Base mobility pattern (weekly cycle)
            base_mobility = 100 + 20 * np.sin(2 * np.pi * t / 7)
            
            # Weekend reduction
            day_of_week = np.array([d.weekday() for d in dates])
            weekend_mask = (day_of_week >= 5)
            base_mobility[weekend_mask] *= 0.7
            
            # Add noise
            mobility_index = base_mobility + np.random.normal(0, 5, self.num_days)
            mobility_index = np.clip(mobility_index, 0, 200)
            
            # Inter-region travel (based on adjacency)
            travel_volume = np.zeros(self.num_days)
            for j in range(self.num_regions):
                if self.adjacency_matrix[region_id, j] > 0:
                    travel_volume += (
                        self.adjacency_matrix[region_id, j] * 
                        np.random.normal(100, 20, self.num_days)
                    )
            
            # Population density (static)
            population_density = np.random.uniform(100, 10000)
            
            for i, date in enumerate(dates):
                data.append({
                    'date': date,
                    'region_id': region_id,
                    'mobility_index': mobility_index[i],
                    'travel_volume': travel_volume[i],
                    'population_density': population_density
                })
        
        return pd.DataFrame(data)
    
    def generate_full_dataset(self) -> pd.DataFrame:
        """
        Generate complete dataset with all features.
        
        Returns:
            Combined DataFrame
        """
        clinical = self.generate_clinical_data()
        environmental = self.generate_environmental_data()
        mobility = self.generate_mobility_data()
        
        # Merge datasets
        combined = clinical.merge(
            environmental,
            on=['date', 'region_id'],
            how='outer'
        )
        
        combined = combined.merge(
            mobility,
            on=['date', 'region_id'],
            how='outer'
        )
        
        # Create target variable (outbreak in next 7 days)
        combined['target'] = combined.groupby('region_id')['is_outbreak'].shift(-7)
        
        return combined
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """Get the adjacency matrix for regions."""
        return self.adjacency_matrix
    
    def get_region_coordinates(self) -> List[Tuple[float, float]]:
        """Get coordinates for all regions."""
        return self.region_coords


def generate_dengue_data(
    num_regions: int = 20,
    num_days: int = 730
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic dengue outbreak data.
    
    Args:
        num_regions: Number of regions
        num_days: Number of days
        
    Returns:
        Tuple of (DataFrame, adjacency_matrix)
    """
    generator = SyntheticDataGenerator(
        num_regions=num_regions,
        num_days=num_days,
        base_infection_rate=0.1,
        seed=42
    )
    
    data = generator.generate_full_dataset()
    adj_matrix = generator.get_adjacency_matrix()
    
    return data, adj_matrix


def generate_covid_data(
    num_regions: int = 50,
    num_days: int = 365
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Generate synthetic COVID-19 outbreak data.
    
    Args:
        num_regions: Number of regions
        num_days: Number of days
        
    Returns:
        Tuple of (DataFrame, adjacency_matrix)
    """
    generator = SyntheticDataGenerator(
        num_regions=num_regions,
        num_days=num_days,
        base_infection_rate=0.15,
        seed=123
    )
    
    data = generator.generate_full_dataset()
    adj_matrix = generator.get_adjacency_matrix()
    
    return data, adj_matrix
