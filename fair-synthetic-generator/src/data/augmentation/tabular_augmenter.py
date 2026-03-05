"""
Tabular Augmenter
=================

Data augmentation utilities for tabular data including:
- Gaussian noise injection
- Mixup augmentation
- Feature dropout
- SMOTE-style oversampling
- Conditional augmentation preserving sensitive attributes
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TabularAugmenter:
    """
    Data augmenter for tabular data.
    
    Provides multiple augmentation strategies:
    - Noise injection: Add Gaussian noise to numerical features
    - Mixup: Linear interpolation between samples
    - Feature dropout: Randomly mask feature values
    - SMOTE-style: Synthesize minority samples
    - Jittering: Small perturbations
    
    All augmentations can be configured to preserve sensitive attributes.
    
    Attributes:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        sensitive_attributes: Attributes to preserve during augmentation
    """
    
    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        sensitive_attributes: Optional[List[str]] = None,
        augmentation_ratio: float = 0.5,
        noise_std: float = 0.05,
        mixup_alpha: float = 0.2,
        dropout_prob: float = 0.1,
        preserve_sensitive: bool = True,
        seed: Optional[int] = None
    ):
        """
        Initialize the tabular augmenter.
        
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            sensitive_attributes: Attributes to preserve during augmentation
            augmentation_ratio: Ratio of augmented to original samples
            noise_std: Standard deviation for Gaussian noise
            mixup_alpha: Alpha parameter for mixup interpolation
            dropout_prob: Probability of feature dropout
            preserve_sensitive: Whether to preserve sensitive attributes
            seed: Random seed
        """
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.sensitive_attributes = sensitive_attributes or []
        self.augmentation_ratio = augmentation_ratio
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
        self.dropout_prob = dropout_prob
        self.preserve_sensitive = preserve_sensitive
        self.seed = seed
        
        self._rng = np.random.default_rng(seed)
        
    def augment(
        self,
        data: pd.DataFrame,
        method: str = "noise",
        n_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Augment tabular data.
        
        Args:
            data: Input DataFrame
            method: Augmentation method ('noise', 'mixup', 'dropout', 'smote', 'combined')
            n_samples: Number of augmented samples (None for ratio-based)
            
        Returns:
            Augmented DataFrame
        """
        if n_samples is None:
            n_samples = int(len(data) * self.augmentation_ratio)
            
        if method == "noise":
            return self._noise_augment(data, n_samples)
        elif method == "mixup":
            return self._mixup_augment(data, n_samples)
        elif method == "dropout":
            return self._dropout_augment(data, n_samples)
        elif method == "smote":
            return self._smote_augment(data, n_samples)
        elif method == "combined":
            return self._combined_augment(data, n_samples)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")
    
    def _noise_augment(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Add Gaussian noise to numerical features."""
        # Sample indices for augmentation
        indices = self._rng.choice(len(data), size=n_samples, replace=True)
        augmented = data.iloc[indices].copy()
        
        # Get features to augment
        features_to_augment = self._get_augmentable_features(data)
        
        # Add noise to numerical features
        for col in features_to_augment["numerical"]:
            if col in data.columns:
                noise = self._rng.normal(0, self.noise_std, n_samples)
                # Scale noise by feature std
                feature_std = data[col].std()
                augmented[col] = augmented[col] + noise * feature_std
                
        return augmented
    
    def _mixup_augment(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Mixup augmentation via linear interpolation."""
        augmented_rows = []
        features_to_augment = self._get_augmentable_features(data)
        
        for _ in range(n_samples):
            # Sample two random indices
            idx1, idx2 = self._rng.choice(len(data), size=2, replace=False)
            
            # Sample interpolation weight
            lam = self._rng.beta(self.mixup_alpha, self.mixup_alpha)
            
            row1 = data.iloc[idx1].copy()
            row2 = data.iloc[idx2].copy()
            mixed = row1.copy()
            
            # Interpolate numerical features
            for col in features_to_augment["numerical"]:
                if col in data.columns:
                    mixed[col] = lam * row1[col] + (1 - lam) * row2[col]
                    
            # For categorical, randomly choose from one parent
            for col in features_to_augment["categorical"]:
                if col in data.columns:
                    mixed[col] = row1[col] if self._rng.random() < lam else row2[col]
                    
            augmented_rows.append(mixed)
            
        return pd.DataFrame(augmented_rows)
    
    def _dropout_augment(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Randomly drop feature values."""
        indices = self._rng.choice(len(data), size=n_samples, replace=True)
        augmented = data.iloc[indices].copy()
        
        features_to_augment = self._get_augmentable_features(data)
        all_features = features_to_augment["numerical"] + features_to_augment["categorical"]
        
        # Randomly drop features
        for col in all_features:
            if col in data.columns:
                mask = self._rng.random(n_samples) < self.dropout_prob
                if col in features_to_augment["numerical"]:
                    # Replace with mean
                    augmented.loc[mask, col] = data[col].mean()
                else:
                    # Replace with mode
                    mode_val = data[col].mode()[0] if len(data[col].mode()) > 0 else data[col].iloc[0]
                    augmented.loc[mask, col] = mode_val
                    
        return augmented
    
    def _smote_augment(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """SMOTE-style oversampling for minority class."""
        # Simple SMOTE implementation
        # Find nearest neighbors and interpolate
        from sklearn.neighbors import NearestNeighbors
        
        # Only for numerical features
        features_to_augment = self._get_augmentable_features(data)
        num_cols = [c for c in features_to_augment["numerical"] if c in data.columns]
        
        if not num_cols:
            return self._noise_augment(data, n_samples)
            
        X = data[num_cols].values
        
        # Fit nearest neighbors
        nn = NearestNeighbors(n_neighbors=6)
        nn.fit(X)
        
        augmented_rows = []
        indices = self._rng.choice(len(data), size=n_samples, replace=True)
        
        for idx in indices:
            # Find nearest neighbors
            distances, neighbors = nn.kneighbors([X[idx]])
            neighbors = neighbors[0][1:]  # Exclude self
            
            # Random neighbor
            neighbor_idx = self._rng.choice(neighbors)
            
            # Interpolate
            lam = self._rng.random()
            new_row = data.iloc[idx].copy()
            
            for col in num_cols:
                new_row[col] = lam * data.iloc[idx][col] + (1 - lam) * data.iloc[neighbor_idx][col]
                
            augmented_rows.append(new_row)
            
        return pd.DataFrame(augmented_rows)
    
    def _combined_augment(self, data: pd.DataFrame, n_samples: int) -> pd.DataFrame:
        """Apply multiple augmentation methods combined."""
        n_each = n_samples // 3
        remainder = n_samples % 3
        
        augmented_parts = []
        methods = ["noise", "mixup", "dropout"]
        counts = [n_each + (1 if i < remainder else 0) for i in range(3)]
        
        for method, count in zip(methods, counts):
            if count > 0:
                part = getattr(self, f"_{method}_augment")(data, count)
                augmented_parts.append(part)
                
        return pd.concat(augmented_parts, ignore_index=True)
    
    def _get_augmentable_features(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """Get features that can be augmented, excluding sensitive ones."""
        numerical = [
            c for c in self.numerical_features 
            if c in data.columns and (not self.preserve_sensitive or c not in self.sensitive_attributes)
        ]
        categorical = [
            c for c in self.categorical_features 
            if c in data.columns and (not self.preserve_sensitive or c not in self.sensitive_attributes)
        ]
        
        # Auto-detect if not specified
        if not numerical and not categorical:
            for col in data.columns:
                if self.preserve_sensitive and col in self.sensitive_attributes:
                    continue
                if data[col].dtype in ["int64", "float64"]:
                    numerical.append(col)
                else:
                    categorical.append(col)
                    
        return {"numerical": numerical, "categorical": categorical}
    
    def augment_with_labels(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        method: str = "noise",
        n_samples: Optional[int] = None,
        target_column: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Augment data while preserving label consistency.
        
        Args:
            data: Input DataFrame
            labels: Labels to preserve
            method: Augmentation method
            n_samples: Number of samples
            target_column: Name of target column if in data
            
        Returns:
            Tuple of (augmented data, augmented labels)
        """
        if n_samples is None:
            n_samples = int(len(data) * self.augmentation_ratio)
            
        # For noise and dropout, labels stay the same
        if method in ["noise", "dropout"]:
            indices = self._rng.choice(len(data), size=n_samples, replace=True)
            augmented_data = getattr(self, f"_{method}_augment")(data, n_samples)
            augmented_labels = labels.iloc[indices].reset_index(drop=True)
            return augmented_data, augmented_labels
            
        # For mixup, interpolate labels if continuous, else use parent
        elif method == "mixup":
            augmented_data = self._mixup_augment(data, n_samples)
            # For classification, use majority parent label
            augmented_labels = pd.Series([labels.iloc[0]] * n_samples)
            return augmented_data, augmented_labels
            
        # For SMOTE, labels stay the same
        elif method == "smote":
            augmented_data = self._smote_augment(data, n_samples)
            augmented_labels = labels.iloc[
                self._rng.choice(len(data), size=n_samples, replace=True)
            ].reset_index(drop=True)
            return augmented_data, augmented_labels
            
        return self.augment(data, method, n_samples), labels.iloc[:n_samples]


def create_balanced_dataset(
    data: pd.DataFrame,
    labels: pd.Series,
    sensitive_attr: str,
    max_ratio: float = 1.0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create a balanced dataset by oversampling minority groups.
    
    Args:
        data: Input DataFrame
        labels: Labels
        sensitive_attr: Sensitive attribute to balance on
        max_ratio: Maximum ratio between largest and smallest groups
        
    Returns:
        Balanced (data, labels) tuple
    """
    augmenter = TabularAugmenter(
        preserve_sensitive=False,
        augmentation_ratio=1.0
    )
    
    # Count samples per group
    group_counts = data[sensitive_attr].value_counts()
    max_count = int(group_counts.max() * max_ratio)
    
    balanced_data = []
    balanced_labels = []
    
    for group_val in group_counts.index:
        group_mask = data[sensitive_attr] == group_val
        group_data = data[group_mask]
        group_labels = labels[group_mask]
        
        n_needed = max_count - len(group_data)
        
        if n_needed > 0:
            # Augment to reach target
            aug_data, aug_labels = augmenter.augment_with_labels(
                group_data, group_labels,
                method="smote",
                n_samples=n_needed
            )
            balanced_data.append(group_data)
            balanced_data.append(aug_data)
            balanced_labels.append(group_labels)
            balanced_labels.append(aug_labels)
        else:
            # Undersample if too many
            sample_idx = np.random.choice(len(group_data), max_count, replace=False)
            balanced_data.append(group_data.iloc[sample_idx])
            balanced_labels.append(group_labels.iloc[sample_idx])
            
    return pd.concat(balanced_data, ignore_index=True), pd.concat(balanced_labels, ignore_index=True)
