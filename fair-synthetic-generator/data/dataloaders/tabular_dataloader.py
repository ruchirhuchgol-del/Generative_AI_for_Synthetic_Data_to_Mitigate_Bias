"""
Tabular Dataloader
==================

Data loader for tabular data with support for numerical and categorical features.
Includes fairness-aware sampling and preprocessing utilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .base_dataloader import BaseDataset, BaseDataLoader, FairnessAwareSampler

logger = logging.getLogger(__name__)


class TabularDataset(BaseDataset):
    """
    Dataset for tabular data with numerical and categorical features.
    
    Features:
    - Automatic feature type detection
    - Sensitive attribute handling
    - Missing value support
    - Normalization and encoding
    """
    
    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        data: Optional[pd.DataFrame] = None,
        schema_path: Optional[Union[str, Path]] = None,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        target_column: Optional[str] = None,
        sensitive_attributes: Optional[List[str]] = None,
        normalize: bool = True,
        encode_categorical: bool = True,
        transform: Optional[Any] = None
    ):
        """
        Initialize the tabular dataset.
        
        Args:
            data_path: Path to CSV/Parquet file
            data: Direct DataFrame input (alternative to data_path)
            schema_path: Path to schema JSON file
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            target_column: Name of target variable column
            sensitive_attributes: List of sensitive attribute names
            normalize: Whether to normalize numerical features
            encode_categorical: Whether to encode categorical features
            transform: Optional transform function
        """
        super().__init__(data_path, transform, sensitive_attributes)
        
        self.data: Optional[pd.DataFrame] = data
        self.schema_path = Path(schema_path) if schema_path else None
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.target_column = target_column
        self.normalize = normalize
        self.encode_categorical = encode_categorical
        
        # Encoders and normalizers
        self._label_encoders: Dict[str, Dict] = {}
        self._normalizer_params: Dict[str, Dict] = {}
        self._feature_indices: Dict[str, int] = {}
        
        # Load schema if provided
        if self.schema_path:
            self._load_schema()
            
        # Load data if path provided
        if self.data_path and self.data is None:
            self.load_data()
            
    def _load_schema(self) -> None:
        """Load schema from JSON file."""
        if not self.schema_path.exists():
            logger.warning(f"Schema file not found: {self.schema_path}")
            return
            
        with open(self.schema_path) as f:
            schema = json.load(f)
            
        # Extract feature names from schema
        self.numerical_features = [
            f["name"] for f in schema.get("numerical_features", [])
        ]
        self.categorical_features = [
            f["name"] for f in schema.get("categorical_features", [])
        ]
        
        if "target_variable" in schema:
            self.target_column = schema["target_variable"]["name"]
            
        self.sensitive_attributes = [
            attr["name"] for attr in schema.get("protected_attributes", [])
        ]
        
        self._metadata["schema"] = schema
        
    def load_data(self) -> None:
        """Load data from file."""
        if not self.data_path:
            raise ValueError("No data path specified")
            
        # Load based on file extension
        suffix = self.data_path.suffix.lower()
        if suffix == ".csv":
            self.data = pd.read_csv(self.data_path)
        elif suffix == ".parquet":
            self.data = pd.read_parquet(self.data_path)
        elif suffix == ".json":
            self.data = pd.read_json(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
            
        # Auto-detect feature types if not specified
        if not self.numerical_features and not self.categorical_features:
            self._auto_detect_features()
            
        # Preprocess data
        self._preprocess()
        
        logger.info(f"Loaded {len(self.data)} samples with {len(self.numerical_features)} "
                   f"numerical and {len(self.categorical_features)} categorical features")
        
    def _auto_detect_features(self) -> None:
        """Automatically detect numerical and categorical features."""
        for col in self.data.columns:
            if col == self.target_column:
                continue
            if self.data[col].dtype in ["int64", "float64"]:
                # Check if it's actually categorical (few unique values)
                if self.data[col].nunique() < 10:
                    self.categorical_features.append(col)
                else:
                    self.numerical_features.append(col)
            else:
                self.categorical_features.append(col)
                
        logger.info(f"Auto-detected {len(self.numerical_features)} numerical and "
                   f"{len(self.categorical_features)} categorical features")
        
    def _preprocess(self) -> None:
        """Preprocess the data."""
        if self.data is None:
            return
            
        # Handle missing values
        for col in self.numerical_features:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna(self.data[col].median())
                
        for col in self.categorical_features:
            if col in self.data.columns:
                self.data[col] = self.data[col].fillna("missing")
                
        # Normalize numerical features
        if self.normalize:
            for col in self.numerical_features:
                if col in self.data.columns:
                    mean = self.data[col].mean()
                    std = self.data[col].std()
                    if std > 0:
                        self.data[col] = (self.data[col] - mean) / std
                    self._normalizer_params[col] = {"mean": mean, "std": std}
                    
        # Encode categorical features
        if self.encode_categorical:
            for col in self.categorical_features:
                if col in self.data.columns:
                    unique_values = sorted(self.data[col].unique())
                    self._label_encoders[col] = {
                        "mapping": {v: i for i, v in enumerate(unique_values)},
                        "inverse": {i: v for i, v in enumerate(unique_values)}
                    }
                    self.data[f"{col}_encoded"] = self.data[col].map(
                        self._label_encoders[col]["mapping"]
                    )
                    
        # Build feature index mapping
        all_features = self._get_all_features()
        for i, feat in enumerate(all_features):
            self._feature_indices[feat] = i
            
    def _get_all_features(self) -> List[str]:
        """Get list of all features in order."""
        features = []
        features.extend(self.numerical_features)
        if self.encode_categorical:
            features.extend([f"{col}_encoded" for col in self.categorical_features])
        else:
            features.extend(self.categorical_features)
        return features
        
    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.data) if self.data is not None else 0
        
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary with features, target, and sensitive attributes
        """
        if self.data is None:
            raise ValueError("Data not loaded")
            
        row = self.data.iloc[index]
        
        # Build feature tensor
        feature_list = []
        for col in self.numerical_features:
            if col in self.data.columns:
                feature_list.append(row[col])
                
        for col in self.categorical_features:
            if self.encode_categorical:
                feature_list.append(row[f"{col}_encoded"])
            else:
                feature_list.append(row[col])
                
        sample = {
            "features": torch.tensor(feature_list, dtype=torch.float32),
            "index": index
        }
        
        # Add target if present
        if self.target_column and self.target_column in self.data.columns:
            target = row[self.target_column]
            if isinstance(target, str) and self.target_column in self._label_encoders:
                target = self._label_encoders[self.target_column]["mapping"].get(target, 0)
            sample["target"] = torch.tensor(target, dtype=torch.float32)
            
        # Add sensitive attributes
        if self.sensitive_attributes:
            sensitive_values = []
            for attr in self.sensitive_attributes:
                if attr in self.data.columns:
                    val = row[attr]
                    if isinstance(val, str) and attr in self._label_encoders:
                        val = self._label_encoders[attr]["mapping"].get(val, 0)
                    sensitive_values.append(val)
            sample["sensitive"] = torch.tensor(sensitive_values, dtype=torch.long)
            
        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names."""
        return self._get_all_features()
    
    def get_feature_types(self) -> Dict[str, str]:
        """Get feature type mapping."""
        types = {}
        for feat in self.numerical_features:
            types[feat] = "numerical"
        for feat in self.categorical_features:
            types[feat] = "categorical"
        return types
    
    def get_sensitive_attribute(self, index: int) -> Optional[torch.Tensor]:
        """Get sensitive attribute values for a sample."""
        sample = self[index]
        return sample.get("sensitive")
    
    def get_group_labels(self, index: int) -> Optional[torch.Tensor]:
        """Get group labels (combination of sensitive attributes)."""
        sample = self[index]
        sensitive = sample.get("sensitive")
        if sensitive is not None:
            # Create group label by combining sensitive attributes
            # Simple approach: use product of values
            return sensitive
        return None
    
    def get_encoded_feature_dim(self) -> int:
        """Get the dimension of encoded features."""
        dim = len(self.numerical_features)
        dim += len(self.categorical_features)
        return dim
    
    def inverse_transform(self, features: torch.Tensor) -> pd.DataFrame:
        """
        Convert encoded features back to original format.
        
        Args:
            features: Encoded feature tensor [batch_size, num_features]
            
        Returns:
            DataFrame with original feature values
        """
        features_np = features.cpu().numpy()
        result = {}
        
        idx = 0
        for col in self.numerical_features:
            if col in self._normalizer_params:
                params = self._normalizer_params[col]
                result[col] = features_np[:, idx] * params["std"] + params["mean"]
            else:
                result[col] = features_np[:, idx]
            idx += 1
            
        for col in self.categorical_features:
            if col in self._label_encoders:
                encoded_vals = features_np[:, idx].astype(int)
                result[col] = [
                    self._label_encoders[col]["inverse"].get(v, "unknown")
                    for v in encoded_vals
                ]
            else:
                result[col] = features_np[:, idx]
            idx += 1
            
        return pd.DataFrame(result)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = super().get_statistics()
        
        if self.data is not None:
            stats.update({
                "numerical_features": len(self.numerical_features),
                "categorical_features": len(self.categorical_features),
                "feature_dim": self.get_encoded_feature_dim(),
                "missing_values": self.data.isnull().sum().to_dict(),
            })
            
            # Distribution of sensitive attributes
            if self.sensitive_attributes:
                stats["sensitive_distributions"] = {}
                for attr in self.sensitive_attributes:
                    if attr in self.data.columns:
                        stats["sensitive_distributions"][attr] = (
                            self.data[attr].value_counts().to_dict()
                        )
                        
        return stats


class TabularDataLoader(BaseDataLoader):
    """
    Data loader for tabular data with fairness-aware sampling.
    """
    
    def __init__(
        self,
        dataset: TabularDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        sampler_strategy: Optional[str] = None
    ):
        """
        Initialize the tabular data loader.
        
        Args:
            dataset: Tabular dataset
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of workers
            pin_memory: Pin memory for GPU
            drop_last: Drop last incomplete batch
            sampler_strategy: Fairness sampling strategy
        """
        # Compute group labels for fairness-aware sampling
        group_labels = None
        if sampler_strategy and dataset.sensitive_attributes:
            group_labels = self._compute_group_labels(dataset)
            
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            sampler_strategy=sampler_strategy,
            group_labels=group_labels
        )
        
    def _compute_group_labels(self, dataset: TabularDataset) -> np.ndarray:
        """Compute group labels for fairness-aware sampling."""
        group_labels = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            sensitive = sample.get("sensitive")
            if sensitive is not None:
                # Create single group label from combination
                group_label = tuple(sensitive.tolist())
                group_labels.append(group_label)
            else:
                group_labels.append((0,))
                
        # Convert to numeric labels
        unique_groups = list(set(group_labels))
        group_mapping = {g: i for i, g in enumerate(unique_groups)}
        
        return np.array([group_mapping[g] for g in group_labels])
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        return self.dataset.get_encoded_feature_dim()
    
    def get_batch_df(self) -> pd.DataFrame:
        """
        Get a batch as a DataFrame.
        
        Returns:
            DataFrame representation of a batch
        """
        batch = next(iter(self._dataloader))
        return self.dataset.inverse_transform(batch["features"])
