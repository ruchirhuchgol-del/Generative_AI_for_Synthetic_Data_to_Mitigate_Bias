"""
Base Dataloader
===============

Abstract base classes for all data loaders in the Fair Synthetic Data Generator.
Provides common functionality for data loading, batching, and fairness-aware sampling.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from pathlib import Path
import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)


class BaseDataset(Dataset, ABC):
    """
    Abstract base dataset class with common functionality.
    
    All specific datasets (tabular, text, image, multimodal) should inherit from this class.
    """
    
    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        transform: Optional[Any] = None,
        sensitive_attributes: Optional[List[str]] = None
    ):
        """
        Initialize the base dataset.
        
        Args:
            data_path: Path to the data file or directory
            transform: Optional transform to apply to data
            sensitive_attributes: List of sensitive attribute names
        """
        self.data_path = Path(data_path) if data_path else None
        self.transform = transform
        self.sensitive_attributes = sensitive_attributes or []
        self._data: Dict[str, Any] = {}
        self._metadata: Dict[str, Any] = {}
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError
    
    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get a single sample from the dataset.
        
        Args:
            index: Sample index
            
        Returns:
            Dictionary containing the sample data
        """
        pass
    
    @abstractmethod
    def load_data(self) -> None:
        """Load data from the data path."""
        pass
    
    def get_sensitive_attribute(self, index: int) -> Optional[torch.Tensor]:
        """
        Get sensitive attribute values for a sample.
        
        Args:
            index: Sample index
            
        Returns:
            Tensor of sensitive attribute values or None
        """
        return None
    
    def get_group_labels(self, index: int) -> Optional[torch.Tensor]:
        """
        Get group labels for fairness analysis.
        
        Args:
            index: Sample index
            
        Returns:
            Tensor of group labels or None
        """
        return None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata."""
        return self._metadata.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute and return dataset statistics.
        
        Returns:
            Dictionary of dataset statistics
        """
        return {
            "num_samples": len(self),
            "sensitive_attributes": self.sensitive_attributes,
        }
    
    def validate(self) -> List[str]:
        """
        Validate the dataset for consistency.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if self.data_path and not self.data_path.exists():
            errors.append(f"Data path does not exist: {self.data_path}")
            
        return errors


class FairnessAwareSampler(Sampler):
    """
    Sampler that ensures fair representation of different groups.
    
    Supports multiple sampling strategies:
    - balanced: Equal number of samples from each group
    - proportional: Maintain original distribution
    - oversample_minority: Oversample minority groups
    - undersample_majority: Undersample majority groups
    """
    
    def __init__(
        self,
        group_labels: np.ndarray,
        strategy: str = "balanced",
        num_samples: Optional[int] = None
    ):
        """
        Initialize the fairness-aware sampler.
        
        Args:
            group_labels: Array of group labels for each sample
            strategy: Sampling strategy
            num_samples: Total number of samples per epoch (None for auto)
        """
        self.group_labels = np.array(group_labels)
        self.strategy = strategy
        self.num_samples = num_samples
        
        # Compute group statistics
        self.unique_groups, self.group_counts = np.unique(
            self.group_labels, return_counts=True
        )
        self.num_groups = len(self.unique_groups)
        
        # Compute weights based on strategy
        self.weights = self._compute_weights()
        self._num_samples = self._compute_num_samples()
        
    def _compute_weights(self) -> torch.Tensor:
        """Compute sampling weights based on strategy."""
        weights = np.ones(len(self.group_labels), dtype=np.float32)
        
        if self.strategy == "balanced":
            # Equal weight for all groups
            for group in self.unique_groups:
                mask = self.group_labels == group
                weights[mask] = 1.0 / self.group_counts[np.where(self.unique_groups == group)[0][0]]
                
        elif self.strategy == "oversample_minority":
            # Oversample minority to match majority
            max_count = self.group_counts.max()
            for i, group in enumerate(self.unique_groups):
                mask = self.group_labels == group
                weights[mask] = max_count / self.group_counts[i]
                
        elif self.strategy == "undersample_majority":
            # Undersample majority to match minority
            min_count = self.group_counts.min()
            for i, group in enumerate(self.unique_groups):
                mask = self.group_labels == group
                weights[mask] = min_count / self.group_counts[i]
                
        elif self.strategy == "proportional":
            # Maintain original distribution (uniform weights)
            weights = np.ones(len(self.group_labels), dtype=np.float32)
        else:
            logger.warning(f"Unknown sampling strategy: {self.strategy}, using uniform weights")
            
        # Normalize weights
        weights = weights / weights.sum()
        
        return torch.from_numpy(weights)
    
    def _compute_num_samples(self) -> int:
        """Compute the number of samples per epoch."""
        if self.num_samples is not None:
            return self.num_samples
            
        if self.strategy == "balanced":
            # Equal samples from each group
            return len(self.group_labels)
        elif self.strategy == "oversample_minority":
            return int(self.num_groups * self.group_counts.max())
        elif self.strategy == "undersample_majority":
            return int(self.num_groups * self.group_counts.min())
        else:
            return len(self.group_labels)
    
    def __iter__(self) -> Iterator[int]:
        """Generate indices for one epoch."""
        indices = torch.multinomial(
            self.weights,
            self._num_samples,
            replacement=True
        ).tolist()
        return iter(indices)
    
    def __len__(self) -> int:
        """Return the number of samples per epoch."""
        return self._num_samples


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    Provides common functionality for creating PyTorch DataLoaders
    with fairness-aware sampling support.
    """
    
    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        sampler_strategy: Optional[str] = None,
        group_labels: Optional[np.ndarray] = None
    ):
        """
        Initialize the base data loader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for GPU transfer
            drop_last: Whether to drop the last incomplete batch
            sampler_strategy: Fairness-aware sampling strategy
            group_labels: Group labels for fairness-aware sampling
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.sampler_strategy = sampler_strategy
        self._group_labels = group_labels
        
        self._dataloader: Optional[DataLoader] = None
        self._create_dataloader()
        
    def _create_dataloader(self) -> None:
        """Create the PyTorch DataLoader with appropriate sampler."""
        sampler = None
        
        if self.sampler_strategy and self._group_labels is not None:
            sampler = FairnessAwareSampler(
                group_labels=self._group_labels,
                strategy=self.sampler_strategy
            )
            self.shuffle = False  # Sampler handles shuffling
            
        self._dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last
        )
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over batches."""
        return iter(self._dataloader)
    
    def __len__(self) -> int:
        """Return the number of batches."""
        return len(self._dataloader)
    
    def get_batch(self) -> Dict[str, Any]:
        """
        Get a single batch (for manual iteration).
        
        Returns:
            Dictionary containing batch data
        """
        return next(iter(self._dataloader))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get data loader statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "num_samples": len(self.dataset),
            "num_batches": len(self),
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }
        
        if self._group_labels is not None:
            unique, counts = np.unique(self._group_labels, return_counts=True)
            stats["group_distribution"] = dict(zip(unique.tolist(), counts.tolist()))
            
        return stats
    
    def set_batch_size(self, batch_size: int) -> None:
        """
        Update the batch size.
        
        Args:
            batch_size: New batch size
        """
        self.batch_size = batch_size
        self._create_dataloader()
        
    def to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Move batch to device.
        
        Args:
            batch: Batch dictionary
            device: Target device
            
        Returns:
            Batch with tensors on target device
        """
        result = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.to(device)
            elif isinstance(value, dict):
                result[key] = self.to_device(value, device)
            else:
                result[key] = value
        return result


def collate_fn_variable_length(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for handling variable-length sequences.
    
    Args:
        batch: List of sample dictionaries
        
    Returns:
        Collated batch dictionary
    """
    result = {}
    
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]
        
        if isinstance(values[0], torch.Tensor):
            if values[0].dim() == 0:
                # Scalar tensor
                result[key] = torch.stack(values)
            elif len(values[0].shape) == 1:
                # 1D tensor - pad to max length
                max_len = max(v.size(0) for v in values)
                padded = torch.zeros(len(values), max_len, dtype=values[0].dtype)
                mask = torch.zeros(len(values), max_len, dtype=torch.bool)
                for i, v in enumerate(values):
                    padded[i, :v.size(0)] = v
                    mask[i, :v.size(0)] = True
                result[key] = padded
                result[f"{key}_mask"] = mask
            else:
                # Higher dimensional tensor - stack
                result[key] = torch.stack(values)
        elif isinstance(values[0], (int, float)):
            result[key] = torch.tensor(values)
        elif isinstance(values[0], str):
            result[key] = values
        else:
            result[key] = values
            
    return result
