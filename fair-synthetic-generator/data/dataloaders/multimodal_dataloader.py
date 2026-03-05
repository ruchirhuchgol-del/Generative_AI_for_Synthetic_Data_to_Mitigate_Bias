"""
Multimodal Data Loader
======================

Data loaders for multimodal datasets with fairness-aware sampling.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler

from src.data.dataset import MultimodalDataset


class FairBatchSampler(Sampler):
    """
    Batch sampler that ensures balanced representation of sensitive groups.
    
    Creates batches with equal representation from each sensitive group
    to ensure fairness-aware training.
    """
    
    def __init__(
        self,
        dataset: MultimodalDataset,
        batch_size: int,
        drop_last: bool = False,
        balance_strategy: str = "equal"
    ):
        """
        Initialize the fair batch sampler.
        
        Args:
            dataset: MultimodalDataset instance
            batch_size: Number of samples per batch
            drop_last: Whether to drop the last incomplete batch
            balance_strategy: Balancing strategy ('equal', 'proportional', 'oversample')
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.balance_strategy = balance_strategy
        
        self.group_indices = dataset.get_sensitive_groups()
        self.num_groups = len(self.group_indices)
        
        # Samples per group per batch
        if balance_strategy == "equal":
            self.samples_per_group = batch_size // self.num_groups
        elif balance_strategy == "proportional":
            total = len(dataset)
            self.samples_per_group = {
                k: int(batch_size * len(v) / total)
                for k, v in self.group_indices.items()
            }
        else:
            self.samples_per_group = batch_size // self.num_groups
    
    def __iter__(self):
        """Iterate over batches."""
        # Shuffle indices within each group
        group_iters = {}
        for key, indices in self.group_indices.items():
            shuffled = np.random.permutation(indices).tolist()
            group_iters[key] = iter(shuffled)
        
        num_batches = len(self)
        
        for _ in range(num_batches):
            batch_indices = []
            
            for key in self.group_indices.keys():
                if self.balance_strategy == "equal":
                    n = self.samples_per_group
                else:
                    n = self.samples_per_group.get(key, self.batch_size // self.num_groups)
                
                for _ in range(n):
                    try:
                        idx = next(group_iters[key])
                        batch_indices.append(idx)
                    except StopIteration:
                        # Reshuffle and continue
                        shuffled = np.random.permutation(self.group_indices[key]).tolist()
                        group_iters[key] = iter(shuffled)
                        idx = next(group_iters[key])
                        batch_indices.append(idx)
            
            np.random.shuffle(batch_indices)
            yield batch_indices
    
    def __len__(self) -> int:
        """Get number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


class MultimodalDataLoader:
    """
    Data loader for multimodal datasets.
    
    Handles batching, shuffling, and collation of multimodal data
    with support for fairness-aware sampling.
    """
    
    def __init__(
        self,
        dataset: MultimodalDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        balance_sensitive_groups: bool = False,
        balance_strategy: str = "equal",
        collate_fn: Optional[Callable] = None,
    ):
        """
        Initialize the data loader.
        
        Args:
            dataset: MultimodalDataset instance
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            drop_last: Whether to drop last incomplete batch
            balance_sensitive_groups: Whether to balance sensitive groups in batches
            balance_strategy: Strategy for balancing ('equal', 'proportional')
            collate_fn: Optional custom collation function
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.balance_sensitive_groups = balance_sensitive_groups
        
        if collate_fn is None:
            collate_fn = self._default_collate
        
        if balance_sensitive_groups:
            batch_sampler = FairBatchSampler(
                dataset, batch_size, drop_last, balance_strategy
            )
            self.dataloader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=collate_fn,
            )
        else:
            self.dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                drop_last=drop_last,
                collate_fn=collate_fn,
            )
    
    def _default_collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate multimodal batch.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Collated batch dictionary
        """
        collated = {}
        
        # Handle tabular data
        tabular = [s["tabular"] for s in batch if s["tabular"] is not None]
        if tabular:
            collated["tabular"] = torch.stack(tabular)
        
        # Handle text data
        text = [s["text"] for s in batch if s["text"] is not None]
        if text:
            # Pad text sequences
            max_len = max(t.shape[0] for t in text)
            padded = torch.zeros(len(text), max_len, dtype=torch.long)
            for i, t in enumerate(text):
                padded[i, :t.shape[0]] = t
            collated["text"] = padded
        
        # Handle image data
        images = [s["image"] for s in batch if s["image"] is not None]
        if images:
            collated["image"] = torch.stack(images)
        
        # Handle sensitive attributes
        sensitive_attrs = {}
        for s in batch:
            for k, v in s["sensitive_attrs"].items():
                if k not in sensitive_attrs:
                    sensitive_attrs[k] = []
                sensitive_attrs[k].append(v)
        
        for k, v in sensitive_attrs.items():
            collated[f"sensitive_{k}"] = torch.tensor(v)
        
        collated["sensitive_attrs"] = sensitive_attrs
        
        # Handle targets
        targets = [s["target"] for s in batch if s["target"] is not None]
        if targets:
            collated["target"] = torch.stack(targets)
        
        # Handle metadata
        collated["metadata"] = [s["metadata"] for s in batch]
        
        return collated
    
    def __iter__(self):
        """Iterate over batches."""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.dataloader)
    
    def get_batch(self) -> Dict[str, Any]:
        """Get a single batch (for testing)."""
        return next(iter(self.dataloader))


def create_dataloader(
    dataset: MultimodalDataset,
    mode: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    balance_sensitive_groups: bool = False,
    **kwargs
) -> MultimodalDataLoader:
    """
    Factory function to create a data loader.
    
    Args:
        dataset: MultimodalDataset instance
        mode: 'train', 'val', or 'test'
        batch_size: Number of samples per batch
        num_workers: Number of worker processes
        balance_sensitive_groups: Whether to balance sensitive groups
        **kwargs: Additional arguments
        
    Returns:
        MultimodalDataLoader instance
    """
    shuffle = (mode == "train")
    drop_last = (mode == "train")
    
    return MultimodalDataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        balance_sensitive_groups=balance_sensitive_groups and mode == "train",
        **kwargs
    )
