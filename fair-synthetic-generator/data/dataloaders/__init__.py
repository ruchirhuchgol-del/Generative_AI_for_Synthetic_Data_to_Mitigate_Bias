"""
Dataloaders Package
===================

This package provides data loading utilities for the Fair Synthetic Data Generator.
Supports tabular, text, image, and multimodal data loading with fairness-aware sampling.
"""

from .base_dataloader import BaseDataLoader, BaseDataset
from .tabular_dataloader import TabularDataLoader, TabularDataset
from .text_dataloader import TextDataLoader, TextDataset
from .image_dataloader import ImageDataLoader, ImageDataset
from .multimodal_dataloader import MultimodalDataLoader, MultimodalDataset

__all__ = [
    # Base classes
    "BaseDataLoader",
    "BaseDataset",
    # Tabular
    "TabularDataLoader",
    "TabularDataset",
    # Text
    "TextDataLoader",
    "TextDataset",
    # Image
    "ImageDataLoader",
    "ImageDataset",
    # Multimodal
    "MultimodalDataLoader",
    "MultimodalDataset",
]
