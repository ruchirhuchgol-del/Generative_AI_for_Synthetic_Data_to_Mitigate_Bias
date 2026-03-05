"""
Data Augmentation Module
========================

This module provides data augmentation utilities for different modalities:
- TabularAugmenter: Noise injection, mixup, SMOTE-style oversampling
- TextAugmenter: Synonym replacement, random insertion/deletion, back-translation
- ImageAugmenter: Geometric and color transformations
"""

from .tabular_augmenter import TabularAugmenter
from .text_augmenter import TextAugmenter
from .image_augmenter import ImageAugmenter

__all__ = [
    "TabularAugmenter",
    "TextAugmenter",
    "ImageAugmenter",
]
