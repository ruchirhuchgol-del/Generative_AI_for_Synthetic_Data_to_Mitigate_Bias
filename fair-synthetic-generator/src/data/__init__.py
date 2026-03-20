"""
Data Module
===========

Provides preprocessing, augmentation, and schema utilities for all data modalities.

Subpackages:
- preprocessing: TabularPreprocessor, TextPreprocessor, ImagePreprocessor, MultimodalPreprocessor
- augmentation: TabularAugmenter, TextAugmenter, ImageAugmenter
- schemas: DataSchema, SensitiveAttribute
"""

from src.data.preprocessing import (
    TabularPreprocessor,
    TextPreprocessor,
    ImagePreprocessor,
    MultimodalPreprocessor,
)

__all__ = [
    "TabularPreprocessor",
    "TextPreprocessor",
    "ImagePreprocessor",
    "MultimodalPreprocessor",
]
