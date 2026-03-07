"""
Encoders Module
===============

Encoder implementations for different data modalities.

This module provides:
- BaseEncoder: Abstract base class for all encoders
- VAEEncoder: VAE-style encoder with reparameterization
- TabularEncoder: PyTorch encoder for tabular data (MLP/TabTransformer)
- TextEncoder: TensorFlow/PyTorch encoder for text data
- ImageEncoder: PyTorch encoder for images (CNN/VAE/ViT/Diffusion)
- MultimodalFusion: Cross-modal attention fusion
"""

from src.models.encoders.base_encoder import BaseEncoder, VAEEncoder
from src.models.encoders.tabular_encoder import (
    TabularEncoder,
    TabularVAEEncoder,
    TabTransformerEncoder,
)
from src.models.encoders.text_encoder import (
    TextEncoder,
    TextVAEEncoder,
    TensorFlowTextEncoder,
)
from src.models.encoders.image_encoder import (
    ImageEncoder,
    ImageVAEEncoder,
    ViTImageEncoder,
    DiffusionImageEncoder,
)
from src.models.encoders.multimodal_fusion import (
    MultimodalFusion,
    CrossModalAttention,
    HierarchicalFusion,
)

__all__ = [
    # Base classes
    "BaseEncoder",
    "VAEEncoder",
    
    # Tabular encoders (PyTorch)
    "TabularEncoder",
    "TabularVAEEncoder",
    "TabTransformerEncoder",
    
    # Text encoders (TensorFlow + PyTorch)
    "TextEncoder",
    "TextVAEEncoder",
    "TensorFlowTextEncoder",
    
    # Image encoders (PyTorch)
    "ImageEncoder",
    "ImageVAEEncoder",
    "ViTImageEncoder",
    "DiffusionImageEncoder",
    
    # Multimodal fusion
    "MultimodalFusion",
    "CrossModalAttention",
    "HierarchicalFusion",
]
