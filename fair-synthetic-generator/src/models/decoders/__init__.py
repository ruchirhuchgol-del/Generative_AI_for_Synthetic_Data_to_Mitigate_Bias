"""
Decoders Module
===============

Decoder implementations for different data modalities.

This module provides decoders that transform latent representations
back to the original data space for each modality:
- Tabular: MLP and Transformer-based decoders
- Text: Transformer decoder with autoregressive generation
- Image: CNN, VAE, ViT, and Diffusion decoders
- Multimodal: Cross-modal attention fusion decoders

Architecture Overview:
    Latent Space (z) → Modality Decoder → Output Space
    
    For VAE:
        z ~ N(0, I) → Decoder → x_recon
        
    For Diffusion:
        x_noisy + t → Denoiser → noise_pred
        
    For Multimodal:
        z → Modality Heads → Cross-Attention → Modality Decoders → {x_tab, x_text, x_img}

Supported Frameworks:
    - PyTorch: All decoders
    - TensorFlow: Text decoder (TensorFlowTextDecoder)

Example Usage:
    >>> from src.models.decoders import (
    ...     TabularDecoder,
    ...     TextDecoder,
    ...     ImageDecoder,
    ...     MultimodalDecoder
    ... )
    >>> 
    >>> # Tabular decoder
    >>> tab_decoder = TabularDecoder(
    ...     num_numerical=10,
    ...     categorical_cardinalities={"cat1": 5, "cat2": 3},
    ...     latent_dim=512
    ... )
    >>> tab_output = tab_decoder(z)
    >>> 
    >>> # Text decoder with generation
    >>> text_decoder = TextDecoder(
    ...     vocab_size=50000,
    ...     latent_dim=512
    ... )
    >>> generated_ids = text_decoder.generate(z, max_length=50)
    >>> 
    >>> # Image decoder
    >>> img_decoder = ImageDecoder(
    ...     output_channels=3,
    ...     resolution=256,
    ...     latent_dim=512
    ... )
    >>> images = img_decoder(z)
    >>> 
    >>> # Multimodal decoder
    >>> mm_decoder = MultimodalDecoder(
    ...     latent_dim=512,
    ...     tabular_config={...},
    ...     text_config={...},
    ...     image_config={...}
    ... )
    >>> outputs = mm_decoder(z)
"""

# Base decoders
from src.models.decoders.base_decoder import (
    BaseDecoder,
    VAEDecoder,
    HierarchicalVAEDecoder,
    ConditionalDecoder,
    DiffusionDecoder,
    TimeEmbedding,
    get_activation,
    get_normalization
)

# Tabular decoders
from src.models.decoders.tabular_decoder import (
    TabularDecoder,
    TabularVAEDecoder,
    TabularConditionalDecoder,
    TabularTransformerDecoder
)

# Text decoders
from src.models.decoders.text_decoder import (
    TextDecoder,
    TextVAEDecoder,
    TensorFlowTextDecoder,
    PositionalEncoding,
    LearnedPositionalEncoding,
    TransformerDecoderLayer
)

# Image decoders
from src.models.decoders.image_decoder import (
    ImageDecoder,
    ImageVAEDecoder,
    ViTImageDecoder,
    DiffusionImageDecoder,
    ConvBlock,
    ResidualBlock,
    UpsampleBlock,
    SelfAttention
)

# Multimodal decoders
from src.models.decoders.multimodal_decoder import (
    MultimodalDecoder,
    MultimodalVAEDecoder,
    ConditionalMultimodalDecoder,
    HierarchicalMultimodalDecoder,
    ModalitySpecificHead,
    CrossModalAttention,
    MultimodalFusionDecoder
)


__all__ = [
    # Base decoders
    "BaseDecoder",
    "VAEDecoder",
    "HierarchicalVAEDecoder",
    "ConditionalDecoder",
    "DiffusionDecoder",
    "TimeEmbedding",
    
    # Tabular decoders
    "TabularDecoder",
    "TabularVAEDecoder",
    "TabularConditionalDecoder",
    "TabularTransformerDecoder",
    
    # Text decoders
    "TextDecoder",
    "TextVAEDecoder",
    "TensorFlowTextDecoder",
    "PositionalEncoding",
    "LearnedPositionalEncoding",
    "TransformerDecoderLayer",
    
    # Image decoders
    "ImageDecoder",
    "ImageVAEDecoder",
    "ViTImageDecoder",
    "DiffusionImageDecoder",
    "ConvBlock",
    "ResidualBlock",
    "UpsampleBlock",
    "SelfAttention",
    
    # Multimodal decoders
    "MultimodalDecoder",
    "MultimodalVAEDecoder",
    "ConditionalMultimodalDecoder",
    "HierarchicalMultimodalDecoder",
    "ModalitySpecificHead",
    "CrossModalAttention",
    "MultimodalFusionDecoder",
    
    # Utility functions
    "get_activation",
    "get_normalization",
]


# Registry for decoder lookup
DECODER_REGISTRY = {
    # Base decoders
    "base": BaseDecoder,
    "vae": VAEDecoder,
    "hierarchical_vae": HierarchicalVAEDecoder,
    "conditional": ConditionalDecoder,
    "diffusion": DiffusionDecoder,
    
    # Tabular decoders
    "tabular": TabularDecoder,
    "tabular_vae": TabularVAEDecoder,
    "tabular_conditional": TabularConditionalDecoder,
    "tabular_transformer": TabularTransformerDecoder,
    
    # Text decoders
    "text": TextDecoder,
    "text_vae": TextVAEDecoder,
    "text_tensorflow": TensorFlowTextDecoder,
    
    # Image decoders
    "image": ImageDecoder,
    "image_vae": ImageVAEDecoder,
    "image_vit": ViTImageDecoder,
    "image_diffusion": DiffusionImageDecoder,
    
    # Multimodal decoders
    "multimodal": MultimodalDecoder,
    "multimodal_vae": MultimodalVAEDecoder,
    "multimodal_conditional": ConditionalMultimodalDecoder,
    "multimodal_hierarchical": HierarchicalMultimodalDecoder,
}


def get_decoder(name: str, **kwargs):
    """
    Get a decoder by name from the registry.
    
    Args:
        name: Decoder name
        **kwargs: Arguments to pass to the decoder constructor
        
    Returns:
        Decoder instance
        
    Raises:
        ValueError: If decoder name is not found
    """
    if name not in DECODER_REGISTRY:
        available = list(DECODER_REGISTRY.keys())
        raise ValueError(
            f"Unknown decoder: {name}. Available decoders: {available}"
        )
    
    return DECODER_REGISTRY[name](**kwargs)


def list_decoders() -> list:
    """
    List all available decoder names.
    
    Returns:
        List of decoder names
    """
    return list(DECODER_REGISTRY.keys())
