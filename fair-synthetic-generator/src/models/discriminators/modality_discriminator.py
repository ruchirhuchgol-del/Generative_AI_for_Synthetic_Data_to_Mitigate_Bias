"""
Modality Discriminator
======================

Modality-specific discriminators for tabular, text, and image data.
Supports GAN training and domain adaptation across modalities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.discriminators.base_discriminator import (
    BaseDiscriminator,
    BinaryDiscriminator,
    MultiClassDiscriminator,
    PatchDiscriminator
)


# ============================================================================
# Tabular Discriminators
# ============================================================================

class TabularDiscriminator(BaseDiscriminator):
    """
    Discriminator for tabular data with mixed features.
    
    Handles:
    - Numerical features: Direct input
    - Categorical features: Embedding
    
    Example:
        >>> discriminator = TabularDiscriminator(
        ...     num_numerical=10,
        ...     categorical_cardinalities={"cat1": 5, "cat2": 3},
        ...     hidden_dims=[256, 128]
        ... )
        >>> logits = discriminator(numerical=x_num, categorical={"cat1": c1, "cat2": c2})
    """
    
    def __init__(
        self,
        num_numerical: int = 0,
        categorical_cardinalities: Optional[Dict[str, int]] = None,
        hidden_dims: List[int] = [256, 128],
        embedding_dim: int = 32,
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        spectral_norm: bool = True,
        output_type: str = "binary",  # binary, multi_class
        num_classes: int = 2,
        name: str = "tabular_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize tabular discriminator.
        
        Args:
            num_numerical: Number of numerical features
            categorical_cardinalities: Dict mapping categorical names to cardinalities
            hidden_dims: Hidden layer dimensions
            embedding_dim: Categorical embedding dimension
            activation: Activation function
            dropout: Dropout rate
            spectral_norm: Apply spectral normalization
            output_type: Output type (binary or multi_class)
            num_classes: Number of classes (for multi_class)
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.num_numerical = num_numerical
        self.categorical_cardinalities = categorical_cardinalities or {}
        self.output_type = output_type
        self.num_classes = num_classes
        
        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleDict()
        total_emb_dim = 0
        for cat_name, cardinality in self.categorical_cardinalities.items():
            emb_dim = min(embedding_dim, (cardinality + 1) // 2)
            self.categorical_embeddings[cat_name] = nn.Embedding(cardinality, emb_dim)
            total_emb_dim += emb_dim
        
        # Total input dimension
        self._input_dim = num_numerical + total_emb_dim
        
        # Build network
        layers = []
        prev_dim = self._input_dim
        
        for dim in hidden_dims:
            linear = nn.Linear(prev_dim, dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LayerNorm(dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.layers = nn.Sequential(*layers)
        
        # Output layer
        output_dim = 1 if output_type == "binary" else num_classes
        self.output = nn.Linear(prev_dim, output_dim)
        if spectral_norm:
            self.output = nn.utils.spectral_norm(self.output)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def discriminate(
        self,
        numerical: Optional[torch.Tensor] = None,
        categorical: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute discrimination output.
        
        Args:
            numerical: Numerical features (batch, num_numerical)
            categorical: Dict of categorical tensors
            
        Returns:
            Discrimination logits
        """
        features = []
        
        # Numerical features
        if numerical is not None:
            features.append(numerical)
        
        # Categorical embeddings
        if categorical is not None:
            for cat_name, cat_values in categorical.items():
                if cat_name in self.categorical_embeddings:
                    emb = self.categorical_embeddings[cat_name](cat_values)
                    features.append(emb)
        
        if not features:
            raise ValueError("No features provided")
        
        x = torch.cat(features, dim=-1)
        
        # Forward pass
        h = self.layers(x)
        return self.output(h)


class TabularTransformerDiscriminator(BaseDiscriminator):
    """
    Transformer-based discriminator for tabular data.
    
    Uses self-attention over feature columns for better feature interaction modeling.
    
    Architecture:
        Features → Column Embeddings → Transformer → Classification
    """
    
    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        activation: str = "gelu",
        dropout: float = 0.1,
        output_type: str = "binary",
        name: str = "tabular_transformer_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize transformer discriminator.
        
        Args:
            num_features: Number of features
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            activation: Activation function
            dropout: Dropout rate
            output_type: Output type
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.num_features = num_features
        self.d_model = d_model
        
        # Feature projection (each feature → token)
        self.feature_projection = nn.Linear(1, d_model)
        
        # Positional encoding for columns
        self.positional_encoding = nn.Parameter(
            torch.randn(1, num_features, d_model) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output head
        output_dim = 1 if output_type == "binary" else 2
        self.output = nn.Sequential(
            nn.Linear(d_model * num_features, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, output_dim)
        )
    
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute discrimination output.
        
        Args:
            x: Input tensor (batch, num_features)
            
        Returns:
            Discrimination logits
        """
        batch_size = x.shape[0]
        
        # Project each feature to token
        x = x.unsqueeze(-1)  # (batch, num_features, 1)
        x = self.feature_projection(x)  # (batch, num_features, d_model)
        
        # Add positional encoding
        x = x + self.positional_encoding
        
        # Apply transformer
        x = self.transformer(x)  # (batch, num_features, d_model)
        
        # Flatten and classify
        x = x.flatten(1)
        return self.output(x)


# ============================================================================
# Text Discriminators
# ============================================================================

class TextDiscriminator(BaseDiscriminator):
    """
    Discriminator for text data using transformer architecture.
    
    Supports:
    - Token-level discrimination
    - Sequence-level discrimination
    - Pre-trained encoder integration
    
    Example:
        >>> discriminator = TextDiscriminator(
        ...     vocab_size=50000,
        ...     hidden_dim=768,
        ...     num_layers=6
        ... )
        >>> logits = discriminator(input_ids, attention_mask=mask)
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 256,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_dim: int = 2048,
        pooling_strategy: str = "cls",  # cls, mean, max
        activation: str = "gelu",
        dropout: float = 0.1,
        output_type: str = "binary",
        pretrained_model: Optional[str] = None,
        name: str = "text_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize text discriminator.
        
        Args:
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            intermediate_dim: Feedforward dimension
            pooling_strategy: Pooling strategy
            activation: Activation function
            dropout: Dropout rate
            output_type: Output type
            pretrained_model: Optional pre-trained model name
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.pooling_strategy = pooling_strategy
        self.output_type = output_type
        
        if pretrained_model is not None:
            self._init_pretrained(pretrained_model)
        else:
            self._init_from_scratch(
                vocab_size, hidden_dim, num_layers, num_heads,
                intermediate_dim, activation, dropout
            )
        
        # Output head
        output_dim = 1 if output_type == "binary" else 2
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def _init_from_scratch(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        intermediate_dim: int,
        activation: str,
        dropout: float
    ):
        """Initialize model from scratch."""
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(1, self.max_length, hidden_dim) * 0.02
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self._is_pretrained = False
    
    def _init_pretrained(self, model_name: str):
        """Initialize from pre-trained model."""
        try:
            from transformers import AutoModel, AutoConfig
            
            self.pretrained = AutoModel.from_pretrained(model_name)
            self.config_pretrained = AutoConfig.from_pretrained(model_name)
            self.hidden_dim = self.config_pretrained.hidden_size
            self._is_pretrained = True
        except ImportError:
            raise ImportError(
                "transformers library required for pre-trained models"
            )
    
    def discriminate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute discrimination output.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Discrimination logits
        """
        if self._is_pretrained:
            outputs = self.pretrained(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            hidden_states = outputs.last_hidden_state
        else:
            # Token embedding
            x = self.token_embedding(input_ids)
            x = x + self.positional_encoding[:, :x.shape[1]]
            
            # Create padding mask
            src_key_padding_mask = None
            if attention_mask is not None:
                src_key_padding_mask = (attention_mask == 0)
            
            x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
            hidden_states = x
        
        # Pool
        if self.pooling_strategy == "cls":
            pooled = hidden_states[:, 0]
        elif self.pooling_strategy == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = hidden_states.mean(dim=1)
        else:  # max
            pooled = hidden_states.max(dim=1)[0]
        
        return self.output(pooled)


class TextCNNDiscriminator(BaseDiscriminator):
    """
    CNN-based discriminator for text classification.
    
    Uses multiple filter sizes for n-gram detection.
    Faster than transformer for simple classification tasks.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embedding_dim: int = 300,
        num_filters: int = 100,
        filter_sizes: List[int] = [3, 4, 5],
        hidden_dim: int = 256,
        dropout: float = 0.5,
        output_type: str = "binary",
        name: str = "text_cnn_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize CNN discriminator.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            num_filters: Number of filters per size
            filter_sizes: List of filter sizes (n-grams)
            hidden_dim: Hidden dimension
            dropout: Dropout rate
            output_type: Output type
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Parallel convolutions
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim))
            for k in filter_sizes
        ])
        
        # Output layers
        output_dim = 1 if output_type == "binary" else 2
        self.output = nn.Sequential(
            nn.Linear(num_filters * len(filter_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def discriminate(
        self,
        input_ids: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute discrimination output.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            
        Returns:
            Discrimination logits
        """
        # Embed
        x = self.embedding(input_ids)  # (batch, seq_len, emb_dim)
        x = x.unsqueeze(1)  # (batch, 1, seq_len, emb_dim)
        
        # Convolve and pool
        conv_outputs = []
        for conv in self.convs:
            h = conv(x).squeeze(-1)  # (batch, num_filters, seq_len - k + 1)
            h = F.relu(h)
            h = F.max_pool1d(h, h.shape[-1]).squeeze(-1)  # (batch, num_filters)
            conv_outputs.append(h)
        
        # Concatenate
        x = torch.cat(conv_outputs, dim=-1)
        
        return self.output(x)


# ============================================================================
# Image Discriminators
# ============================================================================

class ImageDiscriminator(BaseDiscriminator):
    """
    CNN-based discriminator for image data.
    
    Architecture:
        Image → Conv Blocks → Global Pooling → Classification
        
    Supports:
    - Spectral normalization
    - Self-attention layers
    - Progressive growing
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dims: List[int] = [64, 128, 256, 512],
        kernel_size: int = 3,
        activation: str = "leaky_relu",
        normalization: str = "instance",
        dropout: float = 0.0,
        spectral_norm: bool = True,
        attention_layers: List[int] = [],
        num_heads: int = 8,
        output_type: str = "binary",
        name: str = "image_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize image discriminator.
        
        Args:
            input_channels: Number of input channels
            hidden_dims: Hidden channel dimensions
            kernel_size: Convolution kernel size
            activation: Activation function
            normalization: Normalization type
            dropout: Dropout rate
            spectral_norm: Apply spectral normalization
            attention_layers: Layers to add self-attention
            num_heads: Number of attention heads
            output_type: Output type
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.input_channels = input_channels
        self.hidden_dims = hidden_dims
        self.attention_layers = attention_layers
        
        # Build convolutional layers
        self.layers = nn.ModuleList()
        prev_channels = input_channels
        
        for i, out_channels in enumerate(hidden_dims):
            # Convolutional block
            layers = []
            
            # Conv
            conv = nn.Conv2d(
                prev_channels, out_channels, kernel_size, 
                stride=2, padding=kernel_size // 2
            )
            if spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.append(conv)
            
            # Normalization
            if normalization == "instance":
                layers.append(nn.InstanceNorm2d(out_channels))
            elif normalization == "batch":
                layers.append(nn.BatchNorm2d(out_channels))
            elif normalization == "layer":
                layers.append(nn.GroupNorm(1, out_channels))
            
            # Activation
            layers.append(self._get_activation(activation))
            
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            
            self.layers.append(nn.Sequential(*layers))
            
            # Self-attention
            if i in attention_layers:
                self.layers.append(
                    SelfAttention2d(out_channels, num_heads)
                )
            
            prev_channels = out_channels
        
        # Output head
        output_dim = 1 if output_type == "binary" else 2
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dims[-1], output_dim)
        )
        
        if spectral_norm:
            self.output[-1] = nn.utils.spectral_norm(self.output[-1])
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute discrimination output.
        
        Args:
            x: Image tensor (batch, channels, height, width)
            
        Returns:
            Discrimination logits
        """
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


class SelfAttention2d(nn.Module):
    """
    Self-attention module for 2D feature maps.
    
    Computes attention over spatial dimensions.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention.
        
        Args:
            x: Feature tensor (batch, channels, height, width)
            
        Returns:
            Attended features
        """
        b, c, h, w = x.shape
        
        # Normalize and reshape
        x_norm = self.norm(x)
        x_seq = x_norm.flatten(2).transpose(1, 2)  # (batch, h*w, channels)
        
        # Self-attention
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        
        # Project and residual
        return x + self.proj(attn_out)


class ResNetDiscriminator(BaseDiscriminator):
    """
    ResNet-style discriminator with residual blocks.
    
    Deep architecture with skip connections for stable training.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        base_channels: int = 64,
        num_blocks: List[int] = [2, 2, 2, 2],
        spectral_norm: bool = True,
        output_type: str = "binary",
        name: str = "resnet_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ResNet discriminator.
        
        Args:
            input_channels: Number of input channels
            base_channels: Base number of channels
            num_blocks: Number of residual blocks per stage
            spectral_norm: Apply spectral normalization
            output_type: Output type
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.input_channels = input_channels
        
        # Initial convolution
        self.conv_in = nn.Conv2d(input_channels, base_channels, 3, 1, 1)
        if spectral_norm:
            self.conv_in = nn.utils.spectral_norm(self.conv_in)
        
        # Residual stages
        self.stages = nn.ModuleList()
        channels = base_channels
        
        for i, num_block in enumerate(num_blocks):
            out_channels = channels * 2 if i > 0 else channels
            stage = self._make_stage(
                channels, out_channels, num_block, 
                stride=2 if i > 0 else 1,
                spectral_norm=spectral_norm
            )
            self.stages.append(stage)
            channels = out_channels
        
        # Output head
        output_dim = 1 if output_type == "binary" else 2
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, output_dim)
        )
        
        if spectral_norm:
            self.output[-1] = nn.utils.spectral_norm(self.output[-1])
    
    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        spectral_norm: bool
    ) -> nn.Sequential:
        """Create a stage of residual blocks."""
        layers = []
        
        for i in range(num_blocks):
            layers.append(ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                stride=stride if i == 0 else 1,
                spectral_norm=spectral_norm
            ))
        
        return nn.Sequential(*layers)
    
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute discrimination output."""
        x = self.conv_in(x)
        for stage in self.stages:
            x = stage(x)
        return self.output(x)


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        spectral_norm: bool = True
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride, 1
        )
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        
        if spectral_norm:
            self.conv1 = nn.utils.spectral_norm(self.conv1)
            self.conv2 = nn.utils.spectral_norm(self.conv2)
        
        self.norm1 = nn.InstanceNorm2d(in_channels)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride)
            if spectral_norm:
                self.skip = nn.utils.spectral_norm(self.skip)
        else:
            self.skip = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.leaky_relu(h, 0.2)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)
        
        if self.skip is not None:
            x = self.skip(x)
        
        return x + h


# ============================================================================
# Multimodal Discriminators
# ============================================================================

class MultimodalDiscriminator(BaseDiscriminator):
    """
    Multimodal discriminator combining multiple modalities.
    
    Supports:
    - Joint discrimination across modalities
    - Modality-specific branches
    - Cross-modal attention
    """
    
    def __init__(
        self,
        tabular_discriminator: Optional[nn.Module] = None,
        text_discriminator: Optional[nn.Module] = None,
        image_discriminator: Optional[nn.Module] = None,
        fusion_dim: int = 512,
        fusion_type: str = "concat",  # concat, attention, mean
        hidden_dim: int = 256,
        output_type: str = "binary",
        name: str = "multimodal_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multimodal discriminator.
        
        Args:
            tabular_discriminator: Tabular discriminator module
            text_discriminator: Text discriminator module
            image_discriminator: Image discriminator module
            fusion_dim: Fusion dimension
            fusion_type: How to fuse modality features
            hidden_dim: Hidden dimension for output head
            output_type: Output type
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.tabular_discriminator = tabular_discriminator
        self.text_discriminator = text_discriminator
        self.image_discriminator = image_discriminator
        
        self.modalities = []
        if tabular_discriminator is not None:
            self.modalities.append("tabular")
        if text_discriminator is not None:
            self.modalities.append("text")
        if image_discriminator is not None:
            self.modalities.append("image")
        
        self.fusion_type = fusion_type
        
        # Modality projection layers
        self.modality_projections = nn.ModuleDict()
        for modality in self.modalities:
            self.modality_projections[modality] = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU()
            )
        
        # Cross-modal fusion
        if fusion_type == "attention":
            self.fusion = CrossModalAttention(fusion_dim, len(self.modalities))
        else:
            self.fusion = None
        
        # Output head
        output_dim = 1 if output_type == "binary" else 2
        if fusion_type == "concat":
            input_dim = fusion_dim * len(self.modalities)
        else:
            input_dim = fusion_dim
        
        self.output = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def discriminate(
        self,
        tabular: Optional[Dict[str, torch.Tensor]] = None,
        text: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute multimodal discrimination output.
        
        Args:
            tabular: Tabular data dict
            text: Text tensor
            image: Image tensor
            
        Returns:
            Discrimination logits
        """
        features = []
        
        # Extract features from each modality
        if tabular is not None and self.tabular_discriminator is not None:
            tab_features = self._extract_features(
                self.tabular_discriminator, tabular
            )
            tab_features = self.modality_projections["tabular"](tab_features)
            features.append(("tabular", tab_features))
        
        if text is not None and self.text_discriminator is not None:
            text_features = self._extract_features(
                self.text_discriminator, text
            )
            text_features = self.modality_projections["text"](text_features)
            features.append(("text", text_features))
        
        if image is not None and self.image_discriminator is not None:
            img_features = self._extract_features(
                self.image_discriminator, image
            )
            img_features = self.modality_projections["image"](img_features)
            features.append(("image", img_features))
        
        # Fuse features
        if self.fusion_type == "concat":
            fused = torch.cat([f[1] for f in features], dim=-1)
        elif self.fusion_type == "attention" and self.fusion is not None:
            feature_list = [f[1] for f in features]
            fused = self.fusion(feature_list)
            fused = torch.mean(torch.stack(fused), dim=0)
        else:  # mean
            fused = torch.mean(torch.stack([f[1] for f in features]), dim=0)
        
        return self.output(fused)
    
    def _extract_features(
        self, 
        discriminator: nn.Module, 
        x: Any
    ) -> torch.Tensor:
        """Extract features from discriminator."""
        # Try to get intermediate features
        if hasattr(discriminator, 'layers'):
            if isinstance(x, dict):
                h = discriminator.layers(x.get('numerical'))
            else:
                h = x
                for layer in discriminator.layers:
                    h = layer(h)
            return h.flatten(1)
        else:
            # Fallback: use output
            if isinstance(x, dict):
                return discriminator(**x)
            return discriminator(x)


class CrossModalAttention(nn.Module):
    """Cross-modal attention for feature fusion."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_modalities: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attentions = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_modalities)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_modalities)
        ])
    
    def forward(
        self,
        features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Apply cross-modal attention.
        
        Args:
            features: List of feature tensors per modality
            
        Returns:
            List of attended features
        """
        output_features = []
        
        for i, (feat, attn, norm) in enumerate(zip(features, self.attentions, self.norms)):
            # Attend to all other modalities
            other_features = torch.cat([f for j, f in enumerate(features) if j != i], dim=1)
            feat_3d = feat.unsqueeze(1)  # (batch, 1, dim)
            
            attn_out, _ = attn(feat_3d, other_features.unsqueeze(1), other_features.unsqueeze(1))
            output_features.append(norm(feat + attn_out.squeeze(1)))
        
        return output_features
