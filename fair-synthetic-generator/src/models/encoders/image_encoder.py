"""
Image Encoder
=============

Encoder for image data using CNN, VAE, Vision Transformer (ViT), and Diffusion architectures.
All implementations are in PyTorch.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders.base_encoder import (
    BaseEncoder,
    VAEEncoder,
    get_activation
)


# ============================================================================
# Basic Building Blocks
# ============================================================================

class ConvBlock(nn.Module):
    """
    Convolutional block with normalization and activation.
    
    Architecture:
        Conv2d -> [Norm] -> Activation -> [Dropout]
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        use_norm: bool = True,
        activation: str = "silu",
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding,
                bias=not use_norm
            )
        ]
        
        if use_norm:
            layers.append(nn.GroupNorm(8, out_channels))
            
        layers.append(get_activation(activation))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
            
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutions and skip connection.
    
    Architecture:
        x -> Norm -> Act -> Conv -> Norm -> Act -> Dropout -> Conv -> + x
    """
    
    def __init__(
        self,
        channels: int,
        dropout: float = 0.0,
        activation: str = "silu"
    ):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            get_activation(activation),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.GroupNorm(8, channels),
            get_activation(activation),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class DownsampleBlock(nn.Module):
    """
    Downsampling block with optional residual connection.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_residual: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.conv = ConvBlock(in_channels, out_channels, stride=2)
        self.residual = ResidualBlock(out_channels, dropout) if use_residual else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.residual is not None:
            x = self.residual(x)
        return x


class SelfAttention(nn.Module):
    """
    Self-attention module for image feature maps.
    
    Reshapes spatial dimensions to sequence and applies multi-head attention.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm = nn.GroupNorm(8, channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads, dropout=dropout, batch_first=True
        )
        self.proj = nn.Linear(channels, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention to image features.
        
        Args:
            x: Feature tensor (batch, channels, height, width)
            
        Returns:
            Attended features (batch, channels, height, width)
        """
        b, c, h, w = x.shape
        
        # Normalize
        x_norm = self.norm(x)
        
        # Reshape to sequence
        x_seq = x_norm.flatten(2).transpose(1, 2)  # (batch, h*w, channels)
        
        # Self-attention
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        attn_out = self.proj(attn_out)
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).reshape(b, c, h, w)
        
        return x + attn_out


# ============================================================================
# Basic CNN Image Encoder
# ============================================================================

class ImageEncoder(BaseEncoder):
    """
    CNN-based encoder for image data.
    
    Architecture:
        Input -> Conv Blocks (downsampling) -> [Self-Attention] -> Flatten -> Linear -> Latent
        
    Features:
    - Configurable depth and width
    - Optional self-attention at specified layers
    - Residual connections
    - Multiple pooling strategies
    
    Example:
        >>> encoder = ImageEncoder(
        ...     input_channels=3,
        ...     resolution=256,
        ...     hidden_dims=[64, 128, 256, 512],
        ...     latent_dim=512
        ... )
        >>> z = encoder(images)
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        resolution: int = 256,
        hidden_dims: List[int] = [64, 128, 256, 512],
        latent_dim: int = 512,
        attention_layers: List[int] = [],
        num_heads: int = 8,
        activation: str = "silu",
        dropout: float = 0.0,
        use_residual: bool = True,
        name: str = "image_encoder",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the image encoder.
        
        Args:
            input_channels: Number of input channels
            resolution: Input image resolution
            hidden_dims: List of channel dimensions per stage
            latent_dim: Output latent dimension
            attention_layers: Indices of layers with self-attention
            num_heads: Number of attention heads
            activation: Activation function
            dropout: Dropout rate
            use_residual: Use residual blocks
            name: Encoder name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.input_channels = input_channels
        self.resolution = resolution
        self.hidden_dims = hidden_dims
        self.attention_layers = attention_layers
        
        # Build encoder
        self._build_encoder(
            hidden_dims, attention_layers, num_heads, 
            activation, dropout, use_residual
        )
        
        # Calculate final feature dimensions
        self.final_resolution = resolution // (2 ** len(hidden_dims))
        self.final_channels = hidden_dims[-1]
        self.final_feature_dim = self.final_channels * self.final_resolution ** 2
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_feature_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def _build_encoder(
        self,
        hidden_dims: List[int],
        attention_layers: List[int],
        num_heads: int,
        activation: str,
        dropout: float,
        use_residual: bool
    ) -> None:
        """Build convolutional encoder."""
        self.encoder = nn.ModuleList()
        
        in_channels = self.input_channels
        
        for i, out_channels in enumerate(hidden_dims):
            # Downsample block
            self.encoder.append(
                DownsampleBlock(
                    in_channels, out_channels,
                    use_residual=use_residual,
                    dropout=dropout
                )
            )
            
            # Optional self-attention
            if i in attention_layers:
                self.encoder.append(
                    SelfAttention(out_channels, num_heads, dropout)
                )
                
            in_channels = out_channels
            
    @property
    def input_dim(self) -> Tuple[int, int, int]:
        """Return input dimension (C, H, W)."""
        return (self.input_channels, self.resolution, self.resolution)
    
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode image to latent representation.
        
        Args:
            x: Image tensor (batch, channels, height, width)
            
        Returns:
            Latent tensor (batch, latent_dim)
        """
        # Pass through encoder
        for layer in self.encoder:
            x = layer(x)
            
        # Project to latent
        z = self.output_projection(x)
        
        return z
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self.encode(x, **kwargs)
    
    def get_intermediate_features(
        self, 
        x: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get intermediate feature maps.
        
        Args:
            x: Input image
            
        Returns:
            List of feature tensors from each stage
        """
        features = []
        
        for layer in self.encoder:
            x = layer(x)
            if isinstance(layer, DownsampleBlock):
                features.append(x)
                
        return features


class ImageVAEEncoder(VAEEncoder):
    """
    VAE-style encoder for image data with reparameterization.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        resolution: int = 256,
        hidden_dims: List[int] = [64, 128, 256, 512],
        latent_dim: int = 512,
        attention_layers: List[int] = [2, 3],
        name: str = "image_vae_encoder",
        **kwargs
    ):
        # Build CNN encoder
        self.cnn_encoder = ImageEncoder(
            input_channels=input_channels,
            resolution=resolution,
            hidden_dims=hidden_dims,
            latent_dim=hidden_dims[-1],  # Project to hidden_dims[-1] first
            attention_layers=attention_layers,
            name=name,
            **{k: v for k, v in kwargs.items() if k not in ['latent_dim', 'name']}
        )
        
        # Initialize VAE components
        nn.Module.__init__(self)
        self._latent_dim = latent_dim
        self.min_logvar = -10.0
        self.max_logvar = 10.0
        self.hidden_dims = hidden_dims
        self.input_channels = input_channels
        self.resolution = resolution
        
        # Feature dimension
        self.final_resolution = resolution // (2 ** len(hidden_dims))
        self.final_channels = hidden_dims[-1]
        self.feature_dim = self.final_channels * self.final_resolution ** 2
        
        # Build encoder layers (rebuild to get feature map)
        self._build_encoder(hidden_dims, attention_layers, kwargs.get('dropout', 0.0))
        
        # VAE heads
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)
        
    def _build_encoder(
        self,
        hidden_dims: List[int],
        attention_layers: List[int],
        dropout: float
    ) -> None:
        """Build encoder."""
        self.encoder = nn.ModuleList()
        
        in_channels = self.input_channels
        num_heads = 8
        use_residual = True
        
        for i, out_channels in enumerate(hidden_dims):
            self.encoder.append(
                DownsampleBlock(in_channels, out_channels, use_residual, dropout)
            )
            if i in attention_layers:
                self.encoder.append(SelfAttention(out_channels, num_heads, dropout))
            in_channels = out_channels
            
    def _encode_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Encode to intermediate features."""
        for layer in self.encoder:
            x = layer(x)
        return x.flatten(1)
    
    def encode(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters."""
        h = self._encode_features(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar
    
    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
        return_params: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass."""
        mu, logvar = self.encode(x)
        
        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
            
        if return_params:
            return z, mu, logvar
        return z


# ============================================================================
# Vision Transformer (ViT) Image Encoder
# ============================================================================

class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings.
    
    Splits image into patches and projects each patch to embedding dimension.
    """
    
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create patch embeddings.
        
        Args:
            x: Image tensor (batch, channels, height, width)
            
        Returns:
            Patch embeddings (batch, num_patches, embed_dim)
        """
        x = self.proj(x)  # (batch, embed_dim, h/patch, w/patch)
        x = x.flatten(2).transpose(1, 2)
        return x


class ViTImageEncoder(BaseEncoder):
    """
    Vision Transformer (ViT) encoder for images.
    
    Treats image patches as tokens and applies transformer encoder.
    
    Architecture:
        Image -> Patches -> Linear Projection -> Position Embedding -> Transformer -> CLS Token -> Latent
        
    Example:
        >>> encoder = ViTImageEncoder(
        ...     image_size=256,
        ...     patch_size=16,
        ...     embed_dim=768,
        ...     num_layers=12,
        ...     latent_dim=512
        ... )
        >>> z = encoder(images)
    """
    
    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        latent_dim: int = 512,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pretrained: Optional[str] = None,
        name: str = "vit_encoder",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ViT encoder.
        
        Args:
            image_size: Input image size
            patch_size: Patch size for tokenization
            in_channels: Number of input channels
            embed_dim: Embedding dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dim ratio
            latent_dim: Output latent dimension
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            pretrained: Optional pre-trained weights path
            name: Encoder name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
        # Load pretrained if specified
        if pretrained:
            self.load_state_dict(torch.load(pretrained))
            
    def _init_weights(self):
        """Initialize weights."""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    @property
    def input_dim(self) -> Tuple[int, int, int]:
        """Return input dimension."""
        return (3, self.image_size, self.image_size)
    
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode image using Vision Transformer.
        
        Args:
            x: Image tensor (batch, channels, height, width)
            
        Returns:
            Latent tensor (batch, latent_dim)
        """
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer encoder
        x = self.transformer(x)
        x = self.norm(x)
        
        # Use CLS token for representation
        cls_output = x[:, 0]
        
        # Project to latent
        z = self.output_projection(cls_output)
        
        return z
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass."""
        return self.encode(x, **kwargs)
    
    def get_patch_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get patch-level embeddings (for downstream tasks).
        
        Args:
            x: Image tensor
            
        Returns:
            Patch embeddings (batch, num_patches, embed_dim)
        """
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:]  # Exclude CLS position
        x = self.transformer(x)
        return x


# ============================================================================
# Diffusion Image Encoder
# ============================================================================

class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion models.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time embeddings.
        
        Args:
            t: Timestep tensor (batch,)
            
        Returns:
            Time embeddings (batch, dim)
        """
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ResBlock(nn.Module):
    """
    Residual block with time embedding for diffusion.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )
        
        # Residual connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()
            
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with time conditioning.
        
        Args:
            x: Feature tensor
            t: Time embedding
            
        Returns:
            Output features
        """
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Add time embedding
        h = h + self.time_mlp(t)[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.residual(x)


class DiffusionImageEncoder(BaseEncoder):
    """
    Diffusion-style encoder with time conditioning.
    
    Used for encoding images in diffusion models with optional
    timestep conditioning for consistency.
    
    Architecture:
        Image -> Time-conditioned Conv Blocks -> [Attention] -> Flatten -> Latent
        
    Example:
        >>> encoder = DiffusionImageEncoder(
        ...     resolution=256,
        ...     hidden_dims=[64, 128, 256, 512],
        ...     latent_dim=512
        ... )
        >>> z = encoder(images, timestep=0)  # timestep optional
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        resolution: int = 256,
        hidden_dims: List[int] = [64, 128, 256, 512],
        latent_dim: int = 512,
        attention_layers: List[int] = [2, 3],
        num_heads: int = 8,
        dropout: float = 0.1,
        time_embed_dim: int = 512,
        name: str = "diffusion_encoder",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize diffusion encoder.
        
        Args:
            input_channels: Input channels
            resolution: Image resolution
            hidden_dims: Hidden dimensions per stage
            latent_dim: Output latent dimension
            attention_layers: Layers with self-attention
            num_heads: Number of attention heads
            dropout: Dropout rate
            time_embed_dim: Time embedding dimension
            name: Encoder name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.input_channels = input_channels
        self.resolution = resolution
        self.hidden_dims = hidden_dims
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Build encoder blocks
        self.encoder = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        in_channels = input_channels
        
        for i, out_channels in enumerate(hidden_dims):
            # Residual block
            self.encoder.append(
                ResBlock(in_channels, out_channels, time_embed_dim, dropout)
            )
            
            # Optional attention
            if i in attention_layers:
                self.encoder.append(
                    SelfAttention(out_channels, num_heads, dropout)
                )
                
            # Downsampling
            self.downsamplers.append(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
            )
            
            in_channels = out_channels
            
        # Final feature dimensions
        self.final_resolution = resolution // (2 ** len(hidden_dims))
        self.final_channels = hidden_dims[-1]
        self.final_feature_dim = self.final_channels * self.final_resolution ** 2
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_feature_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    @property
    def input_dim(self) -> Tuple[int, int, int]:
        """Return input dimension."""
        return (self.input_channels, self.resolution, self.resolution)
    
    def encode(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode image to latent representation.
        
        Args:
            x: Image tensor (batch, channels, height, width)
            timestep: Optional timestep for conditioning (batch,)
            
        Returns:
            Latent tensor (batch, latent_dim)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Default timestep (t=0)
        if timestep is None:
            timestep = torch.zeros(batch_size, device=device)
            
        # Time embedding
        t_emb = self.time_embed(timestep)
        
        # Pass through encoder
        h = x
        layer_idx = 0
        
        for i, out_channels in enumerate(self.hidden_dims):
            # ResBlock
            h = self.encoder[layer_idx](h, t_emb)
            layer_idx += 1
            
            # Attention if present
            if i in [idx for idx in range(len(self.hidden_dims)) if idx in [2, 3]]:
                if layer_idx < len(self.encoder) and isinstance(
                    self.encoder[layer_idx], SelfAttention
                ):
                    h = self.encoder[layer_idx](h)
                    layer_idx += 1
                    
            # Downsample
            h = self.downsamplers[i](h)
            
        # Project to latent
        z = self.output_projection(h)
        
        return z
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass."""
        return self.encode(x, timestep, **kwargs)

