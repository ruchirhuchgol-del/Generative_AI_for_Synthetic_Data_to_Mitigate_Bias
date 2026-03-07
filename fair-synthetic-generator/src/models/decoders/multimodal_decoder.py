"""
Multimodal Decoder
==================

Decoder for multimodal data combining tabular, text, and image modalities.
Supports joint decoding and modality-specific decoders with cross-attention.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoders.base_decoder import (
    BaseDecoder,
    VAEDecoder,
    ConditionalDecoder,
    get_activation
)
from src.models.decoders.tabular_decoder import TabularDecoder
from src.models.decoders.text_decoder import TextDecoder
from src.models.decoders.image_decoder import ImageDecoder


class ModalitySpecificHead(nn.Module):
    """
    Modality-specific decoding head.
    
    Projects from shared latent space to modality-specific representation
    before final decoding.
    """
    
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.projection = nn.Sequential(*layers)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project latent to modality-specific representation."""
        return self.projection(z)


class CrossModalAttention(nn.Module):
    """
    Cross-attention between modalities.
    
    Allows each modality decoder to attend to other modality representations
    for coherent multimodal generation.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            query_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Project keys if dimensions differ
        if query_dim != key_dim:
            self.key_proj = nn.Linear(key_dim, query_dim)
            self.value_proj = nn.Linear(key_dim, query_dim)
        else:
            self.key_proj = None
            self.value_proj = None
            
        self.norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            query: Query tensor (batch, seq_q, query_dim)
            key: Key tensor (batch, seq_k, key_dim)
            value: Value tensor (batch, seq_k, key_dim)
            
        Returns:
            Attended query tensor
        """
        # Project if needed
        if self.key_proj is not None:
            key = self.key_proj(key)
            value = self.value_proj(value)
            
        # Ensure 3D tensors
        if query.dim() == 2:
            query = query.unsqueeze(1)
        if key.dim() == 2:
            key = key.unsqueeze(1)
        if value.dim() == 2:
            value = value.unsqueeze(1)
            
        # Cross-attention
        attn_out, _ = self.attention(query, key, value)
        
        # Residual + norm
        output = self.norm(query + self.dropout(attn_out))
        
        return output.squeeze(1) if output.shape[1] == 1 else output


class MultimodalFusionDecoder(nn.Module):
    """
    Multimodal fusion decoder for cross-modal attention.
    
    Fuses information across modalities before final decoding.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_modalities: int = 3,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_modalities = num_modalities
        
        # Cross-attention layers for each modality pair
        self.cross_attentions = nn.ModuleList([
            nn.ModuleList([
                CrossModalAttention(latent_dim, latent_dim, num_heads, dropout)
                if i != j else nn.Identity()
                for j in range(num_modalities)
            ])
            for i in range(num_modalities)
        ])
        
        # Self-attention for fusion
        self.self_attentions = nn.ModuleList([
            nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(latent_dim)
            for _ in range(num_layers * 2)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        modality_latents: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Fuse modality latents through cross-attention.
        
        Args:
            modality_latents: List of latent tensors, one per modality
            
        Returns:
            List of fused latent tensors
        """
        # Ensure all latents are 3D
        latents_3d = []
        for z in modality_latents:
            if z.dim() == 2:
                latents_3d.append(z.unsqueeze(1))
            else:
                latents_3d.append(z)
                
        # Cross-modal attention
        fused_latents = []
        for i, z in enumerate(latents_3d):
            # Attend to all other modalities
            cross_attn_out = z
            for j, other_z in enumerate(latents_3d):
                if i != j:
                    cross_attn_out = cross_attn_out + self.cross_attentions[i][j](
                        cross_attn_out, other_z, other_z
                    )
            fused_latents.append(cross_attn_out)
            
        # Stack for self-attention
        stacked = torch.stack(fused_latents, dim=1)  # (batch, num_modalities, latent_dim)
        
        # Self-attention layers
        for i, self_attn in enumerate(self.self_attentions):
            # Self-attention
            attn_out, _ = self_attn(stacked, stacked, stacked)
            stacked = self.layer_norms[i * 2](stacked + self.dropout(attn_out))
            
        # Separate modalities
        output_latents = [stacked[:, i, :].squeeze(1) for i in range(self.num_modalities)]
        
        return output_latents


class MultimodalDecoder(BaseDecoder):
    """
    Multimodal decoder for tabular, text, and image data.
    
    Architecture:
        Shared Latent -> Modality-Specific Heads -> Cross-Modal Fusion -> Modality Decoders
        
    Supports:
    - Joint generation of all modalities
    - Conditional generation (some modalities given)
    - Cross-modal attention for coherent generation
    
    Example:
        >>> decoder = MultimodalDecoder(
        ...     latent_dim=512,
        ...     tabular_config={...},
        ...     text_config={...},
        ...     image_config={...}
        ... )
        >>> output = decoder(z)
        >>> tabular_out = output["tabular"]
        >>> text_out = output["text"]
        >>> image_out = output["image"]
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        tabular_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        image_config: Optional[Dict[str, Any]] = None,
        use_cross_modal_attention: bool = True,
        fusion_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        name: str = "multimodal_decoder",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multimodal decoder.
        
        Args:
            latent_dim: Shared latent dimension
            tabular_config: Configuration for tabular decoder
            text_config: Configuration for text decoder
            image_config: Configuration for image decoder
            use_cross_modal_attention: Whether to use cross-modal attention
            fusion_layers: Number of fusion layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            name: Decoder name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.use_cross_modal_attention = use_cross_modal_attention
        self.modalities = []
        
        # Modality-specific projection heads
        self.modality_heads = nn.ModuleDict()
        self.modality_decoders = nn.ModuleDict()
        
        # Tabular decoder
        if tabular_config is not None:
            self.modalities.append("tabular")
            self.modality_heads["tabular"] = ModalitySpecificHead(
                latent_dim, 
                tabular_config.get("hidden_dim", latent_dim),
                dropout=dropout
            )
            self.modality_decoders["tabular"] = TabularDecoder(
                latent_dim=tabular_config.get("hidden_dim", latent_dim),
                **{k: v for k, v in tabular_config.items() if k != "hidden_dim"}
            )
            
        # Text decoder
        if text_config is not None:
            self.modalities.append("text")
            self.modality_heads["text"] = ModalitySpecificHead(
                latent_dim,
                text_config.get("hidden_dim", latent_dim),
                dropout=dropout
            )
            self.modality_decoders["text"] = TextDecoder(
                latent_dim=text_config.get("hidden_dim", latent_dim),
                **{k: v for k, v in text_config.items() if k != "hidden_dim"}
            )
            
        # Image decoder
        if image_config is not None:
            self.modalities.append("image")
            self.modality_heads["image"] = ModalitySpecificHead(
                latent_dim,
                image_config.get("hidden_dim", latent_dim),
                dropout=dropout
            )
            self.modality_decoders["image"] = ImageDecoder(
                latent_dim=image_config.get("hidden_dim", latent_dim),
                **{k: v for k, v in image_config.items() if k != "hidden_dim"}
            )
            
        # Cross-modal fusion
        if use_cross_modal_attention and len(self.modalities) > 1:
            self.fusion = MultimodalFusionDecoder(
                latent_dim=latent_dim,
                num_modalities=len(self.modalities),
                num_layers=fusion_layers,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.fusion = None
            
    @property
    def output_dim(self) -> Dict[str, Any]:
        """Return output dimensions for each modality."""
        output_dims = {}
        for modality in self.modalities:
            decoder = self.modality_decoders[modality]
            output_dims[modality] = decoder.output_dim
        return output_dims
    
    def decode(
        self,
        z: torch.Tensor,
        modalities: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent to multimodal output.
        
        Args:
            z: Shared latent tensor (batch, latent_dim)
            modalities: Subset of modalities to decode (default: all)
            **kwargs: Modality-specific arguments
            
        Returns:
            Dictionary of modality outputs
        """
        if modalities is None:
            modalities = self.modalities
            
        # Project to modality-specific latents
        modality_latents = {}
        for modality in modalities:
            if modality in self.modality_heads:
                modality_latents[modality] = self.modality_heads[modality](z)
                
        # Apply cross-modal fusion
        if self.fusion is not None and len(modality_latents) > 1:
            latent_list = [modality_latents[m] for m in modalities if m in modality_latents]
            fused_latents = self.fusion(latent_list)
            
            # Map back to modalities
            for i, modality in enumerate([m for m in modalities if m in modality_latents]):
                modality_latents[modality] = fused_latents[i]
                
        # Decode each modality
        outputs = {}
        for modality, latent in modality_latents.items():
            modality_kwargs = kwargs.get(modality, {})
            outputs[modality] = self.modality_decoders[modality](latent, **modality_kwargs)
            
        return outputs
    
    def forward(
        self,
        z: torch.Tensor,
        modalities: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.decode(z, modalities, **kwargs)
    
    def generate(
        self,
        n_samples: int,
        modalities: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate multimodal samples.
        
        Args:
            n_samples: Number of samples to generate
            modalities: Modalities to generate
            device: Device for generation
            **kwargs: Generation arguments
            
        Returns:
            Dictionary of generated samples
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Sample from prior
        z = torch.randn(n_samples, self._latent_dim, device=device)
        
        return self.decode(z, modalities, **kwargs)
    
    def decode_modality(
        self,
        z: torch.Tensor,
        modality: str,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode for a single modality.
        
        Args:
            z: Latent tensor
            modality: Modality name
            **kwargs: Modality-specific arguments
            
        Returns:
            Modality output
        """
        if modality not in self.modality_decoders:
            raise ValueError(f"Unknown modality: {modality}")
            
        # Project to modality-specific latent
        modality_latent = self.modality_heads[modality](z)
        
        # Decode
        return self.modality_decoders[modality](modality_latent, **kwargs)


class MultimodalVAEDecoder(VAEDecoder):
    """
    VAE-style multimodal decoder.
    
    Combines multimodal decoding with VAE generation capabilities.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        tabular_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        image_config: Optional[Dict[str, Any]] = None,
        use_cross_modal_attention: bool = True,
        name: str = "multimodal_vae_decoder",
        **kwargs
    ):
        """
        Initialize multimodal VAE decoder.
        
        Args:
            latent_dim: Latent dimension
            tabular_config: Tabular decoder config
            text_config: Text decoder config
            image_config: Image decoder config
            use_cross_modal_attention: Use cross-modal attention
            name: Decoder name
        """
        # Build multimodal decoder
        self.multimodal_decoder = MultimodalDecoder(
            latent_dim=latent_dim,
            tabular_config=tabular_config,
            text_config=text_config,
            image_config=image_config,
            use_cross_modal_attention=use_cross_modal_attention,
            name=name,
            **kwargs
        )
        
        # Initialize as nn.Module
        nn.Module.__init__(self)
        self._latent_dim = latent_dim
        self.name = name
        
    def _decode(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Decode latent to multimodal output."""
        return self.multimodal_decoder.decode(z, **kwargs)
    
    @property
    def output_dim(self) -> Dict[str, Any]:
        """Return output dimensions."""
        return self.multimodal_decoder.output_dim
    
    def decode(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Decode latent tensor."""
        return self._decode(z, **kwargs)
    
    def generate(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Generate multimodal samples."""
        z = self.sample_prior(n_samples, device)
        return self.decode(z, **kwargs)


class ConditionalMultimodalDecoder(ConditionalDecoder):
    """
    Conditional multimodal decoder for fair synthetic data generation.
    
    Supports conditioning on sensitive attributes for counterfactual
    multimodal generation.
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        num_classes: Optional[int] = None,
        condition_dim: Optional[int] = None,
        tabular_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        image_config: Optional[Dict[str, Any]] = None,
        use_cross_modal_attention: bool = True,
        sensitive_attributes: Optional[List[str]] = None,
        name: str = "conditional_multimodal_decoder",
        **kwargs
    ):
        """
        Initialize conditional multimodal decoder.
        
        Args:
            latent_dim: Latent dimension
            num_classes: Number of classes
            condition_dim: Condition dimension
            tabular_config: Tabular config
            text_config: Text config
            image_config: Image config
            use_cross_modal_attention: Use cross-modal attention
            sensitive_attributes: List of sensitive attributes
            name: Decoder name
        """
        ConditionalDecoder.__init__(
            self,
            name=name,
            latent_dim=latent_dim,
            num_classes=num_classes,
            condition_dim=condition_dim
        )
        
        self.sensitive_attributes = sensitive_attributes or []
        
        # Build multimodal decoder core
        self.decoder_core = MultimodalDecoder(
            latent_dim=latent_dim,
            tabular_config=tabular_config,
            text_config=text_config,
            image_config=image_config,
            use_cross_modal_attention=use_cross_modal_attention,
            name=f"{name}_core",
            **kwargs
        )
        
    def _decode(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Decode combined latent to multimodal output."""
        return self.decoder_core.decode(z, **kwargs)
    
    @property
    def output_dim(self) -> Dict[str, Any]:
        """Return output dimensions."""
        return self.decoder_core.output_dim
    
    def generate_counterfactual(
        self,
        z: torch.Tensor,
        original_sensitive: torch.Tensor,
        target_sensitive: torch.Tensor,
        sensitive_attribute: str,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactual multimodal samples.
        
        Args:
            z: Latent tensor
            original_sensitive: Original sensitive values
            target_sensitive: Target sensitive values
            sensitive_attribute: Sensitive attribute name
            **kwargs: Additional arguments
            
        Returns:
            Counterfactual multimodal output
        """
        return self.decode(z, condition=target_sensitive.float(), **kwargs)


class HierarchicalMultimodalDecoder(BaseDecoder):
    """
    Hierarchical multimodal decoder for multi-scale generation.
    
    Generates different modalities at different hierarchical levels,
    allowing for coarse-to-fine generation.
    """
    
    def __init__(
        self,
        latent_dims: List[int] = [1024, 512, 256],
        modality_at_level: Optional[Dict[int, List[str]]] = None,
        tabular_config: Optional[Dict[str, Any]] = None,
        text_config: Optional[Dict[str, Any]] = None,
        image_config: Optional[Dict[str, Any]] = None,
        name: str = "hierarchical_multimodal_decoder",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hierarchical multimodal decoder.
        
        Args:
            latent_dims: Latent dimensions for each level
            modality_at_level: Mapping of level index to modalities
            tabular_config: Tabular config
            text_config: Text config
            image_config: Image config
            name: Decoder name
            config: Optional config
        """
        super().__init__(name=name, latent_dim=latent_dims[-1], config=config)
        
        self.latent_dims = latent_dims
        self.num_levels = len(latent_dims)
        
        # Default: all modalities at finest level
        if modality_at_level is None:
            modality_at_level = {self.num_levels - 1: ["tabular", "text", "image"]}
        self.modality_at_level = modality_at_level
        
        # Level projections
        self.level_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dims[i], latent_dims[i+1] if i < len(latent_dims)-1 else latent_dims[i]),
                nn.LayerNorm(latent_dims[i+1] if i < len(latent_dims)-1 else latent_dims[i]),
                nn.GELU()
            )
            for i in range(self.num_levels)
        ])
        
        # Modality decoders per level
        self.level_decoders = nn.ModuleList()
        for level in range(self.num_levels):
            level_decoder = nn.ModuleDict()
            
            if level in modality_at_level:
                latent_dim = latent_dims[level]
                
                if "tabular" in modality_at_level[level] and tabular_config:
                    level_decoder["tabular"] = TabularDecoder(
                        latent_dim=latent_dim, **tabular_config
                    )
                if "text" in modality_at_level[level] and text_config:
                    level_decoder["text"] = TextDecoder(
                        latent_dim=latent_dim, **text_config
                    )
                if "image" in modality_at_level[level] and image_config:
                    level_decoder["image"] = ImageDecoder(
                        latent_dim=latent_dim, **image_config
                    )
                    
            self.level_decoders.append(level_decoder)
            
    @property
    def output_dim(self) -> Dict[str, Any]:
        """Return output dimensions."""
        output_dims = {}
        for level, decoders in enumerate(self.level_decoders):
            for modality, decoder in decoders.items():
                output_dims[f"level_{level}_{modality}"] = decoder.output_dim
        return output_dims
    
    def decode(
        self,
        z_list: List[torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Decode hierarchical latents to multimodal output.
        
        Args:
            z_list: List of latent tensors, one per level
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of outputs at each level
        """
        outputs = {}
        
        for level, z in enumerate(z_list):
            # Project latent
            z_proj = self.level_projections[level](z)
            
            # Decode modalities at this level
            for modality, decoder in self.level_decoders[level].items():
                outputs[f"level_{level}_{modality}"] = decoder(z_proj)
                
        return outputs
    
    def forward(
        self,
        z_list: List[torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.decode(z_list, **kwargs)
