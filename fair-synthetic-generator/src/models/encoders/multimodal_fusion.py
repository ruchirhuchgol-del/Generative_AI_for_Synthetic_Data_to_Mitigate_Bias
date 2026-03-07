"""
Multimodal Fusion
=================

Fusion mechanisms for combining representations from multiple modalities.
Supports concatenation, cross-attention, gated, and hierarchical fusion.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Cross-Modal Attention
# ============================================================================

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing different modalities.
    
    Allows each modality to attend to features from other modalities,
    enabling information exchange between modalities.
    
    Architecture:
        Query (modality A) -> Multi-Head Attention over Key/Value (modality B)
        
    Example:
        >>> attention = CrossModalAttention(dim=512, num_heads=8)
        >>> fused = attention(query=text_features, key_value=image_features)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        """
        Initialize cross-modal attention.
        
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            bias: Whether to use bias in projections
        """
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)
        self.out_proj = nn.Linear(dim, dim, bias=bias)
        
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        kv_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-modal attention.
        
        Args:
            query: Query tensor from modality A (batch, seq_q, dim)
            key_value: Key-Value tensor from modality B (batch, seq_kv, dim)
            query_mask: Optional query mask
            kv_mask: Optional key-value mask
            
        Returns:
            Attended tensor (batch, seq_q, dim)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if kv_mask is not None:
            attn = attn.masked_fill(kv_mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        # Residual connection and normalization
        return self.norm(query + self.dropout(out))


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention with separate heads for each modality pair.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_modalities: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_modalities = num_modalities
        
        # Cross-attention for each modality pair
        self.cross_attns = nn.ModuleDict()
        
        for i in range(num_modalities):
            for j in range(num_modalities):
                if i != j:
                    self.cross_attns[f"mod_{i}_{j}"] = CrossModalAttention(
                        dim, num_heads, dropout
                    )
                    
    def forward(
        self,
        modality_features: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Apply cross-attention between all modality pairs.
        
        Args:
            modality_features: List of feature tensors, one per modality
            
        Returns:
            List of updated feature tensors
        """
        outputs = []
        
        for i, feat_i in enumerate(modality_features):
            updated = feat_i
            
            for j, feat_j in enumerate(modality_features):
                if i != j:
                    key = f"mod_{i}_{j}"
                    if key in self.cross_attns:
                        updated = self.cross_attns[key](updated, feat_j)
                        
            outputs.append(updated)
            
        return outputs


# ============================================================================
# Fusion Mechanisms
# ============================================================================

class ConcatenationFusion(nn.Module):
    """
    Simple concatenation fusion with projection.
    
    Concatenates all modality features and projects to unified space.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_modalities: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(latent_dim * num_modalities, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(
        self,
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """Concatenate and project features."""
        concat = torch.cat(features, dim=-1)
        return self.projection(concat)


class GatedFusion(nn.Module):
    """
    Gated fusion with learned modality importance weights.
    
    Computes gates for each modality and combines features accordingly.
    
    Architecture:
        1. Compute gate weights from concatenated features
        2. Weight each modality feature by its gate
        3. Sum weighted features
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_modalities: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.gate_network = nn.Sequential(
            nn.Linear(latent_dim * num_modalities, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, num_modalities),
            nn.Softmax(dim=-1)
        )
        
        self.modality_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.Dropout(dropout)
            )
            for _ in range(num_modalities)
        ])
        
    def forward(
        self,
        features: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply gated fusion.
        
        Args:
            features: List of modality feature tensors
            
        Returns:
            Fused representation
        """
        # Compute gates
        concat = torch.cat(features, dim=-1)
        gates = self.gate_network(concat)
        
        # Weight and sum
        fused = torch.zeros_like(features[0])
        
        for i, (feat, proj) in enumerate(zip(features, self.modality_projections)):
            fused = fused + gates[:, i:i+1] * proj(feat)
            
        return fused


class HierarchicalFusion(nn.Module):
    """
    Hierarchical fusion with modality-specific processing stages.
    
    Architecture:
        1. Modality-specific transformer processing
        2. Cross-modal attention fusion
        3. Final joint representation
    """
    
    def __init__(
        self,
        latent_dim: int,
        modalities: List[str],
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.modalities = modalities
        self.num_layers = num_layers
        
        # Modality-specific transformers
        self.modality_transformers = nn.ModuleDict({
            mod: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=latent_dim * 4,
                    dropout=dropout,
                    batch_first=True
                ),
                num_layers=2
            )
            for mod in modalities
        })
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            MultiHeadCrossAttention(
                latent_dim, num_heads, len(modalities), dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final projection
        self.output_projection = nn.Sequential(
            nn.Linear(latent_dim * len(modalities), latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply hierarchical fusion.
        
        Args:
            modality_features: Dict mapping modality names to features
            
        Returns:
            Fused representation
        """
        # Get features in consistent order
        features = [
            modality_features[mod] for mod in self.modalities
            if mod in modality_features
        ]
        
        # Add sequence dimension if needed
        features = [
            f.unsqueeze(1) if f.dim() == 2 else f
            for f in features
        ]
        
        # Modality-specific processing
        processed = []
        for i, (mod, feat) in enumerate(zip(self.modalities, features)):
            trans_feat = self.modality_transformers[mod](feat)
            processed.append(trans_feat)
            
        # Cross-modal fusion layers
        for cross_layer in self.cross_modal_layers:
            processed = cross_layer(processed)
            
        # Concatenate and project
        concat = torch.cat([f.squeeze(1) for f in processed], dim=-1)
        
        return self.output_projection(concat)


# ============================================================================
# Main Multimodal Fusion Module
# ============================================================================

class MultimodalFusion(nn.Module):
    """
    Main multimodal fusion module supporting multiple fusion strategies.
    
    Strategies:
    - concatenation: Simple concat + projection
    - cross_attention: Cross-modal attention between modalities
    - gated: Learned gating weights
    - hierarchical: Hierarchical processing + cross-attention
    
    Example:
        >>> fusion = MultimodalFusion(
        ...     latent_dim=512,
        ...     modalities=["tabular", "text", "image"],
        ...     fusion_method="cross_attention"
        ... )
        >>> fused = fusion({
        ...     "tabular": z_tab,
        ...     "text": z_text,
        ...     "image": z_img
        ... })
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        modalities: List[str] = ["tabular", "text", "image"],
        fusion_method: str = "cross_attention",
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_positional_encoding: bool = True
    ):
        """
        Initialize multimodal fusion.
        
        Args:
            latent_dim: Dimension of latent space
            modalities: List of modality names
            fusion_method: Fusion strategy
            num_layers: Number of fusion layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_positional_encoding: Add modality positional encoding
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.modalities = modalities
        self.fusion_method = fusion_method
        self.num_layers = num_layers
        
        # Modality embeddings (learnable)
        self.modality_embeddings = nn.ParameterDict({
            mod: nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)
            for mod in modalities
        })
        
        # Build fusion based on method
        if fusion_method == "concatenation":
            self.fusion = ConcatenationFusion(latent_dim, len(modalities), dropout)
            
        elif fusion_method == "cross_attention":
            self.fusion = self._build_cross_attention_fusion(
                latent_dim, num_layers, num_heads, dropout
            )
            
        elif fusion_method == "gated":
            self.fusion = GatedFusion(latent_dim, len(modalities), dropout)
            
        elif fusion_method == "hierarchical":
            self.fusion = HierarchicalFusion(
                latent_dim, modalities, num_layers, num_heads, dropout
            )
            
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
            
        # Output normalization
        self.output_norm = nn.LayerNorm(latent_dim)
        
    def _build_cross_attention_fusion(
        self,
        latent_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float
    ) -> nn.ModuleList:
        """Build cross-attention fusion layers."""
        layers = nn.ModuleList()
        
        for _ in range(num_layers):
            layers.append(
                MultiHeadCrossAttention(
                    latent_dim, num_heads, len(self.modalities), dropout
                )
            )
            
        return layers
        
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Fuse multimodal features.
        
        Args:
            modality_features: Dict mapping modality names to feature tensors
                              Each tensor: (batch, latent_dim)
            
        Returns:
            Fused representation (batch, latent_dim)
        """
        # Get available modalities
        available = [m for m in self.modalities if m in modality_features]
        
        if not available:
            raise ValueError("No modality features provided")
            
        # Add modality embeddings
        features = []
        for mod in available:
            feat = modality_features[mod]
            if feat.dim() == 2:
                feat = feat.unsqueeze(1)
            feat = feat + self.modality_embeddings[mod]
            features.append(feat)
            
        # Apply fusion
        if self.fusion_method == "concatenation":
            # Flatten and concatenate
            flat_features = [f.squeeze(1) for f in features]
            fused = self.fusion(flat_features)
            
        elif self.fusion_method == "cross_attention":
            # Apply cross-attention layers
            for layer in self.fusion:
                features = layer(features)
            # Mean pool
            fused = torch.cat(features, dim=1).mean(dim=1)
            
        elif self.fusion_method == "gated":
            flat_features = [f.squeeze(1) for f in features]
            fused = self.fusion(flat_features)
            
        elif self.fusion_method == "hierarchical":
            fused = self.fusion(modality_features)
            
        return self.output_norm(fused)
    
    def get_modality_importance(
        self,
        modality_features: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute importance of each modality.
        
        Returns attention-like importance weights for each modality.
        
        Args:
            modality_features: Dict of modality features
            
        Returns:
            Dict mapping modality names to importance scores
        """
        available = [m for m in self.modalities if m in modality_features]
        
        if self.fusion_method == "gated":
            # Use gates as importance
            features = [modality_features[m] for m in available]
            concat = torch.cat(features, dim=-1)
            gates = self.fusion.gate_network(concat)
            
            return {
                mod: gates[:, i].mean().item()
                for i, mod in enumerate(available)
            }
        else:
            # Compute norm-based importance
            importance = {}
            for mod in available:
                feat = modality_features[mod]
                importance[mod] = feat.norm(dim=-1).mean().item()
                
            # Normalize
            total = sum(importance.values())
            return {k: v / total for k, v in importance.items()}


# ============================================================================
# Modality-Specific Processing
# ============================================================================

class ModalityEncoder(nn.Module):
    """
    Wrapper for encoding single modality with modality-specific processing.
    """
    
    def __init__(
        self,
        latent_dim: int,
        modality: str,
        encoder: nn.Module
    ):
        super().__init__()
        
        self.modality = modality
        self.encoder = encoder
        
        # Modality embedding
        self.modality_embed = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode and add modality embedding."""
        z = self.encoder(x)
        return z + self.modality_embed


# ============================================================================
# Utility Functions
# ============================================================================

def create_fusion_module(
    latent_dim: int,
    modalities: List[str],
    fusion_method: str = "cross_attention",
    **kwargs
) -> Union[MultimodalFusion, HierarchicalFusion]:
    """
    Factory function to create fusion module.
    
    Args:
        latent_dim: Latent dimension
        modalities: List of modality names
        fusion_method: Fusion strategy
        **kwargs: Additional arguments
        
    Returns:
        Fusion module
    """
    return MultimodalFusion(
        latent_dim=latent_dim,
        modalities=modalities,
        fusion_method=fusion_method,
        **kwargs
    )
