"""
Text Encoder
============

Encoder for text data using transformer architectures.
Supports both PyTorch and TensorFlow implementations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders.base_encoder import (
    BaseEncoder,
    VAEEncoder,
    get_activation
)


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence position information.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to embeddings.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding as learnable embeddings.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add learned positional embeddings.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional embeddings added
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        pos_emb = self.pos_embedding(positions)
        return self.dropout(x + pos_emb)


# ============================================================================
# Transformer Components
# ============================================================================

class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer with self-attention and feedforward.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-6
    ):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        # Activation
        self.activation = get_activation(activation)
        
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through encoder layer.
        
        Args:
            src: Input tensor (batch, seq, d_model)
            src_mask: Attention mask
            src_key_padding_mask: Padding mask
            is_causal: Whether to use causal attention
            
        Returns:
            Output tensor
        """
        # Self-attention block
        src2, _ = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            is_causal=is_causal
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward block
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        
        return src


# ============================================================================
# PyTorch Text Encoder
# ============================================================================

class TextEncoder(BaseEncoder):
    """
    Transformer-based encoder for text data in PyTorch.
    
    Architecture:
        Token Embedding -> Positional Encoding -> Transformer Layers -> Pooling -> Latent
        
    Supports:
    - From-scratch training
    - Pre-trained models (via Hugging Face)
    - Multiple pooling strategies
    - Various positional encoding schemes
    
    Example:
        >>> encoder = TextEncoder(
        ...     vocab_size=50000,
        ...     hidden_dim=768,
        ...     num_layers=6,
        ...     latent_dim=512
        ... )
        >>> z = encoder(input_ids, attention_mask=mask)
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 256,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_dim: int = 2048,
        latent_dim: int = 512,
        activation: str = "gelu",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        pooling_strategy: str = "cls",  # cls, mean, max, attention
        positional_encoding: str = "sinusoidal",  # sinusoidal, learned
        pretrained_model: Optional[str] = None,
        freeze_pretrained: bool = False,
        name: str = "text_encoder",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the text encoder.
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            hidden_dim: Hidden dimension (d_model)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            intermediate_dim: Feedforward intermediate dimension
            latent_dim: Output latent dimension
            activation: Activation function
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            pooling_strategy: Pooling strategy
            positional_encoding: Positional encoding type
            pretrained_model: Optional pre-trained model name
            freeze_pretrained: Whether to freeze pre-trained weights
            name: Encoder name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.pooling_strategy = pooling_strategy
        self.pretrained_model = pretrained_model
        
        if pretrained_model is not None:
            self._init_pretrained(pretrained_model, freeze_pretrained)
        else:
            self._init_from_scratch(
                vocab_size, hidden_dim, num_layers, num_heads,
                intermediate_dim, activation, dropout, attention_dropout,
                positional_encoding
            )
        
        # Pooling
        if pooling_strategy == "attention":
            self.attention_pooling = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1, bias=False)
            )
        else:
            self.attention_pooling = None
            
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    def _init_from_scratch(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        intermediate_dim: int,
        activation: str,
        dropout: float,
        attention_dropout: float,
        positional_encoding: str
    ) -> None:
        """Initialize model from scratch."""
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        if positional_encoding == "sinusoidal":
            self.pos_encoding = PositionalEncoding(hidden_dim, self.max_length, dropout)
        else:
            self.pos_encoding = LearnedPositionalEncoding(hidden_dim, self.max_length, dropout)
            
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim, num_heads, intermediate_dim,
                dropout, activation
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        self._is_pretrained = False
        
    def _init_pretrained(self, model_name: str, freeze: bool) -> None:
        """Initialize from pre-trained model."""
        try:
            from transformers import AutoModel, AutoConfig
            
            self.pretrained = AutoModel.from_pretrained(model_name)
            self.config_pretrained = AutoConfig.from_pretrained(model_name)
            self.hidden_dim = self.config_pretrained.hidden_size
            
            if freeze:
                for param in self.pretrained.parameters():
                    param.requires_grad = False
                    
            self._is_pretrained = True
            
        except ImportError:
            raise ImportError(
                "transformers library required for pre-trained models. "
                "Install with: pip install transformers"
            )
            
    @property
    def input_dim(self) -> int:
        """Return the input dimension (max sequence length)."""
        return self.max_length
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode text to latent representation.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            token_type_ids: Token type IDs for BERT-like models
            
        Returns:
            Latent tensor (batch, latent_dim)
        """
        if self._is_pretrained:
            hidden_states = self._encode_pretrained(
                input_ids, attention_mask, token_type_ids
            )
        else:
            hidden_states = self._encode_from_scratch(input_ids, attention_mask)
            
        # Pool
        pooled = self._pool(hidden_states, attention_mask)
        
        # Project to latent
        z = self.output_projection(pooled)
        
        return z
    
    def _encode_pretrained(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        token_type_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Encode using pre-trained model."""
        outputs = self.pretrained(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        return outputs.last_hidden_state
    
    def _encode_from_scratch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Encode using custom transformer."""
        # Token embedding
        x = self.token_embedding(input_ids)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Create padding mask for transformer
        src_key_padding_mask = None
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
            
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
            
        # Final normalization
        return self.final_norm(x)
    
    def _pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Pool hidden states to single vector."""
        if self.pooling_strategy == "cls":
            return hidden_states[:, 0]
            
        elif self.pooling_strategy == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return hidden_states.mean(dim=1)
            
        elif self.pooling_strategy == "max":
            if attention_mask is not None:
                hidden_states = hidden_states.masked_fill(
                    attention_mask.unsqueeze(-1) == 0,
                    float("-inf")
                )
            return hidden_states.max(dim=1)[0]
            
        elif self.pooling_strategy == "attention":
            # Attention-weighted pooling
            attn_weights = self.attention_pooling(hidden_states).squeeze(-1)
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
            attn_weights = F.softmax(attn_weights, dim=1)
            return (hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)
            
        else:
            return hidden_states[:, 0]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass."""
        return self.encode(input_ids, attention_mask, **kwargs)


class TextVAEEncoder(VAEEncoder):
    """
    VAE-style text encoder with reparameterization.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 256,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        latent_dim: int = 512,
        name: str = "text_vae_encoder",
        **kwargs
    ):
        # Initialize as text encoder
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            max_length=max_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            latent_dim=hidden_dim,  # Project to hidden_dim first
            name=name,
            **kwargs
        )
        
        # Initialize VAE components
        nn.Module.__init__(self)
        self._latent_dim = latent_dim
        self.min_logvar = -10.0
        self.max_logvar = 10.0
        self.hidden_dim = hidden_dim
        
        # VAE heads
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def _encode_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Encode to intermediate features."""
        return self.text_encoder(input_ids, attention_mask)
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters."""
        h = self._encode_features(input_ids, attention_mask)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        sample: bool = True,
        return_params: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass."""
        mu, logvar = self.encode(input_ids, attention_mask)
        
        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
            
        if return_params:
            return z, mu, logvar
        return z


# ============================================================================
# TensorFlow Text Encoder
# ============================================================================

class TensorFlowTextEncoder:
    """
    TensorFlow/Keras implementation of text encoder.
    
    Provides the same interface as PyTorch version but uses
    TensorFlow for the backend.
    
    Note: Requires TensorFlow to be installed.
    
    Example:
        >>> encoder = TensorFlowTextEncoder(
        ...     vocab_size=50000,
        ...     hidden_dim=768,
        ...     latent_dim=512
        ... )
        >>> z = encoder(input_ids, attention_mask=mask)
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 256,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        intermediate_dim: int = 2048,
        latent_dim: int = 512,
        dropout: float = 0.1,
        pretrained_model: Optional[str] = None,
        name: str = "tf_text_encoder"
    ):
        """
        Initialize TensorFlow text encoder.
        
        Args:
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            intermediate_dim: Feedforward dimension
            latent_dim: Output latent dimension
            dropout: Dropout rate
            pretrained_model: Optional pre-trained model
            name: Encoder name
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.name = name
        
        # Check TensorFlow availability
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for TensorFlowTextEncoder. "
                "Install with: pip install tensorflow"
            )
            
        # Build model
        if pretrained_model is not None:
            self._build_pretrained(pretrained_model, latent_dim)
        else:
            self._build_from_scratch(
                vocab_size, hidden_dim, num_layers, num_heads,
                intermediate_dim, latent_dim, dropout
            )
            
    def _build_from_scratch(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        intermediate_dim: int,
        latent_dim: int,
        dropout: float
    ) -> None:
        """Build transformer encoder from scratch."""
        tf = self.tf
        
        # Input layers
        input_ids = tf.keras.layers.Input(
            shape=(None,), 
            dtype=tf.int32, 
            name="input_ids"
        )
        attention_mask = tf.keras.layers.Input(
            shape=(None,), 
            dtype=tf.float32, 
            name="attention_mask"
        )
        
        # Token embedding
        embedding = tf.keras.layers.Embedding(
            vocab_size, 
            hidden_dim,
            name="token_embedding"
        )(input_ids)
        
        # Positional encoding
        positions = tf.range(start=0, limit=tf.shape(input_ids)[1], delta=1)
        position_embedding = tf.keras.layers.Embedding(
            self.max_length, 
            hidden_dim,
            name="position_embedding"
        )(positions)
        x = embedding + position_embedding
        
        # Transformer encoder layers
        for i in range(num_layers):
            x = self._transformer_encoder_layer(
                x, hidden_dim, num_heads, intermediate_dim, dropout, f"layer_{i}"
            )
            
        # Pooling (use [CLS] token or mean)
        cls_token = x[:, 0]
        
        # Output projection
        output = tf.keras.layers.Dense(
            latent_dim,
            name="output_projection"
        )(cls_token)
        output = tf.keras.layers.LayerNormalization(name="output_norm")(output)
        
        # Create model
        self.model = tf.keras.Model(
            inputs=[input_ids, attention_mask],
            outputs=output,
            name=self.name
        )
        
    def _transformer_encoder_layer(
        self,
        x,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        dropout: float,
        name: str
    ):
        """Create a single transformer encoder layer."""
        tf = self.tf
        
        # Multi-head attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout,
            name=f"{name}_attention"
        )(x, x)
        
        # Add & normalize
        x = tf.keras.layers.Add(name=f"{name}_add_1")([x, attn_output])
        x = tf.keras.layers.LayerNormalization(name=f"{name}_norm_1")(x)
        
        # Feedforward
        ff = tf.keras.layers.Dense(
            intermediate_dim, 
            activation="gelu",
            name=f"{name}_ff_1"
        )(x)
        ff = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")(ff)
        ff = tf.keras.layers.Dense(hidden_dim, name=f"{name}_ff_2")(ff)
        
        # Add & normalize
        x = tf.keras.layers.Add(name=f"{name}_add_2")([x, ff])
        x = tf.keras.layers.LayerNormalization(name=f"{name}_norm_2")(x)
        
        return x
    
    def _build_pretrained(self, model_name: str, latent_dim: int) -> None:
        """Build from pre-trained model."""
        try:
            from transformers import TFAutoModel
            
            self.pretrained = TFAutoModel.from_pretrained(model_name)
            
            # Create wrapper model
            input_ids = self.tf.keras.Input(shape=(None,), dtype=tf.int32)
            attention_mask = self.tf.keras.Input(shape=(None,), dtype=tf.int32)
            
            outputs = self.pretrained(input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]
            
            output = self.tf.keras.layers.Dense(latent_dim)(pooled)
            
            self.model = self.tf.keras.Model(
                inputs=[input_ids, attention_mask],
                outputs=output
            )
            
        except ImportError:
            raise ImportError(
                "transformers library required for pre-trained models"
            )
            
    @property
    def input_dim(self) -> int:
        """Return input dimension."""
        return self.max_length
    
    def encode(
        self,
        input_ids,
        attention_mask=None,
        **kwargs
    ):
        """
        Encode text to latent representation.
        
        Args:
            input_ids: Token IDs (numpy array or tf.Tensor)
            attention_mask: Attention mask
            
        Returns:
            Latent tensor
        """
        import numpy as np
        
        # Convert to tensors if needed
        if isinstance(input_ids, np.ndarray):
            input_ids = self.tf.convert_to_tensor(input_ids)
        if attention_mask is not None and isinstance(attention_mask, np.ndarray):
            attention_mask = self.tf.convert_to_tensor(attention_mask)
            
        # Run model
        return self.model([input_ids, attention_mask])
    
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        **kwargs
    ):
        """Call encode method."""
        return self.encode(input_ids, attention_mask, **kwargs)
    
    def save(self, path: str) -> None:
        """Save model to path."""
        self.model.save(path)
        
    def load(self, path: str) -> None:
        """Load model from path."""
        self.model = self.tf.keras.models.load_model(path)
