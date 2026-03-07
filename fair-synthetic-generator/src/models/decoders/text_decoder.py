"""
Text Decoder
============

Decoder for text data using transformer architectures.
Supports both PyTorch and TensorFlow implementations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoders.base_decoder import (
    BaseDecoder,
    VAEDecoder,
    ConditionalDecoder,
    get_activation
)


# ============================================================================
# Positional Encoding
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence generation.
    
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

class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with self-attention, cross-attention, and feedforward.
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
        
        # Self-attention (masked)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Cross-attention to latent
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        
        # Activation
        self.activation = get_activation(activation)
        
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through decoder layer.
        
        Args:
            tgt: Target sequence (batch, tgt_len, d_model)
            memory: Memory from encoder/latent (batch, src_len, d_model)
            tgt_mask: Target mask for self-attention
            memory_mask: Memory mask for cross-attention
            tgt_key_padding_mask: Target padding mask
            memory_key_padding_mask: Memory padding mask
            is_causal: Whether to use causal attention
            
        Returns:
            Output tensor
        """
        # Self-attention block (masked/causal)
        tgt2, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            is_causal=is_causal
        )
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention block
        tgt2, _ = self.cross_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # Feedforward block
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


# ============================================================================
# PyTorch Text Decoder
# ============================================================================

class TextDecoder(BaseDecoder):
    """
    Transformer-based decoder for text data in PyTorch.
    
    Architecture:
        Latent -> Memory Projection -> Transformer Decoder -> Token Prediction
        
    Supports:
    - Autoregressive generation
    - Teacher forcing for training
    - Various decoding strategies (greedy, beam, sampling)
    
    Example:
        >>> decoder = TextDecoder(
        ...     vocab_size=50000,
        ...     hidden_dim=768,
        ...     num_layers=6,
        ...     latent_dim=512
        ... )
        >>> logits = decoder(z, tgt_ids)  # Training with teacher forcing
        >>> generated = decoder.generate(z, max_length=50)  # Inference
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
        positional_encoding: str = "sinusoidal",
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        name: str = "text_decoder",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the text decoder.
        
        Args:
            vocab_size: Size of vocabulary
            max_length: Maximum sequence length
            hidden_dim: Hidden dimension (d_model)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            intermediate_dim: Feedforward intermediate dimension
            latent_dim: Input latent dimension
            activation: Activation function
            dropout: Dropout rate
            attention_dropout: Attention dropout rate
            positional_encoding: Positional encoding type
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            name: Decoder name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        if positional_encoding == "sinusoidal":
            self.pos_encoding = PositionalEncoding(hidden_dim, max_length, dropout)
        else:
            self.pos_encoding = LearnedPositionalEncoding(hidden_dim, max_length, dropout)
        
        # Latent to memory projection
        self.latent_to_memory = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                hidden_dim, num_heads, intermediate_dim,
                dropout, activation
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
    @property
    def output_dim(self) -> int:
        """Return the output dimension (vocab size)."""
        return self.vocab_size
    
    def decode(
        self, 
        z: torch.Tensor,
        tgt_ids: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode latent tensor to text logits.
        
        Args:
            z: Latent tensor (batch, latent_dim)
            tgt_ids: Target token IDs for teacher forcing (batch, tgt_len)
            tgt_mask: Target mask
            tgt_key_padding_mask: Padding mask for target
            
        Returns:
            Logits tensor (batch, tgt_len, vocab_size)
        """
        batch_size = z.shape[0]
        device = z.device
        
        # Project latent to memory
        memory = self.latent_to_memory(z)  # (batch, hidden_dim)
        memory = memory.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # If no target provided, start with BOS token
        if tgt_ids is None:
            tgt_ids = torch.full(
                (batch_size, 1), 
                self.bos_token_id, 
                dtype=torch.long, 
                device=device
            )
        
        # Token embedding
        tgt_emb = self.token_embedding(tgt_ids)
        
        # Positional encoding
        tgt_emb = self.pos_encoding(tgt_emb)
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_len = tgt_ids.shape[1]
            tgt_mask = torch.triu(
                torch.full((tgt_len, tgt_len), float('-inf'), device=device),
                diagonal=1
            )
        
        # Pass through decoder layers
        for layer in self.layers:
            tgt_emb = layer(
                tgt_emb, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
        
        # Final normalization
        tgt_emb = self.final_norm(tgt_emb)
        
        # Output projection
        logits = self.output_projection(tgt_emb)
        
        return logits
    
    def generate(
        self,
        z: torch.Tensor,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            z: Latent tensor (batch, latent_dim)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k filtering (0 to disable)
            top_p: Nucleus sampling threshold (1.0 to disable)
            do_sample: Whether to sample or use greedy decoding
            
        Returns:
            Generated token IDs (batch, seq_len)
        """
        if max_length is None:
            max_length = self.max_length
            
        batch_size = z.shape[0]
        device = z.device
        
        # Start with BOS token
        generated = torch.full(
            (batch_size, 1), 
            self.bos_token_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - 1):
            # Get logits
            logits = self.decode(z, tgt_ids=generated)
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')
            
            # Apply nucleus sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == self.eos_token_id)
            if finished.all():
                break
        
        return generated
    
    def forward(
        self,
        z: torch.Tensor,
        tgt_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass."""
        return self.decode(z, tgt_ids, **kwargs)


class TextVAEDecoder(VAEDecoder):
    """
    VAE-style text decoder for generation.
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        max_length: int = 256,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 8,
        latent_dim: int = 512,
        name: str = "text_vae_decoder",
        **kwargs
    ):
        # Build text decoder
        self.text_decoder = TextDecoder(
            vocab_size=vocab_size,
            max_length=max_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            latent_dim=latent_dim,
            name=name,
            **kwargs
        )
        
        # Initialize as nn.Module
        nn.Module.__init__(self)
        self._latent_dim = latent_dim
        self.name = name
        
    def _decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode latent to text logits."""
        return self.text_decoder.decode(z, **kwargs)
    
    @property
    def output_dim(self) -> int:
        """Return output dimension."""
        return self.text_decoder.output_dim
    
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decode latent tensor."""
        return self._decode(z, **kwargs)
    
    def generate(self, n_samples: int, device=None, **kwargs) -> torch.Tensor:
        """Generate text samples."""
        z = self.sample_prior(n_samples, device)
        return self.text_decoder.generate(z, **kwargs)


# ============================================================================
# TensorFlow Text Decoder
# ============================================================================

class TensorFlowTextDecoder:
    """
    TensorFlow/Keras implementation of text decoder.
    
    Provides the same interface as PyTorch version but uses
    TensorFlow for the backend.
    
    Note: Requires TensorFlow to be installed.
    
    Example:
        >>> decoder = TensorFlowTextDecoder(
        ...     vocab_size=50000,
        ...     hidden_dim=768,
        ...     latent_dim=512
        ... )
        >>> logits = decoder(z, tgt_ids)
        >>> generated = decoder.generate(z, max_length=50)
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
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        name: str = "tf_text_decoder"
    ):
        """
        Initialize TensorFlow text decoder.
        
        Args:
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
            hidden_dim: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            intermediate_dim: Feedforward dimension
            latent_dim: Input latent dimension
            dropout: Dropout rate
            pad_token_id: Padding token ID
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            name: Decoder name
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.name = name
        
        # Check TensorFlow availability
        try:
            import tensorflow as tf
            self.tf = tf
        except ImportError:
            raise ImportError(
                "TensorFlow is required for TensorFlowTextDecoder. "
                "Install with: pip install tensorflow"
            )
            
        # Build model
        self._build_model(
            vocab_size, hidden_dim, num_layers, num_heads,
            intermediate_dim, latent_dim, dropout
        )
        
    def _build_model(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        intermediate_dim: int,
        latent_dim: int,
        dropout: float
    ) -> None:
        """Build transformer decoder model."""
        tf = self.tf
        
        # Input layers
        latent_input = tf.keras.layers.Input(
            shape=(latent_dim,), 
            name="latent_input"
        )
        tgt_ids = tf.keras.layers.Input(
            shape=(None,), 
            dtype=tf.int32, 
            name="target_ids"
        )
        
        # Token embedding
        token_emb = tf.keras.layers.Embedding(
            vocab_size, 
            hidden_dim,
            name="token_embedding"
        )(tgt_ids)
        
        # Positional encoding
        positions = tf.range(start=0, limit=tf.shape(tgt_ids)[1], delta=1)
        pos_emb = tf.keras.layers.Embedding(
            self.max_length, 
            hidden_dim,
            name="position_embedding"
        )(positions)
        x = token_emb + pos_emb
        
        # Latent to memory
        memory = tf.keras.layers.Dense(hidden_dim, name="latent_to_memory")(latent_input)
        memory = tf.keras.layers.LayerNormalization(name="memory_norm")(memory)
        memory = tf.expand_dims(memory, axis=1)  # (batch, 1, hidden_dim)
        
        # Transformer decoder layers
        for i in range(num_layers):
            x = self._transformer_decoder_layer(
                x, memory, hidden_dim, num_heads, intermediate_dim, dropout, f"layer_{i}"
            )
            
        # Output projection
        output = tf.keras.layers.Dense(
            vocab_size,
            name="output_projection"
        )(x)
        
        # Create model
        self.model = tf.keras.Model(
            inputs=[latent_input, tgt_ids],
            outputs=output,
            name=self.name
        )
        
    def _transformer_decoder_layer(
        self,
        x,
        memory,
        hidden_dim: int,
        num_heads: int,
        intermediate_dim: int,
        dropout: float,
        name: str
    ):
        """Create a single transformer decoder layer."""
        tf = self.tf
        
        # Self-attention (causal)
        seq_len = tf.shape(x)[1]
        causal_mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len)), -1, 0
        )
        causal_mask = tf.expand_dims(tf.expand_dims(causal_mask, 0), 0)
        
        self_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout,
            name=f"{name}_self_attention"
        )(x, x, attention_mask=causal_mask)
        
        x = tf.keras.layers.Add(name=f"{name}_add_1")([x, self_attn])
        x = tf.keras.layers.LayerNormalization(name=f"{name}_norm_1")(x)
        
        # Cross-attention
        cross_attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=hidden_dim // num_heads,
            dropout=dropout,
            name=f"{name}_cross_attention"
        )(x, memory)
        
        x = tf.keras.layers.Add(name=f"{name}_add_2")([x, cross_attn])
        x = tf.keras.layers.LayerNormalization(name=f"{name}_norm_2")(x)
        
        # Feedforward
        ff = tf.keras.layers.Dense(
            intermediate_dim, 
            activation="gelu",
            name=f"{name}_ff_1"
        )(x)
        ff = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")(ff)
        ff = tf.keras.layers.Dense(hidden_dim, name=f"{name}_ff_2")(ff)
        
        # Add & normalize
        x = tf.keras.layers.Add(name=f"{name}_add_3")([x, ff])
        x = tf.keras.layers.LayerNormalization(name=f"{name}_norm_3")(x)
        
        return x
    
    @property
    def output_dim(self) -> int:
        """Return output dimension."""
        return self.vocab_size
    
    @property
    def latent_dim(self) -> int:
        """Return latent dimension."""
        return self.latent_dim
    
    def decode(self, z, tgt_ids, **kwargs):
        """
        Decode latent to text logits.
        
        Args:
            z: Latent tensor (numpy array or tf.Tensor)
            tgt_ids: Target token IDs
            
        Returns:
            Logits tensor
        """
        import numpy as np
        
        # Convert to tensors if needed
        if isinstance(z, np.ndarray):
            z = self.tf.convert_to_tensor(z)
        if isinstance(tgt_ids, np.ndarray):
            tgt_ids = self.tf.convert_to_tensor(tgt_ids)
            
        return self.model([z, tgt_ids])
    
    def generate(
        self,
        z,
        max_length: Optional[int] = None,
        temperature: float = 1.0,
        **kwargs
    ):
        """
        Generate text autoregressively.
        
        Args:
            z: Latent tensor
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs
        """
        if max_length is None:
            max_length = self.max_length
            
        tf = self.tf
        batch_size = z.shape[0]
        
        # Start with BOS token
        generated = tf.fill([batch_size, 1], self.bos_token_id)
        
        for _ in range(max_length - 1):
            # Get logits
            logits = self.decode(z, generated)
            
            # Get next token logits
            next_logits = logits[:, -1, :] / temperature
            
            # Sample
            probs = tf.nn.softmax(next_logits, axis=-1)
            next_token = tf.random.categorical(tf.math.log(probs), 1)
            
            # Append
            generated = tf.concat([generated, next_token], axis=1)
            
            # Check for EOS
            if tf.reduce_all(tf.equal(next_token, self.eos_token_id)):
                break
        
        return generated
    
    def __call__(self, z, tgt_ids=None, **kwargs):
        """Call decode or generate based on inputs."""
        if tgt_ids is not None:
            return self.decode(z, tgt_ids, **kwargs)
        return self.generate(z, **kwargs)
    
    def save(self, path: str) -> None:
        """Save model to path."""
        self.model.save(path)
        
    def load(self, path: str) -> None:
        """Load model from path."""
        self.model = self.tf.keras.models.load_model(path)
