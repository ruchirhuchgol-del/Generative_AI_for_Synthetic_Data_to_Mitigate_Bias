"""
Tabular Decoder
===============

Decoder for tabular data with mixed numerical and categorical features.
Supports MLP and Transformer-based architectures in PyTorch.
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
    get_activation,
    get_normalization
)
from src.data.schemas.data_schema import TabularSchema, FeatureType


class TabularDecoder(BaseDecoder):
    """
    Decoder for tabular data with mixed feature types.
    
    Reconstructs tabular data from latent representations:
    - Numerical features: Decoded through MLP with appropriate activation
    - Categorical features: Decoded through separate classification heads
    
    Architecture:
        Latent -> Fusion Layers -> Split into numerical/categorical heads
        
    Example:
        >>> decoder = TabularDecoder(
        ...     num_numerical=10,
        ...     categorical_cardinalities={"cat1": 5, "cat2": 3},
        ...     latent_dim=512
        ... )
        >>> output = decoder(z)
        >>> numerical_out = output["numerical"]
        >>> categorical_out = output["categorical"]
    """
    
    def __init__(
        self,
        schema: Optional[TabularSchema] = None,
        num_numerical: Optional[int] = None,
        categorical_cardinalities: Optional[Dict[str, int]] = None,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 512,
        embedding_dim: int = 32,
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = True,
        residual_connections: bool = True,
        numerical_activation: str = "none",  # none, sigmoid, tanh
        name: str = "tabular_decoder",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the tabular decoder.
        
        Args:
            schema: Tabular data schema (alternative to num_numerical/cardinalities)
            num_numerical: Number of numerical features to decode
            categorical_cardinalities: Dict mapping categorical names to cardinalities
            hidden_dims: Hidden layer dimensions
            latent_dim: Input latent dimension
            embedding_dim: Default embedding dimension (not used in decoder)
            activation: Activation function name
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            residual_connections: Whether to use residual connections
            numerical_activation: Activation for numerical outputs
            name: Decoder name
            config: Optional configuration dictionary
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        # Extract schema information
        if schema is not None:
            self.schema = schema
            self._num_numerical = len(schema.numerical_features)
            self._categorical_cardinalities = {
                f.name: len(f.categories) if f.categories else 2
                for f in schema.categorical_features
            }
            self._categorical_names = [f.name for f in schema.categorical_features]
        else:
            self.schema = None
            self._num_numerical = num_numerical or 0
            self._categorical_cardinalities = categorical_cardinalities or {}
            self._categorical_names = list(self._categorical_cardinalities.keys())
        
        # Store configuration
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.residual_connections = residual_connections
        self.numerical_activation = numerical_activation
        
        # Build layers
        self._build_layers(hidden_dims, activation, dropout)
        
    def _build_layers(
        self, 
        hidden_dims: List[int], 
        activation: str,
        dropout: float
    ) -> None:
        """Build decoder layers."""
        # Build MLP layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        prev_dim = self._latent_dim
        
        for i, dim in enumerate(hidden_dims):
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, dim))
            
            # Normalization
            if self.use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(dim))
            elif self.use_batch_norm:
                self.layer_norms.append(nn.BatchNorm1d(dim))
            else:
                self.layer_norms.append(nn.Identity())
            
            prev_dim = dim
        
        # Activation
        self.activation = get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Residual projections
        if self.residual_connections and len(hidden_dims) > 1:
            self.residual_projections = nn.ModuleList()
            for i in range(len(hidden_dims) - 1):
                if hidden_dims[i] != hidden_dims[i + 1]:
                    self.residual_projections.append(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                    )
                else:
                    self.residual_projections.append(None)
        else:
            self.residual_projections = None
        
        # Output heads
        self._build_output_heads(hidden_dims[-1] if hidden_dims else self._latent_dim)
    
    def _build_output_heads(self, feature_dim: int) -> None:
        """Build numerical and categorical output heads."""
        # Numerical output head
        if self._num_numerical > 0:
            self.numerical_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.LayerNorm(feature_dim // 2),
                get_activation("leaky_relu"),
                nn.Dropout(self.dropout_rate),
                nn.Linear(feature_dim // 2, self._num_numerical)
            )
        else:
            self.numerical_head = None
            
        # Categorical output heads (one per categorical feature)
        self.categorical_heads = nn.ModuleDict()
        for name, cardinality in self._categorical_cardinalities.items():
            self.categorical_heads[name] = nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.LayerNorm(feature_dim // 2),
                get_activation("leaky_relu"),
                nn.Dropout(self.dropout_rate),
                nn.Linear(feature_dim // 2, cardinality)
            )
    
    @property
    def output_dim(self) -> Dict[str, Any]:
        """Return the output dimensions."""
        return {
            "numerical": self._num_numerical,
            "categorical": self._categorical_cardinalities
        }
    
    def decode(
        self, 
        z: torch.Tensor,
        return_logits: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent tensor to tabular output.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            return_logits: If True, return logits for categorical features
            
        Returns:
            Dictionary containing:
                - "numerical": Numerical feature tensor (batch, num_numerical)
                - "categorical": Dict of categorical logits/probs
        """
        x = z
        
        # Pass through MLP layers
        for i, layer in enumerate(self.layers):
            # Linear transform
            h = layer(x)
            
            # Normalization
            h = self.layer_norms[i](h)
            
            # Activation
            h = self.activation(h)
            
            # Dropout
            h = self.dropout(h)
            
            # Residual connection
            if self.residual_connections and i < len(self.layers) - 1:
                if self.residual_projections and self.residual_projections[i] is not None:
                    x = h + self.residual_projections[i](x)
                else:
                    x = h + x
            else:
                x = h
        
        # Decode numerical features
        output = {}
        
        if self.numerical_head is not None:
            numerical_out = self.numerical_head(x)
            
            # Apply activation if specified
            if self.numerical_activation == "sigmoid":
                numerical_out = torch.sigmoid(numerical_out)
            elif self.numerical_activation == "tanh":
                numerical_out = torch.tanh(numerical_out)
                
            output["numerical"] = numerical_out
        
        # Decode categorical features
        categorical_output = {}
        for name, head in self.categorical_heads.items():
            logits = head(x)
            if return_logits:
                categorical_output[name] = logits
            else:
                categorical_output[name] = F.softmax(logits, dim=-1)
                
        output["categorical"] = categorical_output
        
        return output
    
    def forward(
        self, 
        z: torch.Tensor,
        return_logits: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.decode(z, return_logits, **kwargs)
    
    def sample(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Sample from the decoder output distribution.
        
        Args:
            z: Latent tensor
            temperature: Sampling temperature for categorical features
            
        Returns:
            Dictionary of sampled features
        """
        output = self.decode(z, return_logits=True, **kwargs)
        
        samples = {}
        
        # Numerical features (deterministic)
        if "numerical" in output:
            samples["numerical"] = output["numerical"]
        
        # Categorical features (sample from softmax)
        for name, logits in output["categorical"].items():
            if temperature != 1.0:
                logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            samples[name] = torch.multinomial(probs, 1).squeeze(-1)
            
        return samples


class TabularVAEDecoder(VAEDecoder):
    """
    VAE-style decoder for tabular data.
    
    Combines tabular decoding with VAE generation capabilities.
    """
    
    def __init__(
        self,
        schema: Optional[TabularSchema] = None,
        num_numerical: Optional[int] = None,
        categorical_cardinalities: Optional[Dict[str, int]] = None,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 512,
        name: str = "tabular_vae_decoder",
        **kwargs
    ):
        """
        Initialize the tabular VAE decoder.
        
        Args:
            schema: Tabular data schema
            num_numerical: Number of numerical features
            categorical_cardinalities: Categorical cardinalities dict
            hidden_dims: Hidden layer dimensions
            latent_dim: Latent dimension
            name: Decoder name
        """
        # Build tabular decoder
        self.tabular_decoder = TabularDecoder(
            schema=schema,
            num_numerical=num_numerical,
            categorical_cardinalities=categorical_cardinalities,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            name=name,
            **{k: v for k, v in kwargs.items() if k not in ['latent_dim', 'name']}
        )
        
        # Initialize as nn.Module
        nn.Module.__init__(self)
        self._latent_dim = latent_dim
        self.name = name
        
    def _decode(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Decode latent to tabular output."""
        return self.tabular_decoder.decode(z, **kwargs)
    
    @property
    def output_dim(self) -> Dict[str, Any]:
        """Return output dimensions."""
        return self.tabular_decoder.output_dim
    
    def decode(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Decode latent tensor."""
        return self._decode(z, **kwargs)


class TabularConditionalDecoder(ConditionalDecoder):
    """
    Conditional tabular decoder for fair synthetic data generation.
    
    Supports conditioning on:
    - Sensitive attributes for counterfactual generation
    - Class labels for conditional generation
    """
    
    def __init__(
        self,
        schema: Optional[TabularSchema] = None,
        num_numerical: Optional[int] = None,
        categorical_cardinalities: Optional[Dict[str, int]] = None,
        hidden_dims: List[int] = [512, 256],
        latent_dim: int = 512,
        num_classes: Optional[int] = None,
        condition_dim: Optional[int] = None,
        sensitive_attributes: Optional[List[str]] = None,
        name: str = "tabular_conditional_decoder",
        **kwargs
    ):
        """
        Initialize conditional tabular decoder.
        
        Args:
            schema: Tabular schema
            num_numerical: Number of numerical features
            categorical_cardinalities: Categorical cardinalities
            hidden_dims: Hidden dimensions
            latent_dim: Latent dimension
            num_classes: Number of classes for conditional generation
            condition_dim: Dimension of continuous condition
            sensitive_attributes: List of sensitive attribute names
            name: Decoder name
        """
        # Initialize conditional decoder
        ConditionalDecoder.__init__(
            self,
            name=name,
            latent_dim=latent_dim,
            num_classes=num_classes,
            condition_dim=condition_dim
        )
        
        self.sensitive_attributes = sensitive_attributes or []
        
        # Build tabular decoder core
        self.decoder_core = TabularDecoder(
            schema=schema,
            num_numerical=num_numerical,
            categorical_cardinalities=categorical_cardinalities,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            name=f"{name}_core",
            **kwargs
        )
    
    def _decode(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Decode combined latent to tabular output."""
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
        Generate counterfactual samples by changing sensitive attribute.
        
        Args:
            z: Latent tensor
            original_sensitive: Original sensitive attribute values
            target_sensitive: Target sensitive attribute values
            sensitive_attribute: Name of the sensitive attribute
            **kwargs: Additional arguments
            
        Returns:
            Counterfactual output
        """
        # This is a simplified counterfactual generation
        # In practice, you would manipulate the latent to achieve counterfactual
        return self.decode(z, condition=target_sensitive.float(), **kwargs)


class TabularTransformerDecoder(BaseDecoder):
    """
    Transformer-based decoder for tabular data.
    
    Uses column-wise attention to generate tabular features,
    allowing for complex feature interactions.
    
    Architecture:
        Latent -> Expand to columns -> Transformer Decoder -> Column outputs
    """
    
    def __init__(
        self,
        schema: Optional[TabularSchema] = None,
        num_numerical: Optional[int] = None,
        categorical_cardinalities: Optional[Dict[str, int]] = None,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 256,
        latent_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        name: str = "tabular_transformer_decoder",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize transformer decoder.
        
        Args:
            schema: Tabular schema
            num_numerical: Number of numerical features
            categorical_cardinalities: Categorical cardinalities
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            latent_dim: Latent dimension
            dropout: Dropout rate
            activation: Activation function
            name: Decoder name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.d_model = d_model
        
        # Parse schema
        if schema is not None:
            self._num_numerical = len(schema.numerical_features)
            self._categorical_cardinalities = {
                f.name: len(f.categories) if f.categories else 2
                for f in schema.categorical_features
            }
        else:
            self._num_numerical = num_numerical or 0
            self._categorical_cardinalities = categorical_cardinalities or {}
            
        self._total_columns = self._num_numerical + len(self._categorical_cardinalities)
        
        # Latent to sequence projection
        self.latent_to_seq = nn.Sequential(
            nn.Linear(latent_dim, self._total_columns * d_model),
            nn.LayerNorm(self._total_columns * d_model)
        )
        
        # Column positional encoding
        self.column_positional = nn.Parameter(
            torch.randn(1, self._total_columns, d_model) * 0.02
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Memory for cross-attention (learnable)
        self.memory = nn.Parameter(torch.randn(1, 8, d_model) * 0.02)
        
        # Column output heads
        self.column_heads = nn.ModuleList()
        
        # Numerical heads
        for i in range(self._num_numerical):
            self.column_heads.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, 1)
                )
            )
            
        # Categorical heads
        for name, cardinality in self._categorical_cardinalities.items():
            self.column_heads.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, cardinality)
                )
            )
    
    @property
    def output_dim(self) -> Dict[str, Any]:
        """Return output dimensions."""
        return {
            "numerical": self._num_numerical,
            "categorical": self._categorical_cardinalities
        }
    
    def decode(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Decode using transformer architecture.
        
        Args:
            z: Latent tensor (batch, latent_dim)
            
        Returns:
            Dictionary of outputs
        """
        batch_size = z.shape[0]
        
        # Project latent to sequence
        seq = self.latent_to_seq(z)
        seq = seq.reshape(batch_size, self._total_columns, self.d_model)
        
        # Add positional encoding
        seq = seq + self.column_positional
        
        # Expand memory for batch
        memory = self.memory.expand(batch_size, -1, -1)
        
        # Transformer decoder
        seq = self.transformer(seq, memory)
        
        # Column outputs
        output = {"numerical": None, "categorical": {}}
        
        col_idx = 0
        
        # Numerical outputs
        if self._num_numerical > 0:
            numerical_outs = []
            for i in range(self._num_numerical):
                col_out = self.column_heads[col_idx](seq[:, i])
                numerical_outs.append(col_out)
                col_idx += 1
            output["numerical"] = torch.cat(numerical_outs, dim=-1)
        
        # Categorical outputs
        for name in self._categorical_cardinalities.keys():
            logits = self.column_heads[col_idx](seq[:, col_idx])
            output["categorical"][name] = logits
            col_idx += 1
            
        return output
    
    def forward(self, z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        return self.decode(z, **kwargs)
