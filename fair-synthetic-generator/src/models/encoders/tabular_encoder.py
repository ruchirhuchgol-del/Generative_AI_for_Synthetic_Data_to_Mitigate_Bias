"""
Tabular Encoder
===============

Encoder for tabular data with mixed numerical and categorical features.
Supports MLP, TabNet, and TabTransformer architectures in PyTorch.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders.base_encoder import (
    BaseEncoder, 
    VAEEncoder, 
    get_activation,
    get_normalization
)
from src.data.schemas.data_schema import TabularSchema, FeatureType


class TabularEncoder(BaseEncoder):
    """
    Encoder for tabular data with mixed feature types.
    
    Handles:
    - Numerical features: Normalized and passed through MLP
    - Categorical features: Embedded and concatenated
    - Feature-specific processing
    - Residual connections
    
    Architecture:
        Numerical -> BatchNorm -> MLP
        Categorical -> Embedding -> Concatenate -> Fusion Layers -> Latent
        
    Example:
        >>> encoder = TabularEncoder(
        ...     num_features=10,
        ...     categorical_cardinalities={"cat1": 5, "cat2": 3},
        ...     hidden_dims=[256, 512],
        ...     latent_dim=512
        ... )
        >>> z = encoder(numerical=x_num, categorical={"cat1": c1, "cat2": c2})
    """
    
    def __init__(
        self,
        schema: Optional[TabularSchema] = None,
        num_features: Optional[int] = None,
        categorical_cardinalities: Optional[Dict[str, int]] = None,
        hidden_dims: List[int] = [256, 512],
        latent_dim: int = 512,
        embedding_dim: int = 32,
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = True,
        residual_connections: bool = True,
        name: str = "tabular_encoder",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the tabular encoder.
        
        Args:
            schema: Tabular data schema (alternative to num_features/cardinalities)
            num_features: Number of numerical features
            categorical_cardinalities: Dict mapping categorical names to cardinalities
            hidden_dims: Hidden layer dimensions
            latent_dim: Output latent dimension
            embedding_dim: Default embedding dimension for categorical features
            activation: Activation function name
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            residual_connections: Whether to use residual connections
            name: Encoder name
            config: Optional configuration dictionary
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        # Extract schema information
        if schema is not None:
            self.schema = schema
            self._num_numerical = len(schema.numerical_features)
            self._categorical_info = {
                f.name: (
                    len(f.categories) if f.categories else 2,
                    f.embedding_dim or embedding_dim
                )
                for f in schema.categorical_features
            }
        else:
            self.schema = None
            self._num_numerical = num_features or 0
            # Convert cardinalities to (cardinality, embedding_dim) tuples
            self._categorical_info = {
                name: (card, min(embedding_dim, (card + 1) // 2))
                for name, card in (categorical_cardinalities or {}).items()
            }
        
        # Store configuration
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.residual_connections = residual_connections
        self._embedding_dim = embedding_dim
        
        # Build layers
        self._build_layers(hidden_dims, activation, dropout)
        
    def _build_layers(
        self, 
        hidden_dims: List[int], 
        activation: str,
        dropout: float
    ) -> None:
        """Build encoder layers."""
        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleDict()
        for name, (cardinality, emb_dim) in self._categorical_info.items():
            self.categorical_embeddings[name] = nn.Sequential(
                nn.Embedding(cardinality, emb_dim),
                nn.Dropout(dropout)
            )
        
        # Calculate input dimension
        total_emb_dim = sum(emb_dim for _, emb_dim in self._categorical_info.values())
        self._input_dim = self._num_numerical + total_emb_dim
        
        # Numerical feature normalization
        if self._num_numerical > 0:
            self.numerical_norm = nn.BatchNorm1d(self._num_numerical)
            self.numerical_dropout = nn.Dropout(dropout)
        else:
            self.numerical_norm = None
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        prev_dim = self._input_dim
        
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
        
        # Output projection
        self.output_projection = nn.Linear(prev_dim, self._latent_dim)
        self.output_norm = nn.LayerNorm(self._latent_dim)
        
        # Residual projections (for dimension mismatches)
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
    
    @property
    def input_dim(self) -> int:
        """Return the input dimension."""
        return self._input_dim
    
    def encode(
        self, 
        numerical: Optional[torch.Tensor] = None,
        categorical: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode tabular data to latent representation.
        
        Args:
            numerical: Numerical features tensor of shape (batch_size, num_numerical)
            categorical: Dict of categorical feature tensors
            
        Returns:
            Latent tensor of shape (batch_size, latent_dim)
        """
        features = []
        
        # Process numerical features
        if numerical is not None and self._num_numerical > 0:
            if self.numerical_norm is not None:
                numerical = self.numerical_norm(numerical)
            features.append(numerical)
        
        # Process categorical features
        if categorical is not None:
            for name, values in categorical.items():
                if name in self.categorical_embeddings:
                    embedded = self.categorical_embeddings[name](values)
                    features.append(embedded)
        
        # Concatenate all features
        if not features:
            raise ValueError("No features provided for encoding")
        
        x = torch.cat(features, dim=-1)
        
        # Pass through layers
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
        
        # Output projection
        z = self.output_projection(x)
        z = self.output_norm(z)
        
        return z
    
    def forward(
        self,
        numerical: Optional[torch.Tensor] = None,
        categorical: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass."""
        return self.encode(numerical, categorical, **kwargs)


class TabularVAEEncoder(VAEEncoder):
    """
    VAE-style encoder for tabular data.
    
    Combines tabular encoding with VAE reparameterization for
    variational autoencoder training.
    """
    
    def __init__(
        self,
        schema: Optional[TabularSchema] = None,
        num_features: Optional[int] = None,
        categorical_cardinalities: Optional[Dict[str, int]] = None,
        hidden_dims: List[int] = [256, 512],
        latent_dim: int = 512,
        embedding_dim: int = 32,
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        name: str = "tabular_vae_encoder",
        **kwargs
    ):
        """
        Initialize the tabular VAE encoder.
        
        Args:
            schema: Tabular data schema
            num_features: Number of numerical features
            categorical_cardinalities: Dict of categorical cardinalities
            hidden_dims: Hidden layer dimensions
            latent_dim: Latent dimension
            embedding_dim: Categorical embedding dimension
            activation: Activation function
            dropout: Dropout rate
            name: Encoder name
        """
        # Don't call super().__init__ yet to avoid double initialization
        
        # Extract schema information
        if schema is not None:
            self.schema = schema
            self._num_numerical = len(schema.numerical_features)
            self._categorical_info = {
                f.name: (
                    len(f.categories) if f.categories else 2,
                    f.embedding_dim or embedding_dim
                )
                for f in schema.categorical_features
            }
        else:
            self.schema = None
            self._num_numerical = num_features or 0
            self._categorical_info = {
                name: (card, min(embedding_dim, (card + 1) // 2))
                for name, card in (categorical_cardinalities or {}).items()
            }
        
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self._embedding_dim = embedding_dim
        
        # Initialize BaseModule
        nn.Module.__init__(self)
        
        # Build encoder layers
        self._build_encoder(hidden_dims, activation, dropout)
        
        # Create VAE heads
        self._create_heads(hidden_dims[-1])
        
        # Initialize VAE encoder parameters
        self._latent_dim = latent_dim
        self.min_logvar = -10.0
        self.max_logvar = 10.0
        self.name = name
        self.config = kwargs.get('config')
        
    def _build_encoder(
        self, 
        hidden_dims: List[int], 
        activation: str,
        dropout: float
    ) -> None:
        """Build the encoder network."""
        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleDict()
        for name, (cardinality, emb_dim) in self._categorical_info.items():
            self.categorical_embeddings[name] = nn.Sequential(
                nn.Embedding(cardinality, emb_dim),
                nn.Dropout(dropout)
            )
        
        # Input dimension
        total_emb_dim = sum(emb_dim for _, emb_dim in self._categorical_info.values())
        self._input_dim = self._num_numerical + total_emb_dim
        
        # Numerical normalization
        if self._num_numerical > 0:
            self.numerical_norm = nn.BatchNorm1d(self._num_numerical)
        else:
            self.numerical_norm = None
        
        # Build MLP
        layers = []
        prev_dim = self._input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        self.encoder = nn.Sequential(*layers)
        self.feature_dim = hidden_dims[-1]
        
    def _encode_features(
        self,
        numerical: Optional[torch.Tensor] = None,
        categorical: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Encode to intermediate features."""
        features = []
        
        if numerical is not None and self._num_numerical > 0:
            if self.numerical_norm is not None:
                numerical = self.numerical_norm(numerical)
            features.append(numerical)
            
        if categorical is not None:
            for name, values in categorical.items():
                if name in self.categorical_embeddings:
                    features.append(self.categorical_embeddings[name](values))
                    
        if not features:
            raise ValueError("No features provided")
            
        x = torch.cat(features, dim=-1)
        return self.encoder(x)
    
    def encode(
        self,
        numerical: Optional[torch.Tensor] = None,
        categorical: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters."""
        h = self._encode_features(numerical, categorical)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        return mu, logvar
    
    def forward(
        self,
        numerical: Optional[torch.Tensor] = None,
        categorical: Optional[Dict[str, torch.Tensor]] = None,
        sample: bool = True,
        return_params: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with optional sampling."""
        mu, logvar = self.encode(numerical, categorical)
        
        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
            
        if return_params:
            return z, mu, logvar
        return z


class ColumnTransformer(nn.Module):
    """
    Transformer encoder for column-wise attention in tabular data.
    
    Treats each feature as a token and applies self-attention
    to learn feature interactions.
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply column-wise attention.
        
        Args:
            x: Tensor of shape (batch, num_features, d_model)
            
        Returns:
            Transformed tensor
        """
        return self.transformer(x)


class TabTransformerEncoder(BaseEncoder):
    """
    TabTransformer-style encoder using column-wise attention.
    
    Unlike standard MLP encoders, this learns feature interactions
    through self-attention over feature embeddings.
    
    Architecture:
        1. Embed each feature (numerical projection, categorical embedding)
        2. Add column positional encoding
        3. Apply transformer layers for feature interaction
        4. Pool and project to latent space
        
    Based on: "TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
    """
    
    def __init__(
        self,
        schema: Optional[TabularSchema] = None,
        num_features: Optional[int] = None,
        categorical_cardinalities: Optional[Dict[str, int]] = None,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 256,
        latent_dim: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        name: str = "tabtransformer_encoder",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize TabTransformer encoder.
        
        Args:
            schema: Tabular schema
            num_features: Total number of features
            categorical_cardinalities: Categorical cardinalities dict
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            latent_dim: Output latent dimension
            dropout: Dropout rate
            activation: Activation function
            name: Encoder name
            config: Optional config
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.d_model = d_model
        
        # Parse schema
        if schema is not None:
            self._num_numerical = len(schema.numerical_features)
            self._categorical_info = {
                f.name: len(f.categories) if f.categories else 2
                for f in schema.categorical_features
            }
        else:
            self._num_numerical = num_features or 0
            total_cat = sum(1 for _ in (categorical_cardinalities or {}).items())
            self._categorical_info = categorical_cardinalities or {}
            
        self._total_features = self._num_numerical + len(self._categorical_info)
        
        # Numerical feature projection
        if self._num_numerical > 0:
            self.numerical_projection = nn.Sequential(
                nn.Linear(1, d_model),
                nn.LayerNorm(d_model),
                get_activation(activation)
            )
        else:
            self.numerical_projection = None
            
        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleDict()
        for name, cardinality in self._categorical_info.items():
            self.categorical_embeddings[name] = nn.Embedding(cardinality, d_model)
            
        # Column positional encoding
        self.column_positional = nn.Parameter(
            torch.randn(1, self._total_features, d_model) * 0.02
        )
        
        # Column transformer
        self.transformer = ColumnTransformer(
            d_model=d_model,
            nhead=num_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model * self._total_features, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        
    @property
    def input_dim(self) -> int:
        """Return input dimension."""
        return self._total_features
    
    def encode(
        self,
        numerical: Optional[torch.Tensor] = None,
        categorical: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Encode tabular data using column attention.
        
        Args:
            numerical: Numerical features (batch, num_numerical)
            categorical: Dict of categorical tensors
            
        Returns:
            Latent tensor (batch, latent_dim)
        """
        batch_size = numerical.shape[0] if numerical is not None else \
                     next(iter(categorical.values())).shape[0]
        
        column_embeddings = []
        
        # Project numerical features
        if numerical is not None and self.numerical_projection is not None:
            for i in range(numerical.shape[1]):
                col = numerical[:, i:i+1]  # (batch, 1)
                emb = self.numerical_projection(col)  # (batch, d_model)
                column_embeddings.append(emb)
                
        # Embed categorical features
        if categorical is not None:
            for name, values in categorical.items():
                if name in self.categorical_embeddings:
                    emb = self.categorical_embeddings[name](values)
                    column_embeddings.append(emb)
                    
        # Stack columns (batch, num_columns, d_model)
        x = torch.stack(column_embeddings, dim=1)
        
        # Add column positional encoding
        x = x + self.column_positional[:, :x.shape[1]]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Flatten and project
        x = x.flatten(1)
        z = self.output_projection(x)
        
        return z
    
    def forward(
        self,
        numerical: Optional[torch.Tensor] = None,
        categorical: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass."""
        return self.encode(numerical, categorical, **kwargs)
