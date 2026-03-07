"""
Counterfactual Generator Architecture
======================================

Counterfactual generation models for fair synthetic data.

This module implements counterfactual fairness approaches:
- CounterfactualGenerator: Base counterfactual generation
- CausalCounterfactualGenerator: Causal model-based counterfactuals
- LatentCounterfactualGenerator: Latent space counterfactual editing
- CycleConsistentGenerator: Cycle-consistent counterfactual generation

Key Features:
- Counterfactual fairness: What if the sensitive attribute were different?
- Causal modeling for counterfactual inference
- Latent space manipulation for counterfactual editing
- Cycle consistency for preserving non-sensitive attributes

Counterfactual Fairness Definition:
A decision is counterfactually fair if it would remain the same
even if the individual belonged to a different sensitive group.

Example:
    Original: (age=30, gender=M, income=50K)
    Counterfactual: (age=30, gender=F, income=?) 
    What would income be if gender were different?
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base_module import BaseGenerator
from src.models.encoders import TabularEncoder
from src.models.decoders import TabularDecoder
from src.fairness.modules.gradient_reversal import GradientReversalLayer


class CounterfactualGenerator(BaseGenerator):
    """
    Base Counterfactual Generator for fair synthetic data.
    
    Generates counterfactual samples by manipulating the latent representation
    to change sensitive attributes while preserving other properties.
    
    Architecture:
        Encoder: x -> z (latent)
        Sensitive Encoder: x -> s_encoding
        Decoder: z + s_encoding -> x_counterfactual
    
    Training Objectives:
        1. Reconstruction: Reconstruct original data
        2. Counterfactual Consistency: Same z for different sensitive values
        3. Attribute Preservation: Non-sensitive attributes preserved
    
    Example:
        >>> model = CounterfactualGenerator(
        ...     data_dim=100,
        ...     latent_dim=512,
        ...     num_sensitive_groups=2
        ... )
        >>> # Generate counterfactual
        >>> cf = model.generate_counterfactual(x, target_sensitive=1)
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        num_sensitive_groups: int = 2,
        encoder_dims: List[int] = [512, 256],
        decoder_dims: List[int] = [256, 512],
        sensitive_embed_dim: int = 64,
        counterfactual_weight: float = 1.0,
        cycle_weight: float = 1.0,
        kl_weight: float = 0.001,
        dropout: float = 0.3,
        name: str = "counterfactual_generator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Counterfactual Generator.
        
        Args:
            data_dim: Data dimension
            latent_dim: Latent dimension
            num_sensitive_groups: Number of sensitive groups
            encoder_dims: Encoder hidden dimensions
            decoder_dims: Decoder hidden dimensions
            sensitive_embed_dim: Sensitive attribute embedding dimension
            counterfactual_weight: Weight for counterfactual loss
            cycle_weight: Weight for cycle consistency loss
            kl_weight: Weight for KL divergence
            dropout: Dropout rate
            name: Model name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.data_dim = data_dim
        self.num_sensitive_groups = num_sensitive_groups
        self.counterfactual_weight = counterfactual_weight
        self.cycle_weight = cycle_weight
        self.kl_weight = kl_weight
        
        # Build components
        self._build_encoder(encoder_dims, dropout)
        self._build_decoder(decoder_dims, dropout)
        self._build_sensitive_embedding(sensitive_embed_dim)
        self._build_counterfactual_module()
        
        # Prior parameters
        self.register_buffer("prior_mean", torch.zeros(latent_dim))
        self.register_buffer("prior_logvar", torch.zeros(latent_dim))
        
    def _build_encoder(self, hidden_dims: List[int], dropout: float) -> None:
        """Build content encoder (encodes non-sensitive content)."""
        layers = []
        prev_dim = self.data_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        self.encoder_hidden = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, self._latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, self._latent_dim)
        
    def _build_decoder(self, hidden_dims: List[int], dropout: float) -> None:
        """Build decoder that takes latent + sensitive embedding."""
        # Input is latent + sensitive embedding
        input_dim = self._latent_dim + 64  # sensitive_embed_dim
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, self.data_dim))
        self.decoder = nn.Sequential(*layers)
        
    def _build_sensitive_embedding(self, embed_dim: int) -> None:
        """Build sensitive attribute embedding."""
        self.sensitive_embedding = nn.Sequential(
            nn.Embedding(self.num_sensitive_groups, embed_dim),
            nn.Linear(embed_dim, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.2)
        )
        
    def _build_counterfactual_module(self) -> None:
        """Build counterfactual transformation module."""
        # This module learns to transform latent when sensitive changes
        self.cf_transform = nn.Sequential(
            nn.Linear(self._latent_dim + 64, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self._latent_dim)
        )
        
        # Adversary to ensure latent is independent of sensitive
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.sensitive_adversary = nn.Sequential(
            nn.Linear(self._latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_sensitive_groups)
        )
        
    def encode(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode data to latent distribution."""
        hidden = self.encoder_hidden(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_sensitive_embedding(
        self,
        sensitive: torch.Tensor
    ) -> torch.Tensor:
        """Get sensitive attribute embedding."""
        return self.sensitive_embedding(sensitive)
    
    def decode(
        self,
        z: torch.Tensor,
        sensitive_emb: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Decode latent + sensitive to data."""
        combined = torch.cat([z, sensitive_emb], dim=-1)
        return self.decoder(combined)
    
    def sample_latent(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """Sample from prior."""
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(n_samples, self._latent_dim, device=device)
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        sensitive: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples.
        
        Args:
            n_samples: Number of samples
            conditions: Optional conditions
            sensitive: Sensitive attribute values
            device: Device
            
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
            
        z = self.sample_latent(n_samples, device)
        
        if sensitive is None:
            # Random sensitive
            sensitive = torch.randint(0, self.num_sensitive_groups, (n_samples,), device=device)
        
        sensitive_emb = self.get_sensitive_embedding(sensitive)
        return self.decode(z, sensitive_emb)
    
    def generate_counterfactual(
        self,
        x: torch.Tensor,
        original_sensitive: torch.Tensor,
        target_sensitive: torch.Tensor,
        use_transform: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate counterfactual samples.
        
        Args:
            x: Original data
            original_sensitive: Original sensitive attribute
            target_sensitive: Target sensitive attribute
            use_transform: Whether to use learned transform
            
        Returns:
            Counterfactual data
        """
        # Encode original
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Get target sensitive embedding
        target_emb = self.get_sensitive_embedding(target_sensitive)
        
        if use_transform:
            # Transform latent based on sensitive change
            original_emb = self.get_sensitive_embedding(original_sensitive)
            delta_emb = target_emb - original_emb
            z_combined = torch.cat([z, delta_emb], dim=-1)
            z_cf = z + self.cf_transform(z_combined)
        else:
            # Direct: use same latent with different sensitive
            z_cf = z
        
        # Decode with target sensitive
        return self.decode(z_cf, target_emb)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        sensitive: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            x: Input data
            sensitive: Sensitive attribute labels
            
        Returns:
            Dictionary of losses
        """
        batch_size = x.size(0)
        device = x.device
        
        # Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Get sensitive embedding
        sensitive_emb = self.get_sensitive_embedding(sensitive)
        
        # Decode
        x_recon = self.decode(z, sensitive_emb)
        
        losses = {}
        
        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)
        losses["reconstruction"] = recon_loss
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        losses["kl_divergence"] = self.kl_weight * kl_loss
        
        # Adversarial loss (latent should not encode sensitive)
        z_reversed = self.grl(z)
        adv_pred = self.sensitive_adversary(z_reversed)
        adv_loss = F.cross_entropy(adv_pred, sensitive)
        losses["adversarial"] = adv_loss
        
        # Counterfactual consistency loss
        # Generate counterfactual and encode back
        target_sensitive = (sensitive + 1) % self.num_sensitive_groups
        x_cf = self.generate_counterfactual(x, sensitive, target_sensitive)
        
        # Encode counterfactual
        mu_cf, _ = self.encode(x_cf.detach())
        
        # Latent should be similar (counterfactual consistency)
        cf_loss = F.mse_loss(mu_cf, mu.detach())
        losses["counterfactual_consistency"] = self.counterfactual_weight * cf_loss
        
        # Cycle consistency: counterfactual of counterfactual should return original
        x_cf_cf = self.generate_counterfactual(x_cf, target_sensitive, sensitive)
        cycle_loss = F.mse_loss(x_cf_cf, x)
        losses["cycle_consistency"] = self.cycle_weight * cycle_loss
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses
    
    def forward(
        self,
        x: torch.Tensor,
        sensitive: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass."""
        losses = self.compute_loss(x, sensitive)
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        sensitive_emb = self.get_sensitive_embedding(sensitive)
        x_recon = self.decode(z, sensitive_emb)
        
        return {
            "reconstruction": x_recon,
            "latent": z,
            "mu": mu,
            "logvar": logvar,
            "losses": losses
        }


class CausalCounterfactualGenerator(CounterfactualGenerator):
    """
    Causal Counterfactual Generator based on structural causal models.
    
    Implements counterfactual generation using a causal model:
    - Models the causal relationships between variables
    - Generates counterfactuals via causal intervention
    - Ensures counterfactual fairness through causal structure
    
    Causal Model:
        S -> X (sensitive affects some features)
        U -> X (unobserved confounders)
        X = f(S, U, noise)
    
    Counterfactual:
        X(S=s') = f(s', U, noise)  # What if S were different?
    
    Example:
        >>> model = CausalCounterfactualGenerator(
        ...     data_dim=100,
        ...     causal_graph=causal_adjacency_matrix
        ... )
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        num_sensitive_groups: int = 2,
        causal_graph: Optional[torch.Tensor] = None,
        num_causal_layers: int = 3,
        intervention_dim: int = 64,
        name: str = "causal_counterfactual_generator",
        **kwargs
    ):
        """
        Initialize Causal Counterfactual Generator.
        
        Args:
            data_dim: Data dimension
            latent_dim: Latent dimension
            num_sensitive_groups: Number of sensitive groups
            causal_graph: Adjacency matrix for causal graph
            num_causal_layers: Number of causal transformation layers
            intervention_dim: Dimension for intervention representation
            name: Model name
        """
        self.causal_graph = causal_graph
        self.num_causal_layers = num_causal_layers
        self.intervention_dim = intervention_dim
        
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_sensitive_groups=num_sensitive_groups,
            name=name,
            **kwargs
        )
        
        # Build causal layers
        self._build_causal_layers()
        
    def _build_causal_layers(self) -> None:
        """Build causal transformation layers."""
        # Each layer represents one step in the causal chain
        self.causal_layers = nn.ModuleList()
        
        for _ in range(self.num_causal_layers):
            self.causal_layers.append(nn.ModuleDict({
                "transform": nn.Sequential(
                    nn.Linear(self._latent_dim + self.intervention_dim, self._latent_dim),
                    nn.LayerNorm(self._latent_dim),
                    nn.LeakyReLU(0.2)
                ),
                "intervention": nn.Sequential(
                    nn.Linear(64, self.intervention_dim),  # sensitive embed dim
                    nn.LayerNorm(self.intervention_dim),
                    nn.LeakyReLU(0.2)
                )
            }))
        
        # Causal mask (if graph provided)
        if self.causal_graph is not None:
            self.register_buffer("causal_mask", self.causal_graph)
        else:
            self.causal_mask = None
    
    def apply_causal_transform(
        self,
        z: torch.Tensor,
        sensitive_emb: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply causal transformation to latent.
        
        Args:
            z: Latent tensor
            sensitive_emb: Sensitive embedding
            num_steps: Number of causal steps (default: all)
            
        Returns:
            Transformed latent
        """
        if num_steps is None:
            num_steps = self.num_causal_layers
        
        for i, layer in enumerate(self.causal_layers[:num_steps]):
            intervention = layer["intervention"](sensitive_emb)
            combined = torch.cat([z, intervention], dim=-1)
            delta = layer["transform"](combined)
            z = z + delta
        
        return z
    
    def generate_counterfactual(
        self,
        x: torch.Tensor,
        original_sensitive: torch.Tensor,
        target_sensitive: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate causal counterfactual.
        
        Performs causal intervention: do(S = target)
        """
        # Encode original
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Get embeddings
        original_emb = self.get_sensitive_embedding(original_sensitive)
        target_emb = self.get_sensitive_embedding(target_sensitive)
        
        # Reverse causal effect of original sensitive
        z_neutral = self.reverse_causal_effect(z, original_emb)
        
        # Apply causal effect of target sensitive
        z_cf = self.apply_causal_transform(z_neutral, target_emb)
        
        # Decode
        return self.decode(z_cf, target_emb)
    
    def reverse_causal_effect(
        self,
        z: torch.Tensor,
        sensitive_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Remove causal effect of sensitive attribute.
        
        Reverses the causal transformation to get "neutral" latent.
        """
        # Apply negative intervention
        for layer in reversed(self.causal_layers):
            intervention = layer["intervention"](sensitive_emb)
            combined = torch.cat([z, intervention], dim=-1)
            delta = layer["transform"](combined)
            z = z - delta  # Reverse the transformation
        
        return z


class LatentCounterfactualGenerator(CounterfactualGenerator):
    """
    Latent Space Counterfactual Generator.
    
    Generates counterfactuals by manipulating latent directions
    that correspond to sensitive attribute changes.
    
    Key Idea:
        1. Learn latent directions that change sensitive attribute
        2. Apply direction to latent to generate counterfactual
        3. Ensure other attributes are preserved
    
    Methods:
        - Gradient-based: Find direction via gradients
        - Contrastive: Learn direction via contrastive learning
        - Supervised: Learn direction via paired data
    
    Example:
        >>> model = LatentCounterfactualGenerator(
        ...     data_dim=100,
        ...     latent_dim=512,
        ...     direction_method="contrastive"
        ... )
        >>> cf = model.generate_counterfactual(x, target_sensitive=1)
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        num_sensitive_groups: int = 2,
        direction_method: str = "contrastive",
        num_directions: int = 8,
        disentangle_weight: float = 1.0,
        name: str = "latent_counterfactual_generator",
        **kwargs
    ):
        """
        Initialize Latent Counterfactual Generator.
        
        Args:
            data_dim: Data dimension
            latent_dim: Latent dimension
            num_sensitive_groups: Number of sensitive groups
            direction_method: Method for finding directions
            num_directions: Number of latent directions to learn
            disentangle_weight: Weight for disentanglement loss
            name: Model name
        """
        self.direction_method = direction_method
        self.num_directions = num_directions
        self.disentangle_weight = disentangle_weight
        
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_sensitive_groups=num_sensitive_groups,
            name=name,
            **kwargs
        )
        
        # Build direction module
        self._build_direction_module()
        
    def _build_direction_module(self) -> None:
        """Build latent direction learning module."""
        # Learnable directions for each sensitive group transition
        self.directions = nn.ParameterDict()
        
        for i in range(self.num_sensitive_groups):
            for j in range(self.num_sensitive_groups):
                if i != j:
                    key = f"dir_{i}_{j}"
                    direction = torch.randn(self._latent_dim)
                    direction = F.normalize(direction, dim=0)
                    self.directions[key] = nn.Parameter(direction)
        
        # Direction strength predictor
        self.strength_predictor = nn.Sequential(
            nn.Linear(self._latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Contrastive learning projection
        if self.direction_method == "contrastive":
            self.contrastive_proj = nn.Sequential(
                nn.Linear(self._latent_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128)
            )
    
    def get_direction(
        self,
        original: int,
        target: int
    ) -> torch.Tensor:
        """Get direction for sensitive attribute change."""
        key = f"dir_{original}_{target}"
        if key in self.directions:
            return self.directions[key]
        else:
            # Return negative of opposite direction
            return -self.directions[f"dir_{target}_{original}"]
    
    def generate_counterfactual(
        self,
        x: torch.Tensor,
        original_sensitive: torch.Tensor,
        target_sensitive: torch.Tensor,
        strength: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate counterfactual via latent direction.
        
        Args:
            x: Original data
            original_sensitive: Original sensitive values
            target_sensitive: Target sensitive values
            strength: Override direction strength
            
        Returns:
            Counterfactual data
        """
        # Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Get direction
        batch_size = x.size(0)
        device = x.device
        
        # Initialize direction accumulator
        z_cf = z.clone()
        
        for i in range(batch_size):
            orig = original_sensitive[i].item()
            targ = target_sensitive[i].item()
            
            if orig != targ:
                direction = self.get_direction(orig, targ)
                
                # Predict or use given strength
                if strength is None:
                    str_pred = self.strength_predictor(z[i:i+1])
                    str_val = str_pred.item() * 2.0  # Scale factor
                else:
                    str_val = strength
                
                # Apply direction
                z_cf[i] = z[i] + str_val * direction
        
        # Get target sensitive embedding
        target_emb = self.get_sensitive_embedding(target_sensitive)
        
        # Decode
        return self.decode(z_cf, target_emb)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        sensitive: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute losses including direction learning."""
        losses = super().compute_loss(x, sensitive)
        
        # Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Contrastive loss for direction learning
        if self.direction_method == "contrastive":
            # Project to contrastive space
            z_proj = self.contrastive_proj(z)
            
            # Positive pairs: same sensitive group
            # Negative pairs: different sensitive group
            contrastive_loss = self._compute_contrastive_loss(z_proj, sensitive)
            losses["contrastive"] = self.disentangle_weight * contrastive_loss
            losses["total"] = losses["total"] + losses["contrastive"]
        
        # Orthogonality loss between directions
        ortho_loss = self._compute_orthogonality_loss()
        losses["orthogonality"] = ortho_loss
        losses["total"] = losses["total"] + ortho_loss
        
        return losses
    
    def _compute_contrastive_loss(
        self,
        z_proj: torch.Tensor,
        sensitive: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive loss."""
        batch_size = z_proj.size(0)
        
        # Normalize projections
        z_proj = F.normalize(z_proj, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z_proj, z_proj.t())
        
        # Create labels: 1 for same sensitive group, 0 otherwise
        labels = (sensitive.unsqueeze(0) == sensitive.unsqueeze(1)).float()
        
        # Contrastive loss
        pos_loss = -torch.log(torch.exp(sim_matrix[labels == 1]).mean() + 1e-8)
        neg_loss = -torch.log(1 - torch.sigmoid(sim_matrix[labels == 0]) + 1e-8).mean()
        
        return pos_loss + neg_loss
    
    def _compute_orthogonality_loss(self) -> torch.Tensor:
        """Compute orthogonality loss between directions."""
        loss = 0
        count = 0
        
        keys = list(self.directions.keys())
        for i, key1 in enumerate(keys):
            for key2 in keys[i+1:]:
                d1 = self.directions[key1]
                d2 = self.directions[key2]
                loss = loss + (d1 @ d2) ** 2
                count += 1
        
        if count > 0:
            loss = loss / count
        
        return loss


class CycleConsistentGenerator(CounterfactualGenerator):
    """
    Cycle-Consistent Counterfactual Generator.
    
    Uses cycle consistency to ensure counterfactual quality:
        x --[S->T]--> x_cf --[T->S]--> x_recon
    
    The reconstructed x_recon should match original x.
    
    Benefits:
        - Ensures non-sensitive attributes are preserved
        - No need for explicit attribute labels
        - Self-supervised learning
    
    Example:
        >>> model = CycleConsistentGenerator(
        ...     data_dim=100,
        ...     num_sensitive_groups=2
        ... )
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        num_sensitive_groups: int = 2,
        num_cycle_layers: int = 3,
        identity_weight: float = 0.5,
        name: str = "cycle_consistent_generator",
        **kwargs
    ):
        """
        Initialize Cycle-Consistent Generator.
        
        Args:
            data_dim: Data dimension
            latent_dim: Latent dimension
            num_sensitive_groups: Number of sensitive groups
            num_cycle_layers: Number of cycle transformation layers
            identity_weight: Weight for identity mapping loss
            name: Model name
        """
        self.num_cycle_layers = num_cycle_layers
        self.identity_weight = identity_weight
        
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_sensitive_groups=num_sensitive_groups,
            name=name,
            **kwargs
        )
        
        # Build cycle-specific modules
        self._build_cycle_modules()
        
    def _build_cycle_modules(self) -> None:
        """Build cycle transformation modules."""
        # Generator for each sensitive transition
        self.generators = nn.ModuleDict()
        
        for i in range(self.num_sensitive_groups):
            for j in range(self.num_sensitive_groups):
                if i != j:
                    self.generators[f"G_{i}_{j}"] = self._build_transform_net()
        
        # Discriminator for adversarial training
        self.discriminator = nn.Sequential(
            nn.Linear(self.data_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    
    def _build_transform_net(self) -> nn.Module:
        """Build a single transformation network."""
        return nn.Sequential(
            nn.Linear(self._latent_dim + 64, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self._latent_dim)
        )
    
    def transform(
        self,
        z: torch.Tensor,
        sensitive_emb: torch.Tensor,
        source: int,
        target: int
    ) -> torch.Tensor:
        """Apply transformation for sensitive transition."""
        key = f"G_{source}_{target}"
        combined = torch.cat([z, sensitive_emb], dim=-1)
        return z + self.generators[key](combined)
    
    def generate_counterfactual(
        self,
        x: torch.Tensor,
        original_sensitive: torch.Tensor,
        target_sensitive: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Generate counterfactual with cycle consistency."""
        # Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        batch_size = x.size(0)
        device = x.device
        
        # Apply transformation for each sample
        z_cf = z.clone()
        for i in range(batch_size):
            orig = original_sensitive[i].item()
            targ = target_sensitive[i].item()
            
            if orig != targ:
                target_emb = self.get_sensitive_embedding(target_sensitive[i:i+1])
                z_cf[i] = self.transform(z[i:i+1], target_emb, orig, targ)
        
        # Decode
        target_emb = self.get_sensitive_embedding(target_sensitive)
        return self.decode(z_cf, target_emb)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        sensitive: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute cycle-consistent losses."""
        losses = {}
        
        # Encode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        sensitive_emb = self.get_sensitive_embedding(sensitive)
        
        # Reconstruction
        x_recon = self.decode(z, sensitive_emb)
        recon_loss = F.mse_loss(x_recon, x)
        losses["reconstruction"] = recon_loss
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        losses["kl_divergence"] = self.kl_weight * kl_loss
        
        # Cycle consistency loss
        cycle_loss = 0
        count = 0
        
        for i in range(self.num_sensitive_groups):
            for j in range(self.num_sensitive_groups):
                if i != j:
                    # Get samples with sensitive = i
                    mask_i = sensitive == i
                    if mask_i.sum() == 0:
                        continue
                    
                    x_i = x[mask_i]
                    z_i = z[mask_i]
                    
                    # Transform i -> j
                    target_emb_j = self.get_sensitive_embedding(
                        torch.full((x_i.size(0),), j, dtype=torch.long, device=x.device)
                    )
                    z_ij = self.transform(z_i, target_emb_j, i, j)
                    
                    # Decode to counterfactual
                    x_ij = self.decode(z_ij, target_emb_j)
                    
                    # Transform back j -> i
                    target_emb_i = self.get_sensitive_embedding(
                        torch.full((x_i.size(0),), i, dtype=torch.long, device=x.device)
                    )
                    z_iji = self.transform(z_ij, target_emb_i, j, i)
                    
                    # Decode to reconstruction
                    x_iji = self.decode(z_iji, target_emb_i)
                    
                    # Cycle loss
                    cycle_loss = cycle_loss + F.mse_loss(x_iji, x_i)
                    count += 1
        
        if count > 0:
            cycle_loss = cycle_loss / count
        losses["cycle"] = self.cycle_weight * cycle_loss
        
        # Identity loss (transformation with same sensitive should be identity)
        identity_loss = 0
        for i in range(self.num_sensitive_groups):
            mask_i = sensitive == i
            if mask_i.sum() == 0:
                continue
            
            z_i = z[mask_i]
            sensitive_emb_i = sensitive_emb[mask_i]
            
            # Self-transformation should be minimal
            z_self = self.transform(z_i, sensitive_emb_i, i, i)
            identity_loss = identity_loss + F.mse_loss(z_self, z_i)
        
        losses["identity"] = self.identity_weight * identity_loss
        
        # Adversarial loss
        # Discriminator should not distinguish real vs counterfactual
        adv_loss = self._compute_adversarial_loss(x, sensitive, z)
        losses["adversarial"] = adv_loss
        
        # Total
        losses["total"] = sum(losses.values())
        
        return losses
    
    def _compute_adversarial_loss(
        self,
        x: torch.Tensor,
        sensitive: torch.Tensor,
        z: torch.Tensor
    ) -> torch.Tensor:
        """Compute adversarial loss for domain confusion."""
        # Generate counterfactuals
        target_sensitive = (sensitive + 1) % self.num_sensitive_groups
        target_emb = self.get_sensitive_embedding(target_sensitive)
        
        x_cf = []
        for i in range(x.size(0)):
            orig = sensitive[i].item()
            targ = target_sensitive[i].item()
            z_cf = self.transform(z[i:i+1], target_emb[i:i+1], orig, targ)
            x_cf.append(self.decode(z_cf, target_emb[i:i+1]))
        
        x_cf = torch.cat(x_cf, dim=0)
        
        # Discriminator predictions
        d_real = self.discriminator(x)
        d_fake = self.discriminator(x_cf.detach())
        
        # Generator wants discriminator to be confused
        g_loss = F.binary_cross_entropy_with_logits(
            d_fake,
            torch.ones_like(d_fake)
        )
        
        return g_loss


class CounterfactualEvaluator(nn.Module):
    """
    Evaluator for counterfactual quality.
    
    Metrics:
        - Counterfactual Validity: Does counterfactual have target attribute?
        - Proximity: Are non-sensitive attributes preserved?
        - Causal Consistency: Does counterfactual respect causal structure?
        - Diversity: Are counterfactuals diverse?
    """
    
    def __init__(
        self,
        data_dim: int,
        num_sensitive_groups: int,
        attribute_predictor_dims: Optional[List[int]] = None
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.num_sensitive_groups = num_sensitive_groups
        
        # Build attribute predictor for evaluating validity
        dims = attribute_predictor_dims or [256, 128]
        
        layers = []
        prev_dim = data_dim
        for dim in dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = dim
        
        self.attribute_predictor = nn.Sequential(
            *layers,
            nn.Linear(prev_dim, num_sensitive_groups)
        )
    
    def predict_sensitive(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """Predict sensitive attribute from data."""
        return self.attribute_predictor(x)
    
    def evaluate_counterfactual(
        self,
        x_original: torch.Tensor,
        x_counterfactual: torch.Tensor,
        original_sensitive: torch.Tensor,
        target_sensitive: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate counterfactual quality.
        
        Args:
            x_original: Original data
            x_counterfactual: Counterfactual data
            original_sensitive: Original sensitive values
            target_sensitive: Target sensitive values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        with torch.no_grad():
            # 1. Counterfactual Validity
            pred_sensitive = self.predict_sensitive(x_counterfactual).argmax(dim=-1)
            validity = (pred_sensitive == target_sensitive).float().mean().item()
            metrics["validity"] = validity
            
            # 2. Proximity (non-sensitive preservation)
            # Measure how much non-sensitive attributes changed
            distance = F.mse_loss(x_counterfactual, x_original, reduction="none")
            proximity = 1.0 - distance.mean().item()
            metrics["proximity"] = max(0, proximity)
            
            # 3. Feature-wise proximity
            feature_distances = distance.mean(dim=0)
            metrics["mean_feature_distance"] = feature_distances.mean().item()
            metrics["max_feature_distance"] = feature_distances.max().item()
            
            # 4. Diversity (for batch)
            if x_counterfactual.size(0) > 1:
                cf_std = x_counterfactual.std(dim=0).mean().item()
                metrics["diversity"] = cf_std
        
        return metrics


# Registry for Counterfactual Generators
COUNTERFACTUAL_REGISTRY = {
    "counterfactual": CounterfactualGenerator,
    "causal_counterfactual": CausalCounterfactualGenerator,
    "latent_counterfactual": LatentCounterfactualGenerator,
    "cycle_consistent": CycleConsistentGenerator,
}


def get_counterfactual_generator(name: str, **kwargs) -> CounterfactualGenerator:
    """Get a Counterfactual Generator by name."""
    if name not in COUNTERFACTUAL_REGISTRY:
        available = list(COUNTERFACTUAL_REGISTRY.keys())
        raise ValueError(f"Unknown CounterfactualGenerator: {name}. Available: {available}")
    return COUNTERFACTUAL_REGISTRY[name](**kwargs)
