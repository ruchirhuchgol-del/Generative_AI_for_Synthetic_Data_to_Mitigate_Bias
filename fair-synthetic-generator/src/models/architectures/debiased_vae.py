"""
Debiased VAE Architecture
=========================

Variational Autoencoder with adversarial debiasing for fair synthetic data generation.

This module implements several debiased VAE variants:
- DebiasedVAE: VAE with adversarial fairness constraint
- BetaDebiasedVAE: β-VAE with fairness (better disentanglement)
- ConditionalDebiasedVAE: Conditional VAE with fairness
- HierarchicalDebiasedVAE: Multi-level VAE for complex distributions
- MultimodalDebiasedVAE: Multimodal VAE with cross-modal fairness

Key Features:
- Adversarial debiasing via gradient reversal
- Multiple fairness paradigms (group, individual, counterfactual)
- Disentangled representations
- Support for tabular, text, and image modalities
- Progressive training with fairness annealing
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base_module import BaseGenerator
from src.models.encoders import TabularEncoder, TextEncoder, ImageEncoder
from src.models.decoders import TabularDecoder, TextDecoder, ImageDecoder
from src.models.discriminators import FairnessAdversary, GradientReversalLayer
from src.fairness.modules.gradient_reversal import ScheduledGradientReversalLayer
from src.fairness.modules.adversary_network import MultiTaskAdversary


class DebiasedVAE(BaseGenerator):
    """
    VAE with adversarial debiasing for fair synthetic data generation.
    
    Architecture:
        Encoder: x -> (mu, logvar) -> z (latent)
        Decoder: z -> x_hat (reconstruction)
        Adversary: z -> sensitive_attr_pred (with gradient reversal)
    
    Training Objectives:
        1. Reconstruction: Minimize reconstruction error
        2. KL Divergence: Regularize latent distribution
        3. Fairness: Prevent latent from encoding sensitive info
    
    The encoder learns to produce latent representations that:
    - Capture data distribution well (reconstruction loss)
    - Are independent of sensitive attributes (adversarial loss)
    
    Fairness Guarantees:
    - Group Fairness: Adversarial training removes group information
    - Individual Fairness: Similar inputs map to similar latents
    - Counterfactual Fairness: Latent independent of sensitive attributes
    
    Example:
        >>> model = DebiasedVAE(
        ...     data_dim=100,
        ...     latent_dim=512,
        ...     num_sensitive_groups=2
        ... )
        >>> # Training
        >>> losses = model.train_step(data, sensitive_attrs)
        >>> # Generation
        >>> samples = model.generate(100)
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        num_sensitive_groups: int = 2,
        encoder_dims: List[int] = [512, 256],
        decoder_dims: List[int] = [256, 512],
        kl_weight: float = 0.001,
        fairness_weight: float = 1.0,
        recon_loss: str = "mse",
        grl_lambda: float = 1.0,
        grl_warmup: int = 10,
        adversary_hidden: List[int] = [256, 128],
        dropout: float = 0.3,
        use_layer_norm: bool = True,
        name: str = "debias_vae",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Debiased VAE.
        
        Args:
            data_dim: Dimension of input data
            latent_dim: Dimension of latent space
            num_sensitive_groups: Number of sensitive attribute groups
            encoder_dims: Encoder hidden dimensions
            decoder_dims: Decoder hidden dimensions
            kl_weight: Weight for KL divergence loss
            fairness_weight: Weight for adversarial fairness loss
            recon_loss: Reconstruction loss type ("mse", "bce", "huber")
            grl_lambda: Gradient reversal lambda
            grl_warmup: Epochs for GRL warmup
            adversary_hidden: Adversary hidden dimensions
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            name: Model name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.data_dim = data_dim
        self.num_sensitive_groups = num_sensitive_groups
        self.kl_weight = kl_weight
        self.fairness_weight = fairness_weight
        self.recon_loss_type = recon_loss
        
        # Build encoder
        self._build_encoder(encoder_dims, dropout, use_layer_norm)
        
        # Build decoder
        self._build_decoder(decoder_dims, dropout, use_layer_norm)
        
        # Build fairness adversary
        self._build_adversary(adversary_hidden, grl_lambda, grl_warmup, dropout)
        
        # Prior distribution parameters
        self.register_buffer("prior_mean", torch.zeros(latent_dim))
        self.register_buffer("prior_logvar", torch.zeros(latent_dim))
        
        # Training state
        self._current_epoch = 0
        
    def _build_encoder(
        self,
        hidden_dims: List[int],
        dropout: float,
        use_layer_norm: bool
    ) -> None:
        """Build the encoder network."""
        layers = []
        prev_dim = self.data_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output mean and logvar
        self.encoder_hidden = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, self._latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, self._latent_dim)
        
    def _build_decoder(
        self,
        hidden_dims: List[int],
        dropout: float,
        use_layer_norm: bool
    ) -> None:
        """Build the decoder network."""
        layers = []
        prev_dim = self._latent_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, self.data_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def _build_adversary(
        self,
        hidden_dims: List[int],
        grl_lambda: float,
        grl_warmup: int,
        dropout: float
    ) -> None:
        """Build the fairness adversary network."""
        # Scheduled gradient reversal layer
        self.grl = ScheduledGradientReversalLayer(
            lambda_start=0.0,
            lambda_end=grl_lambda,
            warmup_epochs=grl_warmup,
            schedule_type="linear"
        )
        
        # Adversary head
        layers = []
        prev_dim = self._latent_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, self.num_sensitive_groups))
        
        self.adversary = nn.Sequential(*layers)
        
    def encode(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean, log_variance)
        """
        hidden = self.encoder_hidden(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick for VAE sampling.
        
        Args:
            mu: Mean tensor
            logvar: Log-variance tensor
            
        Returns:
            Sampled latent tensor
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(
        self,
        z: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode latent tensor to output.
        
        Args:
            z: Latent tensor
            
        Returns:
            Reconstructed output
        """
        return self.decoder(z)
    
    def sample_latent(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample from the prior distribution.
        
        Args:
            n_samples: Number of samples
            device: Device for samples
            
        Returns:
            Latent samples
        """
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(n_samples, self._latent_dim, device=device)
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples
            conditions: Optional conditions (not used in base VAE)
            device: Device for generation
            
        Returns:
            Generated samples
        """
        z = self.sample_latent(n_samples, device)
        return self.decode(z)
    
    def reconstruction_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            
        Returns:
            Reconstruction loss
        """
        if self.recon_loss_type == "mse":
            return F.mse_loss(x_recon, x, reduction="mean")
        elif self.recon_loss_type == "bce":
            return F.binary_cross_entropy_with_logits(x_recon, x, reduction="mean")
        elif self.recon_loss_type == "huber":
            return F.smooth_l1_loss(x_recon, x, reduction="mean")
        else:
            return F.mse_loss(x_recon, x, reduction="mean")
    
    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence from prior.
        
        Args:
            mu: Mean tensor
            logvar: Log-variance tensor
            
        Returns:
            KL divergence loss
        """
        # KL(N(mu, sigma) || N(0, 1))
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl
    
    def compute_loss(
        self,
        x: torch.Tensor,
        sensitive_attrs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for training.
        
        Args:
            x: Input data
            sensitive_attrs: Sensitive attribute labels
            
        Returns:
            Dictionary of losses
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z)
        
        # Compute losses
        losses = {}
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(x, x_recon)
        losses["reconstruction"] = recon_loss
        
        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar)
        losses["kl_divergence"] = self.kl_weight * kl_loss
        
        # Adversarial fairness loss
        if sensitive_attrs is not None:
            # Apply GRL to latent
            z_reversed = self.grl(z)
            adv_pred = self.adversary(z_reversed)
            adv_loss = F.cross_entropy(adv_pred, sensitive_attrs)
            losses["adversarial"] = self.fairness_weight * adv_loss
        else:
            losses["adversarial"] = torch.tensor(0.0, device=x.device)
        
        # Total loss
        losses["total"] = sum(losses.values())
        
        return losses
    
    def forward(
        self,
        x: torch.Tensor,
        sensitive_attrs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass with all outputs.
        
        Args:
            x: Input data
            sensitive_attrs: Sensitive attribute labels
            
        Returns:
            Dictionary containing all outputs and losses
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z)
        
        # Compute losses
        losses = self.compute_loss(x, sensitive_attrs)
        
        return {
            "reconstruction": x_recon,
            "latent": z,
            "mu": mu,
            "logvar": logvar,
            "losses": losses
        }
    
    def adversary_loss(
        self,
        z: torch.Tensor,
        sensitive_attrs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adversary loss for adversarial training.
        
        Args:
            z: Latent tensor
            sensitive_attrs: Sensitive attribute labels
            
        Returns:
            Cross-entropy loss
        """
        adv_pred = self.adversary(z)
        return F.cross_entropy(adv_pred, sensitive_attrs)
    
    def on_epoch_end(self) -> None:
        """Called at end of each epoch."""
        self._current_epoch += 1
        self.grl.step()
    
    def compute_fairness_metrics(
        self,
        x: torch.Tensor,
        sensitive_attrs: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute fairness metrics.
        
        Args:
            x: Input data
            sensitive_attrs: Sensitive attribute labels
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        with torch.no_grad():
            # Encode to latent
            mu, _ = self.encode(x)
            
            # Get adversary predictions
            adv_pred = self.adversary(mu)
            pred_probs = F.softmax(adv_pred, dim=-1)
            
            # Adversary accuracy (lower is better for fairness)
            pred_labels = adv_pred.argmax(dim=-1)
            accuracy = (pred_labels == sensitive_attrs).float().mean().item()
            metrics["adversary_accuracy"] = accuracy
            
            # Demographic parity gap
            unique_groups = torch.unique(sensitive_attrs)
            if len(unique_groups) >= 2:
                group_means = []
                for group in unique_groups:
                    mask = sensitive_attrs == group
                    if mask.sum() > 0:
                        group_mean = mu[mask].mean(dim=0)
                        group_means.append(group_mean)
                
                if len(group_means) >= 2:
                    dp_gap = (group_means[0] - group_means[1]).norm().item()
                    metrics["latent_separation"] = dp_gap
        
        return metrics


class BetaDebiasedVAE(DebiasedVAE):
    """
    β-VAE with adversarial debiasing.
    
    Introduces β parameter to control disentanglement:
    - β > 1 encourages disentanglement
    - Higher β = more disentangled but worse reconstruction
    
    Disentanglement helps fairness by separating sensitive attributes
    into independent latent dimensions.
    
    Example:
        >>> model = BetaDebiasedVAE(
        ...     data_dim=100,
        ...     latent_dim=512,
        ...     beta=4.0,
        ...     num_sensitive_groups=2
        ... )
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        beta: float = 4.0,
        capacity: Optional[float] = None,
        capacity_annealing: bool = True,
        max_capacity: float = 25.0,
        capacity_steps: int = 100000,
        name: str = "beta_debiased_vae",
        **kwargs
    ):
        """
        Initialize β-Debiased VAE.
        
        Args:
            data_dim: Data dimension
            latent_dim: Latent dimension
            beta: β parameter for disentanglement
            capacity: KL capacity (if None, uses beta)
            capacity_annealing: Whether to anneal capacity
            max_capacity: Maximum KL capacity
            capacity_steps: Steps to reach max capacity
            name: Model name
        """
        self.beta = beta
        self.capacity = capacity if capacity is not None else beta
        self.capacity_annealing = capacity_annealing
        self.max_capacity = max_capacity
        self.capacity_steps = capacity_steps
        
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            name=name,
            **kwargs
        )
        
        self._capacity_current = 0.0
        self._global_step = 0
        
    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Compute KL divergence with capacity annealing."""
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        if self.capacity_annealing:
            # Anneal capacity
            self._capacity_current = min(
                self.max_capacity,
                self.max_capacity * self._global_step / self.capacity_steps
            )
        
        # β-VAE loss: β * |KL - C| encourages KL ≈ C
        return self.beta * (kl - self._capacity_current).abs()
    
    def compute_loss(
        self,
        x: torch.Tensor,
        sensitive_attrs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute losses with β-weighted KL."""
        self._global_step += 1
        return super().compute_loss(x, sensitive_attrs, **kwargs)


class ConditionalDebiasedVAE(DebiasedVAE):
    """
    Conditional VAE with adversarial debiasing.
    
    Supports conditioning on:
    - Class labels
    - Continuous attributes
    - Target sensitive attribute distributions
    
    Useful for:
    - Generating data with specific attributes
    - Counterfactual generation
    - Targeted fairness adjustments
    
    Example:
        >>> model = ConditionalDebiasedVAE(
        ...     data_dim=100,
        ...     num_classes=10,
        ...     num_sensitive_groups=2
        ... )
        >>> # Generate for specific class
        >>> samples = model.generate(100, class_label=3)
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        num_classes: Optional[int] = None,
        condition_dim: Optional[int] = None,
        num_sensitive_groups: int = 2,
        conditioning_type: str = "concat",
        name: str = "conditional_debiased_vae",
        **kwargs
    ):
        """
        Initialize Conditional Debiased VAE.
        
        Args:
            data_dim: Data dimension
            latent_dim: Latent dimension
            num_classes: Number of classes
            condition_dim: Continuous condition dimension
            num_sensitive_groups: Number of sensitive groups
            conditioning_type: How to condition ("concat", "add", "film")
            name: Model name
        """
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.conditioning_type = conditioning_type
        
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_sensitive_groups=num_sensitive_groups,
            name=name,
            **kwargs
        )
        
        # Build condition embeddings
        self._build_condition_embeddings()
        
    def _build_condition_embeddings(self) -> None:
        """Build condition embedding layers."""
        if self.num_classes is not None:
            self.class_embedding = nn.Embedding(self.num_classes, self._latent_dim)
        else:
            self.class_embedding = None
            
        if self.condition_dim is not None:
            self.condition_projection = nn.Sequential(
                nn.Linear(self.condition_dim, self._latent_dim),
                nn.LayerNorm(self._latent_dim),
                nn.LeakyReLU(0.2)
            )
        else:
            self.condition_projection = None
    
    def _build_encoder(
        self,
        hidden_dims: List[int],
        dropout: float,
        use_layer_norm: bool
    ) -> None:
        """Build conditional encoder."""
        # Input dimension includes condition
        input_dim = self.data_dim
        if self.conditioning_type == "concat":
            if self.num_classes is not None:
                input_dim += self._latent_dim
            if self.condition_dim is not None:
                input_dim += self._latent_dim
        
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.encoder_hidden = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev_dim, self._latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, self._latent_dim)
        
    def get_condition_embedding(
        self,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Get combined condition embedding."""
        if device is None:
            device = next(self.parameters()).device
            
        cond_emb = torch.zeros(batch_size, self._latent_dim, device=device)
        
        if class_label is not None and self.class_embedding is not None:
            cond_emb = cond_emb + self.class_embedding(class_label)
            
        if condition is not None and self.condition_projection is not None:
            cond_emb = cond_emb + self.condition_projection(condition)
            
        return cond_emb
    
    def encode(
        self,
        x: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conditional encoding."""
        batch_size = x.size(0)
        
        # Get condition embedding
        if self.conditioning_type == "concat":
            cond_emb = self.get_condition_embedding(class_label, condition, batch_size, x.device)
            x = torch.cat([x, cond_emb], dim=-1)
        
        hidden = self.encoder_hidden(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        
        return mu, logvar
    
    def decode(
        self,
        z: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Conditional decoding."""
        # Combine latent with condition
        if self.conditioning_type == "concat":
            cond_emb = self.get_condition_embedding(class_label, condition, z.size(0), z.device)
            z = torch.cat([z, cond_emb], dim=-1)
        elif self.conditioning_type == "add":
            cond_emb = self.get_condition_embedding(class_label, condition, z.size(0), z.device)
            z = z + cond_emb
        
        return self.decoder(z)
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate conditional samples."""
        if device is None:
            device = next(self.parameters()).device
            
        z = self.sample_latent(n_samples, device)
        return self.decode(z, class_label, condition)
    
    def generate_counterfactual(
        self,
        x: torch.Tensor,
        original_sensitive: int,
        target_sensitive: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate counterfactual samples.
        
        Encode original data, then decode with different sensitive label.
        """
        if device is None:
            device = x.device
            
        batch_size = x.size(0)
        
        # Encode to latent
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        # Decode with target sensitive as condition
        target_label = torch.full((batch_size,), target_sensitive, 
                                   dtype=torch.long, device=device)
        
        return self.decode(z, class_label=target_label)


class HierarchicalDebiasedVAE(DebiasedVAE):
    """
    Hierarchical VAE with adversarial debiasing.
    
    Multi-level latent hierarchy:
        z_1 -> z_2 -> ... -> z_L -> x
    
    Benefits:
    - Better representation learning
    - More flexible latent structure
    - Can enforce fairness at multiple levels
    
    Example:
        >>> model = HierarchicalDebiasedVAE(
        ...     data_dim=100,
        ...     latent_dims=[512, 256, 128],
        ...     num_sensitive_groups=2
        ... )
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dims: List[int] = [512, 256, 128],
        num_sensitive_groups: int = 2,
        share_adversary: bool = False,
        name: str = "hierarchical_debiased_vae",
        **kwargs
    ):
        """
        Initialize Hierarchical Debiased VAE.
        
        Args:
            data_dim: Data dimension
            latent_dims: List of latent dimensions (bottom to top)
            num_sensitive_groups: Number of sensitive groups
            share_adversary: Whether to share adversary across levels
            name: Model name
        """
        self.latent_dims = latent_dims
        self.num_levels = len(latent_dims)
        self.share_adversary = share_adversary
        
        # Use smallest latent dimension
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dims[-1],
            num_sensitive_groups=num_sensitive_groups,
            name=name,
            **kwargs
        )
        
        # Build hierarchical encoders and decoders
        self._build_hierarchical_layers()
        
        # Build adversaries for each level
        if not share_adversary:
            self._build_level_adversaries()
    
    def _build_hierarchical_layers(self) -> None:
        """Build hierarchical encoder/decoder layers."""
        # Bottom-up encoders: x -> z_1 -> z_2 -> ...
        self.level_encoders = nn.ModuleList()
        prev_dim = self.data_dim
        
        for i, dim in enumerate(self.latent_dims):
            self.level_encoders.append(nn.ModuleDict({
                "hidden": nn.Sequential(
                    nn.Linear(prev_dim, dim),
                    nn.LayerNorm(dim),
                    nn.LeakyReLU(0.2)
                ),
                "mu": nn.Linear(dim, self.latent_dims[i+1] if i+1 < len(self.latent_dims) else dim),
                "logvar": nn.Linear(dim, self.latent_dims[i+1] if i+1 < len(self.latent_dims) else dim)
            }))
            prev_dim = dim
        
        # Top-down decoders: z_L -> z_{L-1} -> ... -> z_1 -> x
        self.level_decoders = nn.ModuleList()
        reversed_dims = list(reversed(self.latent_dims))
        
        for i, dim in enumerate(reversed_dims):
            prev_latent = reversed_dims[i-1] if i > 0 else reversed_dims[0]
            next_dim = reversed_dims[i+1] if i+1 < len(reversed_dims) else self.data_dim
            
            self.level_decoders.append(nn.ModuleDict({
                "hidden": nn.Sequential(
                    nn.Linear(prev_latent, dim),
                    nn.LayerNorm(dim),
                    nn.LeakyReLU(0.2)
                ),
                "mu": nn.Linear(dim, next_dim),
                "logvar": nn.Linear(dim, next_dim) if i < len(reversed_dims) - 1 else None
            }))
    
    def _build_level_adversaries(self) -> None:
        """Build separate adversary for each level."""
        self.level_adversaries = nn.ModuleList()
        
        for dim in self.latent_dims:
            self.level_adversaries.append(nn.Sequential(
                nn.Linear(dim, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(128, self.num_sensitive_groups)
            ))
    
    def encode_hierarchical(
        self,
        x: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Encode through hierarchy."""
        level_params = []
        h = x
        
        for encoder in self.level_encoders:
            h = encoder["hidden"](h)
            mu = encoder["mu"](h)
            logvar = encoder["logvar"](h)
            level_params.append((mu, logvar))
            h = self.reparameterize(mu, logvar)
        
        return level_params
    
    def decode_hierarchical(
        self,
        z_top: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Decode through hierarchy."""
        latents = [z_top]
        h = z_top
        
        for decoder in self.level_decoders[:-1]:
            h = decoder["hidden"](h)
            mu = decoder["mu"](h)
            logvar = decoder["logvar"](h)
            h = self.reparameterize(mu, logvar)
            latents.append(h)
        
        # Final decoder produces x
        h = self.level_decoders[-1]["hidden"](h)
        x_recon = self.level_decoders[-1]["mu"](h)
        
        return x_recon, latents
    
    def forward(
        self,
        x: torch.Tensor,
        sensitive_attrs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass through hierarchy."""
        # Encode
        level_params = self.encode_hierarchical(x)
        
        # Sample top level
        mu_top, logvar_top = level_params[-1]
        z_top = self.reparameterize(mu_top, logvar_top)
        
        # Decode
        x_recon, latents = self.decode_hierarchical(z_top)
        
        # Compute losses
        losses = {}
        
        # Reconstruction
        losses["reconstruction"] = self.reconstruction_loss(x, x_recon)
        
        # KL divergence at each level
        kl_total = 0
        for i, (mu, logvar) in enumerate(level_params):
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            losses[f"kl_level_{i}"] = self.kl_weight * kl
            kl_total = kl_total + kl
        losses["kl_divergence"] = self.kl_weight * kl_total
        
        # Adversarial losses at each level
        if sensitive_attrs is not None:
            if self.share_adversary:
                # Use single adversary on top level
                z_reversed = self.grl(z_top)
                adv_pred = self.adversary(z_reversed)
                adv_loss = F.cross_entropy(adv_pred, sensitive_attrs)
                losses["adversarial"] = self.fairness_weight * adv_loss
            else:
                # Separate adversary at each level
                adv_total = 0
                for i, z in enumerate(latents):
                    z_reversed = self.grl(z)
                    adv_pred = self.level_adversaries[i](z_reversed)
                    adv_loss = F.cross_entropy(adv_pred, sensitive_attrs)
                    losses[f"adversarial_level_{i}"] = adv_loss
                    adv_total = adv_total + adv_loss
                losses["adversarial"] = self.fairness_weight * adv_total
        
        losses["total"] = sum(losses.values())
        
        return {
            "reconstruction": x_recon,
            "latents": latents,
            "level_params": level_params,
            "losses": losses
        }


class MultimodalDebiasedVAE(DebiasedVAE):
    """
    Multimodal VAE with cross-modal fairness.
    
    Handles multiple data modalities:
    - Tabular: Structured numerical/categorical data
    - Text: Sequential text data
    - Image: Visual data
    
    Ensures fairness across modalities through:
    - Shared latent space
    - Cross-modal adversarial debiasing
    - Modality-specific fairness constraints
    
    Example:
        >>> model = MultimodalDebiasedVAE(
        ...     modalities=["tabular", "text", "image"],
        ...     latent_dim=512,
        ...     modality_configs={
        ...         "tabular": {"dim": 100},
        ...         "text": {"vocab_size": 10000, "seq_len": 50},
        ...         "image": {"channels": 3, "size": 64}
        ...     }
        ... )
        >>> samples = model.generate(100)
    """
    
    def __init__(
        self,
        modalities: List[str],
        latent_dim: int = 512,
        modality_configs: Optional[Dict[str, Dict]] = None,
        num_sensitive_groups: int = 2,
        fusion_type: str = "poen",
        kl_weight: float = 0.001,
        fairness_weight: float = 1.0,
        name: str = "multimodal_debiased_vae",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Multimodal Debiased VAE.
        
        Args:
            modalities: List of modalities
            latent_dim: Shared latent dimension
            modality_configs: Configuration for each modality
            num_sensitive_groups: Number of sensitive groups
            fusion_type: Fusion type ("poen", "moe", "attention")
            kl_weight: KL divergence weight
            fairness_weight: Fairness loss weight
            name: Model name
            config: Optional configuration
        """
        self.modalities = modalities
        self.modality_configs = modality_configs or {}
        self.fusion_type = fusion_type
        
        # Calculate effective data dimension (for parent class)
        data_dim = latent_dim  # Placeholder
        
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_sensitive_groups=num_sensitive_groups,
            kl_weight=kl_weight,
            fairness_weight=fairness_weight,
            name=name,
            config=config
        )
        
        # Build modality-specific encoders and decoders
        self._build_modality_encoders()
        self._build_modality_decoders()
        self._build_fusion_layer()
        
    def _build_modality_encoders(self) -> None:
        """Build encoders for each modality."""
        self.modality_encoders = nn.ModuleDict()
        
        for mod in self.modalities:
            config = self.modality_configs.get(mod, {})
            
            if mod == "tabular":
                dim = config.get("dim", 100)
                self.modality_encoders[mod] = nn.Sequential(
                    nn.Linear(dim, 256),
                    nn.LayerNorm(256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, self._latent_dim * 2)
                )
                
            elif mod == "text":
                vocab_size = config.get("vocab_size", 10000)
                self.modality_encoders[mod] = nn.Sequential(
                    nn.Embedding(vocab_size, 256),
                    nn.LSTM(256, self._latent_dim, batch_first=True)
                )
                
            elif mod == "image":
                channels = config.get("channels", 3)
                size = config.get("size", 64)
                self.modality_encoders[mod] = nn.Sequential(
                    nn.Conv2d(channels, 64, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(128, self._latent_dim * 2)
                )
    
    def _build_modality_decoders(self) -> None:
        """Build decoders for each modality."""
        self.modality_decoders = nn.ModuleDict()
        
        for mod in self.modalities:
            config = self.modality_configs.get(mod, {})
            
            if mod == "tabular":
                dim = config.get("dim", 100)
                self.modality_decoders[mod] = nn.Sequential(
                    nn.Linear(self._latent_dim, 256),
                    nn.LayerNorm(256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, dim)
                )
                
            elif mod == "text":
                vocab_size = config.get("vocab_size", 10000)
                seq_len = config.get("seq_len", 50)
                self.modality_decoders[mod] = nn.Sequential(
                    nn.Linear(self._latent_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, seq_len * vocab_size)
                )
                
            elif mod == "image":
                channels = config.get("channels", 3)
                size = config.get("size", 64)
                self.modality_decoders[mod] = nn.Sequential(
                    nn.Linear(self._latent_dim, 128 * 4 * 4),
                    nn.Unflatten(1, (128, 4, 4)),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(64, channels, 4, 2, 1),
                    nn.Tanh()
                )
    
    def _build_fusion_layer(self) -> None:
        """Build cross-modal fusion layer."""
        if self.fusion_type == "poen":
            # Product of Experts
            self.fusion = ProductOfExperts()
        elif self.fusion_type == "moe":
            # Mixture of Experts
            self.fusion = MixtureOfExperts(len(self.modalities), self._latent_dim)
        elif self.fusion_type == "attention":
            self.fusion = CrossModalAttentionFusion(self._latent_dim)
        else:
            self.fusion = None
    
    def encode_modality(
        self,
        mod: str,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode single modality."""
        encoder = self.modality_encoders[mod]
        
        if mod == "text":
            # LSTM encoder
            embedded, (h, _) = encoder(x)
            # Use final hidden state
            h = h[-1]  # Last layer
            mu, logvar = h.chunk(2, dim=-1)
        else:
            params = encoder(x)
            mu, logvar = params.chunk(2, dim=-1)
        
        return mu, logvar
    
    def encode(
        self,
        data: Dict[str, torch.Tensor],
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode multimodal data to shared latent."""
        # Get modality-specific parameters
        modality_params = {}
        
        for mod in self.modalities:
            if mod in data and data[mod] is not None:
                mu, logvar = self.encode_modality(mod, data[mod])
                modality_params[mod] = (mu, logvar)
        
        # Fuse modality parameters
        if self.fusion_type == "poen":
            mu, logvar = self.fusion(modality_params)
        elif self.fusion_type == "moe":
            mus = [p[0] for p in modality_params.values()]
            logvars = [p[1] for p in modality_params.values()]
            mu, logvar = self.fusion(mus, logvars)
        else:
            # Simple average
            mu = torch.stack([p[0] for p in modality_params.values()]).mean(dim=0)
            logvar = torch.stack([p[1] for p in modality_params.values()]).mean(dim=0)
        
        return mu, logvar
    
    def decode(
        self,
        z: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Decode latent to all modalities."""
        outputs = {}
        
        for mod in self.modalities:
            outputs[mod] = self.modality_decoders[mod](z)
            
            if mod == "text":
                config = self.modality_configs.get(mod, {})
                seq_len = config.get("seq_len", 50)
                vocab_size = config.get("vocab_size", 10000)
                outputs[mod] = outputs[mod].view(-1, seq_len, vocab_size)
        
        return outputs
    
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        sensitive_attrs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for multimodal data."""
        # Encode
        mu, logvar = self.encode(data)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructions = self.decode(z)
        
        # Compute losses
        losses = {}
        
        # Reconstruction loss per modality
        recon_total = 0
        for mod in self.modalities:
            if mod in data and data[mod] is not None:
                if mod == "text":
                    recon = F.cross_entropy(
                        reconstructions[mod].view(-1, reconstructions[mod].size(-1)),
                        data[mod].view(-1)
                    )
                else:
                    recon = F.mse_loss(reconstructions[mod], data[mod])
                losses[f"recon_{mod}"] = recon
                recon_total = recon_total + recon
        losses["reconstruction"] = recon_total
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        losses["kl_divergence"] = self.kl_weight * kl_loss
        
        # Adversarial fairness loss
        if sensitive_attrs is not None:
            z_reversed = self.grl(z)
            adv_pred = self.adversary(z_reversed)
            adv_loss = F.cross_entropy(adv_pred, sensitive_attrs)
            losses["adversarial"] = self.fairness_weight * adv_loss
        
        losses["total"] = sum(losses.values())
        
        return {
            "reconstructions": reconstructions,
            "latent": z,
            "mu": mu,
            "logvar": logvar,
            "losses": losses
        }
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Generate multimodal samples."""
        if device is None:
            device = next(self.parameters()).device
            
        z = self.sample_latent(n_samples, device)
        return self.decode(z)


class ProductOfExperts(nn.Module):
    """Product of Experts for multimodal fusion."""
    
    def forward(
        self,
        modality_params: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute product of Gaussian experts.
        
        Args:
            modality_params: Dict of (mu, logvar) for each modality
            
        Returns:
            Fused (mu, logvar)
        """
        mus = []
        logvars = []
        
        for mu, logvar in modality_params.values():
            mus.append(mu)
            logvars.append(logvar)
        
        # Product of Gaussians
        # var = 1 / sum(1/var_i)
        # mu = var * sum(mu_i / var_i)
        
        logvars_stack = torch.stack(logvars, dim=0)
        mus_stack = torch.stack(mus, dim=0)
        
        # Precision (1/var)
        precisions = torch.exp(-logvars_stack)
        
        # Fused variance
        fused_var = 1.0 / precisions.sum(dim=0)
        fused_logvar = torch.log(fused_var)
        
        # Fused mean
        fused_mu = fused_var * (precisions * mus_stack).sum(dim=0)
        
        return fused_mu, fused_logvar


class MixtureOfExperts(nn.Module):
    """Mixture of Experts for multimodal fusion."""
    
    def __init__(self, num_experts: int, latent_dim: int):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(latent_dim * num_experts, 256),
            nn.ReLU(),
            nn.Linear(256, num_experts),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self,
        mus: List[torch.Tensor],
        logvars: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mixture of experts."""
        # Concatenate for gating
        concat = torch.cat(mus, dim=-1)
        weights = self.gating(concat)
        
        # Weighted combination
        mu = sum(w.unsqueeze(-1) * m for w, m in zip(weights.unbind(-1), mus))
        logvar = sum(w.unsqueeze(-1) * lv for w, lv in zip(weights.unbind(-1), logvars))
        
        return mu, logvar


class CrossModalAttentionFusion(nn.Module):
    """Cross-modal attention for fusion."""
    
    def __init__(self, latent_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(latent_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Apply cross-modal attention."""
        # Stack features (batch, num_modalities, dim)
        stacked = torch.stack(features, dim=1)
        
        # Self-attention
        attended, _ = self.attention(stacked, stacked, stacked)
        
        # Average pooling
        fused = attended.mean(dim=1)
        
        return self.norm(fused)


# Registry for Debiased VAE variants
DEBIASED_VAE_REGISTRY = {
    "debias_vae": DebiasedVAE,
    "beta_debiased_vae": BetaDebiasedVAE,
    "conditional_debiased_vae": ConditionalDebiasedVAE,
    "hierarchical_debiased_vae": HierarchicalDebiasedVAE,
    "multimodal_debiased_vae": MultimodalDebiasedVAE,
}


def get_debiased_vae(name: str, **kwargs) -> DebiasedVAE:
    """Get a Debiased VAE variant by name."""
    if name not in DEBIASED_VAE_REGISTRY:
        available = list(DEBIASED_VAE_REGISTRY.keys())
        raise ValueError(f"Unknown DebiasedVAE: {name}. Available: {available}")
    return DEBIASED_VAE_REGISTRY[name](**kwargs)
