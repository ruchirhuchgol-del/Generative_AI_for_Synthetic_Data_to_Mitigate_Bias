"""
Base Encoder
============

Abstract base classes for all encoder modules including standard and VAE-style encoders.
Provides common interface for encoding data into latent representations with fairness support.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base_module import BaseModule


class BaseEncoder(BaseModule):
    """
    Base class for all encoder modules.
    
    Encoders transform input data into latent representations. Each modality 
    (tabular, text, image) has its own encoder implementation.
    
    This base class provides:
    - Common interface for encoding
    - Parameter counting and device management
    - Fairness-aware encoding hooks
    - Checkpoint save/load
    
    Subclasses must implement:
    - encode(): Core encoding logic
    - input_dim property: Input dimension specification
    - latent_dim property: Output latent dimension
    
    Example:
        >>> encoder = MyEncoder(latent_dim=512)
        >>> z = encoder.encode(x)
        >>> print(f"Encoded to shape: {z.shape}")
    """
    
    def __init__(
        self,
        name: str = "encoder",
        latent_dim: int = 512,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the encoder.
        
        Args:
            name: Encoder name for identification
            latent_dim: Dimension of the latent space
            config: Optional configuration dictionary
        """
        super().__init__(name=name, config=config)
        self._latent_dim = latent_dim
        
        # Fairness hooks
        self._fairness_hooks: List[nn.Module] = []
        
    @property
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        return self._latent_dim
    
    @abstractmethod
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode input tensor to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, *)
            
        Returns:
            Latent tensor of shape (batch_size, latent_dim)
        """
        pass
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass calls encode."""
        return self.encode(x, **kwargs)
    
    @property
    @abstractmethod
    def input_dim(self) -> Union[int, Tuple[int, ...]]:
        """Return the dimension of the input."""
        pass
    
    def add_fairness_hook(self, hook: nn.Module) -> None:
        """
        Add a fairness-related hook module.
        
        Fairness hooks can be used to:
        - Remove sensitive information from latent codes
        - Apply adversarial debiasing
        - Add fairness regularization
        
        Args:
            hook: Hook module to add
        """
        self._fairness_hooks.append(hook)
        
    def apply_fairness_hooks(self, z: torch.Tensor) -> torch.Tensor:
        """
        Apply all fairness hooks to latent representation.
        
        Args:
            z: Latent tensor
            
        Returns:
            Transformed latent tensor
        """
        for hook in self._fairness_hooks:
            z = hook(z)
        return z
    
    def get_latent_statistics(self, z: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics of the latent representation.
        
        Args:
            z: Latent tensor
            
        Returns:
            Dictionary of statistics (mean, std, norm, etc.)
        """
        with torch.no_grad():
            return {
                "mean": z.mean().item(),
                "std": z.std().item(),
                "norm_mean": z.norm(dim=1).mean().item(),
                "norm_std": z.norm(dim=1).std().item(),
                "max": z.max().item(),
                "min": z.min().item(),
            }


class VAEEncoder(BaseEncoder):
    """
    VAE-style encoder that outputs parameters of a Gaussian distribution.
    
    Outputs mean (μ) and log-variance (log σ²) for a Gaussian distribution 
    in latent space. Uses the reparameterization trick for differentiable 
    sampling.
    
    Architecture:
        Input -> Feature Encoder -> fc_mu -> μ
                              -> fc_logvar -> log σ²
                              
    Sampling:
        z = μ + σ * ε, where ε ~ N(0, I)
        
    Example:
        >>> encoder = MyVAEEncoder(latent_dim=512)
        >>> mu, logvar = encoder.encode(x)
        >>> z = encoder.reparameterize(mu, logvar)
    """
    
    def __init__(
        self,
        name: str = "vae_encoder",
        latent_dim: int = 512,
        config: Optional[Dict[str, Any]] = None,
        min_logvar: float = -10.0,
        max_logvar: float = 10.0
    ):
        """
        Initialize the VAE encoder.
        
        Args:
            name: Encoder name
            latent_dim: Dimension of the latent space
            config: Optional configuration dictionary
            min_logvar: Minimum value for log-variance (numerical stability)
            max_logvar: Maximum value for log-variance
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        
        # Mean and log-variance heads (to be created by subclass)
        self.fc_mu: Optional[nn.Linear] = None
        self.fc_logvar: Optional[nn.Linear] = None
        
    def _create_heads(self, feature_dim: int) -> None:
        """
        Create the mean and log-variance projection heads.
        
        Args:
            feature_dim: Dimension of features before projection
        """
        self.fc_mu = nn.Linear(feature_dim, self._latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, self._latent_dim)
        
    def encode(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean, log_variance) tensors
        """
        h = self._encode_features(x, **kwargs)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        
        return mu, logvar
    
    @abstractmethod
    def _encode_features(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode input to intermediate features.
        
        Must be implemented by subclasses to define the encoder architecture.
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor before the mu/logvar projections
        """
        pass
    
    def reparameterize(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        During training:
            z = μ + σ * ε, where ε ~ N(0, I)
        During inference:
            z = μ (deterministic)
            
        Args:
            mu: Mean tensor
            logvar: Log-variance tensor
            training: Whether in training mode
            
        Returns:
            Sampled latent tensor
        """
        if training and self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def sample(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Sample multiple latent vectors from the distribution.
        
        Args:
            mu: Mean tensor
            logvar: Log-variance tensor
            n_samples: Number of samples per input
            
        Returns:
            Sampled tensors of shape (batch * n_samples, latent_dim)
        """
        batch_size = mu.shape[0]
        
        # Expand for multiple samples
        mu_exp = mu.unsqueeze(1).expand(-1, n_samples, -1)
        logvar_exp = logvar.unsqueeze(1).expand(-1, n_samples, -1)
        
        std = torch.exp(0.5 * logvar_exp)
        eps = torch.randn_like(std)
        
        samples = (mu_exp + eps * std).reshape(batch_size * n_samples, -1)
        
        return samples
    
    def kl_divergence(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute KL divergence from N(0, I).
        
        KL(N(μ, σ²) || N(0, I)) = -0.5 * sum(1 + log σ² - μ² - σ²)
        
        Args:
            mu: Mean tensor
            logvar: Log-variance tensor
            reduction: Reduction method ('mean', 'sum', 'none')
            
        Returns:
            KL divergence
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        
        if reduction == "mean":
            return kl.mean()
        elif reduction == "sum":
            return kl.sum()
        return kl
    
    def forward(
        self, 
        x: torch.Tensor, 
        sample: bool = True,
        return_params: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass with optional sampling.
        
        Args:
            x: Input tensor
            sample: If True, sample from distribution; else return mean
            return_params: If True, also return mu and logvar
            
        Returns:
            If return_params: (z, mu, logvar)
            Else: z (sampled or mean)
        """
        mu, logvar = self.encode(x)
        
        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
            
        if return_params:
            return z, mu, logvar
        return z


class DiagonalGaussianEncoder(VAEEncoder):
    """
    VAE encoder with diagonal Gaussian distribution assumption.
    
    The latent distribution is modeled as a product of independent
    univariate Gaussians (diagonal covariance matrix).
    
    This is the standard VAE assumption and enables tractable KL computation.
    """
    
    def __init__(
        self,
        name: str = "diagonal_gaussian_encoder",
        latent_dim: int = 512,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
    def log_prob(
        self, 
        z: torch.Tensor, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probability of z under N(μ, diag(σ²)).
        
        Args:
            z: Latent sample
            mu: Mean
            logvar: Log-variance
            
        Returns:
            Log probability
        """
        log_2pi = math.log(2 * math.pi)
        
        log_prob = -0.5 * (
            log_2pi + 
            logvar + 
            (z - mu).pow(2) / logvar.exp()
        )
        
        return log_prob.sum(dim=-1)


class HierarchicalVAEEncoder(VAEEncoder):
    """
    Hierarchical VAE encoder with multiple latent levels.
    
    Enables learning hierarchical latent representations where
    each level captures different levels of abstraction.
    
    Architecture:
        Input -> L1 Features -> μ₁, σ₁
        L1 Features -> L2 Features -> μ₂, σ₂
        ...
        
    Useful for:
    - Multi-scale feature learning
    - Hierarchical generation
    - Better latent space organization
    """
    
    def __init__(
        self,
        name: str = "hierarchical_vae_encoder",
        latent_dims: List[int] = [256, 512, 1024],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hierarchical encoder.
        
        Args:
            name: Encoder name
            latent_dims: List of latent dimensions for each level
            config: Optional configuration
        """
        # Use largest latent dim as the output dimension
        super().__init__(
            name=name, 
            latent_dim=latent_dims[-1], 
            config=config
        )
        
        self.latent_dims = latent_dims
        self.num_levels = len(latent_dims)
        
        # Create projection heads for each level
        self.fc_mu_levels = nn.ModuleList()
        self.fc_logvar_levels = nn.ModuleList()
        
    def _create_level_heads(self, feature_dims: List[int]) -> None:
        """
        Create projection heads for each level.
        
        Args:
            feature_dims: List of feature dimensions before each level
        """
        for i, (feat_dim, lat_dim) in enumerate(zip(feature_dims, self.latent_dims)):
            self.fc_mu_levels.append(nn.Linear(feat_dim, lat_dim))
            self.fc_logvar_levels.append(nn.Linear(feat_dim, lat_dim))
            
    def encode(
        self, 
        x: torch.Tensor, 
        **kwargs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Encode to hierarchical latent parameters.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (list of mus, list of logvars)
        """
        features = self._encode_hierarchical_features(x, **kwargs)
        
        mus = []
        logvars = []
        
        for i, feat in enumerate(features):
            mu = self.fc_mu_levels[i](feat)
            logvar = self.fc_logvar_levels[i](feat)
            logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
            mus.append(mu)
            logvars.append(logvar)
            
        return mus, logvars
    
    @abstractmethod
    def _encode_hierarchical_features(
        self, 
        x: torch.Tensor, 
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Encode to hierarchical features for each level.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature tensors, one per level
        """
        pass
    
    def reparameterize(
        self,
        mus: List[torch.Tensor],
        logvars: List[torch.Tensor],
        training: bool = True
    ) -> List[torch.Tensor]:
        """
        Reparameterize each level.
        Args:
            mus: List of mean tensors
            logvars: List of log-variance tensors
            training: Training mode flag  
        Returns:
            List of sampled latent tensors
        """
        samples = []
        
        for mu, logvar in zip(mus, logvars):
            if training and self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                samples.append(mu + eps * std)
            else:
                samples.append(mu)
                
        return samples
    
    def kl_divergence(
        self,
        mus: List[torch.Tensor],
        logvars: List[torch.Tensor],
        weights: Optional[List[float]] = None,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute weighted KL divergence across levels.
        Args:
            mus: List of mean tensors
            logvars: List of log-variance tensors
            weights: Optional weights for each level
            reduction: Reduction method
            
        Returns:
            Total KL divergence
        """
        if weights is None:
            weights = [1.0] * self.num_levels
            
        total_kl = 0.0
        
        for mu, logvar, weight in zip(mus, logvars, weights):
            kl = super().kl_divergence(mu, logvar, reduction="none")
            total_kl = total_kl + weight * kl
            
        if reduction == "mean":
            return total_kl.mean()
        elif reduction == "sum":
            return total_kl.sum()
        return total_kl
    
    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
        return_params: bool = False
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]]:
        """Forward pass through hierarchical encoder."""
        mus, logvars = self.encode(x)
        samples = self.reparameterize(mus, logvars)
        
        if return_params:
            return samples, mus, logvars
        return samples


def get_activation(name: str) -> nn.Module:
    """
    Get activation function by name.
    Args:
        name: Activation function name   
    Returns:
        Activation module
    """
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.2),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "mish": nn.Mish(),
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    
    return activations[name]


def get_normalization(name: str, num_features: int) -> nn.Module:
    """
    Get normalization layer by name.
    Args:
        name: Normalization type
        num_features: Number of features/channels 
    Returns:
        Normalization module
    """
    if name == "batch_norm":
        return nn.BatchNorm1d(num_features)
    elif name == "layer_norm":
        return nn.LayerNorm(num_features)
    elif name == "instance_norm":
        return nn.InstanceNorm1d(num_features)
    elif name == "group_norm":
        return nn.GroupNorm(min(32, num_features // 4), num_features)
    elif name == "none" or name is None:
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization: {name}")
