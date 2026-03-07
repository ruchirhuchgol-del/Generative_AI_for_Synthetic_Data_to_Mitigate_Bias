"""
Base Decoder
============

Abstract base classes for all decoder modules including standard and VAE-style decoders.
Provides common interface for decoding latent representations back to data space.
"""

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base_module import BaseModule


class BaseDecoder(BaseModule):
    """
    Base class for all decoder modules.
    
    Decoders transform latent representations back to the original data space.
    Each modality (tabular, text, image) has its own decoder implementation.
    
    This base class provides:
    - Common interface for decoding
    - Parameter counting and device management
    - Output shape management
    - Checkpoint save/load
    
    Subclasses must implement:
    - decode(): Core decoding logic
    - latent_dim property: Input latent dimension
    - output_dim property: Output dimension specification
    
    Example:
        >>> decoder = MyDecoder(latent_dim=512, output_dim=100)
        >>> x_recon = decoder.decode(z)
        >>> print(f"Reconstructed shape: {x_recon.shape}")
    """
    
    def __init__(
        self,
        name: str = "decoder",
        latent_dim: int = 512,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the decoder.
        
        Args:
            name: Decoder name for identification
            latent_dim: Dimension of the latent space
            config: Optional configuration dictionary
        """
        super().__init__(name=name, config=config)
        self._latent_dim = latent_dim
        
    @property
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        return self._latent_dim
    
    @abstractmethod
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decode latent tensor to output space.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Output tensor of shape (batch_size, *)
        """
        pass
    
    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass calls decode."""
        return self.decode(z, **kwargs)
    
    @property
    @abstractmethod
    def output_dim(self) -> Union[int, Tuple[int, ...]]:
        """Return the dimension of the output."""
        pass
    
    def get_output_statistics(self, output: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics of the decoder output.
        
        Args:
            output: Decoder output tensor
            
        Returns:
            Dictionary of statistics
        """
        with torch.no_grad():
            return {
                "mean": output.mean().item(),
                "std": output.std().item(),
                "min": output.min().item(),
                "max": output.max().item(),
            }


class VAEDecoder(BaseDecoder):
    """
    VAE-style decoder that samples from latent distribution and decodes.
    
    Works with VAEEncoder to form a complete VAE. Can sample from the
    prior distribution N(0, I) for generation.
    
    Architecture:
        Latent z -> fc -> Feature Decoder -> Output
        
    Example:
        >>> decoder = MyVAEDecoder(latent_dim=512, output_dim=100)
        >>> x_recon = decoder.decode(mu, logvar)  # From encoder
        >>> x_gen = decoder.generate(10)  # Sample from prior
    """
    
    def __init__(
        self,
        name: str = "vae_decoder",
        latent_dim: int = 512,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the VAE decoder.
        
        Args:
            name: Decoder name
            latent_dim: Dimension of the latent space
            config: Optional configuration dictionary
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decode latent sample to output.
        
        Args:
            z: Latent tensor (sampled or from encoder)
            
        Returns:
            Reconstructed output
        """
        return self._decode(z, **kwargs)
    
    @abstractmethod
    def _decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Internal decode method to be implemented by subclasses.
        
        Args:
            z: Latent tensor
            
        Returns:
            Reconstructed output
        """
        pass
    
    def sample_prior(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Sample from the prior distribution N(0, I).
        
        Args:
            batch_size: Number of samples
            device: Device to create tensor on
            
        Returns:
            Latent samples of shape (batch_size, latent_dim)
        """
        if device is None:
            device = next(self.parameters()).device
            
        return torch.randn(batch_size, self._latent_dim, device=device)
    
    def generate(
        self, 
        n_samples: int, 
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples by decoding from the prior.
        
        Args:
            n_samples: Number of samples to generate
            device: Device for generation
            **kwargs: Additional arguments for decode
            
        Returns:
            Generated samples
        """
        z = self.sample_prior(n_samples, device)
        return self.decode(z, **kwargs)
    
    def reconstruct(
        self,
        z: torch.Tensor,
        return_latent: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Reconstruct from latent code.
        
        Args:
            z: Latent tensor
            return_latent: If True, also return the latent
            
        Returns:
            Reconstructed output (and latent if return_latent=True)
        """
        output = self.decode(z)
        if return_latent:
            return output, z
        return output


class HierarchicalVAEDecoder(VAEDecoder):
    """
    Hierarchical VAE decoder with multiple latent levels.
    
    Decodes from hierarchical latent representations where each level
    captures different levels of abstraction.
    
    Architecture:
        L_n -> L_{n-1} -> ... -> L_1 -> Output
        
    Useful for:
    - Multi-scale generation
    - Coarse-to-fine synthesis
    - Better latent space organization
    """
    
    def __init__(
        self,
        name: str = "hierarchical_vae_decoder",
        latent_dims: List[int] = [1024, 512, 256],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize hierarchical decoder.
        
        Args:
            name: Decoder name
            latent_dims: List of latent dimensions for each level (largest to smallest)
            config: Optional configuration
        """
        # Use smallest latent dim as the input dimension
        super().__init__(
            name=name, 
            latent_dim=latent_dims[-1], 
            config=config
        )
        
        self.latent_dims = latent_dims
        self.num_levels = len(latent_dims)
        
    def sample_prior(
        self, 
        batch_size: int, 
        device: Optional[torch.device] = None
    ) -> List[torch.Tensor]:
        """
        Sample from hierarchical priors.
        
        Args:
            batch_size: Number of samples
            device: Device for tensors
            
        Returns:
            List of latent samples, one per level
        """
        if device is None:
            device = next(self.parameters()).device
            
        return [
            torch.randn(batch_size, dim, device=device)
            for dim in self.latent_dims
        ]
    
    def generate(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples from hierarchical latents.
        
        Args:
            n_samples: Number of samples
            device: Device for generation
            **kwargs: Additional arguments
            
        Returns:
            Generated samples
        """
        z_list = self.sample_prior(n_samples, device)
        return self.decode(z_list, **kwargs)
    
    @abstractmethod
    def decode(self, z_list: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """
        Decode hierarchical latents to output.
        
        Args:
            z_list: List of latent tensors, one per level
            
        Returns:
            Reconstructed output
        """
        pass


class ConditionalDecoder(BaseDecoder):
    """
    Decoder with conditional generation support.
    
    Supports conditioning on:
    - Class labels (for conditional generation)
    - Continuous attributes
    - Multi-modal conditions
    
    Example:
        >>> decoder = MyConditionalDecoder(latent_dim=512, num_classes=10)
        >>> x_gen = decoder.generate(10, class_label=3)
    """
    
    def __init__(
        self,
        name: str = "conditional_decoder",
        latent_dim: int = 512,
        num_classes: Optional[int] = None,
        condition_dim: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize conditional decoder.
        
        Args:
            name: Decoder name
            latent_dim: Dimension of the latent space
            num_classes: Number of classes for class-conditional generation
            condition_dim: Dimension of continuous condition
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        
        # Build condition embedding if needed
        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, latent_dim)
        else:
            self.class_embedding = None
            
        if condition_dim is not None:
            self.condition_projection = nn.Linear(condition_dim, latent_dim)
        else:
            self.condition_projection = None
    
    def combine_latent_condition(
        self,
        z: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine latent code with condition.
        
        Args:
            z: Latent tensor
            class_label: Class label indices
            condition: Continuous condition tensor
            
        Returns:
            Combined latent representation
        """
        combined = z
        
        if class_label is not None and self.class_embedding is not None:
            class_emb = self.class_embedding(class_label)
            combined = combined + class_emb
            
        if condition is not None and self.condition_projection is not None:
            cond_emb = self.condition_projection(condition)
            combined = combined + cond_emb
            
        return combined
    
    def decode(
        self,
        z: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode with optional conditioning.
        
        Args:
            z: Latent tensor
            class_label: Optional class label
            condition: Optional continuous condition
            **kwargs: Additional arguments
            
        Returns:
            Reconstructed output
        """
        z_combined = self.combine_latent_condition(z, class_label, condition)
        return self._decode(z_combined, **kwargs)
    
    @abstractmethod
    def _decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Internal decode method.
        
        Args:
            z: Combined latent tensor
            
        Returns:
            Reconstructed output
        """
        pass
    
    def generate(
        self,
        n_samples: int,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate conditional samples.
        
        Args:
            n_samples: Number of samples
            class_label: Class labels for generation
            condition: Continuous conditions
            device: Device for generation
            **kwargs: Additional arguments
            
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
            
        z = torch.randn(n_samples, self._latent_dim, device=device)
        return self.decode(z, class_label, condition, **kwargs)


class DiffusionDecoder(BaseDecoder):
    """
    Decoder with diffusion-style time conditioning.
    
    Used in diffusion models where the decoder (denoiser) is conditioned
    on the noise timestep.
    
    Architecture:
        Noisy Input + Time Embedding -> Denoiser -> Clean Output
        
    Example:
        >>> decoder = MyDiffusionDecoder(latent_dim=512)
        >>> x_clean = decoder(x_noisy, timestep=100)
    """
    
    def __init__(
        self,
        name: str = "diffusion_decoder",
        latent_dim: int = 512,
        time_embed_dim: int = 512,
        num_timesteps: int = 1000,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize diffusion decoder.
        
        Args:
            name: Decoder name
            latent_dim: Dimension of the latent space
            time_embed_dim: Dimension of time embedding
            num_timesteps: Number of diffusion timesteps
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.time_embed_dim = time_embed_dim
        self.num_timesteps = num_timesteps
        
        # Time embedding
        self.time_embed = nn.Sequential(
            TimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
    
    def get_time_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Compute time embedding for given timesteps.
        
        Args:
            timestep: Timestep tensor (batch,)
            
        Returns:
            Time embedding (batch, time_embed_dim)
        """
        return self.time_embed(timestep)
    
    @abstractmethod
    def decode(
        self, 
        z: torch.Tensor, 
        timestep: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode with time conditioning.
        
        Args:
            z: Latent tensor
            timestep: Diffusion timestep
            **kwargs: Additional arguments
            
        Returns:
            Decoded output
        """
        pass


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for diffusion models.
    
    PE(t, 2i) = sin(t / 10000^(2i/dim))
    PE(t, 2i+1) = cos(t / 10000^(2i/dim))
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