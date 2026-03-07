v"""
Base Generator
==============

Abstract base classes for all generative models.
Provides common interface for synthetic data generation with fairness support.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base_module import BaseModule


class BaseGenerator(BaseModule):
    """
    Base class for all generative models.
    
    Generators create synthetic data samples. They combine encoders and decoders
    with various generative mechanisms (VAE, GAN, Diffusion).
    
    This base class provides:
    - Common interface for generation
    - Latent space management
    - Conditional generation support
    - Fairness-aware generation hooks
    
    Subclasses must implement:
    - generate(): Core generation logic
    - sample_latent(): Latent sampling strategy
    - reconstruction_loss(): Reconstruction loss for training
    
    Example:
        >>> generator = MyGenerator(latent_dim=512)
        >>> samples = generator.generate(100)
        >>> print(f"Generated {samples.shape}")
    """
    
    def __init__(
        self,
        name: str = "generator",
        latent_dim: int = 512,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the generator.
        
        Args:
            name: Generator name for identification
            latent_dim: Dimension of the latent space
            config: Optional configuration dictionary
        """
        super().__init__(name=name, config=config)
        self._latent_dim = latent_dim
        
        # Training state
        self._is_trained = False
        self._global_step = 0
        
    @property
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        return self._latent_dim
    
    @abstractmethod
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples to generate
            conditions: Optional conditioning dictionary
            **kwargs: Additional generation parameters
            
        Returns:
            Generated samples tensor or dictionary
        """
        pass
    
    @abstractmethod
    def sample_latent(
        self, 
        n_samples: int, 
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample from the latent prior distribution.
        
        Args:
            n_samples: Number of samples
            device: Device to create samples on
            
        Returns:
            Latent samples tensor of shape (n_samples, latent_dim)
        """
        pass
    
    def encode(
        self, 
        x: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Not all generators support encoding (e.g., pure GANs).
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
            
        Raises:
            NotImplementedError: If generator doesn't support encoding
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support encoding. "
            "Use a VAE-based generator for reconstruction."
        )
    
    def decode(
        self, 
        z: torch.Tensor, 
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Decode latent to output space.
        
        Args:
            z: Latent tensor
            
        Returns:
            Decoded output
            
        Raises:
            NotImplementedError: If not implemented
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support direct decoding."
        )
    
    def reconstruct(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[Union[torch.Tensor, Dict[str, torch.Tensor]], torch.Tensor]:
        """
        Reconstruct input through encode-decode cycle.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed tensor, latent representation)
            
        Raises:
            NotImplementedError: If generator doesn't support reconstruction
        """
        z = self.encode(x, **kwargs)
        x_recon = self.decode(z, **kwargs)
        return x_recon, z
    
    @property
    def is_trained(self) -> bool:
        """Check if generator has been trained."""
        return self._is_trained
    
    def set_trained(self, trained: bool = True) -> None:
        """Set the training status."""
        self._is_trained = trained
        
    def get_generation_config(self) -> Dict[str, Any]:
        """
        Get default generation configuration.
        
        Returns:
            Dictionary of generation parameters
        """
        return {
            "latent_dim": self._latent_dim,
            "device": str(self.device) if hasattr(self, 'device') else "cpu",
        }


class ConditionalGenerator(BaseGenerator):
    """
    Base class for conditional generators.
    
    Supports conditioning on:
    - Class labels
    - Continuous attributes
    - Sensitive attributes (for fair generation)
    
    Example:
        >>> generator = MyConditionalGenerator(
        ...     latent_dim=512,
        ...     num_classes=10
        ... )
        >>> samples = generator.generate(100, class_label=3)
    """
    
    def __init__(
        self,
        name: str = "conditional_generator",
        latent_dim: int = 512,
        num_classes: Optional[int] = None,
        condition_dim: Optional[int] = None,
        sensitive_attributes: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize conditional generator.
        
        Args:
            name: Generator name
            latent_dim: Latent dimension
            num_classes: Number of classes for class-conditional generation
            condition_dim: Dimension of continuous condition
            sensitive_attributes: List of sensitive attribute names
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.sensitive_attributes = sensitive_attributes or []
        
        # Build condition embeddings if needed
        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, latent_dim)
        else:
            self.class_embedding = None
            
        if condition_dim is not None:
            self.condition_projection = nn.Sequential(
                nn.Linear(condition_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.GELU()
            )
        else:
            self.condition_projection = None
    
    def combine_latent_condition(
        self,
        z: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Combine latent with condition.
        
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
    
    def generate_counterfactual(
        self,
        z: torch.Tensor,
        original_sensitive: torch.Tensor,
        target_sensitive: torch.Tensor,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate counterfactual samples.
        
        Change sensitive attribute while keeping other attributes.
        
        Args:
            z: Original latent
            original_sensitive: Original sensitive values
            target_sensitive: Target sensitive values
            
        Returns:
            Counterfactual samples
        """
        raise NotImplementedError(
            "Counterfactual generation must be implemented by subclass."
        )


class FairGenerator(BaseGenerator):
    """
    Base class for fairness-aware generators.
    
    Integrates fairness constraints into the generation process:
    - Group fairness: Demographic parity, equalized odds
    - Individual fairness: Similar individuals get similar outcomes
    - Counterfactual fairness: Independence from sensitive attributes
    
    Example:
        >>> generator = MyFairGenerator(
        ...     latent_dim=512,
        ...     sensitive_attributes=["gender", "race"]
        ... )
        >>> samples = generator.generate_fair(100)
    """
    
    def __init__(
        self,
        name: str = "fair_generator",
        latent_dim: int = 512,
        sensitive_attributes: Optional[List[str]] = None,
        fairness_weights: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize fair generator.
        
        Args:
            name: Generator name
            latent_dim: Latent dimension
            sensitive_attributes: List of sensitive attribute names
            fairness_weights: Weights for different fairness constraints
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.sensitive_attributes = sensitive_attributes or []
        self.fairness_weights = fairness_weights or {
            "demographic_parity": 0.1,
            "equalized_odds": 0.1,
            "counterfactual": 0.1
        }
        
        # Adversary for debiasing (optional)
        self._adversary = None
        
    def set_adversary(self, adversary: nn.Module) -> None:
        """
        Set the adversary network for adversarial debiasing.
        
        Args:
            adversary: Adversary network that predicts sensitive attributes
        """
        self._adversary = adversary
        
    def compute_fairness_penalty(
        self,
        samples: torch.Tensor,
        sensitive: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute fairness penalty for generated samples.
        
        Args:
            samples: Generated samples
            sensitive: Sensitive attribute values
            
        Returns:
            Fairness penalty tensor
        """
        # Base implementation - override in subclass
        penalty = torch.tensor(0.0, device=samples.device)
        return penalty
    
    def generate_fair(
        self,
        n_samples: int,
        target_distribution: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Generate fair synthetic samples.
        
        Args:
            n_samples: Number of samples
            target_distribution: Target distribution for sensitive attributes
            **kwargs: Additional arguments
            
        Returns:
            Fair synthetic samples
        """
        # Base implementation - uses standard generate
        return self.generate(n_samples, **kwargs)


class LatentSpaceMixin:
    """
    Mixin providing latent space manipulation utilities.
    """
    
    def interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        steps: int = 10
    ) -> torch.Tensor:
        """
        Interpolate between two latent codes.
        
        Args:
            z1: First latent code
            z2: Second latent code
            steps: Number of interpolation steps
            
        Returns:
            Interpolated latent codes (steps, latent_dim)
        """
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        interpolations = []
        
        for alpha in alphas:
            z = (1 - alpha) * z1 + alpha * z2
            interpolations.append(z)
            
        return torch.stack(interpolations)
    
    def spherical_interpolate(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        steps: int = 10
    ) -> torch.Tensor:
        """
        Spherical interpolation (SLERP) between latent codes.
        
        Better for high-dimensional latent spaces.
        
        Args:
            z1: First latent code
            z2: Second latent code
            steps: Number of interpolation steps
            
        Returns:
            Spherically interpolated latent codes
        """
        # Normalize
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)
        
        # Compute omega
        omega = torch.arccos(torch.clamp((z1_norm * z2_norm).sum(dim=-1), -1, 1))
        
        # SLERP
        alphas = torch.linspace(0, 1, steps, device=z1.device)
        interpolations = []
        
        for alpha in alphas:
            if omega < 1e-6:
                z = (1 - alpha) * z1 + alpha * z2
            else:
                z = (
                    torch.sin((1 - alpha) * omega) / torch.sin(omega) * z1 +
                    torch.sin(alpha * omega) / torch.sin(omega) * z2
                )
            interpolations.append(z)
            
        return torch.stack(interpolations)
    
    def perturb_latent(
        self,
        z: torch.Tensor,
        noise_scale: float = 0.1
    ) -> torch.Tensor:
        """
        Perturb latent code with noise.
        
        Args:
            z: Latent code
            noise_scale: Scale of perturbation noise
            
        Returns:
            Perturbed latent code
        """
        noise = torch.randn_like(z) * noise_scale
        return z + noise
    
    def get_latent_direction(
        self,
        attribute: str,
        z_pos: torch.Tensor,
        z_neg: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute a direction in latent space for an attribute.
        
        Args:
            attribute: Attribute name
            z_pos: Latent codes with positive attribute
            z_neg: Latent codes with negative attribute
            
        Returns:
            Direction vector (normalized)
        """
        direction = z_pos.mean(dim=0) - z_neg.mean(dim=0)
        return F.normalize(direction, dim=0)
    
    def edit_latent(
        self,
        z: torch.Tensor,
        direction: torch.Tensor,
        strength: float = 1.0
    ) -> torch.Tensor:
        """
        Edit latent code along a direction.
        
        Args:
            z: Latent code
            direction: Edit direction
            strength: Edit strength
            
        Returns:
            Edited latent code
        """
        return z + strength * direction


class ReconstructionMixin:
    """
    Mixin providing reconstruction utilities.
    """
    
    def reconstruction_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Compute reconstruction loss.
        
        Args:
            x: Original input
            x_recon: Reconstructed input
            reduction: Reduction method
            
        Returns:
            Reconstruction loss
        """
        if isinstance(x_recon, dict):
            # Handle multi-modal reconstruction
            loss = 0.0
            for key, recon in x_recon.items():
                if key in x:
                    loss = loss + F.mse_loss(recon, x[key], reduction=reduction)
            return loss
        else:
            return F.mse_loss(x_recon, x, reduction=reduction)


class GenerationCallback:
    """
    Callback for generation process.
    
    Can be used to:
    - Log generated samples
    - Save checkpoints
    - Apply post-processing
    """
    
    def on_generation_start(self, generator: BaseGenerator, **kwargs) -> None:
        """Called at the start of generation."""
        pass
    
    def on_sample_generated(
        self, 
        generator: BaseGenerator, 
        samples: torch.Tensor,
        step: int,
        **kwargs
    ) -> None:
        """Called after each batch of samples is generated."""
        pass
    
    def on_generation_end(
        self, 
        generator: BaseGenerator, 
        samples: torch.Tensor,
        **kwargs
    ) -> None:
        """Called at the end of generation."""
        pass


class ProgressiveGenerator(BaseGenerator):
    """
    Base class for progressive generation (e.g., progressive GAN).
    
    Supports gradual increase in resolution or complexity during training.
    """
    
    def __init__(
        self,
        name: str = "progressive_generator",
        latent_dim: int = 512,
        num_stages: int = 5,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        self.num_stages = num_stages
        self._current_stage = 0
        
    @property
    def current_stage(self) -> int:
        """Get current training stage."""
        return self._current_stage
    
    def set_stage(self, stage: int) -> None:
        """Set the current training stage."""
        if 0 <= stage < self.num_stages:
            self._current_stage = stage
        else:
            raise ValueError(f"Stage must be in [0, {self.num_stages})")
            
    def progressive_generate(
        self,
        n_samples: int,
        stage: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate at a specific stage.
        
        Args:
            n_samples: Number of samples
            stage: Stage to generate at (default: current)
            
        Returns:
            Generated samples
        """
        if stage is None:
            stage = self._current_stage
        return self.generate(n_samples, stage=stage, **kwargs)


def get_noise_schedule(
    schedule_type: str,
    num_steps: int,
    **kwargs
) -> torch.Tensor:
    """
    Get noise schedule for diffusion models.
    
    Args:
        schedule_type: Type of schedule ("linear", "cosine", "quadratic")
        num_steps: Number of diffusion steps
        
    Returns:
        Beta schedule tensor
    """
    if schedule_type == "linear":
        beta_start = kwargs.get("beta_start", 1e-4)
        beta_end = kwargs.get("beta_end", 0.02)
        betas = torch.linspace(beta_start, beta_end, num_steps)
        
    elif schedule_type == "cosine":
        s = kwargs.get("s", 0.008)
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clamp(betas, 0.0001, 0.9999)
        
    elif schedule_type == "quadratic":
        beta_start = kwargs.get("beta_start", 1e-4)
        beta_end = kwargs.get("beta_end", 0.02)
        betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps) ** 2
        
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
        
    return betas
