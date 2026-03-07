"""
Base Discriminator
==================

Abstract base classes for discriminator modules used in adversarial training.
Provides interfaces for GAN discriminators and fairness adversaries.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base_module import BaseModule


class BaseDiscriminator(BaseModule):
    """
    Base class for all discriminator modules.
    
    Discriminators are used in:
    - GAN training: Distinguishing real vs fake samples
    - Fairness adversarial: Predicting sensitive attributes from representations
    - Domain adaptation: Distinguishing source vs target domains
    
    This base class provides:
    - Common interface for discrimination
    - Probability output methods
    - Feature extraction hooks
    - Gradient penalty support
    
    Example:
        >>> discriminator = MyDiscriminator(input_dim=512)
        >>> logits = discriminator(x)
        >>> probs = discriminator.get_probability(x)
    """
    
    def __init__(
        self,
        name: str = "discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the discriminator.
        
        Args:
            name: Discriminator name for identification
            config: Optional configuration dictionary
        """
        super().__init__(name=name, config=config)
        
        # Feature extraction hooks
        self._feature_hooks: List[torch.Tensor] = []
        self._hook_handles = []
        
    @abstractmethod
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute discriminator output.
        
        Args:
            x: Input tensor
            
        Returns:
            Discriminator output (logits or features)
        """
        pass
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass calls discriminate."""
        return self.discriminate(x, **kwargs)
    
    def get_probability(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Get probability output using sigmoid or softmax.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor
        """
        logits = self.discriminate(x, **kwargs)
        if logits.dim() == 1 or logits.shape[-1] == 1:
            return torch.sigmoid(logits)
        else:
            return F.softmax(logits, dim=-1)
    
    def get_binary_prediction(
        self, 
        x: torch.Tensor, 
        threshold: float = 0.5,
        **kwargs
    ) -> torch.Tensor:
        """
        Get binary classification output.
        
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0 or 1)
        """
        probs = self.get_probability(x, **kwargs)
        return (probs > threshold).float()
    
    def enable_feature_extraction(self) -> None:
        """Enable feature extraction hooks."""
        self._feature_hooks = []
        
        def hook_fn(module, input, output):
            self._feature_hooks.append(output)
        
        # Register hooks on linear and conv layers
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handle = module.register_forward_hook(hook_fn)
                self._hook_handles.append(handle)
    
    def disable_feature_extraction(self) -> None:
        """Disable and clear feature extraction hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        self._feature_hooks = []
    
    def get_features(self) -> List[torch.Tensor]:
        """
        Get extracted features from last forward pass.
        
        Returns:
            List of feature tensors from each hooked layer
        """
        return self._feature_hooks
    
    def compute_gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        lambda_gp: float = 10.0
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.
        
        Args:
            real: Real samples
            fake: Fake samples
            lambda_gp: Gradient penalty weight
            
        Returns:
            Gradient penalty loss
        """
        batch_size = real.shape[0]
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, device=real.device)
        while alpha.dim() < real.dim():
            alpha = alpha.unsqueeze(-1)
        
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)
        
        # Forward pass
        d_interpolated = self.discriminate(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten and compute penalty
        gradients = gradients.flatten(1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return lambda_gp * penalty


class BinaryDiscriminator(BaseDiscriminator):
    """
    Binary discriminator for real/fake classification.
    
    Outputs single logit for binary classification.
    Used in standard GAN training.
    
    Example:
        >>> discriminator = BinaryDiscriminator(input_dim=512, hidden_dims=[256, 128])
        >>> logits = discriminator(x)  # Shape: (batch, 1)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        spectral_norm: bool = False,
        name: str = "binary_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize binary discriminator.
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout rate
            spectral_norm: Apply spectral normalization
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.input_dim = input_dim
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            linear = nn.Linear(prev_dim, dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LayerNorm(dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.layers = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(prev_dim, 1)
        if spectral_norm:
            self.output = nn.utils.spectral_norm(self.output)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute binary discrimination output.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Logits of shape (batch, 1)
        """
        h = self.layers(x)
        return self.output(h)


class MultiClassDiscriminator(BaseDiscriminator):
    """
    Multi-class discriminator for classification tasks.
    
    Outputs logits for each class.
    Used in conditional GANs and multi-class discrimination.
    
    Example:
        >>> discriminator = MultiClassDiscriminator(
        ...     input_dim=512,
        ...     num_classes=10
        ... )
        >>> logits = discriminator(x)  # Shape: (batch, 10)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [256, 128],
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        spectral_norm: bool = False,
        name: str = "multiclass_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-class discriminator.
        
        Args:
            input_dim: Input dimension
            num_classes: Number of classes
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout rate
            spectral_norm: Apply spectral normalization
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            linear = nn.Linear(prev_dim, dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LayerNorm(dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.layers = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(prev_dim, num_classes)
        if spectral_norm:
            self.output = nn.utils.spectral_norm(self.output)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute multi-class discrimination output.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Logits of shape (batch, num_classes)
        """
        h = self.layers(x)
        return self.output(h)
    
    def get_class_prediction(
        self, 
        x: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Get predicted class indices.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices
        """
        logits = self.discriminate(x, **kwargs)
        return logits.argmax(dim=-1)


class ProjectionDiscriminator(BaseDiscriminator):
    """
    Projection-based discriminator for conditional GANs.
    
    Uses projection mechanism for conditioning instead of concatenation.
    More parameter-efficient for large number of classes.
    
    Architecture:
        x → Feature Extractor → h
        h → Binary Output + ⟨h, c_emb⟩
        
    Based on: "cGANs with Projection Discriminator"
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [256, 128],
        embedding_dim: Optional[int] = None,
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        spectral_norm: bool = True,
        name: str = "projection_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize projection discriminator.
        
        Args:
            input_dim: Input dimension
            num_classes: Number of classes for conditioning
            hidden_dims: Hidden layer dimensions
            embedding_dim: Class embedding dimension
            activation: Activation function
            dropout: Dropout rate
            spectral_norm: Apply spectral normalization
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim or hidden_dims[-1]
        
        # Build feature extractor
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            linear = nn.Linear(prev_dim, dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LayerNorm(dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Binary output
        self.binary_output = nn.Linear(prev_dim, 1)
        if spectral_norm:
            self.binary_output = nn.utils.spectral_norm(self.binary_output)
        
        # Class embedding for projection
        self.class_embedding = nn.Embedding(num_classes, self.embedding_dim)
        if spectral_norm:
            self.class_embedding = nn.utils.spectral_norm(self.class_embedding)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def discriminate(
        self, 
        x: torch.Tensor, 
        class_label: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute projection discriminator output.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            class_label: Class label tensor of shape (batch,)
            
        Returns:
            Logits of shape (batch, 1)
        """
        # Extract features
        h = self.feature_extractor(x)
        
        # Binary output
        out = self.binary_output(h)
        
        # Add projection term if class label provided
        if class_label is not None:
            # Project features to embedding space if needed
            if h.shape[-1] != self.embedding_dim:
                h_proj = F.linear(h, self.class_embedding.weight)
            else:
                h_proj = h
            
            # Get class embedding
            c_emb = self.class_embedding(class_label)
            
            # Inner product
            out = out + (h_proj * c_emb).sum(dim=-1, keepdim=True)
        
        return out


class PatchDiscriminator(BaseDiscriminator):
    """
    Patch discriminator for image data.
    
    Outputs a grid of predictions instead of single value.
    Used in Pix2Pix and CycleGAN.
    
    Architecture:
        Image → Conv Layers → Patch Grid of logits
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dims: List[int] = [64, 128, 256, 512],
        kernel_size: int = 4,
        activation: str = "leaky_relu",
        normalization: str = "instance",
        dropout: float = 0.0,
        spectral_norm: bool = False,
        name: str = "patch_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize patch discriminator.
        
        Args:
            input_channels: Number of input channels
            hidden_dims: Hidden channel dimensions
            kernel_size: Convolution kernel size
            activation: Activation function
            normalization: Normalization type
            dropout: Dropout rate
            spectral_norm: Apply spectral normalization
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.input_channels = input_channels
        
        # Build layers
        layers = []
        prev_channels = input_channels
        
        for i, out_channels in enumerate(hidden_dims):
            # Convolution
            conv = nn.Conv2d(
                prev_channels, out_channels, 
                kernel_size, stride=2, padding=1
            )
            if spectral_norm:
                conv = nn.utils.spectral_norm(conv)
            layers.append(conv)
            
            # Normalization
            if normalization == "instance":
                layers.append(nn.InstanceNorm2d(out_channels))
            elif normalization == "batch":
                layers.append(nn.BatchNorm2d(out_channels))
            elif normalization == "layer":
                layers.append(nn.GroupNorm(1, out_channels))
            
            # Activation
            layers.append(self._get_activation(activation))
            
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            
            prev_channels = out_channels
        
        self.layers = nn.Sequential(*layers)
        
        # Output convolution
        self.output = nn.Conv2d(prev_channels, 1, kernel_size, padding=1)
        if spectral_norm:
            self.output = nn.utils.spectral_norm(self.output)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute patch discrimination output.
        
        Args:
            x: Image tensor of shape (batch, channels, height, width)
            
        Returns:
            Patch logits of shape (batch, 1, h', w')
        """
        h = self.layers(x)
        return self.output(h)
    
    def get_probability(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Get probability map for each patch.
        
        Args:
            x: Image tensor
            
        Returns:
            Probability map
        """
        logits = self.discriminate(x, **kwargs)
        return torch.sigmoid(logits)


class SpectralNormDiscriminator(BaseDiscriminator):
    """
    Discriminator with spectral normalization for stable GAN training.
    
    Spectral normalization constrains the Lipschitz constant,
    improving training stability.
    
    Based on: "Spectral Normalization for Generative Adversarial Networks"
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 1,
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        name: str = "spectral_norm_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize spectral norm discriminator.
        
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (1 for binary)
            activation: Activation function
            dropout: Dropout rate
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build layers with spectral norm
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            # Apply spectral norm to linear layer
            linear = nn.utils.spectral_norm(nn.Linear(prev_dim, dim))
            layers.append(linear)
            layers.append(nn.LayerNorm(dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.layers = nn.Sequential(*layers)
        
        # Output layer with spectral norm
        self.output = nn.utils.spectral_norm(nn.Linear(prev_dim, output_dim))
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute discriminator output."""
        h = self.layers(x)
        return self.output(h)


class EnsembleDiscriminator(BaseDiscriminator):
    """
    Ensemble of multiple discriminators.
    
    Combines multiple discriminator outputs for improved performance.
    Used in BigGAN and other large-scale GANs.
    """
    
    def __init__(
        self,
        discriminators: List[nn.Module],
        aggregation: str = "mean",  # mean, max, vote
        name: str = "ensemble_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ensemble discriminator.
        
        Args:
            discriminators: List of discriminator modules
            aggregation: How to combine outputs
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.discriminators = nn.ModuleList(discriminators)
        self.aggregation = aggregation
    
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute ensemble discrimination output.
        
        Args:
            x: Input tensor
            
        Returns:
            Aggregated logits
        """
        outputs = [d(x, **kwargs) for d in self.discriminators]
        outputs = torch.stack(outputs, dim=0)
        
        if self.aggregation == "mean":
            return outputs.mean(dim=0)
        elif self.aggregation == "max":
            return outputs.max(dim=0)[0]
        elif self.aggregation == "sum":
            return outputs.sum(dim=0)
        else:
            return outputs.mean(dim=0)
    
    def get_individual_outputs(
        self, 
        x: torch.Tensor, 
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Get outputs from each discriminator.
        
        Args:
            x: Input tensor
            
        Returns:
            List of individual outputs
        """
        return [d(x, **kwargs) for d in self.discriminators]
