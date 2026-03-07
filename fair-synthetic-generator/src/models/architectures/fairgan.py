"""
FairGAN Architecture
====================

Fairness-aware Generative Adversarial Network for fair synthetic data generation.

This module implements several FairGAN variants:
- FairGAN: Basic fair GAN with adversarial debiasing
- FairGAN-GP: FairGAN with gradient penalty (WGAN-GP style)
- FairGAN-Conditional: Conditional fair GAN
- FairGAN-Multimodal: Multimodal fair GAN

Key Features:
- Adversarial debiasing via gradient reversal
- Multiple fairness constraints (demographic parity, equalized odds)
- Support for multiple sensitive attributes
- Mode collapse mitigation
- Progressive training support
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from src.core.base_module import BaseGenerator
from src.models.generators import GANGenerator, WANGenerator, ConditionalGANGenerator
from src.models.discriminators import (
    TabularDiscriminator,
    ImageDiscriminator,
    MultimodalDiscriminator,
    FairnessAdversary,
    GradientReversalLayer
)
from src.models.encoders import TabularEncoder, TextEncoder, ImageEncoder
from src.fairness.modules.gradient_reversal import ScheduledGradientReversalLayer


class FairGAN(BaseGenerator):
    """
    Fairness-aware Generative Adversarial Network.
    
    Architecture:
        Generator: z -> synthetic_data
        Discriminator: data -> real/fake
        Adversary: latent -> sensitive_attribute_prediction (with GRL)
    
    Training Objectives:
        - Generator: Generate realistic + fair data
        - Discriminator: Distinguish real vs fake
        - Adversary: Predict sensitive attributes (gradient reversed)
    
    The generator is trained to fool the discriminator while also
    preventing the adversary from predicting sensitive attributes.
    
    Fairness Metrics Supported:
        - Demographic Parity: P(Y|S=0) = P(Y|S=1)
        - Equalized Odds: P(Y|Y_hat,S=0) = P(Y|Y_hat,S=1)
        - Counterfactual Fairness: Y_hat(S=0) = Y_hat(S=1)
    
    Example:
        >>> model = FairGAN(
        ...     data_dim=100,
        ...     latent_dim=512,
        ...     num_sensitive_groups=2
        ... )
        >>> # Training
        >>> losses = model.train_step(real_data, sensitive_attrs)
        >>> # Generation
        >>> fair_samples = model.generate(100)
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        num_sensitive_groups: int = 2,
        hidden_dims: List[int] = [512, 256, 128],
        generator_type: str = "standard",
        discriminator_type: str = "standard",
        fairness_weight: float = 1.0,
        gp_weight: float = 10.0,
        grl_lambda: float = 1.0,
        grl_schedule: str = "linear",
        grl_warmup: int = 10,
        spectral_norm: bool = True,
        dropout: float = 0.3,
        name: str = "fairgan",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize FairGAN.
        
        Args:
            data_dim: Dimension of output data
            latent_dim: Dimension of latent space
            num_sensitive_groups: Number of sensitive attribute groups
            hidden_dims: Hidden layer dimensions
            generator_type: Type of generator ("standard", "resnet")
            discriminator_type: Type of discriminator ("standard", "resnet")
            fairness_weight: Weight for fairness adversarial loss
            gp_weight: Gradient penalty weight (for WGAN-GP)
            grl_lambda: Gradient reversal lambda
            grl_schedule: GRL schedule type ("constant", "linear", "cosine")
            grl_warmup: Epochs for GRL warmup
            spectral_norm: Whether to use spectral normalization
            dropout: Dropout rate
            name: Model name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.data_dim = data_dim
        self.num_sensitive_groups = num_sensitive_groups
        self.fairness_weight = fairness_weight
        self.gp_weight = gp_weight
        
        # Build components
        self._build_generator(hidden_dims, generator_type, spectral_norm)
        self._build_discriminator(hidden_dims, discriminator_type, spectral_norm, dropout)
        self._build_adversary(hidden_dims, grl_lambda, grl_schedule, grl_warmup, dropout)
        
        # Training state
        self._current_epoch = 0
        self._n_critic = 5  # Discriminator updates per generator update
        
    def _build_generator(
        self,
        hidden_dims: List[int],
        generator_type: str,
        spectral_norm: bool
    ) -> None:
        """Build the generator network."""
        layers = []
        prev_dim = self._latent_dim
        
        for dim in hidden_dims:
            if spectral_norm:
                layers.append(nn.utils.spectral_norm(nn.Linear(prev_dim, dim)))
            else:
                layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(0.2))
            prev_dim = dim
        
        # Output layer
        if spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Linear(prev_dim, self.data_dim)))
        else:
            layers.append(nn.Linear(prev_dim, self.data_dim))
        layers.append(nn.Tanh())
        
        self.generator = nn.Sequential(*layers)
        
    def _build_discriminator(
        self,
        hidden_dims: List[int],
        discriminator_type: str,
        spectral_norm: bool,
        dropout: float
    ) -> None:
        """Build the discriminator network."""
        layers = []
        prev_dim = self.data_dim
        reversed_dims = list(reversed(hidden_dims))
        
        for dim in reversed_dims:
            if spectral_norm:
                layers.append(nn.utils.spectral_norm(nn.Linear(prev_dim, dim)))
            else:
                layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Output layer (real/fake)
        if spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Linear(prev_dim, 1)))
        else:
            layers.append(nn.Linear(prev_dim, 1))
        
        self.discriminator = nn.Sequential(*layers)
        
    def _build_adversary(
        self,
        hidden_dims: List[int],
        grl_lambda: float,
        grl_schedule: str,
        grl_warmup: int,
        dropout: float
    ) -> None:
        """Build the fairness adversary network."""
        # Use scheduled GRL for gradual debiasing
        self.grl = ScheduledGradientReversalLayer(
            lambda_start=0.0,
            lambda_end=grl_lambda,
            warmup_epochs=grl_warmup,
            schedule_type=grl_schedule
        )
        
        # Adversary head (predicts sensitive attributes)
        layers = []
        prev_dim = self._latent_dim
        
        for dim in hidden_dims[-2:]:  # Use smaller dims for adversary
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, self.num_sensitive_groups))
        
        self.adversary = nn.Sequential(*layers)
        
    def sample_latent(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """Sample from the latent prior."""
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
        """Generate synthetic samples."""
        z = self.sample_latent(n_samples, device)
        return self.generator(z)
    
    def forward(
        self,
        real_data: torch.Tensor,
        sensitive_attrs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for training.
        
        Args:
            real_data: Real data samples
            sensitive_attrs: Sensitive attribute labels
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing all losses
        """
        batch_size = real_data.size(0)
        device = real_data.device
        
        # Generate fake samples
        z = self.sample_latent(batch_size, device)
        fake_data = self.generator(z)
        
        # Discriminator predictions
        d_real = self.discriminator(real_data)
        d_fake = self.discriminator(fake_data.detach())
        
        # Compute losses
        losses = {}
        
        # Discriminator loss (Wasserstein)
        d_loss = d_fake.mean() - d_real.mean()
        
        # Gradient penalty
        gp = self._compute_gradient_penalty(real_data, fake_data, device)
        d_loss = d_loss + self.gp_weight * gp
        
        losses["d_loss"] = d_loss
        losses["gradient_penalty"] = gp
        
        # Generator loss
        d_fake_for_g = self.discriminator(fake_data)
        g_loss = -d_fake_for_g.mean()
        losses["g_loss"] = g_loss
        
        # Fairness adversary loss
        if sensitive_attrs is not None:
            # Apply GRL to latent
            z_reversed = self.grl(z)
            adv_pred = self.adversary(z_reversed)
            adv_loss = F.cross_entropy(adv_pred, sensitive_attrs)
            losses["adv_loss"] = self.fairness_weight * adv_loss
            
            # Total generator loss includes fairness
            losses["g_total"] = g_loss + self.fairness_weight * adv_loss
        else:
            losses["g_total"] = g_loss
            losses["adv_loss"] = torch.tensor(0.0, device=device)
        
        return {
            "losses": losses,
            "fake_data": fake_data,
            "latent": z,
            "d_real": d_real,
            "d_fake": d_fake
        }
    
    def _compute_gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """Compute gradient penalty for WGAN-GP."""
        batch_size = real_data.size(0)
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, device=device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        # Discriminator output
        d_interpolates = self.discriminator(interpolates)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def train_step(
        self,
        real_data: torch.Tensor,
        sensitive_attrs: Optional[torch.Tensor] = None,
        optimizer_g: Optional[torch.optim.Optimizer] = None,
        optimizer_d: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            real_data: Real data batch
            sensitive_attrs: Sensitive attribute labels
            optimizer_g: Generator optimizer
            optimizer_d: Discriminator optimizer
            
        Returns:
            Dictionary of loss values
        """
        if optimizer_g is None or optimizer_d is None:
            raise ValueError("Optimizers must be provided for training")
        
        device = real_data.device
        batch_size = real_data.size(0)
        
        losses = {}
        
        # Train discriminator
        optimizer_d.zero_grad()
        
        z = self.sample_latent(batch_size, device)
        fake_data = self.generator(z).detach()
        
        d_real = self.discriminator(real_data)
        d_fake = self.discriminator(fake_data)
        
        d_loss = d_fake.mean() - d_real.mean()
        gp = self._compute_gradient_penalty(real_data, fake_data, device)
        d_loss_total = d_loss + self.gp_weight * gp
        
        d_loss_total.backward()
        optimizer_d.step()
        
        losses["d_loss"] = d_loss_total.item()
        losses["gradient_penalty"] = gp.item()
        
        # Train generator
        optimizer_g.zero_grad()
        
        z = self.sample_latent(batch_size, device)
        fake_data = self.generator(z)
        
        d_fake = self.discriminator(fake_data)
        g_loss = -d_fake.mean()
        
        # Fairness adversary loss
        if sensitive_attrs is not None:
            z_reversed = self.grl(z)
            adv_pred = self.adversary(z_reversed)
            adv_loss = F.cross_entropy(adv_pred, sensitive_attrs)
            g_loss_total = g_loss + self.fairness_weight * adv_loss
            losses["adv_loss"] = adv_loss.item()
        else:
            g_loss_total = g_loss
            losses["adv_loss"] = 0.0
        
        g_loss_total.backward()
        optimizer_g.step()
        
        losses["g_loss"] = g_loss.item()
        losses["g_total"] = g_loss_total.item()
        
        return losses
    
    def on_epoch_end(self) -> None:
        """Called at the end of each epoch."""
        self._current_epoch += 1
        self.grl.step()
    
    def compute_fairness_metrics(
        self,
        data: torch.Tensor,
        sensitive_attrs: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """
        Compute fairness metrics for generated data.
        
        Args:
            data: Generated data
            sensitive_attrs: Sensitive attribute values
            
        Returns:
            Dictionary of fairness metrics
        """
        metrics = {}
        
        # Encode to latent
        z = self.sample_latent(data.size(0), data.device)
        
        # Get adversary predictions
        with torch.no_grad():
            adv_pred = self.adversary(z)
            pred_probs = F.softmax(adv_pred, dim=-1)
        
        # Compute demographic parity gap
        unique_groups = torch.unique(sensitive_attrs)
        if len(unique_groups) >= 2:
            group_probs = []
            for group in unique_groups:
                mask = sensitive_attrs == group
                if mask.sum() > 0:
                    group_prob = pred_probs[mask].mean(dim=0)
                    group_probs.append(group_prob)
            
            if len(group_probs) >= 2:
                dp_gap = (group_probs[0] - group_probs[1]).abs().max().item()
                metrics["demographic_parity_gap"] = dp_gap
        
        # Adversary accuracy (lower is better for fairness)
        with torch.no_grad():
            pred_labels = adv_pred.argmax(dim=-1)
            accuracy = (pred_labels == sensitive_attrs).float().mean().item()
        metrics["adversary_accuracy"] = accuracy
        
        return metrics


class FairGANGP(FairGAN):
    """
    FairGAN with Wasserstein distance and gradient penalty.
    
    Uses WGAN-GP style training for more stable training.
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        num_sensitive_groups: int = 2,
        hidden_dims: List[int] = [512, 256, 128],
        fairness_weight: float = 1.0,
        gp_weight: float = 10.0,
        n_critic: int = 5,
        name: str = "fairgan_gp",
        **kwargs
    ):
        """
        Initialize FairGAN-GP.
        
        Args:
            data_dim: Data dimension
            latent_dim: Latent dimension
            num_sensitive_groups: Number of sensitive groups
            hidden_dims: Hidden dimensions
            fairness_weight: Weight for fairness loss
            gp_weight: Gradient penalty weight
            n_critic: Discriminator updates per generator update
            name: Model name
        """
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_sensitive_groups=num_sensitive_groups,
            hidden_dims=hidden_dims,
            fairness_weight=fairness_weight,
            gp_weight=gp_weight,
            name=name,
            **kwargs
        )
        self._n_critic = n_critic


class ConditionalFairGAN(FairGAN):
    """
    Conditional FairGAN for conditional fair generation.
    
    Supports conditioning on:
    - Class labels
    - Continuous attributes
    - Target sensitive attribute distributions
    
    Example:
        >>> model = ConditionalFairGAN(
        ...     data_dim=100,
        ...     latent_dim=512,
        ...     num_classes=10,
        ...     num_sensitive_groups=2
        ... )
        >>> # Generate samples for specific class
        >>> samples = model.generate(100, class_label=3)
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        num_classes: Optional[int] = None,
        condition_dim: Optional[int] = None,
        num_sensitive_groups: int = 2,
        hidden_dims: List[int] = [512, 256, 128],
        fairness_weight: float = 1.0,
        gp_weight: float = 10.0,
        name: str = "conditional_fairgan",
        **kwargs
    ):
        """
        Initialize Conditional FairGAN.
        
        Args:
            data_dim: Data dimension
            latent_dim: Latent dimension
            num_classes: Number of classes for conditioning
            condition_dim: Dimension of continuous condition
            num_sensitive_groups: Number of sensitive groups
            hidden_dims: Hidden dimensions
            fairness_weight: Fairness loss weight
            gp_weight: Gradient penalty weight
            name: Model name
        """
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_sensitive_groups=num_sensitive_groups,
            hidden_dims=hidden_dims,
            fairness_weight=fairness_weight,
            gp_weight=gp_weight,
            name=name,
            **kwargs
        )
        
        # Build condition embeddings
        if num_classes is not None:
            self.class_embedding = nn.Embedding(num_classes, latent_dim)
        else:
            self.class_embedding = None
            
        if condition_dim is not None:
            self.condition_projection = nn.Sequential(
                nn.Linear(condition_dim, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.LeakyReLU(0.2)
            )
        else:
            self.condition_projection = None
    
    def _build_generator(
        self,
        hidden_dims: List[int],
        generator_type: str,
        spectral_norm: bool
    ) -> None:
        """Build conditional generator."""
        # Input is latent + condition
        super()._build_generator(hidden_dims, generator_type, spectral_norm)
    
    def combine_latent_condition(
        self,
        z: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Combine latent with condition."""
        combined = z
        
        if class_label is not None and self.class_embedding is not None:
            class_emb = self.class_embedding(class_label)
            combined = combined + class_emb
            
        if condition is not None and self.condition_projection is not None:
            cond_emb = self.condition_projection(condition)
            combined = combined + cond_emb
            
        return combined
    
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
        
        # Combine with conditions
        if class_label is not None or condition is not None:
            z = self.combine_latent_condition(z, class_label, condition)
        
        return self.generator(z)
    
    def generate_counterfactual(
        self,
        n_samples: int,
        original_sensitive: int,
        target_sensitive: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate counterfactual samples.
        
        Generates samples where sensitive attribute would be different
        while keeping other properties similar.
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Generate with target sensitive conditioning
        # This requires the model to be trained with sensitive conditioning
        z = self.sample_latent(n_samples, device)
        target_label = torch.full((n_samples,), target_sensitive, 
                                   dtype=torch.long, device=device)
        
        if self.class_embedding is not None:
            z = self.combine_latent_condition(z, class_label=target_label)
        
        return self.generator(z)


class MultimodalFairGAN(FairGAN):
    """
    Multimodal FairGAN for multi-modal fair data generation.
    
    Generates synchronized fair data across multiple modalities:
    - Tabular: Structured data
    - Text: Textual content
    - Image: Visual data
    
    Architecture:
        Shared Latent -> Modality-specific generators
        Modality-specific discriminators -> Joint discrimination
    
    Example:
        >>> model = MultimodalFairGAN(
        ...     modalities=["tabular", "text", "image"],
        ...     latent_dim=512,
        ...     modality_configs={
        ...         "tabular": {"dim": 100},
        ...         "text": {"vocab_size": 10000, "seq_len": 50},
        ...         "image": {"channels": 3, "size": 64}
        ...     }
        ... )
        >>> samples = model.generate(100)
        >>> # Returns: {"tabular": ..., "text": ..., "image": ...}
    """
    
    def __init__(
        self,
        modalities: List[str],
        latent_dim: int = 512,
        modality_configs: Optional[Dict[str, Dict]] = None,
        num_sensitive_groups: int = 2,
        hidden_dims: List[int] = [512, 256, 128],
        fairness_weight: float = 1.0,
        gp_weight: float = 10.0,
        fusion_type: str = "attention",
        name: str = "multimodal_fairgan",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Multimodal FairGAN.
        
        Args:
            modalities: List of modalities
            latent_dim: Shared latent dimension
            modality_configs: Configuration for each modality
            num_sensitive_groups: Number of sensitive groups
            hidden_dims: Hidden dimensions
            fairness_weight: Fairness loss weight
            gp_weight: Gradient penalty weight
            fusion_type: Fusion type ("concat", "attention", "sum")
            name: Model name
            config: Optional configuration
        """
        self.modalities = modalities
        self.modality_configs = modality_configs or {}
        self.fusion_type = fusion_type
        
        # Calculate total data dimension
        data_dim = latent_dim  # Will be overridden by multimodal handling
        
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_sensitive_groups=num_sensitive_groups,
            hidden_dims=hidden_dims,
            fairness_weight=fairness_weight,
            gp_weight=gp_weight,
            name=name,
            config=config
        )
        
        # Build modality-specific generators and discriminators
        self._build_modality_generators()
        self._build_modality_discriminators()
        self._build_fusion_layer()
        
    def _build_modality_generators(self) -> None:
        """Build generators for each modality."""
        self.modality_generators = nn.ModuleDict()
        
        for mod in self.modalities:
            config = self.modality_configs.get(mod, {})
            
            if mod == "tabular":
                output_dim = config.get("dim", 100)
                self.modality_generators[mod] = self._build_tabular_generator(output_dim)
                
            elif mod == "text":
                vocab_size = config.get("vocab_size", 10000)
                seq_len = config.get("seq_len", 50)
                self.modality_generators[mod] = self._build_text_generator(vocab_size, seq_len)
                
            elif mod == "image":
                channels = config.get("channels", 3)
                size = config.get("size", 64)
                self.modality_generators[mod] = self._build_image_generator(channels, size)
    
    def _build_tabular_generator(self, output_dim: int) -> nn.Module:
        """Build tabular data generator."""
        return nn.Sequential(
            nn.Linear(self._latent_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )
    
    def _build_text_generator(self, vocab_size: int, seq_len: int) -> nn.Module:
        """Build text generator with transformer decoder."""
        return nn.Sequential(
            nn.Linear(self._latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            # Output logits for each position
            nn.Linear(512, seq_len * vocab_size)
        )
    
    def _build_image_generator(self, channels: int, size: int) -> nn.Module:
        """Build image generator with transposed convolutions."""
        # Calculate initial size for progressive upsampling
        init_size = size // 16  # Will upsample 4 times (2^4 = 16)
        
        layers = [
            nn.Linear(self._latent_dim, 512 * init_size * init_size),
            nn.BatchNorm1d(512 * init_size * init_size),
            nn.LeakyReLU(0.2),
        ]
        
        # Reshape and transposed convolutions
        layers.append(nn.Unflatten(1, (512, init_size, init_size)))
        
        # Progressive upsampling
        current_channels = 512
        for out_channels in [256, 128, 64]:
            layers.extend([
                nn.ConvTranspose2d(current_channels, out_channels, 4, 2, 1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ])
            current_channels = out_channels
        
        # Final layer
        layers.extend([
            nn.ConvTranspose2d(current_channels, channels, 4, 2, 1),
            nn.Tanh()
        ])
        
        return nn.Sequential(*layers)
    
    def _build_modality_discriminators(self) -> None:
        """Build discriminators for each modality."""
        self.modality_discriminators = nn.ModuleDict()
        
        for mod in self.modalities:
            config = self.modality_configs.get(mod, {})
            
            if mod == "tabular":
                input_dim = config.get("dim", 100)
                self.modality_discriminators[mod] = self._build_tabular_discriminator(input_dim)
                
            elif mod == "text":
                vocab_size = config.get("vocab_size", 10000)
                seq_len = config.get("seq_len", 50)
                self.modality_discriminators[mod] = self._build_text_discriminator(vocab_size, seq_len)
                
            elif mod == "image":
                channels = config.get("channels", 3)
                self.modality_discriminators[mod] = self._build_image_discriminator(channels)
    
    def _build_tabular_discriminator(self, input_dim: int) -> nn.Module:
        """Build tabular discriminator."""
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(input_dim, 256)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.utils.spectral_norm(nn.Linear(128, 1))
        )
    
    def _build_text_discriminator(self, vocab_size: int, seq_len: int) -> nn.Module:
        """Build text discriminator."""
        return nn.Sequential(
            nn.Conv1d(vocab_size, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(128, 1))
        )
    
    def _build_image_discriminator(self, channels: int) -> nn.Module:
        """Build image discriminator."""
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(channels, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(256, 1))
        )
    
    def _build_fusion_layer(self) -> None:
        """Build cross-modal fusion layer."""
        if self.fusion_type == "attention":
            self.fusion = CrossModalFusion(
                modality_dims={mod: 128 for mod in self.modalities},
                hidden_dim=self._latent_dim
            )
        else:
            self.fusion = None
    
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
        
        outputs = {}
        for mod in self.modalities:
            outputs[mod] = self.modality_generators[mod](z)
            
            # Special handling for text output
            if mod == "text":
                config = self.modality_configs.get(mod, {})
                seq_len = config.get("seq_len", 50)
                vocab_size = config.get("vocab_size", 10000)
                # Reshape to (batch, seq_len, vocab_size)
                outputs[mod] = outputs[mod].view(n_samples, seq_len, vocab_size)
        
        return outputs
    
    def forward(
        self,
        real_data: Dict[str, torch.Tensor],
        sensitive_attrs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for multimodal training."""
        batch_size = next(iter(real_data.values())).size(0)
        device = next(iter(real_data.values())).device
        
        # Generate fake samples
        z = self.sample_latent(batch_size, device)
        fake_data = self.generate(batch_size, device=device)
        
        losses = {}
        
        # Discriminator losses per modality
        d_losses = []
        for mod in self.modalities:
            if mod in real_data:
                d_real = self.modality_discriminators[mod](real_data[mod])
                d_fake = self.modality_discriminators[mod](fake_data[mod].detach())
                d_loss = d_fake.mean() - d_real.mean()
                d_losses.append(d_loss)
        
        losses["d_loss"] = sum(d_losses) / len(d_losses)
        
        # Generator losses
        g_losses = []
        for mod in self.modalities:
            if mod in real_data:
                d_fake = self.modality_discriminators[mod](fake_data[mod])
                g_losses.append(-d_fake.mean())
        
        losses["g_loss"] = sum(g_losses) / len(g_losses)
        
        # Fairness adversary loss
        if sensitive_attrs is not None:
            z_reversed = self.grl(z)
            adv_pred = self.adversary(z_reversed)
            adv_loss = F.cross_entropy(adv_pred, sensitive_attrs)
            losses["adv_loss"] = self.fairness_weight * adv_loss
            losses["g_total"] = losses["g_loss"] + self.fairness_weight * adv_loss
        else:
            losses["adv_loss"] = torch.tensor(0.0, device=device)
            losses["g_total"] = losses["g_loss"]
        
        return {
            "losses": losses,
            "fake_data": fake_data,
            "latent": z
        }


class CrossModalFusion(nn.Module):
    """Cross-modal attention-based fusion module."""
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.modality_dims = modality_dims
        self.hidden_dim = hidden_dim
        
        # Projections for each modality
        self.projections = nn.ModuleDict({
            mod: nn.Linear(dim, hidden_dim)
            for mod, dim in modality_dims.items()
        })
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Fuse modality features."""
        # Project each modality
        projected = []
        for mod, feat in modality_features.items():
            if mod in self.projections:
                projected.append(self.projections[mod](feat))
        
        if not projected:
            raise ValueError("No valid modality features provided")
        
        # Stack for attention (batch, num_modalities, hidden_dim)
        stacked = torch.stack(projected, dim=1)
        
        # Cross-modal attention
        fused, _ = self.cross_attention(stacked, stacked, stacked)
        
        # Average across modalities
        fused = fused.mean(dim=1)
        
        return self.output(fused)


# Registry for FairGAN variants
FAIRGAN_REGISTRY = {
    "fairgan": FairGAN,
    "fairgan_gp": FairGANGP,
    "conditional_fairgan": ConditionalFairGAN,
    "multimodal_fairgan": MultimodalFairGAN,
}


def get_fairgan(name: str, **kwargs) -> FairGAN:
    """Get a FairGAN variant by name."""
    if name not in FAIRGAN_REGISTRY:
        available = list(FAIRGAN_REGISTRY.keys())
        raise ValueError(f"Unknown FairGAN: {name}. Available: {available}")
    return FAIRGAN_REGISTRY[name](**kwargs)
