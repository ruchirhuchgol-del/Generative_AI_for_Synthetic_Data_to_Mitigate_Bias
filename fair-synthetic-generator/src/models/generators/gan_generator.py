"""
GAN Generator
=============

Generative Adversarial Network (GAN) generators for synthetic data.
Supports GAN, DCGAN, WGAN, WGAN-GP, and StyleGAN variants.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.generators.base_generator import (
    BaseGenerator,
    ConditionalGenerator,
    FairGenerator,
    LatentSpaceMixin,
    ProgressiveGenerator
)


class GANGenerator(BaseGenerator, LatentSpaceMixin):
    """
    Standard GAN Generator.
    
    Architecture:
        Generator: z → x̂
        Discriminator: x → real/fake
        
    Training:
        min_G max_D E[log D(x)] + E[log(1 - D(G(z)))]
        
    Example:
        >>> generator = GANGenerator(
        ...     generator=generator_net,
        ...     discriminator=discriminator_net,
        ...     latent_dim=512
        ... )
        >>> samples = generator.generate(100)
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        latent_dim: int = 512,
        loss_type: str = "vanilla",  # vanilla, ls (least squares), hinge
        noise_dim: Optional[int] = None,
        name: str = "gan_generator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize GAN generator.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
            latent_dim: Latent dimension
            loss_type: Type of GAN loss
            noise_dim: Additional noise dimension (optional)
            name: Generator name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.generator = generator
        self.discriminator = discriminator
        self.loss_type = loss_type
        self.noise_dim = noise_dim or latent_dim
        
    def sample_latent(
        self, 
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample from prior.
        
        Args:
            n_samples: Number of samples
            device: Device for samples
            
        Returns:
            Latent samples
        """
        if device is None:
            device = next(self.generator.parameters()).device
        return torch.randn(n_samples, self.noise_dim, device=device)
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples
            conditions: Optional conditions (ignored for base GAN)
            
        Returns:
            Generated samples
        """
        z = self.sample_latent(n_samples)
        return self.generator(z, **kwargs)
    
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decode latent to output.
        
        Args:
            z: Latent tensor
            
        Returns:
            Generated output
        """
        return self.generator(z, **kwargs)
    
    def compute_generator_loss(
        self,
        batch_size: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute generator loss.
        
        Args:
            batch_size: Batch size for generation
            
        Returns:
            Generator loss
        """
        # Generate fake samples
        z = self.sample_latent(batch_size)
        fake = self.generator(z)
        
        # Discriminator output on fake
        d_fake = self.discriminator(fake)
        
        # Compute loss based on type
        if self.loss_type == "vanilla":
            # Non-saturating loss
            loss = F.binary_cross_entropy_with_logits(
                d_fake, torch.ones_like(d_fake)
            )
        elif self.loss_type == "ls":
            # Least squares GAN
            loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        elif self.loss_type == "hinge":
            # Hinge loss
            loss = -d_fake.mean()
        else:
            loss = F.binary_cross_entropy_with_logits(
                d_fake, torch.ones_like(d_fake)
            )
            
        return loss
    
    def compute_discriminator_loss(
        self,
        real: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute discriminator loss.
        
        Args:
            real: Real samples
            
        Returns:
            Discriminator loss
        """
        batch_size = real.shape[0]
        
        # Generate fake samples
        z = self.sample_latent(batch_size)
        with torch.no_grad():
            fake = self.generator(z)
        
        # Discriminator outputs
        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake)
        
        # Compute loss based on type
        if self.loss_type == "vanilla":
            loss_real = F.binary_cross_entropy_with_logits(
                d_real, torch.ones_like(d_real)
            )
            loss_fake = F.binary_cross_entropy_with_logits(
                d_fake, torch.zeros_like(d_fake)
            )
            loss = (loss_real + loss_fake) / 2
            
        elif self.loss_type == "ls":
            loss_real = F.mse_loss(d_real, torch.ones_like(d_real))
            loss_fake = F.mse_loss(d_fake, torch.zeros_like(d_fake))
            loss = (loss_real + loss_fake) / 2
            
        elif self.loss_type == "hinge":
            loss_real = F.relu(1 - d_real).mean()
            loss_fake = F.relu(1 + d_fake).mean()
            loss = loss_real + loss_fake
            
        else:
            loss_real = F.binary_cross_entropy_with_logits(
                d_real, torch.ones_like(d_real)
            )
            loss_fake = F.binary_cross_entropy_with_logits(
                d_fake, torch.zeros_like(d_fake)
            )
            loss = (loss_real + loss_fake) / 2
            
        return loss
    
    def compute_loss(
        self,
        real: torch.Tensor,
        return_components: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total GAN loss (for training both networks).
        
        Args:
            real: Real samples
            return_components: Return loss components
            
        Returns:
            Total loss or dictionary
        """
        g_loss = self.compute_generator_loss(real.shape[0])
        d_loss = self.compute_discriminator_loss(real)
        
        if return_components:
            return {
                "generator_loss": g_loss,
                "discriminator_loss": d_loss,
            }
        
        return g_loss + d_loss


class WANGenerator(GANGenerator):
    """
    Wasserstein GAN with Gradient Penalty (WGAN-GP).
    
    Uses Wasserstein distance with gradient penalty for stable training.
    
    Loss:
        L_D = D(x_fake) - D(x_real) + λ · GP
        L_G = -D(G(z))
        
    GP = E[(||∇D(x_interp)||_2 - 1)²]
        
    Example:
        >>> generator = WANGenerator(
        ...     generator=generator_net,
        ...     discriminator=discriminator_net,
        ...     latent_dim=512,
        ...     gp_weight=10.0
        ... )
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        latent_dim: int = 512,
        gp_weight: float = 10.0,
        drift_weight: float = 0.001,
        n_critic: int = 5,
        name: str = "wgan_generator",
        **kwargs
    ):
        """
        Initialize WGAN-GP generator.
        
        Args:
            generator: Generator network
            discriminator: Critic network
            latent_dim: Latent dimension
            gp_weight: Gradient penalty weight
            drift_weight: Drift regularization weight
            n_critic: Critic iterations per generator iteration
            name: Generator name
        """
        super().__init__(
            generator=generator,
            discriminator=discriminator,
            latent_dim=latent_dim,
            loss_type="hinge",  # WGAN uses linear outputs
            name=name,
            **kwargs
        )
        
        self.gp_weight = gp_weight
        self.drift_weight = drift_weight
        self.n_critic = n_critic
        
    def compute_gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty.
        
        Args:
            real: Real samples
            fake: Fake samples
            
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
        
        # Compute gradient
        d_interpolated = self.discriminator(interpolated)
        
        # Compute gradient w.r.t. interpolated
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # Flatten gradients
        gradients = gradients.flatten(1)
        
        # Compute gradient penalty
        gradient_norm = gradients.norm(2, dim=1)
        gp = ((gradient_norm - 1) ** 2).mean()
        
        return gp
    
    def compute_discriminator_loss(
        self,
        real: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute critic (discriminator) loss with gradient penalty.
        
        Args:
            real: Real samples
            
        Returns:
            Critic loss
        """
        batch_size = real.shape[0]
        
        # Generate fake samples
        z = self.sample_latent(batch_size)
        with torch.no_grad():
            fake = self.generator(z)
        
        # Wasserstein distance
        d_real = self.discriminator(real)
        d_fake = self.discriminator(fake)
        
        wasserstein_loss = d_fake.mean() - d_real.mean()
        
        # Gradient penalty
        gp = self.compute_gradient_penalty(real, fake)
        
        # Drift regularization
        drift = (d_real ** 2).mean()
        
        # Total loss
        loss = wasserstein_loss + self.gp_weight * gp + self.drift_weight * drift
        
        return loss
    
    def compute_generator_loss(
        self,
        batch_size: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute generator loss.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Generator loss
        """
        z = self.sample_latent(batch_size)
        fake = self.generator(z)
        d_fake = self.discriminator(fake)
        
        # Maximize critic output on fake samples
        return -d_fake.mean()


class ConditionalGANGenerator(ConditionalGenerator, GANGenerator):
    """
    Conditional GAN (CGAN) Generator.
    
    Conditions generation on labels or continuous attributes.
    
    Architecture:
        Generator: (z, c) → x̂
        Discriminator: (x, c) → real/fake
        
    Example:
        >>> generator = ConditionalGANGenerator(
        ...     generator=generator_net,
        ...     discriminator=discriminator_net,
        ...     latent_dim=512,
        ...     num_classes=10
        ... )
        >>> samples = generator.generate(100, class_label=3)
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        latent_dim: int = 512,
        num_classes: Optional[int] = None,
        condition_dim: Optional[int] = None,
        conditioning_strategy: str = "concat",  # concat, embedding, projection
        loss_type: str = "vanilla",
        name: str = "conditional_gan_generator",
        **kwargs
    ):
        """
        Initialize conditional GAN.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
            latent_dim: Latent dimension
            num_classes: Number of classes
            condition_dim: Continuous condition dimension
            conditioning_strategy: How to incorporate conditions
            loss_type: GAN loss type
            name: Generator name
        """
        ConditionalGenerator.__init__(
            self,
            name=name,
            latent_dim=latent_dim,
            num_classes=num_classes,
            condition_dim=condition_dim
        )
        GANGenerator.__init__(
            self,
            generator=generator,
            discriminator=discriminator,
            latent_dim=latent_dim,
            loss_type=loss_type,
            name=name,
            **kwargs
        )
        
        self.conditioning_strategy = conditioning_strategy
        
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate conditional samples.
        
        Args:
            n_samples: Number of samples
            conditions: Conditions dict
            class_label: Class label
            condition: Continuous condition
            
        Returns:
            Generated samples
        """
        if conditions is not None:
            class_label = conditions.get("class_label", class_label)
            condition = conditions.get("condition", condition)
            
        z = self.sample_latent(n_samples)
        z_cond = self.combine_latent_condition(z, class_label, condition)
        
        return self.generator(z_cond, **kwargs)
    
    def compute_generator_loss(
        self,
        batch_size: int,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute conditional generator loss.
        
        Args:
            batch_size: Batch size
            class_label: Class labels
            condition: Continuous conditions
            
        Returns:
            Generator loss
        """
        z = self.sample_latent(batch_size)
        z_cond = self.combine_latent_condition(z, class_label, condition)
        fake = self.generator(z_cond)
        
        # Condition discriminator
        if class_label is not None and self.class_embedding is not None:
            c_emb = self.class_embedding(class_label)
            fake_cond = torch.cat([fake, c_emb], dim=1)
        else:
            fake_cond = fake
            
        d_fake = self.discriminator(fake_cond)
        
        if self.loss_type == "vanilla":
            loss = F.binary_cross_entropy_with_logits(
                d_fake, torch.ones_like(d_fake)
            )
        elif self.loss_type == "ls":
            loss = F.mse_loss(d_fake, torch.ones_like(d_fake))
        elif self.loss_type == "hinge":
            loss = -d_fake.mean()
        else:
            loss = F.binary_cross_entropy_with_logits(
                d_fake, torch.ones_like(d_fake)
            )
            
        return loss
    
    def compute_discriminator_loss(
        self,
        real: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute conditional discriminator loss.
        
        Args:
            real: Real samples
            class_label: Class labels
            condition: Continuous conditions
            
        Returns:
            Discriminator loss
        """
        batch_size = real.shape[0]
        
        z = self.sample_latent(batch_size)
        z_cond = self.combine_latent_condition(z, class_label, condition)
        
        with torch.no_grad():
            fake = self.generator(z_cond)
        
        # Concat condition to inputs
        if class_label is not None and self.class_embedding is not None:
            c_emb = self.class_embedding(class_label)
            real_cond = torch.cat([real, c_emb], dim=1)
            fake_cond = torch.cat([fake, c_emb], dim=1)
        else:
            real_cond = real
            fake_cond = fake
        
        d_real = self.discriminator(real_cond)
        d_fake = self.discriminator(fake_cond)
        
        if self.loss_type == "vanilla":
            loss_real = F.binary_cross_entropy_with_logits(
                d_real, torch.ones_like(d_real)
            )
            loss_fake = F.binary_cross_entropy_with_logits(
                d_fake, torch.zeros_like(d_fake)
            )
        elif self.loss_type == "ls":
            loss_real = F.mse_loss(d_real, torch.ones_like(d_real))
            loss_fake = F.mse_loss(d_fake, torch.zeros_like(d_fake))
        elif self.loss_type == "hinge":
            loss_real = F.relu(1 - d_real).mean()
            loss_fake = F.relu(1 + d_fake).mean()
        else:
            loss_real = F.binary_cross_entropy_with_logits(
                d_real, torch.ones_like(d_real)
            )
            loss_fake = F.binary_cross_entropy_with_logits(
                d_fake, torch.zeros_like(d_fake)
            )
            
        return (loss_real + loss_fake) / 2


class StyleGANGenerator(BaseGenerator, LatentSpaceMixin):
    """
    StyleGAN Generator with style modulation.
    
    Features:
    - Mapping network: z → w
    - Style injection via AdaIN
    - Progressive growing
    - Noise injection
    
    Architecture:
        z → Mapping Network → w → Style Blocks → Image
        
    Example:
        >>> generator = StyleGANGenerator(
        ...     mapping_network=mapping_net,
        ...     synthesis_network=synthesis_net,
        ...     latent_dim=512,
        ...     style_dim=512
        ... )
    """
    
    def __init__(
        self,
        mapping_network: nn.Module,
        synthesis_network: nn.Module,
        discriminator: nn.Module,
        latent_dim: int = 512,
        style_dim: int = 512,
        num_mapping_layers: int = 8,
        style_mixing_prob: float = 0.9,
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
        name: str = "stylegan_generator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize StyleGAN generator.
        
        Args:
            mapping_network: Mapping network (z → w)
            synthesis_network: Synthesis network (w → image)
            discriminator: Discriminator network
            latent_dim: Input latent dimension
            style_dim: Style vector dimension
            num_mapping_layers: Number of mapping layers
            style_mixing_prob: Probability of style mixing
            truncation_psi: Truncation parameter
            truncation_cutoff: Layer cutoff for truncation
            name: Generator name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.mapping_network = mapping_network
        self.synthesis_network = synthesis_network
        self.discriminator = discriminator
        self.style_dim = style_dim
        self.num_mapping_layers = num_mapping_layers
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff
        
        # Learnable w_avg for truncation
        self.register_buffer("w_avg", torch.zeros(style_dim))
        
    def sample_latent(
        self, 
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """Sample input latent z."""
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(n_samples, self._latent_dim, device=device)
    
    def map_latent(
        self, 
        z: torch.Tensor,
        update_w_avg: bool = False
    ) -> torch.Tensor:
        """
        Map z to w style space.
        
        Args:
            z: Input latent
            update_w_avg: Update running average
            
        Returns:
            Style vector w
        """
        w = self.mapping_network(z)
        
        if update_w_avg:
            with torch.no_grad():
                self.w_avg.lerp_(w.mean(dim=0), 0.01)
                
        return w
    
    def apply_truncation(
        self,
        w: torch.Tensor,
        psi: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply style truncation.
        
        w_new = w_avg + ψ * (w - w_avg)
        
        Args:
            w: Style vector
            psi: Truncation parameter (uses self.truncation_psi if None)
            
        Returns:
            Truncated style vector
        """
        if psi is None:
            psi = self.truncation_psi
            
        if psi < 1.0:
            w = self.w_avg + psi * (w - self.w_avg)
            
        return w
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        truncation_psi: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples.
        
        Args:
            n_samples: Number of samples
            conditions: Optional conditions
            truncation_psi: Truncation parameter
            
        Returns:
            Generated samples
        """
        z = self.sample_latent(n_samples)
        w = self.map_latent(z)
        
        if truncation_psi is not None:
            w = self.apply_truncation(w, truncation_psi)
            
        return self.synthesis_network(w, **kwargs)
    
    def generate_with_style_mixing(
        self,
        n_samples: int,
        mixing_prob: Optional[float] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate with style mixing regularization.
        
        Args:
            n_samples: Number of samples
            mixing_prob: Style mixing probability
            
        Returns:
            Tuple of (generated samples, mixed styles)
        """
        if mixing_prob is None:
            mixing_prob = self.style_mixing_prob
            
        # Sample two latent codes
        z1 = self.sample_latent(n_samples)
        z2 = self.sample_latent(n_samples)
        
        # Map to style space
        w1 = self.map_latent(z1)
        w2 = self.map_latent(z2)
        
        # Random mixing
        if torch.rand(1).item() < mixing_prob:
            # Create mixed style
            mix_layer = torch.randint(1, self.synthesis_network.num_layers, (1,)).item()
            w = w1.clone()
            w[:, mix_layer:] = w2[:, mix_layer:]
        else:
            w = w1
            
        return self.synthesis_network(w), w
    
    def compute_generator_loss(
        self,
        batch_size: int,
        **kwargs
    ) -> torch.Tensor:
        """Compute generator loss."""
        fake = self.generate(batch_size)
        d_fake = self.discriminator(fake)
        return F.softplus(-d_fake).mean()
    
    def compute_discriminator_loss(
        self,
        real: torch.Tensor,
        r1_gamma: float = 10.0,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute discriminator loss with R1 regularization.
        
        Args:
            real: Real samples
            r1_gamma: R1 regularization weight
            
        Returns:
            Discriminator loss
        """
        batch_size = real.shape[0]
        
        # Real samples
        real.requires_grad_(True)
        d_real = self.discriminator(real)
        
        # R1 gradient penalty
        gradients = torch.autograd.grad(
            outputs=d_real.sum(),
            inputs=real,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        r1_penalty = (gradients.flatten(1).norm(2, dim=1) ** 2).mean()
        
        # Fake samples
        fake = self.generate(batch_size)
        d_fake = self.discriminator(fake)
        
        # Non-saturating loss with R1
        loss = F.softplus(d_fake).mean() + F.softplus(-d_real).mean()
        loss = loss + r1_gamma * 0.5 * r1_penalty
        
        return loss


class FairGANGenerator(ConditionalGANGenerator, FairGenerator):
    """
    Fair GAN Generator for bias-aware synthetic data generation.
    
    Integrates fairness constraints through:
    - Adversarial debiasing
    - Fairness-aware discriminator
    - Conditional generation with sensitive attributes
    
    Example:
        >>> generator = FairGANGenerator(
        ...     generator=generator_net,
        ...     discriminator=discriminator_net,
        ...     sensitive_adversary=adversary_net,
        ...     latent_dim=512,
        ...     sensitive_attributes=["gender", "race"]
        ... )
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        sensitive_adversary: Optional[nn.Module] = None,
        latent_dim: int = 512,
        sensitive_attributes: Optional[List[str]] = None,
        fairness_weights: Optional[Dict[str, float]] = None,
        adversarial_weight: float = 0.1,
        name: str = "fair_gan_generator",
        **kwargs
    ):
        """
        Initialize fair GAN generator.
        
        Args:
            generator: Generator network
            discriminator: Discriminator network
            sensitive_adversary: Adversary for sensitive attribute prediction
            latent_dim: Latent dimension
            sensitive_attributes: List of sensitive attribute names
            fairness_weights: Weights for fairness constraints
            adversarial_weight: Weight for adversarial fairness loss
            name: Generator name
        """
        ConditionalGANGenerator.__init__(
            self,
            generator=generator,
            discriminator=discriminator,
            latent_dim=latent_dim,
            name=name,
            **kwargs
        )
        FairGenerator.__init__(
            self,
            name=name,
            latent_dim=latent_dim,
            sensitive_attributes=sensitive_attributes,
            fairness_weights=fairness_weights
        )
        
        self._adversary = sensitive_adversary
        self.adversarial_weight = adversarial_weight
        
    def compute_fairness_penalty(
        self,
        samples: torch.Tensor,
        sensitive: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute fairness penalty using adversary.
        
        Args:
            samples: Generated samples
            sensitive: Sensitive attribute values
            
        Returns:
            Fairness penalty
        """
        if self._adversary is None or sensitive is None:
            return torch.tensor(0.0, device=samples.device)
            
        # Adversary tries to predict sensitive from generated samples
        pred_sensitive = self._adversary(samples)
        
        # We want adversary to fail (high entropy predictions)
        # This encourages fairness by removing sensitive information
        if pred_sensitive.dim() > sensitive.dim():
            # Classification case
            penalty = F.cross_entropy(pred_sensitive, sensitive)
        else:
            # Regression case
            penalty = F.mse_loss(pred_sensitive, sensitive.float())
            
        # Negative because we want to maximize adversary confusion
        return -self.adversarial_weight * penalty
    
    def compute_generator_loss(
        self,
        batch_size: int,
        sensitive: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute fair generator loss.
        
        Args:
            batch_size: Batch size
            sensitive: Sensitive attributes
            
        Returns:
            Generator loss with fairness penalty
        """
        # Standard GAN loss
        g_loss = super().compute_generator_loss(batch_size, **kwargs)
        
        # Generate samples
        z = self.sample_latent(batch_size)
        samples = self.generator(z)
        
        # Fairness penalty
        fairness_penalty = self.compute_fairness_penalty(samples, sensitive)
        
        return g_loss + fairness_penalty
    
    def generate_fair(
        self,
        n_samples: int,
        target_distribution: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate fair samples.
        
        Args:
            n_samples: Number of samples
            target_distribution: Target distribution for sensitive attributes
            
        Returns:
            Fair synthetic samples
        """
        # Sample from balanced latent distribution
        z = self.sample_latent(n_samples)
        
        # Generate samples
        samples = self.generator(z)
        
        # Apply post-hoc fairness if target distribution specified
        if target_distribution is not None:
            samples = self._apply_distribution_matching(samples, target_distribution)
            
        return samples
    
    def _apply_distribution_matching(
        self,
        samples: torch.Tensor,
        target_distribution: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Apply distribution matching for fairness.
        
        Args:
            samples: Generated samples
            target_distribution: Target distribution
            
        Returns:
            Rebalanced samples
        """
        # Placeholder - implement based on specific fairness requirements
        return samples
