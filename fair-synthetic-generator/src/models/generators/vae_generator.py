"""
VAE Generator
=============

Variational Autoencoder (VAE) generators for synthetic data generation.
Supports VAE, β-VAE, VAE-GAN, and Conditional VAE (CVAE).
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
    ReconstructionMixin
)


class VAEGenerator(BaseGenerator, LatentSpaceMixin, ReconstructionMixin):
    """
    Variational Autoencoder (VAE) Generator.
    
    Architecture:
        Encoder: x → (μ, log σ²) → z = μ + σ·ε
        Decoder: z → x̂
        
    Loss:
        L = Reconstruction Loss + β · KL(q(z|x) || p(z))
        
    Supports:
    - β-VAE for disentanglement
    - Hierarchical VAE
    - Importance weighted VAE
    
    Example:
        >>> generator = VAEGenerator(
        ...     encoder=encoder,
        ...     decoder=decoder,
        ...     latent_dim=512,
        ...     beta=1.0
        ... )
        >>> # Training
        >>> loss_dict = generator.compute_loss(x)
        >>> # Generation
        >>> samples = generator.generate(100)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int = 512,
        beta: float = 1.0,
        beta_schedule: str = "constant",  # constant, cyclic, monotonic
        reconstruction_loss: str = "mse",  # mse, bce, gaussian
        use_importance_sampling: bool = False,
        num_importance_samples: int = 5,
        name: str = "vae_generator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize VAE generator.
        
        Args:
            encoder: Encoder network (outputs μ and log σ²)
            decoder: Decoder network
            latent_dim: Latent dimension
            beta: KL divergence weight (β-VAE)
            beta_schedule: Beta scheduling strategy
            reconstruction_loss: Type of reconstruction loss
            use_importance_sampling: Use importance weighted VAE
            num_importance_samples: Number of importance samples
            name: Generator name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.beta_schedule = beta_schedule
        self.reconstruction_loss_type = reconstruction_loss
        self.use_importance_sampling = use_importance_sampling
        self.num_importance_samples = num_importance_samples
        
        # Min/max logvar for numerical stability
        self.min_logvar = -10.0
        self.max_logvar = 10.0
        
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
        encoder_output = self.encoder(x, **kwargs)
        
        if isinstance(encoder_output, tuple):
            mu, logvar = encoder_output
        else:
            # Single output - split in half
            mu, logvar = torch.chunk(encoder_output, 2, dim=-1)
            
        # Clamp logvar for stability
        logvar = torch.clamp(logvar, self.min_logvar, self.max_logvar)
        
        return mu, logvar
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        training: bool = True
    ) -> torch.Tensor:
        """
        Reparameterization trick.
        
        z = μ + σ · ε, where ε ~ N(0, I)
        
        Args:
            mu: Mean tensor
            logvar: Log-variance tensor
            training: Training mode flag
            
        Returns:
            Sampled latent tensor
        """
        if training and self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(
        self, 
        z: torch.Tensor, 
        **kwargs
    ) -> torch.Tensor:
        """
        Decode latent to output.
        
        Args:
            z: Latent tensor
            
        Returns:
            Decoded output
        """
        return self.decoder(z, **kwargs)
    
    def sample_latent(
        self, 
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample from prior N(0, I).
        
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
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples from prior.
        
        Args:
            n_samples: Number of samples
            conditions: Optional conditions (ignored for base VAE)
            
        Returns:
            Generated samples
        """
        z = self.sample_latent(n_samples)
        return self.decode(z, **kwargs)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        return_components: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute VAE loss.
        
        L = L_recon + β · L_KL
        
        Args:
            x: Input tensor
            return_components: If True, return loss components
            
        Returns:
            Total loss or dictionary of loss components
        """
        # Encode
        mu, logvar = self.encode(x, **kwargs)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decode(z, **kwargs)
        
        # Reconstruction loss
        recon_loss = self._compute_reconstruction_loss(x, x_recon)
        
        # KL divergence
        kl_loss = self._compute_kl_divergence(mu, logvar)
        
        # Get current beta
        current_beta = self._get_current_beta()
        
        # Total loss
        total_loss = recon_loss + current_beta * kl_loss
        
        if return_components:
            return {
                "total_loss": total_loss,
                "reconstruction_loss": recon_loss,
                "kl_loss": kl_loss,
                "beta": current_beta,
                "mu_mean": mu.mean().item(),
                "logvar_mean": logvar.mean().item(),
            }
            
        return total_loss
    
    def _compute_reconstruction_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss."""
        if self.reconstruction_loss_type == "mse":
            return F.mse_loss(x_recon, x, reduction="mean")
        elif self.reconstruction_loss_type == "bce":
            return F.binary_cross_entropy_with_logits(x_recon, x, reduction="mean")
        elif self.reconstruction_loss_type == "gaussian":
            # Gaussian NLL with learned variance
            if isinstance(x_recon, tuple):
                mu, logvar = x_recon
                return F.gaussian_nll_loss(mu, x, logvar.exp(), reduction="mean")
            return F.mse_loss(x_recon, x, reduction="mean")
        else:
            return F.mse_loss(x_recon, x, reduction="mean")
    
    def _compute_kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence from N(0, I).
        
        KL(N(μ, σ²) || N(0, I)) = -0.5 * Σ(1 + log σ² - μ² - σ²)
        """
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()
    
    def _get_current_beta(self) -> float:
        """Get current beta value based on schedule."""
        if self.beta_schedule == "constant":
            return self.beta
        elif self.beta_schedule == "cyclic":
            # Cyclic beta schedule
            cycle_length = 10000
            cycle_pos = self._global_step % cycle_length
            cycle_progress = cycle_pos / cycle_length
            return self.beta * (0.5 + 0.5 * math.cos(math.pi * cycle_progress))
        elif self.beta_schedule == "monotonic":
            # Monotonically increasing beta
            warmup_steps = 10000
            if self._global_step < warmup_steps:
                return self.beta * (self._global_step / warmup_steps)
            return self.beta
        return self.beta
    
    def compute_elbo(
        self,
        x: torch.Tensor,
        n_samples: int = 1
    ) -> torch.Tensor:
        """
        Compute importance weighted ELBO.
        
        Args:
            x: Input tensor
            n_samples: Number of importance samples
            
        Returns:
            ELBO estimate
        """
        if n_samples == 1:
            loss = self.compute_loss(x, return_components=True)
            return -loss["total_loss"]
            
        # Importance weighted
        elbos = []
        for _ in range(n_samples):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z)
            
            log_p_x_z = -self._compute_reconstruction_loss(x, x_recon)
            log_p_z = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(dim=-1)
            log_q_z_x = -0.5 * (logvar + math.log(2 * math.pi) + 
                               (z - mu).pow(2) / logvar.exp()).sum(dim=-1)
            
            elbo = log_p_x_z + log_p_z - log_q_z_x
            elbos.append(elbo)
            
        # Importance weighted average
        elbos = torch.stack(elbos, dim=0)
        return torch.logsumexp(elbos, dim=0).mean() - math.log(n_samples)
    
    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returns reconstruction, mu, logvar.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x, **kwargs)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, **kwargs)
        return x_recon, mu, logvar


class BetaVAEGenerator(VAEGenerator):
    """
    β-VAE Generator for disentangled representation learning.
    
    Higher β values encourage disentanglement but may hurt reconstruction.
    
    Loss:
        L = L_recon + β · L_KL
        
    Typical β values:
    - β = 1: Standard VAE
    - β = 4: Good disentanglement
    - β > 10: Very disentangled but poor reconstruction
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int = 512,
        beta: float = 4.0,
        gamma: Optional[float] = None,  # For β-TC-VAE
        name: str = "beta_vae_generator",
        **kwargs
    ):
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            beta=beta,
            name=name,
            **kwargs
        )
        self.gamma = gamma
        
    def _compute_kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL with total correlation term for β-TC-VAE.
        
        Decomposes KL into:
        - Index-wise KL
        - Total Correlation
        - Dimension-wise KL
        """
        if self.gamma is None:
            return super()._compute_kl_divergence(mu, logvar)
            
        batch_size = mu.shape[0]
        
        # Standard KL
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total Correlation
        # log q(z) - Σ log q(z_j)
        log_q_z = self._compute_log_density(mu, logvar)
        log_q_z_product = self._compute_log_product_density(mu, logvar)
        tc = (log_q_z - log_q_z_product).mean()
        
        # Total loss
        return (self.beta * kl.mean() + self.gamma * tc)


class ConditionalVAEGenerator(ConditionalGenerator, VAEGenerator):
    """
    Conditional VAE (CVAE) Generator.
    
    Conditions the generation on labels or continuous attributes.
    
    Architecture:
        Encoder: (x, c) → (μ, log σ²) → z
        Decoder: (z, c) → x̂
        
    Example:
        >>> generator = ConditionalVAEGenerator(
        ...     encoder=encoder,
        ...     decoder=decoder,
        ...     latent_dim=512,
        ...     num_classes=10
        ... )
        >>> samples = generator.generate(100, class_label=3)
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dim: int = 512,
        num_classes: Optional[int] = None,
        condition_dim: Optional[int] = None,
        beta: float = 1.0,
        name: str = "conditional_vae_generator",
        **kwargs
    ):
        """
        Initialize CVAE generator.
        
        Args:
            encoder: Encoder network
            decoder: Decoder network
            latent_dim: Latent dimension
            num_classes: Number of classes
            condition_dim: Continuous condition dimension
            beta: KL weight
            name: Generator name
        """
        ConditionalGenerator.__init__(
            self,
            name=name,
            latent_dim=latent_dim,
            num_classes=num_classes,
            condition_dim=condition_dim
        )
        VAEGenerator.__init__(
            self,
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            beta=beta,
            name=name,
            **kwargs
        )
        
    def encode(
        self,
        x: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode with conditioning.
        
        Args:
            x: Input tensor
            class_label: Class label
            condition: Continuous condition
            
        Returns:
            Tuple of (mu, logvar)
        """
        # Get condition embedding
        if class_label is not None and self.class_embedding is not None:
            c_emb = self.class_embedding(class_label)
            x = torch.cat([x, c_emb], dim=-1)
        if condition is not None and self.condition_projection is not None:
            c_emb = self.condition_projection(condition)
            x = torch.cat([x, c_emb], dim=-1)
            
        return super().encode(x, **kwargs)
    
    def decode(
        self,
        z: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Decode with conditioning.
        
        Args:
            z: Latent tensor
            class_label: Class label
            condition: Continuous condition
            
        Returns:
            Decoded output
        """
        z = self.combine_latent_condition(z, class_label, condition)
        return self.decoder(z, **kwargs)
    
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
        return self.decode(z, class_label, condition, **kwargs)
    
    def generate_counterfactual(
        self,
        z: torch.Tensor,
        original_sensitive: torch.Tensor,
        target_sensitive: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate counterfactual by changing sensitive attribute.
        
        Args:
            z: Original latent
            original_sensitive: Original sensitive values
            target_sensitive: Target sensitive values
            
        Returns:
            Counterfactual output
        """
        return self.decode(z, condition=target_sensitive.float(), **kwargs)


class VAEGANGenerator(VAEGenerator):
    """
    VAE-GAN Generator combining VAE reconstruction with GAN adversarial loss.
    
    Architecture:
        Encoder: x → z
        Generator: z → x̂
        Discriminator: x or x̂ → real/fake
        
    Loss:
        L_VAE = L_recon + β·KL
        L_GAN = Adversarial loss
        L_total = L_VAE + λ·L_GAN
        
    Example:
        >>> generator = VAEGANGenerator(
        ...     encoder=encoder,
        ...     decoder=decoder,
        ...     discriminator=discriminator,
        ...     latent_dim=512
        ... )
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        discriminator: nn.Module,
        latent_dim: int = 512,
        beta: float = 1.0,
        adversarial_weight: float = 0.5,
        feature_matching: bool = True,
        name: str = "vae_gan_generator",
        **kwargs
    ):
        """
        Initialize VAE-GAN generator.
        
        Args:
            encoder: Encoder network
            decoder: Decoder/Generator network
            discriminator: Discriminator network
            latent_dim: Latent dimension
            beta: KL weight
            adversarial_weight: Weight for adversarial loss
            feature_matching: Use feature matching loss
            name: Generator name
        """
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=latent_dim,
            beta=beta,
            name=name,
            **kwargs
        )
        
        self.discriminator = discriminator
        self.adversarial_weight = adversarial_weight
        self.feature_matching = feature_matching
        
    def compute_loss(
        self,
        x: torch.Tensor,
        return_components: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute VAE-GAN loss.
        
        Args:
            x: Input tensor
            return_components: Return loss components
            
        Returns:
            Total loss or dictionary
        """
        # VAE forward pass
        mu, logvar = self.encode(x, **kwargs)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, **kwargs)
        
        # VAE losses
        recon_loss = self._compute_reconstruction_loss(x, x_recon)
        kl_loss = self._compute_kl_divergence(mu, logvar)
        current_beta = self._get_current_beta()
        
        vae_loss = recon_loss + current_beta * kl_loss
        
        # GAN losses
        # Real samples
        d_real = self.discriminator(x)
        # Fake samples
        d_fake = self.discriminator(x_recon.detach())
        
        # Discriminator loss
        d_loss_real = F.binary_cross_entropy_with_logits(
            d_real, torch.ones_like(d_real)
        )
        d_loss_fake = F.binary_cross_entropy_with_logits(
            d_fake, torch.zeros_like(d_fake)
        )
        d_loss = (d_loss_real + d_loss_fake) / 2
        
        # Generator adversarial loss
        d_fake_for_g = self.discriminator(x_recon)
        g_adv_loss = F.binary_cross_entropy_with_logits(
            d_fake_for_g, torch.ones_like(d_fake_for_g)
        )
        
        # Feature matching loss
        if self.feature_matching:
            fm_loss = self._compute_feature_matching_loss(x, x_recon)
        else:
            fm_loss = torch.tensor(0.0, device=x.device)
        
        # Total generator loss
        total_loss = (
            vae_loss + 
            self.adversarial_weight * g_adv_loss +
            0.1 * fm_loss
        )
        
        if return_components:
            return {
                "total_loss": total_loss,
                "vae_loss": vae_loss,
                "reconstruction_loss": recon_loss,
                "kl_loss": kl_loss,
                "generator_adv_loss": g_adv_loss,
                "discriminator_loss": d_loss,
                "feature_matching_loss": fm_loss,
                "beta": current_beta,
            }
            
        return total_loss
    
    def _compute_feature_matching_loss(
        self,
        x_real: torch.Tensor,
        x_fake: torch.Tensor
    ) -> torch.Tensor:
        """Compute feature matching loss on discriminator features."""
        # Get intermediate features
        real_features = self._get_discriminator_features(x_real)
        fake_features = self._get_discriminator_features(x_fake)
        
        if real_features is None or fake_features is None:
            return torch.tensor(0.0, device=x_real.device)
        
        # Match features
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss = loss + F.l1_loss(fake_feat, real_feat.detach())
            
        return loss
    
    def _get_discriminator_features(
        self,
        x: torch.Tensor
    ) -> Optional[List[torch.Tensor]]:
        """Get intermediate features from discriminator."""
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
            
        # Register hooks on intermediate layers
        hooks = []
        for module in self.discriminator.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn))
                
        # Forward pass
        _ = self.discriminator(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return features if features else None
    
    def get_discriminator_loss(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Get discriminator loss for separate training.
        
        Args:
            x: Real input tensor
            
        Returns:
            Discriminator loss
        """
        with torch.no_grad():
            mu, logvar = self.encode(x, **kwargs)
            z = self.reparameterize(mu, logvar)
            x_recon = self.decode(z, **kwargs)
        
        d_real = self.discriminator(x)
        d_fake = self.discriminator(x_recon)
        
        d_loss_real = F.binary_cross_entropy_with_logits(
            d_real, torch.ones_like(d_real)
        )
        d_loss_fake = F.binary_cross_entropy_with_logits(
            d_fake, torch.zeros_like(d_fake)
        )
        
        return (d_loss_real + d_loss_fake) / 2


class HierarchicalVAEGenerator(VAEGenerator):
    """
    Hierarchical VAE with multiple latent levels.
    
    Architecture:
        x → L1 → L2 → ... → Ln (bottom-up)
        Ln → L{n-1} → ... → L1 → x̂ (top-down)
        
    Each level captures different levels of abstraction.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_dims: List[int] = [256, 512, 1024],
        beta: float = 1.0,
        name: str = "hierarchical_vae_generator",
        **kwargs
    ):
        """
        Initialize hierarchical VAE.
        
        Args:
            encoder: Hierarchical encoder
            decoder: Hierarchical decoder
            latent_dims: Latent dimensions for each level
            beta: KL weight (can be list for per-level weights)
            name: Generator name
        """
        # Use largest latent dim as the main dimension
        super().__init__(
            encoder=encoder,
            decoder=decoder,
            latent_dim=max(latent_dims),
            beta=beta,
            name=name,
            **kwargs
        )
        
        self.latent_dims = latent_dims
        self.num_levels = len(latent_dims)
        
        # Per-level beta weights
        if isinstance(beta, list):
            self.beta_per_level = beta
        else:
            self.beta_per_level = [beta] * self.num_levels
    
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
        return self.encoder(x, **kwargs)
    
    def reparameterize(
        self,
        mus: List[torch.Tensor],
        logvars: List[torch.Tensor],
        training: bool = True
    ) -> List[torch.Tensor]:
        """
        Reparameterize each level.
        
        Args:
            mus: List of means
            logvars: List of log-variances
            training: Training mode
            
        Returns:
            List of sampled latents
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
    
    def decode(
        self,
        z_list: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Decode hierarchical latents.
        
        Args:
            z_list: List of latent tensors
            
        Returns:
            Decoded output
        """
        return self.decoder(z_list, **kwargs)
    
    def sample_latent(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> List[torch.Tensor]:
        """
        Sample from hierarchical priors.
        
        Args:
            n_samples: Number of samples
            device: Device for samples
            
        Returns:
            List of latent samples
        """
        if device is None:
            device = next(self.parameters()).device
            
        return [
            torch.randn(n_samples, dim, device=device)
            for dim in self.latent_dims
        ]
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate from hierarchical priors.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Generated samples
        """
        z_list = self.sample_latent(n_samples)
        return self.decode(z_list, **kwargs)
    
    def compute_loss(
        self,
        x: torch.Tensor,
        return_components: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute hierarchical VAE loss.
        
        Args:
            x: Input tensor
            return_components: Return components
            
        Returns:
            Total loss or dictionary
        """
        # Encode
        mus, logvars = self.encode(x, **kwargs)
        
        # Sample
        z_list = self.reparameterize(mus, logvars)
        
        # Decode
        x_recon = self.decode(z_list, **kwargs)
        
        # Reconstruction loss
        recon_loss = self._compute_reconstruction_loss(x, x_recon)
        
        # KL loss per level
        kl_losses = []
        for i, (mu, logvar) in enumerate(zip(mus, logvars)):
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
            kl_losses.append(self.beta_per_level[i] * kl.mean())
        
        total_kl = sum(kl_losses)
        total_loss = recon_loss + total_kl
        
        if return_components:
            components = {
                "total_loss": total_loss,
                "reconstruction_loss": recon_loss,
                "total_kl": total_kl,
            }
            for i, kl in enumerate(kl_losses):
                components[f"kl_level_{i}"] = kl
            return components
            
        return total_loss
    
    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstruction, mus, logvars)
        """
        mus, logvars = self.encode(x, **kwargs)
        z_list = self.reparameterize(mus, logvars)
        x_recon = self.decode(z_list, **kwargs)
        return x_recon, mus, logvars
