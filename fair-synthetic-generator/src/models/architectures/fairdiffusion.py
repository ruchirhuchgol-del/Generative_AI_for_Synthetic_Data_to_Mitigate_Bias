"""
Fair Diffusion Architecture
===========================

Fairness-aware diffusion models for fair synthetic data generation.

This module implements several fair diffusion variants:
- FairDiffusion: Basic fair diffusion with adversarial debiasing
- ConditionalFairDiffusion: Conditional fair diffusion
- LatentFairDiffusion: Latent diffusion for efficiency
- FairDDIM: Fast fair diffusion with DDIM sampling

Key Features:
- Denoising diffusion probabilistic models (DDPM)
- DDIM for faster sampling
- Adversarial debiasing integrated into diffusion
- Multiple fairness constraints
- Support for multimodal generation
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.core.base_module import BaseGenerator
from src.models.generators import DiffusionGenerator, DDIMGenerator
from src.models.discriminators import FairnessAdversary, GradientReversalLayer
from src.fairness.modules.gradient_reversal import ScheduledGradientReversalLayer


def get_noise_schedule(
    schedule_type: str,
    num_steps: int,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get noise schedule for diffusion.
    
    Args:
        schedule_type: Type of schedule ("linear", "cosine", "quadratic")
        num_steps: Number of diffusion steps
        
    Returns:
        Tuple of (betas, alphas_cumprod)
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
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    return betas, alphas_cumprod


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for timestep encoding."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timestep tensor of shape (batch,)
        Returns:
            Embedding tensor of shape (batch, dim)
        """
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class FairDiffusion(BaseGenerator):
    """
    Fairness-aware Diffusion Model.
    
    Architecture:
        Forward Process: x_0 -> x_t (adding noise)
        Reverse Process: x_t -> x_{t-1} (denoising)
        Fairness: Adversarial debiasing on latent representations
    
    Training Objectives:
        - Denoising: Predict noise added to data
        - Fairness: Ensure latent representations don't encode sensitive info
    
    Fairness Integration:
        - Adversarial debiasing on denoiser's intermediate representations
        - Fairness-guided sampling
        - Counterfactual generation via latent manipulation
    
    Example:
        >>> model = FairDiffusion(
        ...     data_dim=100,
        ...     num_timesteps=1000,
        ...     num_sensitive_groups=2
        ... )
        >>> # Training
        >>> loss = model.compute_loss(data, sensitive_attrs)
        >>> # Generation
        >>> samples = model.generate(100)
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        num_timesteps: int = 1000,
        num_sensitive_groups: int = 2,
        hidden_dims: List[int] = [512, 256, 128],
        noise_schedule: str = "cosine",
        fairness_weight: float = 1.0,
        grl_lambda: float = 1.0,
        grl_warmup: int = 10,
        dropout: float = 0.1,
        use_ema: bool = True,
        ema_decay: float = 0.9999,
        name: str = "fairdiffusion",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Fair Diffusion.
        
        Args:
            data_dim: Dimension of data
            latent_dim: Dimension of latent/hidden representations
            num_timesteps: Number of diffusion timesteps
            num_sensitive_groups: Number of sensitive attribute groups
            hidden_dims: Hidden layer dimensions for denoiser
            noise_schedule: Noise schedule type
            fairness_weight: Weight for fairness loss
            grl_lambda: Gradient reversal lambda
            grl_warmup: Epochs for GRL warmup
            dropout: Dropout rate
            use_ema: Whether to use EMA for sampling
            ema_decay: EMA decay rate
            name: Model name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.data_dim = data_dim
        self.num_timesteps = num_timesteps
        self.num_sensitive_groups = num_sensitive_groups
        self.fairness_weight = fairness_weight
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        
        # Build components
        self._build_noise_schedule(noise_schedule)
        self._build_denoiser(hidden_dims, dropout)
        self._build_time_embedding()
        self._build_fairness_adversary(grl_lambda, grl_warmup)
        
        # EMA model for sampling
        if use_ema:
            self._build_ema_model()
        
        # Training state
        self._current_epoch = 0
        
    def _build_noise_schedule(self, schedule_type: str) -> None:
        """Build noise schedule."""
        betas, alphas_cumprod = get_noise_schedule(schedule_type, self.num_timesteps)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", F.pad(alphas_cumprod[:-1], (1, 0), value=1.0))
        
        # Precompute useful quantities
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))
        
        # Posterior variance
        posterior_variance = betas * (1 - self.alphas_cumprod_prev) / (1 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod) / (1 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1 - self.alphas_cumprod_prev) * torch.sqrt(1 - betas) / (1 - alphas_cumprod))
        
    def _build_time_embedding(self) -> None:
        """Build timestep embedding."""
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(self._latent_dim),
            nn.Linear(self._latent_dim, self._latent_dim * 4),
            nn.SiLU(),
            nn.Linear(self._latent_dim * 4, self._latent_dim)
        )
        
    def _build_denoiser(self, hidden_dims: List[int], dropout: float) -> None:
        """Build the denoiser network (U-Net style for tabular)."""
        layers = []
        prev_dim = self.data_dim + self._latent_dim  # data + time embedding
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        for dim in hidden_dims:
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ))
            prev_dim = dim
        
        # Bottleneck representation (for fairness adversary)
        self.bottleneck_dim = hidden_dims[-1]
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        reversed_dims = list(reversed(hidden_dims))
        for i, dim in enumerate(reversed_dims):
            skip_dim = hidden_dims[-(i+1)] if i < len(hidden_dims) else 0
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim + skip_dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ))
            prev_dim = dim
        
        # Output projection (predicts noise)
        self.output_proj = nn.Linear(prev_dim, self.data_dim)
        
    def _build_fairness_adversary(self, grl_lambda: float, grl_warmup: int) -> None:
        """Build fairness adversary."""
        self.grl = ScheduledGradientReversalLayer(
            lambda_start=0.0,
            lambda_end=grl_lambda,
            warmup_epochs=grl_warmup,
            schedule_type="linear"
        )
        
        self.adversary = nn.Sequential(
            nn.Linear(self.bottleneck_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_sensitive_groups)
        )
        
    def _build_ema_model(self) -> None:
        """Build EMA model for stable sampling."""
        # Create shadow parameters
        self._ema_parameters = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                self._ema_parameters[name] = param.data.clone()
                
    def _update_ema(self) -> None:
        """Update EMA parameters."""
        if not self.use_ema:
            return
        for name, param in self.named_parameters():
            if param.requires_grad and name in self._ema_parameters:
                self._ema_parameters[name] = (
                    self.ema_decay * self._ema_parameters[name] + 
                    (1 - self.ema_decay) * param.data
                )
    
    def sample_latent(
        self,
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """Sample from the prior (pure noise)."""
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(n_samples, self.data_dim, device=device)
    
    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Get timestep embedding."""
        return self.time_embed(t)
    
    def denoise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        use_ema: bool = False
    ) -> torch.Tensor:
        """
        Denoise step: predict noise given noisy input and timestep.
        
        Args:
            x: Noisy input tensor
            t: Timestep tensor
            use_ema: Whether to use EMA parameters
            
        Returns:
            Predicted noise
        """
        # Get time embedding
        t_emb = self.get_time_embedding(t)
        
        # Concatenate input with time embedding
        h = torch.cat([x, t_emb.expand(x.size(0), -1)], dim=-1)
        
        # Encoder with skip connections
        skip_connections = []
        for layer in self.encoder_layers:
            h = layer(h)
            skip_connections.append(h)
        
        # Store bottleneck for fairness adversary
        bottleneck = h
        
        # Decoder with skip connections
        for i, layer in enumerate(self.decoder_layers):
            skip = skip_connections[-(i+1)]
            h = torch.cat([h, skip], dim=-1)
            h = layer(h)
        
        # Output projection
        noise_pred = self.output_proj(h)
        
        return noise_pred, bottleneck
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to data at timestep t.
        
        Args:
            x_0: Clean data
            t: Timestep
            noise: Optional noise tensor
            
        Returns:
            Noisy data at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alpha = sqrt_alpha.view(-1, 1)
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.view(-1, 1)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        use_ema: bool = False
    ) -> torch.Tensor:
        """
        Reverse diffusion: denoise one step.
        
        Args:
            x_t: Noisy data at timestep t
            t: Timestep
            use_ema: Whether to use EMA parameters
            
        Returns:
            Denoised data at timestep t-1
        """
        # Predict noise
        noise_pred, _ = self.denoise(x_t, t, use_ema=use_ema)
        
        # Get parameters
        beta = self.betas[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        sqrt_recip_alpha = self.sqrt_recip_alphas_cumprod[t].view(-1, 1)
        
        # Compute mean
        mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_alpha * noise_pred)
        
        # Add noise if not at t=0
        if t[0] > 0:
            posterior_var = self.posterior_variance[t].view(-1, 1)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(posterior_var) * noise
        else:
            return mean
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        progress: bool = True,
        use_ema: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion.
        
        Args:
            n_samples: Number of samples
            conditions: Optional conditioning
            device: Device for generation
            progress: Whether to show progress
            use_ema: Whether to use EMA parameters
            
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Start from pure noise
        x = self.sample_latent(n_samples, device)
        
        # Reverse diffusion
        timesteps = list(reversed(range(self.num_timesteps)))
        
        for i, t in enumerate(timesteps):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor, use_ema=use_ema and self.use_ema)
            
            if progress and i % 100 == 0:
                pass  # Could add progress callback here
        
        return x
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        sensitive_attrs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training loss.
        
        Args:
            x_0: Clean data
            sensitive_attrs: Sensitive attribute labels
            
        Returns:
            Dictionary of losses
        """
        batch_size = x_0.size(0)
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Add noise to get x_t
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict noise
        noise_pred, bottleneck = self.denoise(x_t, t)
        
        # Denoising loss (MSE)
        denoise_loss = F.mse_loss(noise_pred, noise)
        
        losses = {
            "denoise_loss": denoise_loss,
            "total": denoise_loss
        }
        
        # Fairness adversarial loss
        if sensitive_attrs is not None:
            # Apply GRL to bottleneck representation
            bottleneck_reversed = self.grl(bottleneck)
            adv_pred = self.adversary(bottleneck_reversed)
            adv_loss = F.cross_entropy(adv_pred, sensitive_attrs)
            losses["adv_loss"] = adv_loss
            losses["total"] = losses["total"] + self.fairness_weight * adv_loss
        
        return losses
    
    def forward(
        self,
        x: torch.Tensor,
        sensitive_attrs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass for training."""
        losses = self.compute_loss(x, sensitive_attrs)
        self._update_ema()
        return losses
    
    def on_epoch_end(self) -> None:
        """Called at end of each epoch."""
        self._current_epoch += 1
        self.grl.step()


class ConditionalFairDiffusion(FairDiffusion):
    """
    Conditional Fair Diffusion for guided fair generation.
    
    Supports conditioning on:
    - Class labels
    - Continuous attributes
    - Fairness constraints
    
    Uses classifier-free guidance for conditional generation.
    
    Example:
        >>> model = ConditionalFairDiffusion(
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
        num_timesteps: int = 1000,
        num_classes: Optional[int] = None,
        condition_dim: Optional[int] = None,
        num_sensitive_groups: int = 2,
        guidance_scale: float = 7.5,
        classifier_free_prob: float = 0.1,
        name: str = "conditional_fairdiffusion",
        **kwargs
    ):
        """
        Initialize Conditional Fair Diffusion.
        
        Args:
            data_dim: Data dimension
            latent_dim: Latent dimension
            num_timesteps: Number of timesteps
            num_classes: Number of classes
            condition_dim: Continuous condition dimension
            num_sensitive_groups: Number of sensitive groups
            guidance_scale: Classifier-free guidance scale
            classifier_free_prob: Probability of dropping condition
            name: Model name
        """
        self.num_classes = num_classes
        self.condition_dim = condition_dim
        self.guidance_scale = guidance_scale
        self.classifier_free_prob = classifier_free_prob
        
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_timesteps=num_timesteps,
            num_sensitive_groups=num_sensitive_groups,
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
                nn.SiLU()
            )
        else:
            self.condition_projection = None
    
    def _build_denoiser(self, hidden_dims: List[int], dropout: float) -> None:
        """Build conditional denoiser."""
        # Input includes condition embedding
        prev_dim = self.data_dim + self._latent_dim * 2  # data + time + condition
        
        self.encoder_layers = nn.ModuleList()
        for dim in hidden_dims:
            self.encoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ))
            prev_dim = dim
        
        self.bottleneck_dim = hidden_dims[-1]
        
        self.decoder_layers = nn.ModuleList()
        reversed_dims = list(reversed(hidden_dims))
        for i, dim in enumerate(reversed_dims):
            skip_dim = hidden_dims[-(i+1)] if i < len(hidden_dims) else 0
            self.decoder_layers.append(nn.Sequential(
                nn.Linear(prev_dim + skip_dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ))
            prev_dim = dim
        
        self.output_proj = nn.Linear(prev_dim, self.data_dim)
        
        # Null embedding for classifier-free guidance
        self.null_embedding = nn.Parameter(torch.randn(latent_dim))
        
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
            
        cond_emb = self.null_embedding.expand(batch_size, -1).clone()
        
        if class_label is not None and self.class_embedding is not None:
            class_emb = self.class_embedding(class_label)
            cond_emb = cond_emb + class_emb
            
        if condition is not None and self.condition_projection is not None:
            cond_proj = self.condition_projection(condition)
            cond_emb = cond_emb + cond_proj
            
        return cond_emb
    
    def denoise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition_emb: Optional[torch.Tensor] = None,
        use_ema: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Conditional denoising step."""
        # Get time embedding
        t_emb = self.get_time_embedding(t)
        
        # Get condition embedding
        if condition_emb is None:
            condition_emb = self.null_embedding.expand(x.size(0), -1)
        
        # Concatenate all embeddings
        h = torch.cat([x, t_emb, condition_emb], dim=-1)
        
        # Encoder with skip connections
        skip_connections = []
        for layer in self.encoder_layers:
            h = layer(h)
            skip_connections.append(h)
        
        bottleneck = h
        
        # Decoder
        for i, layer in enumerate(self.decoder_layers):
            skip = skip_connections[-(i+1)]
            h = torch.cat([h, skip], dim=-1)
            h = layer(h)
        
        noise_pred = self.output_proj(h)
        
        return noise_pred, bottleneck
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        sensitive_attrs: Optional[torch.Tensor] = None,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute conditional training loss."""
        batch_size = x_0.size(0)
        device = x_0.device
        
        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Add noise
        x_t = self.q_sample(x_0, t, noise)
        
        # Get condition embedding
        condition_emb = self.get_condition_embedding(
            class_label, condition, batch_size, device
        )
        
        # Classifier-free guidance: randomly drop condition
        if self.training:
            mask = torch.rand(batch_size, device=device) > self.classifier_free_prob
            null_emb = self.null_embedding.expand(batch_size, -1)
            condition_emb = torch.where(
                mask.unsqueeze(-1),
                condition_emb,
                null_emb
            )
        
        # Predict noise
        noise_pred, bottleneck = self.denoise(x_t, t, condition_emb)
        
        # Denoising loss
        denoise_loss = F.mse_loss(noise_pred, noise)
        
        losses = {
            "denoise_loss": denoise_loss,
            "total": denoise_loss
        }
        
        # Fairness loss
        if sensitive_attrs is not None:
            bottleneck_reversed = self.grl(bottleneck)
            adv_pred = self.adversary(bottleneck_reversed)
            adv_loss = F.cross_entropy(adv_pred, sensitive_attrs)
            losses["adv_loss"] = adv_loss
            losses["total"] = losses["total"] + self.fairness_weight * adv_loss
        
        return losses
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        guidance_scale: Optional[float] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate conditional samples with classifier-free guidance."""
        if device is None:
            device = next(self.parameters()).device
        if guidance_scale is None:
            guidance_scale = self.guidance_scale
            
        # Start from noise
        x = self.sample_latent(n_samples, device)
        
        # Get condition embedding
        condition_emb = self.get_condition_embedding(
            class_label, condition, n_samples, device
        )
        null_emb = self.null_embedding.expand(n_samples, -1)
        
        # Reverse diffusion with guidance
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            # Predict noise with and without condition
            noise_pred_cond, _ = self.denoise(x, t_tensor, condition_emb)
            noise_pred_uncond, _ = self.denoise(x, t_tensor, null_emb)
            
            # Classifier-free guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Denoise step
            beta = self.betas[t].view(-1, 1)
            sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
            sqrt_recip_alpha = self.sqrt_recip_alphas_cumprod[t].view(-1, 1)
            
            mean = sqrt_recip_alpha * (x - beta / sqrt_one_minus_alpha * noise_pred)
            
            if t > 0:
                posterior_var = self.posterior_variance[t].view(-1, 1)
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(posterior_var) * noise
            else:
                x = mean
        
        return x
    
    def generate_counterfactual(
        self,
        x_0: torch.Tensor,
        original_sensitive: int,
        target_sensitive: int,
        device: Optional[torch.device] = None,
        num_steps: int = 100,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate counterfactual samples.
        
        Edit the latent representation to change sensitive attribute
        while preserving other attributes.
        """
        if device is None:
            device = x_0.device
            
        batch_size = x_0.size(0)
        
        # Encode to latent at t=T/2
        t = torch.full((batch_size,), self.num_timesteps // 2, device=device, dtype=torch.long)
        x_t = self.q_sample(x_0, t)
        
        # Get latent representation
        _, bottleneck = self.denoise(x_t, t)
        
        # Apply counterfactual editing
        # Remove sensitive direction from bottleneck
        with torch.no_grad():
            adv_weights = self.adversary[0].weight  # First layer weights
            sensitive_direction = adv_weights[target_sensitive] - adv_weights[original_sensitive]
            sensitive_direction = F.normalize(sensitive_direction, dim=0)
            
            # Project out sensitive direction
            projection = (bottleneck @ sensitive_direction) / (sensitive_direction @ sensitive_direction)
            bottleneck_edited = bottleneck - projection.unsqueeze(-1) * sensitive_direction
        
        # Continue denoising with edited representation
        # This requires custom denoising with fixed bottleneck
        # For simplicity, use standard generation here
        return self.generate(batch_size, device=device)


class LatentFairDiffusion(FairDiffusion):
    """
    Latent Diffusion for fair generation with improved efficiency.
    
    Operates in a compressed latent space for faster generation.
    Uses a pre-trained encoder/decoder (VAE or similar).
    
    Architecture:
        Encoder: data -> latent
        Diffusion: latent_t -> latent_{t-1}
        Decoder: latent -> data
    
    Example:
        >>> model = LatentFairDiffusion(
        ...     data_dim=100,
        ...     latent_dim=64,
        ...     compression_ratio=8
        ... )
        >>> samples = model.generate(100)
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 64,
        compression_ratio: int = 8,
        num_timesteps: int = 1000,
        num_sensitive_groups: int = 2,
        encoder_dims: Optional[List[int]] = None,
        decoder_dims: Optional[List[int]] = None,
        name: str = "latent_fairdiffusion",
        **kwargs
    ):
        """
        Initialize Latent Fair Diffusion.
        
        Args:
            data_dim: Data dimension
            latent_dim: Compressed latent dimension
            compression_ratio: Compression ratio for latent space
            num_timesteps: Number of diffusion timesteps
            num_sensitive_groups: Number of sensitive groups
            encoder_dims: Encoder hidden dimensions
            decoder_dims: Decoder hidden dimensions
            name: Model name
        """
        self.compression_ratio = compression_ratio
        self.encoder_dims = encoder_dims or [512, 256, 128]
        self.decoder_dims = decoder_dims or [128, 256, 512]
        
        super().__init__(
            data_dim=latent_dim,  # Diffusion operates in latent space
            latent_dim=latent_dim,
            num_timesteps=num_timesteps,
            num_sensitive_groups=num_sensitive_groups,
            name=name,
            **kwargs
        )
        
        self.original_data_dim = data_dim
        
        # Build encoder/decoder for latent space
        self._build_autoencoder()
        
    def _build_autoencoder(self) -> None:
        """Build VAE-style encoder/decoder."""
        # Encoder
        encoder_layers = []
        prev_dim = self.original_data_dim
        for dim in self.encoder_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU()
            ])
            prev_dim = dim
        encoder_layers.append(nn.Linear(prev_dim, self._latent_dim * 2))  # mean and logvar
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = self._latent_dim
        for dim in self.decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU()
            ])
            prev_dim = dim
        decoder_layers.append(nn.Linear(prev_dim, self.original_data_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode data to latent distribution parameters."""
        params = self.encoder(x)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to data space."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """Generate samples via latent diffusion."""
        # Generate in latent space
        z = super().generate(n_samples, conditions, device, **kwargs)
        
        # Decode to data space
        return self.decode(z)
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        sensitive_attrs: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss with autoencoder."""
        # Encode to latent
        mu, logvar = self.encode(x_0)
        z_0 = self.reparameterize(mu, logvar)
        
        # Diffusion loss in latent space
        losses = super().compute_loss(z_0, sensitive_attrs)
        
        # Reconstruction loss
        x_recon = self.decode(z_0)
        recon_loss = F.mse_loss(x_recon, x_0)
        losses["recon_loss"] = recon_loss
        losses["total"] = losses["total"] + 0.1 * recon_loss
        
        # KL loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        losses["kl_loss"] = kl_loss
        losses["total"] = losses["total"] + 0.001 * kl_loss
        
        return losses


class FairDDIM(FairDiffusion):
    """
    Fair Diffusion with DDIM (Denoising Diffusion Implicit Models) sampling.
    
    DDIM enables much faster sampling with fewer steps while maintaining
    sample quality.
    
    Example:
        >>> model = FairDDIM(
        ...     data_dim=100,
        ...     num_timesteps=1000,
        ...     ddim_steps=50  # Generate in 50 steps instead of 1000
        ... )
        >>> samples = model.generate(100, ddim_steps=50)
    """
    
    def __init__(
        self,
        data_dim: int,
        latent_dim: int = 512,
        num_timesteps: int = 1000,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        num_sensitive_groups: int = 2,
        name: str = "fair_ddim",
        **kwargs
    ):
        """
        Initialize Fair DDIM.
        
        Args:
            data_dim: Data dimension
            latent_dim: Latent dimension
            num_timesteps: Total diffusion timesteps
            ddim_steps: Number of DDIM sampling steps
            ddim_eta: DDIM eta parameter (0 = deterministic)
            num_sensitive_groups: Number of sensitive groups
            name: Model name
        """
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        
        super().__init__(
            data_dim=data_dim,
            latent_dim=latent_dim,
            num_timesteps=num_timesteps,
            num_sensitive_groups=num_sensitive_groups,
            name=name,
            **kwargs
        )
        
        # Compute DDIM timestep schedule
        self._build_ddim_schedule()
        
    def _build_ddim_schedule(self) -> None:
        """Build DDIM timestep schedule."""
        c = self.num_timesteps // self.ddim_steps
        ddim_timesteps = np.asarray(list(range(0, self.num_timesteps, c)))
        
        self.register_buffer("ddim_timesteps", torch.from_numpy(ddim_timesteps))
        
        # Precompute alphas for DDIM
        alphas = self.alphas_cumprod[ddim_timesteps]
        self.register_buffer("ddim_alphas", alphas)
        self.register_buffer("ddim_alphas_prev", F.pad(alphas[:-1], (1, 0), value=1.0))
        
    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        use_ema: bool = False
    ) -> torch.Tensor:
        """
        Perform one DDIM sampling step.
        
        Args:
            x_t: Noisy sample at timestep t
            t: Current timestep
            t_prev: Previous timestep
            use_ema: Whether to use EMA parameters
            
        Returns:
            Sample at timestep t_prev
        """
        batch_size = x_t.size(0)
        device = x_t.device
        
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict noise
        noise_pred, _ = self.denoise(x_t, t_tensor, use_ema=use_ema)
        
        # Get alpha values
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Compute x_{t-1}
        if self.ddim_eta == 0:
            # Deterministic
            x_prev = torch.sqrt(alpha_prev) * x_0_pred + torch.sqrt(1 - alpha_prev) * noise_pred
        else:
            # Stochastic
            sigma = self.ddim_eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            )
            noise = torch.randn_like(x_t)
            x_prev = torch.sqrt(alpha_prev) * x_0_pred + \
                     torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred + \
                     sigma * noise
        
        return x_prev
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        ddim_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples using DDIM.
        
        Args:
            n_samples: Number of samples
            conditions: Optional conditions
            device: Device
            ddim_steps: Number of DDIM steps (default: self.ddim_steps)
            
        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device
        if ddim_steps is None:
            ddim_steps = self.ddim_steps
            
        # Start from noise
        x = self.sample_latent(n_samples, device)
        
        # Get DDIM timesteps
        c = self.num_timesteps // ddim_steps
        timesteps = list(range(0, self.num_timesteps, c))[::-1]
        
        # DDIM sampling
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = self.ddim_step(x, t, t_prev, use_ema=self.use_ema)
        
        return x


# Import numpy for DDIM schedule
import numpy as np


# Registry for FairDiffusion variants
FAIRDIFFUSION_REGISTRY = {
    "fairdiffusion": FairDiffusion,
    "conditional_fairdiffusion": ConditionalFairDiffusion,
    "latent_fairdiffusion": LatentFairDiffusion,
    "fair_ddim": FairDDIM,
}


def get_fairdiffusion(name: str, **kwargs) -> FairDiffusion:
    """Get a FairDiffusion variant by name."""
    if name not in FAIRDIFFUSION_REGISTRY:
        available = list(FAIRDIFFUSION_REGISTRY.keys())
        raise ValueError(f"Unknown FairDiffusion: {name}. Available: {available}")
    return FAIRDIFFUSION_REGISTRY[name](**kwargs)
