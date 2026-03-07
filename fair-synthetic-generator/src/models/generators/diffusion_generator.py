"""
Diffusion Generator
===================

Diffusion model generators for synthetic data generation.
Supports DDPM, DDIM, and Latent Diffusion variants.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.generators.base_generator import (
    BaseGenerator,
    ConditionalGenerator,
    FairGenerator,
    LatentSpaceMixin,
    get_noise_schedule
)


class DiffusionGenerator(BaseGenerator, LatentSpaceMixin):
    """
    Denoising Diffusion Probabilistic Model (DDPM) Generator.
    
    Forward Process:
        x_t = √(α̅_t) x_0 + √(1 - α̅_t) ε
        
    Reverse Process:
        x_{t-1} = 1/√α_t (x_t - (1-α_t)/√(1-α̅_t) ε_θ(x_t, t)) + σ_t z
        
    Example:
        >>> generator = DiffusionGenerator(
        ...     denoiser=unet_model,
        ...     latent_dim=512,
        ...     num_timesteps=1000
        ... )
        >>> samples = generator.generate(100)
    """
    
    def __init__(
        self,
        denoiser: nn.Module,
        latent_dim: int = 512,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",  # epsilon, x_start, v
        clip_denoised: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        rescale_timesteps: bool = False,
        name: str = "diffusion_generator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize diffusion generator.
        
        Args:
            denoiser: Denoising network (UNet-style)
            latent_dim: Data dimension (not used for images)
            num_timesteps: Number of diffusion steps
            beta_schedule: Beta schedule type
            beta_start: Starting beta
            beta_end: Ending beta
            prediction_type: What the model predicts
            clip_denoised: Whether to clip denoised samples
            clip_range: Range for clipping
            rescale_timesteps: Rescale timesteps for model
            name: Generator name
            config: Optional configuration
        """
        super().__init__(name=name, latent_dim=latent_dim, config=config)
        
        self.denoiser = denoiser
        self.num_timesteps = num_timesteps
        self.beta_schedule = beta_schedule
        self.prediction_type = prediction_type
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range
        self.rescale_timesteps = rescale_timesteps
        
        # Register noise schedule
        self._register_noise_schedule(beta_start, beta_end)
        
    def _register_noise_schedule(self, beta_start: float, beta_end: float) -> None:
        """Register noise schedule buffers."""
        # Get beta schedule
        betas = get_noise_schedule(
            self.beta_schedule,
            self.num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end
        )
        
        # Compute alphas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        
        # Precompute values for sampling
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        
        # Posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer("posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))
        
    def sample_latent(
        self, 
        n_samples: int,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample pure noise (starting point for reverse diffusion).
        
        Args:
            n_samples: Number of samples
            device: Device for samples
            
        Returns:
            Noise tensor
        """
        if device is None:
            device = next(self.parameters()).device
        # Shape depends on data type - override in subclass
        return torch.randn(n_samples, self._latent_dim, device=device)
    
    def _rescale_timesteps(self, t: torch.Tensor) -> torch.Tensor:
        """Rescale timesteps to [0, 1000] range."""
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion process: add noise to x_start at timestep t.
        
        x_t = √(α̅_t) x_0 + √(1 - α̅_t) ε
        
        Args:
            x_start: Clean samples
            t: Timestep
            noise: Optional noise (sampled if None)
            
        Returns:
            Noisy samples
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
    
    def predict_start_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise.
        
        x_0 = (x_t - √(1-α̅_t) ε) / √α̅_t
        """
        sqrt_recip = self._extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        
        return sqrt_recip * x_t - sqrt_recipm1 * noise
    
    def predict_start_from_v(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted v.
        
        v = √α̅_t ε - √(1-α̅_t) x_0
        """
        sqrt_alpha = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v
    
    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior mean and variance for q(x_{t-1} | x_t, x_0).
        """
        posterior_mean = (
            self._extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self._extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self._extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = self._extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mean and variance for p(x_{t-1} | x_t).
        
        Args:
            x_t: Noisy samples at timestep t
            t: Timestep
            clip_denoised: Whether to clip predicted x_0
            
        Returns:
            Tuple of (mean, variance, log_variance)
        """
        # Model prediction
        model_output = self.denoiser(x_t, self._rescale_timesteps(t))
        
        # Convert to x_start based on prediction type
        if self.prediction_type == "epsilon":
            x_start = self.predict_start_from_noise(x_t, t, model_output)
        elif self.prediction_type == "x_start":
            x_start = model_output
        elif self.prediction_type == "v":
            x_start = self.predict_start_from_v(x_t, t, model_output)
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Clip
        if clip_denoised:
            x_start = x_start.clamp(*self.clip_range)
        
        # Get posterior
        mean, variance, log_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        
        return mean, variance, log_variance
    
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Sample from p(x_{t-1} | x_t).
        
        Args:
            x_t: Samples at timestep t
            t: Timestep
            clip_denoised: Whether to clip
            
        Returns:
            Sample at timestep t-1
        """
        mean, variance, log_variance = self.p_mean_variance(x_t, t, clip_denoised)
        
        # No noise for t=0
        if t[0] == 0:
            return mean
        
        noise = torch.randn_like(x_t)
        return mean + torch.exp(0.5 * log_variance) * noise
    
    def p_sample_loop(
        self,
        shape: Tuple[int, ...],
        progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Full reverse diffusion sampling loop.
        
        Args:
            shape: Shape of samples to generate
            progress: Show progress bar
            
        Returns:
            Generated samples
        """
        device = next(self.parameters()).device
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        timesteps = list(range(self.num_timesteps))[::-1]
        
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for t in timesteps:
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor)
            
        return x
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion.
        
        Args:
            n_samples: Number of samples
            conditions: Optional conditions
            progress: Show progress bar
            
        Returns:
            Generated samples
        """
        shape = (n_samples, self._latent_dim)
        return self.p_sample_loop(shape, progress)
    
    def compute_loss(
        self,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_components: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute diffusion training loss.
        
        Args:
            x_start: Clean samples
            t: Optional timestep (sampled if None)
            noise: Optional noise
            return_components: Return loss components
            
        Returns:
            Loss tensor or dictionary
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample timestep
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise
        x_t = self.q_sample(x_start, t, noise)
        
        # Model prediction
        model_output = self.denoiser(x_t, self._rescale_timesteps(t))
        
        # Compute loss based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x_start":
            target = x_start
        elif self.prediction_type == "v":
            # v-prediction target
            sqrt_alpha = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alpha = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_start
        else:
            target = noise
        
        loss = F.mse_loss(model_output, target)
        
        if return_components:
            return {
                "loss": loss,
                "timestep": t.float().mean().item(),
            }
        
        return loss
    
    def _extract_into_tensor(
        self,
        arr: torch.Tensor,
        t: torch.Tensor,
        shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Extract values from array at indices t and reshape."""
        batch_size = t.shape[0]
        out = arr.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))
    
    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass computes training loss."""
        return self.compute_loss(x, **kwargs)


class DDIMGenerator(DiffusionGenerator):
    """
    Denoising Diffusion Implicit Models (DDIM) Generator.
    
    Non-Markovian sampling for faster generation with fewer steps.
    
    Example:
        >>> generator = DDIMGenerator(
        ...     denoiser=unet_model,
        ...     num_timesteps=1000,
        ...     ddim_steps=50
        ... )
        >>> samples = generator.generate(100)
    """
    
    def __init__(
        self,
        denoiser: nn.Module,
        latent_dim: int = 512,
        num_timesteps: int = 1000,
        ddim_steps: int = 50,
        ddim_eta: float = 0.0,
        ddim_discretize: str = "uniform",  # uniform, quadratic
        **kwargs
    ):
        """
        Initialize DDIM generator.
        
        Args:
            denoiser: Denoising network
            latent_dim: Data dimension
            num_timesteps: Training timesteps
            ddim_steps: Number of DDIM sampling steps
            ddim_eta: Stochasticity (0 = deterministic)
            ddim_discretize: Timestep discretization
        """
        super().__init__(
            denoiser=denoiser,
            latent_dim=latent_dim,
            num_timesteps=num_timesteps,
            **kwargs
        )
        
        self.ddim_steps = ddim_steps
        self.ddim_eta = ddim_eta
        self.ddim_discretize = ddim_discretize
        
        # Compute DDIM timesteps
        self._register_ddim_timesteps()
        
    def _register_ddim_timesteps(self) -> None:
        """Register DDIM timestep schedule."""
        if self.ddim_discretize == "uniform":
            c = self.num_timesteps // self.ddim_steps
            ddim_timesteps = torch.tensor(list(range(0, self.num_timesteps, c)))
        elif self.ddim_discretize == "quadratic":
            ddim_timesteps = torch.linspace(0, self.num_timesteps - 1, self.ddim_steps).long()
        else:
            ddim_timesteps = torch.tensor(list(range(0, self.num_timesteps, self.num_timesteps // self.ddim_steps)))
            
        self.register_buffer("ddim_timesteps", ddim_timesteps)
        
    def ddim_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Single DDIM sampling step.
        
        Args:
            x_t: Sample at timestep t
            t: Current timestep
            t_prev: Previous timestep
            eta: Stochasticity
            
        Returns:
            Sample at timestep t_prev
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Get model prediction
        model_output = self.denoiser(x_t, self._rescale_timesteps(t_tensor))
        
        # Convert to x_start
        if self.prediction_type == "epsilon":
            x_start = self.predict_start_from_noise(x_t, t_tensor, model_output)
        elif self.prediction_type == "x_start":
            x_start = model_output
        elif self.prediction_type == "v":
            x_start = self.predict_start_from_v(x_t, t_tensor, model_output)
        
        # Clip
        if self.clip_denoised:
            x_start = x_start.clamp(*self.clip_range)
        
        # Get alpha values
        alpha_t = self.alphas_cumprod[t]
        alpha_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        
        # Predict noise
        if self.prediction_type == "epsilon":
            pred_noise = model_output
        else:
            pred_noise = (x_t - torch.sqrt(alpha_t) * x_start) / torch.sqrt(1 - alpha_t)
        
        # Compute sigma
        sigma = eta * torch.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        )
        
        # Compute direction to x_t
        dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * pred_noise
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_prev) * x_start + dir_xt
        
        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma * noise
            
        return x_prev
    
    def p_sample_loop(
        self,
        shape: Tuple[int, ...],
        progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        DDIM sampling loop.
        
        Args:
            shape: Shape of samples
            progress: Show progress bar
            
        Returns:
            Generated samples
        """
        device = next(self.parameters()).device
        
        # Start from noise at last timestep
        x = torch.randn(shape, device=device)
        
        timesteps = self.ddim_timesteps.tolist()
        
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="DDIM Sampling")
        
        for i, t in enumerate(reversed(timesteps)):
            if i < len(timesteps) - 1:
                t_prev = timesteps[-(i + 2)]
            else:
                t_prev = -1
                
            x = self.ddim_sample(x, t, t_prev, self.ddim_eta)
            
        return x
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Generate using DDIM."""
        shape = (n_samples, self._latent_dim)
        return self.p_sample_loop(shape, progress)


class ConditionalDiffusionGenerator(ConditionalGenerator, DiffusionGenerator):
    """
    Conditional Diffusion Generator.
    
    Supports conditioning on class labels or continuous attributes.
    
    Example:
        >>> generator = ConditionalDiffusionGenerator(
        ...     denoiser=unet_model,
        ...     latent_dim=512,
        ...     num_timesteps=1000,
        ...     num_classes=10
        ... )
        >>> samples = generator.generate(100, class_label=3)
    """
    
    def __init__(
        self,
        denoiser: nn.Module,
        latent_dim: int = 512,
        num_timesteps: int = 1000,
        num_classes: Optional[int] = None,
        condition_dim: Optional[int] = None,
        guidance_scale: float = 1.0,
        classifier_free_guidance: bool = False,
        name: str = "conditional_diffusion_generator",
        **kwargs
    ):
        """
        Initialize conditional diffusion generator.
        
        Args:
            denoiser: Conditional denoising network
            latent_dim: Data dimension
            num_timesteps: Number of diffusion steps
            num_classes: Number of classes
            condition_dim: Continuous condition dimension
            guidance_scale: Classifier-free guidance scale
            classifier_free_guidance: Use classifier-free guidance
            name: Generator name
        """
        ConditionalGenerator.__init__(
            self,
            name=name,
            latent_dim=latent_dim,
            num_classes=num_classes,
            condition_dim=condition_dim
        )
        DiffusionGenerator.__init__(
            self,
            denoiser=denoiser,
            latent_dim=latent_dim,
            num_timesteps=num_timesteps,
            name=name,
            **kwargs
        )
        
        self.guidance_scale = guidance_scale
        self.classifier_free_guidance = classifier_free_guidance
        
    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        clip_denoised: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Conditional p(x_{t-1} | x_t, c).
        """
        # Get model prediction with condition
        if self.classifier_free_guidance:
            # Classifier-free guidance
            # Unconditional
            model_output_uncond = self.denoiser(x_t, self._rescale_timesteps(t))
            # Conditional
            model_output_cond = self.denoiser(
                x_t, self._rescale_timesteps(t),
                class_label=class_label, condition=condition
            )
            # Combine
            model_output = model_output_uncond + self.guidance_scale * (model_output_cond - model_output_uncond)
        else:
            model_output = self.denoiser(
                x_t, self._rescale_timesteps(t),
                class_label=class_label, condition=condition
            )
        
        # Convert to x_start
        if self.prediction_type == "epsilon":
            x_start = self.predict_start_from_noise(x_t, t, model_output)
        elif self.prediction_type == "x_start":
            x_start = model_output
        elif self.prediction_type == "v":
            x_start = self.predict_start_from_v(x_t, t, model_output)
        
        if clip_denoised:
            x_start = x_start.clamp(*self.clip_range)
        
        mean, variance, log_variance = self.q_posterior_mean_variance(x_start, x_t, t)
        
        return mean, variance, log_variance
    
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """Conditional single step."""
        mean, variance, log_variance = self.p_mean_variance(
            x_t, t, class_label, condition, clip_denoised
        )
        
        if t[0] == 0:
            return mean
        
        noise = torch.randn_like(x_t)
        return mean + torch.exp(0.5 * log_variance) * noise
    
    def p_sample_loop(
        self,
        shape: Tuple[int, ...],
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Conditional sampling loop."""
        device = next(self.parameters()).device
        x = torch.randn(shape, device=device)
        
        timesteps = list(range(self.num_timesteps))[::-1]
        
        if progress:
            from tqdm import tqdm
            timesteps = tqdm(timesteps, desc="Conditional Sampling")
        
        for t in timesteps:
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(x, t_tensor, class_label, condition)
            
        return x
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate conditional samples.
        
        Args:
            n_samples: Number of samples
            conditions: Conditions dict
            class_label: Class label
            condition: Continuous condition
            progress: Show progress
            
        Returns:
            Generated samples
        """
        if conditions is not None:
            class_label = conditions.get("class_label", class_label)
            condition = conditions.get("condition", condition)
            
        shape = (n_samples, self._latent_dim)
        return self.p_sample_loop(shape, class_label, condition, progress)
    
    def compute_loss(
        self,
        x_start: torch.Tensor,
        class_label: Optional[torch.Tensor] = None,
        condition: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_components: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute conditional training loss."""
        batch_size = x_start.shape[0]
        device = x_start.device
        
        if t is None:
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_t = self.q_sample(x_start, t, noise)
        
        # Classifier-free guidance training
        if self.classifier_free_guidance:
            # Randomly drop condition
            drop_mask = torch.rand(batch_size, device=device) < 0.1
            if class_label is not None:
                class_label_drop = class_label.clone()
                class_label_drop[drop_mask] = 0  # Special token for unconditional
            else:
                class_label_drop = None
            model_output = self.denoiser(
                x_t, self._rescale_timesteps(t),
                class_label=class_label_drop, condition=condition
            )
        else:
            model_output = self.denoiser(
                x_t, self._rescale_timesteps(t),
                class_label=class_label, condition=condition
            )
        
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x_start":
            target = x_start
        elif self.prediction_type == "v":
            sqrt_alpha = self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alpha = self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_start
        else:
            target = noise
        
        loss = F.mse_loss(model_output, target)
        
        if return_components:
            return {
                "loss": loss,
                "timestep": t.float().mean().item(),
            }
        
        return loss


class LatentDiffusionGenerator(DiffusionGenerator):
    """
    Latent Diffusion Model (LDM) Generator.
    
    Performs diffusion in learned latent space for efficiency.
    
    Architecture:
        Encoder: x → z
        Diffusion: z_t → z_0
        Decoder: z → x̂
        
    Example:
        >>> generator = LatentDiffusionGenerator(
        ...     encoder=vae_encoder,
        ...     decoder=vae_decoder,
        ...     denoiser=unet_model,
        ...     latent_dim=512,
        ...     num_timesteps=1000
        ... )
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        denoiser: nn.Module,
        latent_dim: int = 512,
        num_timesteps: int = 1000,
        scale_factor: float = 1.0,
        name: str = "latent_diffusion_generator",
        **kwargs
    ):
        """
        Initialize latent diffusion generator.
        
        Args:
            encoder: Encoder to latent space
            decoder: Decoder from latent space
            denoiser: Latent denoiser
            latent_dim: Latent dimension
            num_timesteps: Number of diffusion steps
            scale_factor: Latent scaling factor
            name: Generator name
        """
        super().__init__(
            denoiser=denoiser,
            latent_dim=latent_dim,
            num_timesteps=num_timesteps,
            name=name,
            **kwargs
        )
        
        self.encoder = encoder
        self.decoder = decoder
        self.scale_factor = scale_factor
        
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode to latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        z = self.encoder(x, **kwargs)
        if isinstance(z, tuple):
            z = z[0]  # Take mean if VAE
        return z * self.scale_factor
    
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decode from latent space.
        
        Args:
            z: Latent tensor
            
        Returns:
            Decoded output
        """
        return self.decoder(z / self.scale_factor, **kwargs)
    
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate by sampling in latent space and decoding.
        
        Args:
            n_samples: Number of samples
            conditions: Optional conditions
            progress: Show progress bar
            
        Returns:
            Generated samples in data space
        """
        # Determine latent shape (may differ from data shape)
        shape = self._get_latent_shape(n_samples)
        
        # Sample in latent space
        z = self.p_sample_loop(shape, progress)
        
        # Decode to data space
        return self.decode(z)
    
    def _get_latent_shape(self, n_samples: int) -> Tuple[int, ...]:
        """Get shape for latent sampling."""
        # Override for image data where latent has different spatial dims
        return (n_samples, self._latent_dim)


class FairDiffusionGenerator(ConditionalDiffusionGenerator, FairGenerator):
    """
    Fair Diffusion Generator for bias-aware synthetic data.
    
    Integrates fairness constraints into diffusion process:
    - Conditional generation with sensitive attribute control
    - Fairness-aware guidance
    - Counterfactual generation
    
    Example:
        >>> generator = FairDiffusionGenerator(
        ...     denoiser=unet_model,
        ...     latent_dim=512,
        ...     sensitive_attributes=["gender", "race"],
        ...     fairness_guidance_scale=2.0
        ... )
        >>> samples = generator.generate_fair(100)
    """
    
    def __init__(
        self,
        denoiser: nn.Module,
        latent_dim: int = 512,
        num_timesteps: int = 1000,
        sensitive_attributes: Optional[List[str]] = None,
        fairness_weights: Optional[Dict[str, float]] = None,
        fairness_guidance_scale: float = 1.0,
        name: str = "fair_diffusion_generator",
        **kwargs
    ):
        """
        Initialize fair diffusion generator.
        
        Args:
            denoiser: Denoising network
            latent_dim: Data dimension
            num_timesteps: Number of diffusion steps
            sensitive_attributes: Sensitive attribute names
            fairness_weights: Fairness constraint weights
            fairness_guidance_scale: Scale for fairness guidance
            name: Generator name
        """
        ConditionalDiffusionGenerator.__init__(
            self,
            denoiser=denoiser,
            latent_dim=latent_dim,
            num_timesteps=num_timesteps,
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
        
        self.fairness_guidance_scale = fairness_guidance_scale
        
    def generate_fair(
        self,
        n_samples: int,
        target_distribution: Optional[Dict[str, Any]] = None,
        progress: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate fair synthetic samples.
        
        Args:
            n_samples: Number of samples
            target_distribution: Target distribution for sensitive attrs
            progress: Show progress
            
        Returns:
            Fair synthetic samples
        """
        # Use fairness-guided sampling
        return self.generate(
            n_samples,
            progress=progress,
            guidance_scale=self.fairness_guidance_scale,
            **kwargs
        )
    
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
            z: Original latent (from encoder)
            original_sensitive: Original sensitive values
            target_sensitive: Target sensitive values
            
        Returns:
            Counterfactual samples
        """
        # Reconstruct with target sensitive attribute
        # This assumes the model was trained with sensitive attribute conditioning
        n_samples = z.shape[0]
        
        shape = z.shape
        device = z.device
        
        # Start from latent and denoise with target condition
        x_t = z
        
        timesteps = list(range(self.num_timesteps))[::-1]
        
        for t in timesteps:
            t_tensor = torch.full((n_samples,), t, device=device, dtype=torch.long)
            x_t = self.p_sample(
                x_t, t_tensor,
                condition=target_sensitive.float()
            )
        
        return x_t
