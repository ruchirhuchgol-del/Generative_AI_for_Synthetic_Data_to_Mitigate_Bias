"""
Generators Module
=================

Generator implementations for synthetic data generation.

This module provides comprehensive generative models for fair synthetic data:

## Model Types

### VAE-based
- VAEGenerator: Standard Variational Autoencoder
- BetaVAEGenerator: β-VAE for disentanglement
- ConditionalVAEGenerator: Conditional VAE (CVAE)
- VAEGANGenerator: VAE-GAN hybrid
- HierarchicalVAEGenerator: Multi-level VAE

### GAN-based
- GANGenerator: Standard GAN
- WANGenerator: Wasserstein GAN with Gradient Penalty
- ConditionalGANGenerator: Conditional GAN (CGAN)
- StyleGANGenerator: StyleGAN with style modulation
- FairGANGenerator: Fairness-aware GAN

### Diffusion-based
- DiffusionGenerator: DDPM (Denoising Diffusion)
- DDIMGenerator: DDIM (faster sampling)
- ConditionalDiffusionGenerator: Conditional diffusion
- LatentDiffusionGenerator: Latent diffusion (LDM)
- FairDiffusionGenerator: Fairness-aware diffusion

### Multimodal
- MultimodalGenerator: Joint multimodal generation
- MultimodalVAEGenerator: VAE-based multimodal
- MultimodalDiffusionGenerator: Diffusion-based multimodal
- ConditionalMultimodalGenerator: Conditional multimodal
- FairMultimodalGenerator: Fair multimodal generation

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │         Generative Model            │
                    └─────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │  VAE Family  │          │  GAN Family  │          │ Diffusion    │
    │  - VAE       │          │  - GAN       │          │  - DDPM      │
    │  - β-VAE     │          │  - WGAN-GP   │          │  - DDIM      │
    │  - VAE-GAN   │          │  - StyleGAN  │          │  - LDM       │
    │  - CVAE      │          │  - CGAN      │          │  - Cond.     │
    └──────────────┘          └──────────────┘          └──────────────┘
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     Multimodal Fusion & Fairness    │
                    └─────────────────────────────────────┘
```

## Example Usage

### VAE Generator
```python
from src.models.generators import VAEGenerator

generator = VAEGenerator(
    encoder=encoder_net,
    decoder=decoder_net,
    latent_dim=512,
    beta=1.0
)

# Training
loss = generator.compute_loss(x)

# Generation
samples = generator.generate(100)
```

### GAN Generator
```python
from src.models.generators import WANGenerator

generator = WANGenerator(
    generator=gen_net,
    discriminator=disc_net,
    latent_dim=512,
    gp_weight=10.0
)

# Training
g_loss = generator.compute_generator_loss(batch_size)
d_loss = generator.compute_discriminator_loss(real_data)

# Generation
samples = generator.generate(100)
```

### Diffusion Generator
```python
from src.models.generators import DDIMGenerator

generator = DDIMGenerator(
    denoiser=unet_net,
    latent_dim=512,
    num_timesteps=1000,
    ddim_steps=50
)

# Training
loss = generator.compute_loss(x)

# Generation
samples = generator.generate(100, progress=True)
```

### Multimodal Generator
```python
from src.models.generators import MultimodalVAEGenerator

generator = MultimodalVAEGenerator(
    encoders={"tabular": tab_enc, "text": text_enc, "image": img_enc},
    decoders={"tabular": tab_dec, "text": text_dec, "image": img_dec},
    latent_dim=512
)

# Generation
samples = generator.generate(100)
# Returns: {"tabular": ..., "text": ..., "image": ...}
```
"""

# Base generators
from src.models.generators.base_generator import (
    BaseGenerator,
    ConditionalGenerator,
    FairGenerator,
    ProgressiveGenerator,
    LatentSpaceMixin,
    ReconstructionMixin,
    GenerationCallback,
    get_noise_schedule
)

# VAE generators
from src.models.generators.vae_generator import (
    VAEGenerator,
    BetaVAEGenerator,
    ConditionalVAEGenerator,
    VAEGANGenerator,
    HierarchicalVAEGenerator
)

# GAN generators
from src.models.generators.gan_generator import (
    GANGenerator,
    WANGenerator,
    ConditionalGANGenerator,
    StyleGANGenerator,
    FairGANGenerator
)

# Diffusion generators
from src.models.generators.diffusion_generator import (
    DiffusionGenerator,
    DDIMGenerator,
    ConditionalDiffusionGenerator,
    LatentDiffusionGenerator,
    FairDiffusionGenerator
)

# Multimodal generators
from src.models.generators.multimodal_generator import (
    MultimodalGenerator,
    MultimodalVAEGenerator,
    MultimodalDiffusionGenerator,
    ConditionalMultimodalGenerator,
    FairMultimodalGenerator,
    CrossModalFusion
)


__all__ = [
    # Base generators
    "BaseGenerator",
    "ConditionalGenerator",
    "FairGenerator",
    "ProgressiveGenerator",
    "LatentSpaceMixin",
    "ReconstructionMixin",
    "GenerationCallback",
    "get_noise_schedule",
    
    # VAE generators
    "VAEGenerator",
    "BetaVAEGenerator",
    "ConditionalVAEGenerator",
    "VAEGANGenerator",
    "HierarchicalVAEGenerator",
    
    # GAN generators
    "GANGenerator",
    "WANGenerator",
    "ConditionalGANGenerator",
    "StyleGANGenerator",
    "FairGANGenerator",
    
    # Diffusion generators
    "DiffusionGenerator",
    "DDIMGenerator",
    "ConditionalDiffusionGenerator",
    "LatentDiffusionGenerator",
    "FairDiffusionGenerator",
    
    # Multimodal generators
    "MultimodalGenerator",
    "MultimodalVAEGenerator",
    "MultimodalDiffusionGenerator",
    "ConditionalMultimodalGenerator",
    "FairMultimodalGenerator",
    "CrossModalFusion",
]


# Registry for generator lookup
GENERATOR_REGISTRY = {
    # Base generators
    "base": BaseGenerator,
    "conditional": ConditionalGenerator,
    "fair": FairGenerator,
    "progressive": ProgressiveGenerator,
    
    # VAE generators
    "vae": VAEGenerator,
    "beta_vae": BetaVAEGenerator,
    "conditional_vae": ConditionalVAEGenerator,
    "vae_gan": VAEGANGenerator,
    "hierarchical_vae": HierarchicalVAEGenerator,
    
    # GAN generators
    "gan": GANGenerator,
    "wgan": WANGenerator,
    "conditional_gan": ConditionalGANGenerator,
    "stylegan": StyleGANGenerator,
    "fair_gan": FairGANGenerator,
    
    # Diffusion generators
    "diffusion": DiffusionGenerator,
    "ddpm": DiffusionGenerator,
    "ddim": DDIMGenerator,
    "conditional_diffusion": ConditionalDiffusionGenerator,
    "latent_diffusion": LatentDiffusionGenerator,
    "fair_diffusion": FairDiffusionGenerator,
    
    # Multimodal generators
    "multimodal": MultimodalGenerator,
    "multimodal_vae": MultimodalVAEGenerator,
    "multimodal_diffusion": MultimodalDiffusionGenerator,
    "conditional_multimodal": ConditionalMultimodalGenerator,
    "fair_multimodal": FairMultimodalGenerator,
}


def get_generator(name: str, **kwargs):
    """
    Get a generator by name from the registry.
    
    Args:
        name: Generator name
        **kwargs: Arguments to pass to the generator constructor
        
    Returns:
        Generator instance
        
    Raises:
        ValueError: If generator name is not found
    """
    if name not in GENERATOR_REGISTRY:
        available = list(GENERATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown generator: {name}. Available generators: {available}"
        )
    
    return GENERATOR_REGISTRY[name](**kwargs)


def list_generators() -> list:
    """
    List all available generator names.
    
    Returns:
        List of generator names
    """
    return list(GENERATOR_REGISTRY.keys())


def get_generator_for_data_type(
    data_type: str,
    fairness: bool = False,
    conditional: bool = False
) -> type:
    """
    Get recommended generator class for a data type.
    
    Args:
        data_type: Type of data ("tabular", "text", "image", "multimodal")
        fairness: Whether fairness constraints are needed
        conditional: Whether conditional generation is needed
        
    Returns:
        Recommended generator class
    """
    recommendations = {
        "tabular": {
            (False, False): VAEGenerator,
            (False, True): ConditionalVAEGenerator,
            (True, False): FairGenerator,
            (True, True): ConditionalVAEGenerator,
        },
        "text": {
            (False, False): VAEGenerator,
            (False, True): ConditionalVAEGenerator,
            (True, False): FairGenerator,
            (True, True): ConditionalVAEGenerator,
        },
        "image": {
            (False, False): DiffusionGenerator,
            (False, True): ConditionalDiffusionGenerator,
            (True, False): FairDiffusionGenerator,
            (True, True): FairDiffusionGenerator,
        },
        "multimodal": {
            (False, False): MultimodalVAEGenerator,
            (False, True): ConditionalMultimodalGenerator,
            (True, False): FairMultimodalGenerator,
            (True, True): FairMultimodalGenerator,
        },
    }
    
    if data_type not in recommendations:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return recommendations[data_type][(fairness, conditional)]
