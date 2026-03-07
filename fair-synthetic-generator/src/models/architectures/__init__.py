"""
Model Architectures Module
==========================

End-to-end model architectures for fair synthetic data generation.

This module provides complete fair generative model architectures:

## Architecture Types

### FairGAN Family
- FairGAN: GAN with adversarial debiasing
- FairGANGP: Wasserstein GAN with gradient penalty
- ConditionalFairGAN: Conditional fair GAN
- MultimodalFairGAN: Multimodal fair GAN

### FairDiffusion Family
- FairDiffusion: Diffusion with adversarial debiasing
- ConditionalFairDiffusion: Conditional fair diffusion
- LatentFairDiffusion: Latent space diffusion
- FairDDIM: Fast fair diffusion with DDIM

### DebiasedVAE Family
- DebiasedVAE: VAE with adversarial debiasing
- BetaDebiasedVAE: β-VAE with fairness
- ConditionalDebiasedVAE: Conditional fair VAE
- HierarchicalDebiasedVAE: Multi-level VAE
- MultimodalDebiasedVAE: Multimodal fair VAE

### Counterfactual Family
- CounterfactualGenerator: Base counterfactual generation
- CausalCounterfactualGenerator: Causal model-based counterfactuals
- LatentCounterfactualGenerator: Latent space counterfactual editing
- CycleConsistentGenerator: Cycle-consistent counterfactual generation

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │     Fair Synthetic Data Generation   │
                    └─────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │   FairGAN    │          │FairDiffusion │          │ DebiasedVAE  │
    │  - Adversary │          │  - Denoising │          │  - VAE + Adv │
    │  - WGAN-GP   │          │  - DDIM      │          │  - β-VAE     │
    │  - Multimodal│          │  - Latent    │          │  - Hierarch. │
    └──────────────┘          └──────────────┘          └──────────────┘
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     Counterfactual Generation       │
                    │  - Causal Models                    │
                    │  - Latent Editing                   │
                    │  - Cycle Consistency                │
                    └─────────────────────────────────────┘
```

## Fairness Paradigms

### Group Fairness
- Demographic Parity
- Equalized Odds
- Disparate Impact

### Individual Fairness
- Similar individuals → similar outcomes
- Lipschitz continuity

### Counterfactual Fairness
- What if sensitive attribute were different?
- Causal intervention

## Example Usage

### FairGAN
```python
from src.models.architectures import FairGAN

model = FairGAN(
    data_dim=100,
    latent_dim=512,
    num_sensitive_groups=2
)

# Training
losses = model.train_step(real_data, sensitive_attrs)

# Generation
fair_samples = model.generate(100)
```

### FairDiffusion
```python
from src.models.architectures import FairDiffusion

model = FairDiffusion(
    data_dim=100,
    num_timesteps=1000,
    num_sensitive_groups=2
)

# Training
loss = model.compute_loss(data, sensitive_attrs)

# Generation
samples = model.generate(100)
```

### DebiasedVAE
```python
from src.models.architectures import DebiasedVAE

model = DebiasedVAE(
    data_dim=100,
    latent_dim=512,
    num_sensitive_groups=2
)

# Training
outputs = model(data, sensitive_attrs)
loss = outputs["losses"]["total"]

# Generation
samples = model.generate(100)
```

### Counterfactual Generation
```python
from src.models.architectures import CounterfactualGenerator

model = CounterfactualGenerator(
    data_dim=100,
    latent_dim=512,
    num_sensitive_groups=2
)

# Generate counterfactual
target_sensitive = torch.ones_like(original_sensitive)
cf_samples = model.generate_counterfactual(
    x, 
    original_sensitive, 
    target_sensitive
)
```

### Multimodal Generation
```python
from src.models.architectures import MultimodalFairGAN

model = MultimodalFairGAN(
    modalities=["tabular", "text", "image"],
    latent_dim=512,
    modality_configs={
        "tabular": {"dim": 100},
        "text": {"vocab_size": 10000, "seq_len": 50},
        "image": {"channels": 3, "size": 64}
    }
)

# Generate multimodal samples
samples = model.generate(100)
# Returns: {"tabular": ..., "text": ..., "image": ...}
```
"""

# FairGAN architectures
from src.models.architectures.fairgan import (
    FairGAN,
    FairGANGP,
    ConditionalFairGAN,
    MultimodalFairGAN,
    CrossModalFusion,
    FAIRGAN_REGISTRY,
    get_fairgan
)

# FairDiffusion architectures
from src.models.architectures.fairdiffusion import (
    FairDiffusion,
    ConditionalFairDiffusion,
    LatentFairDiffusion,
    FairDDIM,
    SinusoidalPositionEmbedding,
    get_noise_schedule,
    FAIRDIFFUSION_REGISTRY,
    get_fairdiffusion
)

# DebiasedVAE architectures
from src.models.architectures.debiased_vae import (
    DebiasedVAE,
    BetaDebiasedVAE,
    ConditionalDebiasedVAE,
    HierarchicalDebiasedVAE,
    MultimodalDebiasedVAE,
    ProductOfExperts,
    MixtureOfExperts,
    CrossModalAttentionFusion,
    DEBIASED_VAE_REGISTRY,
    get_debiased_vae
)

# Counterfactual architectures
from src.models.architectures.counterfactual_generator import (
    CounterfactualGenerator,
    CausalCounterfactualGenerator,
    LatentCounterfactualGenerator,
    CycleConsistentGenerator,
    CounterfactualEvaluator,
    COUNTERFACTUAL_REGISTRY,
    get_counterfactual_generator
)


__all__ = [
    # FairGAN family
    "FairGAN",
    "FairGANGP",
    "ConditionalFairGAN",
    "MultimodalFairGAN",
    "CrossModalFusion",
    "FAIRGAN_REGISTRY",
    "get_fairgan",
    
    # FairDiffusion family
    "FairDiffusion",
    "ConditionalFairDiffusion",
    "LatentFairDiffusion",
    "FairDDIM",
    "SinusoidalPositionEmbedding",
    "get_noise_schedule",
    "FAIRDIFFUSION_REGISTRY",
    "get_fairdiffusion",
    
    # DebiasedVAE family
    "DebiasedVAE",
    "BetaDebiasedVAE",
    "ConditionalDebiasedVAE",
    "HierarchicalDebiasedVAE",
    "MultimodalDebiasedVAE",
    "ProductOfExperts",
    "MixtureOfExperts",
    "CrossModalAttentionFusion",
    "DEBIASED_VAE_REGISTRY",
    "get_debiased_vae",
    
    # Counterfactual family
    "CounterfactualGenerator",
    "CausalCounterfactualGenerator",
    "LatentCounterfactualGenerator",
    "CycleConsistentGenerator",
    "CounterfactualEvaluator",
    "COUNTERFACTUAL_REGISTRY",
    "get_counterfactual_generator",
]


# Master registry for all architectures
ARCHITECTURE_REGISTRY = {
    # FairGAN variants
    **{f"fairgan_{k}": v for k, v in FAIRGAN_REGISTRY.items()},
    **{f"gan_{k}": v for k, v in FAIRGAN_REGISTRY.items()},
    
    # FairDiffusion variants
    **{f"diffusion_{k}": v for k, v in FAIRDIFFUSION_REGISTRY.items()},
    
    # DebiasedVAE variants
    **{f"vae_{k}": v for k, v in DEBIASED_VAE_REGISTRY.items()},
    
    # Counterfactual variants
    **{f"cf_{k}": v for k, v in COUNTERFACTUAL_REGISTRY.items()},
}

# Add simple aliases
ARCHITECTURE_REGISTRY.update({
    "fairgan": FairGAN,
    "fairgan_gp": FairGANGP,
    "fairdiffusion": FairDiffusion,
    "fair_ddim": FairDDIM,
    "debias_vae": DebiasedVAE,
    "counterfactual": CounterfactualGenerator,
})


def get_architecture(name: str, **kwargs):
    """
    Get an architecture by name from the registry.
    
    Args:
        name: Architecture name (e.g., "fairgan", "fairdiffusion", "debias_vae")
        **kwargs: Arguments to pass to the architecture constructor
        
    Returns:
        Architecture instance
        
    Raises:
        ValueError: If architecture name is not found
        
    Example:
        >>> model = get_architecture("fairgan", data_dim=100, latent_dim=512)
    """
    if name not in ARCHITECTURE_REGISTRY:
        available = list(ARCHITECTURE_REGISTRY.keys())
        raise ValueError(
            f"Unknown architecture: {name}. Available architectures: {available}"
        )
    
    return ARCHITECTURE_REGISTRY[name](**kwargs)


def list_architectures() -> List[str]:
    """
    List all available architecture names.
    
    Returns:
        List of architecture names
    """
    return list(ARCHITECTURE_REGISTRY.keys())


def get_architecture_for_task(
    task: str = "generation",
    fairness: bool = True,
    modality: str = "tabular",
    conditional: bool = False
) -> type:
    """
    Get recommended architecture for a task.
    
    Args:
        task: Task type ("generation", "counterfactual", "editing")
        fairness: Whether fairness constraints are needed
        modality: Data modality ("tabular", "text", "image", "multimodal")
        conditional: Whether conditional generation is needed
        
    Returns:
        Recommended architecture class
        
    Example:
        >>> ArchClass = get_architecture_for_task(
        ...     task="generation",
        ...     fairness=True,
        ...     modality="tabular"
        ... )
        >>> model = ArchClass(data_dim=100, latent_dim=512)
    """
    if task == "counterfactual":
        return CounterfactualGenerator
    
    if task == "editing":
        return LatentCounterfactualGenerator
    
    recommendations = {
        "generation": {
            ("tabular", False, False): DebiasedVAE,
            ("tabular", False, True): ConditionalDebiasedVAE,
            ("tabular", True, False): DebiasedVAE,
            ("tabular", True, True): ConditionalDebiasedVAE,
            
            ("image", False, False): FairDiffusion,
            ("image", False, True): ConditionalFairDiffusion,
            ("image", True, False): FairDiffusion,
            ("image", True, True): ConditionalFairDiffusion,
            
            ("text", False, False): DebiasedVAE,
            ("text", False, True): ConditionalDebiasedVAE,
            ("text", True, False): DebiasedVAE,
            ("text", True, True): ConditionalDebiasedVAE,
            
            ("multimodal", False, False): MultimodalDebiasedVAE,
            ("multimodal", False, True): MultimodalFairGAN,
            ("multimodal", True, False): MultimodalDebiasedVAE,
            ("multimodal", True, True): MultimodalFairGAN,
        }
    }
    
    if task not in recommendations:
        raise ValueError(f"Unknown task: {task}")
    
    key = (modality, fairness, conditional)
    
    if key not in recommendations[task]:
        # Default fallback
        return DebiasedVAE
    
    return recommendations[task][key]


def get_default_config(
    architecture: str,
    data_dim: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get default configuration for an architecture.
    
    Args:
        architecture: Architecture name
        data_dim: Optional data dimension
        **kwargs: Additional overrides
        
    Returns:
        Default configuration dictionary
    """
    configs = {
        "fairgan": {
            "latent_dim": 512,
            "hidden_dims": [512, 256, 128],
            "fairness_weight": 1.0,
            "gp_weight": 10.0,
            "spectral_norm": True,
        },
        "fairdiffusion": {
            "latent_dim": 512,
            "num_timesteps": 1000,
            "hidden_dims": [512, 256, 128],
            "noise_schedule": "cosine",
            "fairness_weight": 1.0,
        },
        "debias_vae": {
            "latent_dim": 512,
            "encoder_dims": [512, 256],
            "decoder_dims": [256, 512],
            "kl_weight": 0.001,
            "fairness_weight": 1.0,
        },
        "counterfactual": {
            "latent_dim": 512,
            "encoder_dims": [512, 256],
            "decoder_dims": [256, 512],
            "counterfactual_weight": 1.0,
            "cycle_weight": 1.0,
        },
    }
    
    config = configs.get(architecture, {})
    config.update(kwargs)
    
    if data_dim is not None:
        config["data_dim"] = data_dim
    
    return config
