"""
Discriminators Module
=====================

Discriminator implementations for adversarial training and fairness debiasing.

This module provides:

## Core Components
- BaseDiscriminator: Abstract base class for all discriminators
- BinaryDiscriminator: Binary real/fake classification
- MultiClassDiscriminator: Multi-class discrimination
- ProjectionDiscriminator: Projection-based conditional discriminator
- PatchDiscriminator: Patch-level image discrimination (Pix2Pix)
- SpectralNormDiscriminator: Spectral normalized discriminator
- EnsembleDiscriminator: Ensemble of multiple discriminators

## Modality-Specific Discriminators
- TabularDiscriminator: Tabular data discrimination
- TabularTransformerDiscriminator: Transformer-based tabular discriminator
- TextDiscriminator: Transformer-based text discriminator
- TextCNNDiscriminator: CNN-based text discriminator
- ImageDiscriminator: CNN image discriminator
- ResNetDiscriminator: ResNet-style image discriminator
- MultimodalDiscriminator: Joint multimodal discrimination

## Fairness Discriminators
- GradientReversalLayer: Gradient reversal for adversarial training
- SensitiveAttributeDiscriminator: Predict sensitive attributes
- MultiSensitiveDiscriminator: Multiple sensitive attributes
- FairnessAdversary: Complete fairness adversary module
- ContrastiveFairnessDiscriminator: Individual fairness via contrastive learning
- CounterfactualDiscriminator: Counterfactual fairness
- DomainDiscriminator: Domain adaptation for fairness

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │         BaseDiscriminator          │
                    └─────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │   Modality   │          │   Fairness   │          │   Special    │
    │ Discriminators│         │ Discriminators│         │ Discriminators│
    │ - Tabular    │          │ - Sensitive  │          │ - Patch      │
    │ - Text       │          │ - Multi-Sens │          │ - Projection │
    │ - Image      │          │ - Counterf.  │          │ - Ensemble   │
    │ - Multimodal │          │ - Contrast.  │          │ - Spectral   │
    └──────────────┘          └──────────────┘          └──────────────┘
```

## Example Usage

### GAN Training
```python
from src.models.discriminators import TabularDiscriminator

discriminator = TabularDiscriminator(
    num_numerical=10,
    categorical_cardinalities={"cat1": 5, "cat2": 3},
    hidden_dims=[256, 128],
    spectral_norm=True
)

# Training
d_real = discriminator(numerical=x_num_real, categorical=x_cat_real)
d_fake = discriminator(numerical=x_num_fake, categorical=x_cat_fake)
loss = F.binary_cross_entropy_with_logits(d_real, ones) + \\
       F.binary_cross_entropy_with_logits(d_fake, zeros)
```

### Fairness Adversarial Training
```python
from src.models.discriminators import FairnessAdversary

adversary = FairnessAdversary(
    input_dim=512,
    sensitive_configs=[
        {"name": "gender", "num_classes": 2},
        {"name": "race", "num_classes": 5}
    ],
    grl_lambda=1.0
)

# During main model training
loss, metrics = adversary.compute_fairness_loss(z, sensitive_labels)
# loss.backward() will reverse gradients through GRL
```

### Image Discrimination
```python
from src.models.discriminators import ImageDiscriminator, PatchDiscriminator

# Standard image discriminator
discriminator = ImageDiscriminator(
    input_channels=3,
    hidden_dims=[64, 128, 256, 512],
    spectral_norm=True
)

# Patch discriminator for Pix2Pix
patch_disc = PatchDiscriminator(
    input_channels=3,
    hidden_dims=[64, 128, 256, 512]
)
patch_logits = patch_disc(images)  # (batch, 1, h', w')
```

### Gradient Penalty
```python
from src.models.discriminators import ImageDiscriminator

discriminator = ImageDiscriminator(input_channels=3, hidden_dims=[64, 128])
gp_loss = discriminator.compute_gradient_penalty(real_images, fake_images, lambda_gp=10.0)
```
"""

# Base discriminators
from src.models.discriminators.base_discriminator import (
    BaseDiscriminator,
    BinaryDiscriminator,
    MultiClassDiscriminator,
    ProjectionDiscriminator,
    PatchDiscriminator,
    SpectralNormDiscriminator,
    EnsembleDiscriminator
)

# Modality discriminators
from src.models.discriminators.modality_discriminator import (
    TabularDiscriminator,
    TabularTransformerDiscriminator,
    TextDiscriminator,
    TextCNNDiscriminator,
    ImageDiscriminator,
    SelfAttention2d,
    ResNetDiscriminator,
    ResidualBlock,
    MultimodalDiscriminator,
    CrossModalAttention
)

# Fairness discriminators
from src.models.discriminators.fairness_discriminator import (
    GradientReversalFunction,
    GradientReversalLayer,
    SensitiveAttributeDiscriminator,
    MultiSensitiveDiscriminator,
    FairnessAdversary,
    ContrastiveFairnessDiscriminator,
    CounterfactualDiscriminator,
    DomainDiscriminator
)


__all__ = [
    # Base discriminators
    "BaseDiscriminator",
    "BinaryDiscriminator",
    "MultiClassDiscriminator",
    "ProjectionDiscriminator",
    "PatchDiscriminator",
    "SpectralNormDiscriminator",
    "EnsembleDiscriminator",
    
    # Modality discriminators
    "TabularDiscriminator",
    "TabularTransformerDiscriminator",
    "TextDiscriminator",
    "TextCNNDiscriminator",
    "ImageDiscriminator",
    "SelfAttention2d",
    "ResNetDiscriminator",
    "ResidualBlock",
    "MultimodalDiscriminator",
    "CrossModalAttention",
    
    # Fairness discriminators
    "GradientReversalFunction",
    "GradientReversalLayer",
    "SensitiveAttributeDiscriminator",
    "MultiSensitiveDiscriminator",
    "FairnessAdversary",
    "ContrastiveFairnessDiscriminator",
    "CounterfactualDiscriminator",
    "DomainDiscriminator",
]


# Registry for discriminator lookup
DISCRIMINATOR_REGISTRY = {
    # Base discriminators
    "base": BaseDiscriminator,
    "binary": BinaryDiscriminator,
    "multiclass": MultiClassDiscriminator,
    "projection": ProjectionDiscriminator,
    "patch": PatchDiscriminator,
    "spectral_norm": SpectralNormDiscriminator,
    "ensemble": EnsembleDiscriminator,
    
    # Modality discriminators
    "tabular": TabularDiscriminator,
    "tabular_transformer": TabularTransformerDiscriminator,
    "text": TextDiscriminator,
    "text_cnn": TextCNNDiscriminator,
    "image": ImageDiscriminator,
    "resnet": ResNetDiscriminator,
    "multimodal": MultimodalDiscriminator,
    
    # Fairness discriminators
    "sensitive_attribute": SensitiveAttributeDiscriminator,
    "multi_sensitive": MultiSensitiveDiscriminator,
    "fairness_adversary": FairnessAdversary,
    "contrastive_fairness": ContrastiveFairnessDiscriminator,
    "counterfactual": CounterfactualDiscriminator,
    "domain": DomainDiscriminator,
}


def get_discriminator(name: str, **kwargs):
    """
    Get a discriminator by name from the registry.
    
    Args:
        name: Discriminator name
        **kwargs: Arguments to pass to the discriminator constructor
        
    Returns:
        Discriminator instance
        
    Raises:
        ValueError: If discriminator name is not found
    """
    if name not in DISCRIMINATOR_REGISTRY:
        available = list(DISCRIMINATOR_REGISTRY.keys())
        raise ValueError(
            f"Unknown discriminator: {name}. Available discriminators: {available}"
        )
    
    return DISCRIMINATOR_REGISTRY[name](**kwargs)


def list_discriminators() -> list:
    """
    List all available discriminator names.
    
    Returns:
        List of discriminator names
    """
    return list(DISCRIMINATOR_REGISTRY.keys())


def get_discriminator_for_task(
    task: str,
    fairness: bool = False,
    modality: str = "tabular"
) -> type:
    """
    Get recommended discriminator class for a task.
    
    Args:
        task: Task type ("gan", "fairness", "domain")
        fairness: Whether fairness constraints are needed
        modality: Data modality ("tabular", "text", "image", "multimodal")
        
    Returns:
        Recommended discriminator class
    """
    if task == "fairness":
        return FairnessAdversary if fairness else SensitiveAttributeDiscriminator
    
    recommendations = {
        "gan": {
            "tabular": TabularDiscriminator,
            "text": TextDiscriminator,
            "image": ImageDiscriminator,
            "multimodal": MultimodalDiscriminator,
        },
        "domain": {
            "tabular": DomainDiscriminator,
            "text": DomainDiscriminator,
            "image": DomainDiscriminator,
            "multimodal": DomainDiscriminator,
        },
    }
    
    if task not in recommendations:
        raise ValueError(f"Unknown task: {task}")
    
    if modality not in recommendations[task]:
        raise ValueError(f"Unknown modality: {modality}")
    
    return recommendations[task][modality]
