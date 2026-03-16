"""
Fairness Loss Functions Module
==============================

Loss functions for fairness-aware training and debiasing.

This module provides:

## Core Losses
- FairnessLoss: Base class for fairness losses
- MultiObjectiveFairnessLoss: Combines multiple fairness objectives

## Adversarial Losses
- AdversarialDebiasingLoss: Main adversarial debiasing loss
- MultiAdversaryLoss: Loss for multiple adversaries
- ContrastiveFairnessLoss: Contrastive learning for fairness
- MutualInformationLoss: MI minimization for independence

## Multi-Objective Optimization
- MultiObjectiveLoss: Scalarization methods for multi-objective optimization
- DynamicFairnessWeightScheduler: Adaptive weight scheduling
- EpsilonConstraintHandler: Epsilon-constraint method

## Example Usage

### Adversarial Debiasing
```python
from src.fairness.losses import AdversarialDebiasingLoss

loss_fn = AdversarialDebiasingLoss(
    adversary=adversary_network,
    fairness_weight=1.0,
    use_grl=True
)

# During training
result = loss_fn(latent, sensitive_attrs)
result["total"].backward()
```

### Multi-Objective Optimization
```python
from src.fairness.losses import MultiObjectiveLoss

loss_fn = MultiObjectiveLoss(
    objectives={
        "task": task_loss_fn,
        "fairness": fairness_constraint
    },
    weights={"task": 1.0, "fairness": 0.5},
    method="weighted_sum"
)

losses = loss_fn(predictions, groups, labels)
losses["total"].backward()
```
"""

from typing import List

# Base losses
from src.fairness.losses.fairness_loss import (
    FairnessLoss,
    MultiObjectiveFairnessLoss,
    AdversarialDebiasingLoss as AdversarialLoss,
)

# Adversarial losses
from src.fairness.losses.adversarial_loss import (
    AdversarialDebiasingLoss,
    MultiAdversaryLoss,
    ContrastiveFairnessLoss,
    MutualInformationLoss,
)

# Multi-objective losses
from src.fairness.losses.multi_objective_loss import (
    MultiObjectiveLoss,
    DynamicFairnessWeightScheduler,
    EpsilonConstraintHandler,
    ScalarizationMethod,
    WeightUpdateStrategy,
)


__all__ = [
    # Base losses
    "FairnessLoss",
    "MultiObjectiveFairnessLoss",
    "AdversarialLoss",
    
    # Adversarial losses
    "AdversarialDebiasingLoss",
    "MultiAdversaryLoss",
    "ContrastiveFairnessLoss",
    "MutualInformationLoss",
    
    # Multi-objective optimization
    "MultiObjectiveLoss",
    "DynamicFairnessWeightScheduler",
    "EpsilonConstraintHandler",
    "ScalarizationMethod",
    "WeightUpdateStrategy",
]


# Loss Registry
LOSS_REGISTRY = {
    "fairness": FairnessLoss,
    "multi_objective_fairness": MultiObjectiveFairnessLoss,
    "adversarial_debiasing": AdversarialDebiasingLoss,
    "multi_adversary": MultiAdversaryLoss,
    "contrastive_fairness": ContrastiveFairnessLoss,
    "mutual_information": MutualInformationLoss,
    "multi_objective": MultiObjectiveLoss,
}


def get_loss(name: str, **kwargs):
    """
    Get a loss function by name from the registry.
    
    Args:
        name: Loss name
        **kwargs: Arguments to pass to the loss constructor
        
    Returns:
        Loss function instance
        
    Raises:
        ValueError: If loss name is not found
    """
    if name not in LOSS_REGISTRY:
        available = list(LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unknown loss: {name}. Available losses: {available}"
        )
    
    return LOSS_REGISTRY[name](**kwargs)


def list_losses() -> List[str]:
    """List all available loss names."""
    return list(LOSS_REGISTRY.keys())
