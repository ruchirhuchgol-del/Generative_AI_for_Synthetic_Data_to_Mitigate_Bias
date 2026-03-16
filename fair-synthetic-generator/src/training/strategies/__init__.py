"""
Training Strategies Module
=======================

This module provides different training strategies for fair synthetic data generation:

## Available Strategies

### Adversarial Training
- Gradient Reversal-based adversarial debiasing
- Alternating GAN-style training
- Pretrain adversary approach

### Multi-Task Training
- Multi-objective optimization
- Gradient balancing (GradNorm, Uncertainty, Dynamic Weight Averaging)
- Task-specific weighting

### Curriculum Learning
- Progressive training with increasing difficulty
- Self-paced learning
- Fairness-aware curriculum

## Example Usage

```python
from src.training.strategies import (
    AdversarialTrainingStrategy,
    MultiTaskTrainingStrategy,
    CurriculumTrainingStrategy,
)

# Initialize strategy
strategy = AdversarialTrainingStrategy(
    model=generator,
    adversary=adversary_net,
    config={"mode": "gradient_reversal"}
)

# Train
for epoch in range(num_epochs):
    metrics = strategy.train_epoch(dataloader, epoch)
```

This module also provides a factory function for creating strategies.
"""

from src.training.strategies.adversarial_training import (
    AdversarialTrainingStrategy,
    AdversarialConfig,
    AdversarialMode,
)

from src.training.strategies.multi_task_training import (
    MultiTaskTrainingStrategy,
    MultiTaskConfig,
    GradNormBalancing,
    UncertaintyBalancing,
    DynamicWeightAveraging,
)

from src.training.strategies.curriculum_training import (
    CurriculumTrainingStrategy,
    CurriculumConfig,
    CurriculumScheduler,
    DifficultySampler,
)


__all__ = [
    # Adversarial training
    "AdversarialTrainingStrategy",
    "AdversarialConfig",
    "AdversarialMode",
    
    # Multi-task training
    "MultiTaskTrainingStrategy",
    "MultiTaskConfig",
    "GradNormBalancing",
    "UncertaintyBalancing",
    "DynamicWeightAveraging",
    
    # Curriculum training
    "CurriculumTrainingStrategy",
    "CurriculumConfig",
    "CurriculumScheduler",
    "DifficultySampler",
]


# Strategy registry
STRATEGY_REGISTRY = {
    "adversarial": AdversarialTrainingStrategy,
    "multi_task": MultiTaskTrainingStrategy,
    "curriculum": CurriculumTrainingStrategy,
}


def get_strategy(name: str, **kwargs):
    """
    Get a training strategy by name.
    
    Args:
        name: Strategy name
        **kwargs: Arguments for strategy constructor
        
    Returns:
        Strategy instance
        
    Raises:
        ValueError: If strategy name is not found
    """
    if name not in STRATEGY_REGISTRY:
        available = list(STRATEGY_REGISTRY.keys())
        raise ValueError(
            f"Unknown strategy: {name}. Available strategies: {available}"
        )
    
    return STRATEGY_REGISTRY[name](**kwargs)
