"""
Training Callbacks Module
=======================

Callbacks for training monitoring and management.

This module provides:
- FairnessCallback: Monitor fairness metrics
- CheckpointCallback: Save and load checkpoints
- LoggingCallback: Log training progress
"""

from src.training.callbacks.fairness_callback import (
    FairnessCallback,
    FairnessVisualizationCallback,
    FairnessMetrics,
)

from src.training.callbacks.checkpoint_callback import (
    CheckpointCallback,
    ModelCheckpointCallback,
    BestModelCheckpointCallback,
)

from src.training.callbacks.logging_callback import (
    LoggingCallback,
    TensorBoardCallback,
    WandBCallback,
    MetricsLogger,
)


__all__ = [
    # Fairness callbacks
    "FairnessCallback",
    "FairnessVisualizationCallback",
    "FairnessMetrics",
    
    # Checkpoint callbacks
    "CheckpointCallback",
    "ModelCheckpointCallback",
    "BestModelCheckpointCallback",
    
    # Logging callbacks
    "LoggingCallback",
    "TensorBoardCallback",
    "WandBCallback",
    "MetricsLogger",
]


# Callback registry
CALLBACK_REGISTRY = {
    "fairness": FairnessCallback,
    "fairness_viz": FairnessVisualizationCallback,
    "checkpoint": CheckpointCallback,
    "model_checkpoint": ModelCheckpointCallback,
    "best_model": BestModelCheckpointCallback,
    "logging": LoggingCallback,
    "tensorboard": TensorBoardCallback,
    "wandb": WandBCallback,
}


def get_callback(name: str, **kwargs):
    """
    Get a callback by name from the registry.
    
    Args:
        name: Callback name
        **kwargs: Arguments for callback constructor
        
    Returns:
        Callback instance
        
    Raises:
        ValueError: If callback name is not found
    """
    if name not in CALLBACK_REGISTRY:
        available = list(CALLBACK_REGISTRY.keys())
        raise ValueError(
            f"Unknown callback: {name}. Available callbacks: {available}"
        )
    
    return CALLBACK_REGISTRY[name](**kwargs)
