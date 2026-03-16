"""
Scheduler Factory
===================

Factory for creating learning rate schedulers.

This module provides:
- SchedulerFactory: Main factory for creating schedulers
- Various scheduler implementations
- Scheduler utilities
"""

from typing import Any, Dict, List, Optional, Union,from dataclasses import dataclass, field
from enum import Enum
import math

import torch
from torch.optim.lr_scheduler import (
    LRScheduler,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiplicativeLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch.optim import Optimizer


class SchedulerType(Enum):
    """Types of learning rate schedulers."""
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WARMUP = "cosine_warmup"
    EXPONENTIAL = "exponential"
    CYCLIC = "cyclic"
    ONECYCLE = "onecycle"
    STEPLR = "step"
    POLY = "poly"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    WARMUP_DECAY = "warmup_decay"
    LAMBDA = "lambda"
    MULTIPLICATIVE = "multiplicative"


@dataclass
class SchedulerConfig:
    """
    Configuration for learning rate scheduler.
    
    Attributes:
        scheduler_type: Type of scheduler
        lr: Base learning rate
        warmup_epochs: Number of warmup epochs (if applicable)
        warmup_start_lr: Starting LR for warmup (if applicable)
        total_epochs: Total training epochs
        min_lr: Minimum learning rate
        max_lr: Maximum learning rate
        gamma: Decay factor (for exponential decay)
        milestones: Milestones for MultiStepLR
        step_size: Step size for decay
        cycle_epochs: Number of epochs per cycle
        cycle_mult: Multiplication factor for cyclic LR
        eta_min: Minimum eta for cosine annealing
        eta_max: Maximum eta for cosine annealing
        T_0: Initial temperature for cosine annealing warmup
        T_mult: Final temperature for cosine annealing warmup
        factor: Decay factor
    """
    scheduler_type: str = "constant"
    lr: float = 1e-4
    warmup_epochs: int = 0
    warmup_start_lr: float = 0.0
    total_epochs: int = 1000
    min_lr: float = 1e-7
    max_lr: float = 1e-3
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=list)
    step_size: int = 10
    cycle_epochs: int = 100
    cycle_mult: float = 2.0
    eta_min: float = 1e-7
    eta_max: float = 1e-4
    T_0: float = 1.0
    T_mult: float = 100.0
    factor: float = 0.5


class SchedulerFactory:
    """
    Factory for creating learning rate schedulers.
    
    Provides a unified interface for creating and configuring
    various learning rate schedulers.
    
    Example:
        >>> factory = SchedulerFactory()
        >>> scheduler = factory.create(
        ...     optimizer=optimizer,
        ...     config=SchedulerConfig(scheduler_type="cosine_warmup")
        ... )
        >>> 
        >>> # Or using from config dict
        >>> scheduler = factory.create_from_dict(
        ...     optimizer=optimizer,
        ...     config_dict={"scheduler_type": "cosine", "lr": 1e-4}
        ... )
    """
    
    @staticmethod
    def create(
        optimizer: Optimizer,
        config: SchedulerConfig
    ) -> Optional[LRScheduler]:
        """
        Create a learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            config: Scheduler configuration
            
        Returns:
            Learning rate scheduler instance
        """
        scheduler_type = SchedulerType(config.scheduler_type)
        
        if scheduler_type == SchedulerType.CONSTANT:
            return None
        
        elif scheduler_type == SchedulerType.LINEAR:
            return LinearLR(
                optimizer,
                start_factor=config.warmup_start_lr,
                total_iters=config.total_epochs
            )
        
        elif scheduler_type == SchedulerType.COSINE:
            return CosineAnnealingLR(
                optimizer,
                T_max=config.total_epochs,
                eta_min=config.eta_min
            )
        
        elif scheduler_type == SchedulerType.COSINE_WARMUP:
            return CosineAnnealingWarmRestarts(
                optimizer,
                T_0=config.T_0,
                T_mult=config.T_mult,
                warmup_epochs=config.warmup_epochs,
                eta_min=config.eta_min
            )
        
        elif scheduler_type == SchedulerType.EXPONENTIAL:
            return ExponentialLR(
                optimizer,
                gamma=config.gamma,
                last_epoch=config.total_epochs
            )
        
        elif scheduler_type == SchedulerType.CYCLIC:
            return CyclicLR(
                optimizer,
                base_lr=config.lr,
                max_lr=config.max_lr,
                step_size_up=config.cycle_epochs,
                mode="triangular",
                gamma=config.cycle_mult
            )
        
        elif scheduler_type == SchedulerType.ONECYCLE:
            return OneCycleLR(
                optimizer,
                max_lr=config.max_lr,
                total_steps=config.total_epochs,
                pct_start=config.warmup_start_lr,
                anneal_strategy="cos",
                final_div_factor=config.factor
            )
        
        elif scheduler_type == SchedulerType.STEPLR:
            return StepLR(
                optimizer,
                step_size=config.step_size,
                gamma=config.gamma
            )
        
        elif scheduler_type == SchedulerType.POLY:
            if not config.milestones:
                raise ValueError("Milestones required for POLY scheduler")
            return LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: config.lr * (config.gamma ** epoch)
            )
        
        elif scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            return ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=config.factor,
                patience=config.milestones[0] if config.milestones else 10,
                threshold=config.min_lr,
                threshold_mode="rel",
                cooldown_epochs=config.total_epochs // config.milestones[0] if config.milestones else 0
            )
        
        elif scheduler_type == SchedulerType.WARMUP_DECAY:
            return LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: SchedulerFactory._warmup_decay_lambda(
                    epoch,
                    config.warmup_epochs,
                    config.total_epochs,
                    config.lr,
                    config.min_lr
                )
            )
        
        elif scheduler_type == SchedulerType.MULTIPLICATIVE:
            return MultiplicativeLR(
                optimizer,
                total_iters=config.total_epochs
            )
        
        elif scheduler_type == SchedulerType.LAMBDA:
            return LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: config.lr
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    @staticmethod
    def create_from_dict(
        optimizer: Optimizer,
        config_dict: Dict[str, Any]
    ) -> Optional[LRScheduler]:
        """
        Create scheduler from dictionary configuration.
        
        Args:
            optimizer: Optimizer to schedule
            config_dict: Configuration dictionary
            
        Returns:
            Learning rate scheduler instance
        """
        config = SchedulerConfig(**config_dict)
        return SchedulerFactory.create(optimizer, config)
    
    @staticmethod
    def _warmup_decay_lambda(
        epoch: int,
        warmup_epochs: int,
        total_epochs: int,
        initial_lr: float,
        min_lr: float
    ) -> float:
        """
        Lambda function for warmup-decay schedule.
        
        Args:
            epoch: Current epoch
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            
        Returns:
            Current learning rate multiplier
        """
        if epoch < warmup_epochs:
            # Linear warmup
            multiplier = epoch / max(warmup_epochs, 1)
            return initial_lr * multiplier
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr + (initial_lr - min_lr) * (0.5 * (1 + math.cos(math.pi * progress)))
    
    @staticmethod
    def get_scheduler_config(
        scheduler_name: str,
        **kwargs
    ) -> SchedulerConfig:
        """
        Get scheduler configuration by name.
        
        Args:
            scheduler_name: Name of scheduler
            **kwargs: Additional configuration parameters
            
        Returns:
            Scheduler configuration
        """
        configs = {
            "constant": SchedulerConfig(
                scheduler_type="constant"
            ),
            "linear": SchedulerConfig(
                scheduler_type="linear",
                total_epochs=kwargs.get("total_epochs", 1000)
            ),
            "cosine": SchedulerConfig(
                scheduler_type="cosine",
                total_epochs=kwargs.get("total_epochs", 1000),
                eta_min=kwargs.get("eta_min", 1e-7)
            ),
            "cosine_warmup": SchedulerConfig(
                scheduler_type="cosine_warmup",
                warmup_epochs=kwargs.get("warmup_epochs", 100),
                total_epochs=kwargs.get("total_epochs", 1000),
                T_0=kwargs.get("T_0", 1.0),
                T_mult=kwargs.get("T_mult", 100.0)
            ),
            "exponential": SchedulerConfig(
                scheduler_type="exponential",
                gamma=kwargs.get("gamma", 0.1),
                total_epochs=kwargs.get("total_epochs", 1000)
            ),
            "onecycle": SchedulerConfig(
                scheduler_type="onecycle",
                max_lr=kwargs.get("max_lr", 1e-3),
                total_epochs=kwargs.get("total_epochs", 1000)
            ),
        }
        
        if scheduler_name not in configs:
            config = configs[scheduler_name]
            for key, value in kwargs.items():
                setattr(config, key, value)
            return config
        
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")


class FairnessScheduler:
    """
    Fairness-Aware Learning Rate Scheduler.
    
    Adjusts learning rate based on fairness metrics during training.
    Gradually increases fairness weight as the model learns fairer representations.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        fairness_objective: Any,
        initial_lr: float = 1e-4,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        fairness_weight: float = 0.1,
        fairness_warmup_epochs: int = 100,
    ):
        """
        Initialize fairness scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            fairness_objective: Fairness objective to track
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            fairness_weight: Weight for fairness objective
            fairness_warmup_epochs: Epochs to warmup fairness
        """
        self.optimizer = optimizer
        self.fairness_objective = fairness_objective
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.fairness_weight = fairness_weight
        self.fairness_warmup_epochs = fairness_warmup_epochs
        
        self.base_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max_lr,
            T_mult=1.0,
            warmup_epochs=fairness_warmup_epochs,
            eta_min=min_lr
        )
        
        self._step_count = 0
        self._current_fairness_metric = 0.0
    
    def step(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Step the scheduler based on fairness metrics.
        
        Args:
            metrics: Dictionary of current metrics
        """
        self._step_count += 1
        
        if metrics is not None:
            fairness_metric = metrics.get("fairness_metric", metrics.get("fairness", 0.0)
            self._current_fairness_metric = fairness_metric
            
            # Adjust LR based on fairness progress
            if self._current_fairness_metric < 0.05:
                # Fairness needs more work, decrease LR
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = max(
                        self.min_lr,
                        param_group["lr"] * 0.9
                    )
            else:
                # Fairness is good, use base scheduler
                self.base_scheduler.step()
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def get_last_lr(self) -> float:
        """Get last learning rate from base scheduler."""
        return self.base_scheduler.get_last_lr()
