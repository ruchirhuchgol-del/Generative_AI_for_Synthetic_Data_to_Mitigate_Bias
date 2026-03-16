"""
Curriculum Learning Strategy
===============================

Implements curriculum learning for fair synthetic data generation.

Curriculum learning progressively increases task difficulty during training
to help models learn better representations and converge faster.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
import math
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.trainer import Trainer, TrainingConfig
from src.core.utils import get_logger, get_device


class CurriculumType(Enum):
    """Types of curriculum scheduling."""
    LINEAR = "linear"              # Linear increase in difficulty
    STEP = "step"                # Step-wise increase
    COSINE = "cosine"            # Cosine annealing
    EXPONENTIAL = "exponential"  # Exponential increase
    SELF_PACED = "self_paced"  # Self-paced based on metrics


@dataclass
class CurriculumConfig:
    """
    Configuration for curriculum learning.
    
    Attributes:
        curriculum_type: Type of curriculum scheduling
        start_difficulty: Starting difficulty (0.0 to 1.0)
        end_difficulty: Ending difficulty (0.0 to 1.0)
        warmup_epochs: Number of warmup epochs
        step_epochs: Epochs per step (for step type)
        self_paced_threshold: Threshold for self-paced advancement
        fairness_weight_schedule: How fairness weight changes
        use_fairness_curriculum: Whether to apply fairness curriculum
    """
    curriculum_type: str = "linear"
    start_difficulty: float = 0.1
    end_difficulty: float = 1.0
    warmup_epochs: int = 10
    step_epochs: int = 100
    self_paced_threshold: float = 0.02
    fairness_weight_schedule: str = "linear"
    use_fairness_curriculum: bool = True


    # Progressive batch sizing
    initial_batch_size: int = 32
    final_batch_size: int = 128
    batch_growth_epochs: int = 50
    
    # Progressive sequence length (for text)
    initial_seq_len: int = 32
    final_seq_len: int = 512


    seq_growth_epochs: int = 50


@dataclass
class DifficultyMetrics:
    """Metrics for tracking difficulty progress."""
    current_difficulty: float = 0.0
    epochs_completed: int = 0
    steps_at_current_difficulty: int = 0
    performance_at_current: float = 0.0
    fairness_at_current: float = 0.0


class CurriculumTrainingStrategy:
    """
    Curriculum Learning Strategy for Fairness.
    
    Implements progressive training with increasing difficulty:
        - Linear: Smooth linear increase
        - Step: Discrete difficulty steps
        - Cosine: Cosine annealing schedule
        - Self-paced: Adaptive based on performance
    
    Supports:
        - Progressive batch sizing
        - Progressive sequence lengths
        - Fairness curriculum (increasing fairness weight)
    
    Example:
        >>> strategy = CurriculumTrainingStrategy(
        ...     model=model,
        ...     train_dataloader=dataloader,
        ...     config=CurriculumConfig()
        ... )
        >>> for epoch in range(num_epochs):
        ...     metrics = strategy.train_epoch(dataloader, epoch)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[CurriculumConfig] = None,
        device: Optional[torch.device] = None,
        val_dataloader: Optional[DataLoader] = None,
        fairness_regularizer: Optional[nn.Module] = None,
        difficulty_fn: Optional[Callable] = None,
    ):
        """
        Initialize curriculum training strategy.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            optimizer: Model optimizer
            config: Curriculum configuration
            device: Device to use
            val_dataloader: Validation data loader
            fairness_regularizer: Fairness regularization module
            difficulty_fn: Function to compute difficulty
        """
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            config=config or TrainingConfig(),
            device=device
        )
        
        self.curriculum_config = config or CurriculumConfig()
        self.fairness_regularizer = fairness_regularizer
        self.difficulty_fn = difficulty_fn
        
        # Difficulty tracking
        self.difficulty_metrics = DifficultyMetrics()
        
        # History for curriculum schedule
        self.difficulty_history = []
        self.batch_size_history = []
        self.fairness_weight_history = []
        
        # Current state
        self._current_difficulty = self.curriculum_config.start_difficulty
        self._current_fairness_weight = 0.1  # Initial fairness weight
        
        self.logger = get_logger("CurriculumStrategy")
        
    def get_current_difficulty(self, epoch: int) -> float:
        """
        Get current difficulty level based on curriculum schedule.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Current difficulty level (0.0 to 1.0)
        """
        total_epochs = self.config.n_epochs
        warmup_epochs = self.curriculum_config.warmup_epochs
        
        if epoch < warmup_epochs:
            # During warmup, progress = epoch / warmup_epochs
            return self._current_difficulty
        
        # After warmup,        if self.curriculum_config.curriculum_type == "linear":
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            difficulty = self.curriculum_config.start_difficulty + \
                progress * (self.curriculum_config.end_difficulty - self.curriculum_config.start_difficulty)
            return difficulty
        
        elif self.curriculum_config.curriculum_type == "step":
            step_idx = epoch // self.curriculum_config.step_epochs
            difficulty = min(
                self.curriculum_config.end_difficulty,
                self.curriculum_config.start_difficulty + 
                (self.curriculum_config.end_difficulty - self.curriculum_config.start_difficulty) * step_idx
            )
            return difficulty
        
        elif self.curriculum_config.curriculum_type == "cosine":
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cos_progress = (1 - math.cos(math.pi * progress)) / 2
            difficulty = self.curriculum_config.start_difficulty + \
                cos_progress * (self.curriculum_config.end_difficulty - self.curriculum_config.start_difficulty)
            return difficulty
        
        elif self.curriculum_config.curriculum_type == "exponential":
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            exp_progress = math.exp(progress * 2) - 1
            difficulty = self.curriculum_config.start_difficulty + \
                exp_progress * (self.curriculum_config.end_difficulty - self.curriculum_config.start_difficulty)
            return difficulty
        
        elif self.curriculum_config.curriculum_type == "self_paced":
            return self._current_difficulty  # Updated dynamically
        
        else:
            return self.curriculum_config.end_difficulty
    
    def get_current_batch_size(self, epoch: int) -> int:
        """Get current batch size based on progressive schedule."""
        config = self.curriculum_config
        
        progress = min(epoch / config.batch_growth_epochs, 1.0)
        
        batch_size = int(
            config.initial_batch_size +
            progress * (config.final_batch_size - config.initial_batch_size)
        )
        
        return batch_size
    
    def get_current_fairness_weight(self, epoch: int) -> float:
        """Get current fairness regularization weight."""
        if not self.curriculum_config.use_fairness_curriculum:
            return self._current_fairness_weight
        
        total_epochs = self.config.n_epochs
        
        # Similar schedule as difficulty
        if self.curriculum_config.fairness_weight_schedule == "linear":
            progress = epoch / total_epochs
            return self._current_fairness_weight * (1 + progress)
        
        return self._current_fairness_weight
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train for one epoch with curriculum learning.
        
        Args:
            dataloader: Training dataloader
            epoch: Current epoch number
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of training metrics
        """
        self.current_epoch = epoch
        
        # Update difficulty
        self._current_difficulty = self.get_current_difficulty(epoch)
        self.difficulty_history.append(self._current_difficulty)
        
        # Update batch size (if progressive)
        current_batch_size = self.get_current_batch_size(epoch)
        
        # Update fairness weight
        if self.curriculum_config.use_fairness_curriculum:
            self._current_fairness_weight = self.get_current_fairness_weight(epoch)
            self.fairness_weight_history.append(self._current_fairness_weight)
        
        # Track difficulty metrics
        self.difficulty_metrics.current_difficulty = self._current_difficulty
        self.difficulty_metrics.epochs_completed = epoch
        
        # Train with base trainer
        metrics = super().train_epoch(dataloader, epoch, **kwargs)
        
        # Apply difficulty to loss if applicable
        if self.difficulty_fn is not None and "difficulty" not in metrics:
            metrics["difficulty"] = self._current_difficulty
        
        # Add curriculum metrics
        metrics["curriculum_difficulty"] = self._current_difficulty
        metrics["curriculum_batch_size"] = current_batch_size
        
        if self.curriculum_config.use_fairness_curriculum:
            metrics["curriculum_fairness_weight"] = self._current_fairness_weight
        
        # Self-paced adjustment
        if self.curriculum_config.curriculum_type == "self_paced":
            self._update_self_paced_difficulty(metrics)
        
        return metrics
    
    def _update_self_paced_difficulty(self, metrics: Dict[str, float]) -> None:
        """
        Update difficulty based on self-paced learning.
        
        Adjusts difficulty based on model performance and fairness metrics.
        """
        threshold = self.curriculum_config.self_paced_threshold
        
        # Check if performance is improving
        if "val_loss" in metrics:
            val_loss = metrics["val_loss"]
            
            # Check fairness
            fairness_score = metrics.get("fairness_score", 0.0)
            
            # Adjust difficulty
            if val_loss < threshold or fairness_score < threshold:
                # Performance is good and fairness is good, increase difficulty
                self._current_difficulty = min(
                    1.0,
                    self._current_difficulty + 0.05
                )
            else:
                # Need more practice, decrease difficulty
                self._current_difficulty = max(
                    self.curriculum_config.start_difficulty,
                    self._current_difficulty - 0.05
                )
        
        self.difficulty_metrics.performance_at_current = val_loss
        self.difficulty_metrics.fairness_at_current  fairness_score
    
    def get_curriculum_state(self) -> Dict[str, Any]:
        """Get current curriculum state."""
        return {
            "current_difficulty": self._current_difficulty,
            "current_fairness_weight": self._current_fairness_weight,
            "difficulty_history": self.difficulty_history,
            "fairness_weight_history": self.fairness_weight_history,
        }
    
    def set_difficulty(self, difficulty: float) -> None:
        """Manually set difficulty level."""
        self._current_difficulty = max(
            self.curriculum_config.start_difficulty,
            min(difficulty, 1.0)
        )
        self._current_difficulty = min(difficulty, self.curriculum_config.end_difficulty)
        
        self.logger.info(f"Manually set difficulty to {self._current_difficulty:.3f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get curriculum metrics summary."""
        return {
            **self.difficulty_metrics.to_dict(),
            "difficulty_history_length": len(self.difficulty_history),
            "fairness_weight_history_length": len(self.fairness_weight_history),
        }


class FairnessCurriculumMixin:
    """
    Mixin for adding fairness-aware curriculum to any trainer.
    
    Gradually increases fairness weight during training to help
    the model learn fair representations.
    """
    
    def __init__(
        self,
        initial_weight: float = 0.1,
        final_weight: float  1.0,
        warmup_epochs: int = 100
    ):
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.warmup_epochs = warmup_epochs
    
    def get_weight(self, epoch: int) -> float:
        """Get fairness weight for current epoch."""
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            return self.initial_weight + progress * (self.final_weight - self.initial_weight)
        return self.final_weight
    
    def on_train_begin(self, trainer: Trainer) -> None:
        """Called at training start."""
        trainer._current_fairness_weight = self.initial_weight
    
    def on_epoch_begin(self, trainer: Trainer, epoch: int) -> None:
        """Called at epoch start."""
        weight = self.get_weight(epoch)
        trainer._current_fairness_weight = weight
    
    def on_epoch_end(self, trainer: Trainer, **kwargs) -> None:
        """Called at epoch end."""
        self.logger.info(
            f"Epoch {trainer.current_epoch}: "
            f"Fairness weight = {trainer._current_fairness_weight:.4f}"
        )
