"""
Checkpoint Callback
======================

Callback for saving and loading model checkpoints.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import os
import time

import torch
import torch.nn as nn

from src.training.trainer import Trainer
from src.core.utils import get_logger


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint."""
    path: str
    epoch: int
    step: int
    loss: float
    timestamp: float
    metrics: Dict[str, float]
    is_best: bool = False


class CheckpointCallback:
    """
    Callback for saving model checkpoints.
    
    Saves model, optimizer, and training state at regular intervals
    to enable training resumption and model selection.
    
    Example:
        >>> callback = CheckpointCallback(
        ...     checkpoint_dir="checkpoints",
        ...     save_frequency=5,
        ...     max_checkpoints=10
        ... )
        >>> trainer.add_callback(callback)
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_frequency: int = 5,
        max_checkpoints: int = 10,
        save_best: bool = True,
        metric_name: str = "val_loss",
        mode: str = "min",
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        filename_pattern: str = "checkpoint_epoch_{epoch:04d}.pt",
        name: str = "checkpoint_callback"
    ):
        """
        Initialize checkpoint callback.
        
        Args:
            checkpoint_dir: Directory for saving checkpoints
            save_frequency: Save checkpoint every N epochs
            max_checkpoints: Maximum number of checkpoints to keep
            save_best: Whether to save the best model
            metric_name: Metric for selecting best model
            mode: "min" or "max" for best model selection
            save_optimizer: Whether to save optimizer state
            save_scheduler: Whether to save scheduler state
            filename_pattern: Pattern for checkpoint filenames
            name: Callback name
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_frequency = save_frequency
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.metric_name = metric_name
        self.mode = mode
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.filename_pattern = filename_pattern
        self.name = name
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Logger
        self.logger = get_logger(self.name)
        
        # State tracking
        self._best_metric = float("inf") if mode == "min" else float("-inf")
        self._checkpoint_history: List[CheckpointInfo] = []
        self._current_checkpoint: Optional[CheckpointInfo] = None
    
    def on_train_begin(self, trainer: Trainer) -> None:
        """Called at training start."""
        self.logger.info(f"Checkpoint callback initialized")
        self.logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """Called at end of each epoch."""
        # Check if we should save a checkpoint
        should_save = (
            epoch % self.save_frequency == 0 or
            epoch == trainer.config.n_epochs - 1
        )
        
        if should_save:
            self._save_checkpoint(trainer, epoch, metrics)
        
        # Check for best model
        if self.save_best:
            self._check_and_save_best(trainer, epoch, metrics)
    
    def on_train_end(self, trainer: Trainer) -> None:
        """Called at training end."""
        # Save final checkpoint
        if self._current_checkpoint is None:
            self._save_checkpoint(
                trainer,
                trainer.current_epoch,
                {"final": True}
            )
        
        # Save checkpoint history
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        with open(history_path, "w") as f:
            json.dump([c.__dict__ for c in self._checkpoint_history], f)
        
        self.logger.info(
            f"Training complete. "
            f"Total checkpoints saved: {len(self._checkpoint_history)}"
        )
    
    def _save_checkpoint(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: Dict[str, float]
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch
            metrics: Current metrics
            
        Returns:
            Path to saved checkpoint
        """
        # Create checkpoint data
        checkpoint_data = {
            "model_state_dict": trainer.model.state_dict(),
            "epoch": epoch,
            "step": trainer.global_step,
            "metrics": metrics,
            "config": trainer.config,
        }
        
        # Add optimizer state
        if self.save_optimizer:
            checkpoint_data["optimizer_state_dict"] = trainer.optimizer.state_dict()
        
        # Add scheduler state
        if self.save_scheduler and trainer.scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = trainer.scheduler.state_dict()
        
        # Add adversary state if present
        if hasattr(trainer, "adversary") and trainer.adversary is not None:
            checkpoint_data["adversary_state_dict"] = trainer.adversary.state_dict()
        
        # Save checkpoint
        filename = self.filename_pattern.format(epoch=epoch)
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Track checkpoint info
        info = CheckpointInfo(
            path=str(checkpoint_path),
            epoch=epoch,
            step=trainer.global_step,
            loss=metrics.get("loss", float("inf")),
            timestamp=time.time(),
            metrics=metrics
        )
        
        self._checkpoint_history.append(info)
        self._current_checkpoint = info
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def _check_and_save_best(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: Dict[str, float]
    ) -> bool:
        """
        Check if current model is the best and save if so.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch
            metrics: Current metrics
            
        Returns:
            True if best model was saved
        """
        metric_value = metrics.get(self.metric_name, float("inf"))
        
        is_best = False
        if self.mode == "min":
            is_best = metric_value < self._best_metric
            if is_best:
                self._best_metric = metric_value
        else:
            is_best = metric_value > self._best_metric
            if is_best:
                self._best_metric = metric_value
        
        if is_best:
            # Save best model
            best_path = self.checkpoint_dir / "best_model.pt"
            
            checkpoint_data = {
                "model_state_dict": trainer.model.state_dict(),
                "epoch": epoch,
                "step": trainer.global_step,
                "metrics": metrics,
                "best_metric": self._best_metric,
            }
            
            torch.save(checkpoint_data, best_path)
            
            self.logger.info(
                f"New best model! {self.metric_name}={metric_value:.4f} "
                f"Saved to {best_path}"
            )
            
            # Update current checkpoint info
            if self._current_checkpoint is not None:
                self._current_checkpoint.is_best = True
            
            return True
        
        return False
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to save disk space."""
        if len(self._checkpoint_history) > self.max_checkpoints:
            # Remove oldest checkpoint (but not the best one)
            for info in self._checkpoint_history[:-self.max_checkpoints]:
                if not info.is_best:
                    try:
                        os.remove(info.path)
                        self.logger.debug(f"Removed old checkpoint: {info.path}")
                    except OSError:
                        pass
            
            # Keep only recent checkpoints in history
            self._checkpoint_history = self._checkpoint_history[-self.max_checkpoints:]
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint."""
        if self._checkpoint_history:
            return self._checkpoint_history[-1].path
        return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint."""
        for info in reversed(self._checkpoint_history):
            if info.is_best:
                return info.path
        return None
    
    def load_checkpoint(
        self,
        trainer: Trainer,
        checkpoint_path: str
    ) -> None:
        """
        Load a checkpoint into the trainer.
        
        Args:
            trainer: Trainer instance
            checkpoint_path: Path to checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        
        # Load model state
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer state
        if self.save_optimizer and "optimizer_state_dict" in checkpoint:
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if self.save_scheduler and "scheduler_state_dict" in checkpoint:
            if trainer.scheduler is not None:
                trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load adversary state
        if "adversary_state_dict" in checkpoint:
            if hasattr(trainer, "adversary") and trainer.adversary is not None:
                trainer.adversary.load_state_dict(checkpoint["adversary_state_dict"])
        
        # Update trainer state
        trainer.current_epoch = checkpoint.get("epoch", 0)
        trainer.global_step = checkpoint.get("step", 0)
        
        self.logger.info(
            f"Loaded checkpoint from {checkpoint_path}. "
            f"Resuming from epoch {trainer.current_epoch}"
        )


class ModelCheckpointCallback(CheckpointCallback):
    """
    Simplified callback for saving model weights only.
    
    Saves only model weights without optimizer or scheduler state.
    Useful for deployment and inference.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "model_weights",
        save_frequency: int = 10,
        filename_pattern: str = "model_epoch_{epoch:04d}.pt",
        **kwargs
    ):
        """
        Initialize model checkpoint callback.
        
        Args:
            checkpoint_dir: Directory for saving weights
            save_frequency: Save every N epochs
            filename_pattern: Pattern for filenames
        """
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            save_frequency=save_frequency,
            save_optimizer=False,
            save_scheduler=False,
            filename_pattern=filename_pattern,
            **kwargs
        )
    
    def _save_checkpoint(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: Dict[str, float]
    ) -> str:
        """Save model weights only."""
        filename = self.filename_pattern.format(epoch=epoch)
        checkpoint_path = self.checkpoint_dir / filename
        
        torch.save(trainer.model.state_dict(), checkpoint_path)
        
        info = CheckpointInfo(
            path=str(checkpoint_path),
            epoch=epoch,
            step=trainer.global_step,
            loss=metrics.get("loss", float("inf")),
            timestamp=time.time(),
            metrics=metrics
        )
        
        self._checkpoint_history.append(info)
        self._cleanup_old_checkpoints()
        
        self.logger.info(f"Saved model weights: {checkpoint_path}")
        
        return str(checkpoint_path)


class BestModelCheckpointCallback(CheckpointCallback):
    """
    Callback that only saves the best model.
    
    Monitors a specific metric and saves only when the metric improves.
    Saves disk space while ensuring the best model is preserved.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "best_models",
        metric_name: str = "val_loss",
        mode: str = "min",
        patience: int = 5,
        **kwargs
    ):
        """
        Initialize best model checkpoint callback.
        
        Args:
            checkpoint_dir: Directory for saving
            metric_name: Metric to monitor
            mode: "min" or "max"
            patience: Number of epochs without improvement before stopping
        """
        super().__init__(
            checkpoint_dir=checkpoint_dir,
            save_frequency=1,  # Check every epoch
            save_best=True,
            metric_name=metric_name,
            mode=mode,
            **kwargs
        )
        
        self.patience = patience
        self._epochs_without_improvement = 0
    
    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """Check for improvement and save if better."""
        was_best = self._check_and_save_best(trainer, epoch, metrics)
        
        if was_best:
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1
        
        # Early stopping check
        if self._epochs_without_improvement >= self.patience:
            self.logger.warning(
                f"No improvement for {self.patience} epochs. "
                f"Consider early stopping."
            )
