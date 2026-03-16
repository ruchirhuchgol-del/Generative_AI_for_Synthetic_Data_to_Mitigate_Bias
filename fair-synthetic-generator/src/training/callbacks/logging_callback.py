"""
Logging Callback
==================

Callback for logging training progress.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

import torch

from src.training.trainer import Trainer
from src.core.utils import get_logger


@dataclass
class LogEntry:
    """Single log entry."""
    step: int
    epoch: int
    timestamp: float
    metrics: Dict[str, float]
    message: Optional[str] = None


class LoggingCallback:
    """
    Basic logging callback for training progress.
    
    Logs training progress to console and/or file with configurable
    frequency and detail level.
    
    Example:
        >>> callback = LoggingCallback(
        ...     log_frequency=100,
        ...     log_dir="logs",
        ...     log_to_file=True
        ... )
        >>> trainer.add_callback(callback)
    """
    
    def __init__(
        self,
        log_frequency: int = 100,
        log_dir: Optional[str] = None,
        log_to_file: bool = False,
        log_to_console: bool = True,
        metrics_to_log: Optional[List[str]] = None,
        name: str = "logging_callback"
    ):
        """
        Initialize logging callback.
        
        Args:
            log_frequency: Log every N steps
            log_dir: Directory for log files
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            metrics_to_log: Specific metrics to log (None = all)
            name: Callback name
        """
        self.log_frequency = log_frequency
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.metrics_to_log = metrics_to_log
        self.name = name
        
        # Logger
        self.logger = get_logger(self.name)
        
        # Log history
        self.log_history: List[LogEntry] = []
        self._log_file = None
    
    def on_train_begin(self, trainer: Trainer) -> None:
        """Called at training start."""
        if self.log_to_file and self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_path = self.log_dir / f"training_{timestamp}.log"
            self._log_file = open(log_path, "w")
            self.logger.info(f"Logging to file: {log_path}")
        
        msg = (
            f"Starting training for {trainer.config.n_epochs} epochs "
            f"with batch size {trainer.config.batch_size}"
        )
        self._log(trainer, msg, {})
    
    def on_train_end(self, trainer: Trainer) -> None:
        """Called at training end."""
        msg = "Training completed"
        self._log(trainer, msg, {})
        
        if self._log_file:
            self._log_file.close()
        
        # Save log history
        if self.log_dir:
            history_path = self.log_dir / "log_history.json"
            with open(history_path, "w") as f:
                json.dump([e.__dict__ for e in self.log_history], f)
    
    def on_epoch_begin(
        self,
        trainer: Trainer,
        epoch: int,
        **kwargs
    ) -> None:
        """Called at start of each epoch."""
        msg = f"Starting epoch {epoch}/{trainer.config.n_epochs}"
        self._log(trainer, msg, {"epoch": epoch})
    
    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """Called at end of each epoch."""
        self._log(trainer, f"Epoch {epoch} completed", metrics)
    
    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """Called at end of each step."""
        if step % self.log_frequency == 0:
            self._log(trainer, f"Step {step}", metrics)
    
    def _log(
        self,
        trainer: Trainer,
        message: str,
        metrics: Dict[str, float]
    ) -> None:
        """
        Log a message with metrics.
        
        Args:
            trainer: Trainer instance
            message: Log message
            metrics: Metrics to log
        """
        # Filter metrics
        if self.metrics_to_log:
            metrics = {k: v for k, v in metrics.items()
                      if k in self.metrics_to_log}
        
        # Create log entry
        entry = LogEntry(
            step=trainer.global_step,
            epoch=trainer.current_epoch,
            timestamp=time.time(),
            metrics=metrics,
            message=message
        )
        
        self.log_history.append(entry)
        
        # Log to console
        if self.log_to_console:
            metrics_str = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            if metrics_str:
                msg = f"{message} | {metrics_str}"
            else:
                msg = message
            self.logger.info(msg)
        
        # Log to file
        if self.log_to_file and self._log_file:
            log_line = f"[{entry.timestamp}] Epoch {entry.epoch} | Step {entry.step} | {message}"
            if metrics:
                log_line += " | " + " | ".join(f"{k}: {v:.6f}" for k, v in metrics.items())
            self._log_file.write(log_line + "\n")
            self._log_file.flush()


class TensorBoardCallback(LoggingCallback):
    """
    Callback for TensorBoard logging.
    
    Logs metrics, histograms, and other data to TensorBoard
    for visualization.
    
    Example:
        >>> callback = TensorBoardCallback(
        ...     log_dir="runs/experiment",
        ...     log_histograms=True
        ... )
    """
    
    def __init__(
        self,
        log_dir: str = "runs/training",
        log_frequency: int = 100,
        log_histograms: bool = True,
        log_embeddings: bool = False,
        log_graph: bool = False,
        name: str = "tensorboard_callback"
    ):
        """
        Initialize TensorBoard callback.
        
        Args:
            log_dir: TensorBoard log directory
            log_frequency: Log every N steps
            log_histograms: Log weight histograms
            log_embeddings: Log embeddings (for high-dimensional data)
            log_graph: Log model graph
            name: Callback name
        """
        super().__init__(
            log_frequency=log_frequency,
            log_dir=log_dir,
            log_to_file=False,
            log_to_console=False,
            name=name
        )
        
        self.log_histograms = log_histograms
        self.log_embeddings = log_embeddings
        self.log_graph = log_graph
        
        self._writer = None
        self._logged_graph = False
    
    def on_train_begin(self, trainer: Trainer) -> None:
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(self.log_dir)
            self.logger.info(f"TensorBoard logging to: {self.log_dir}")
        except ImportError:
            self.logger.warning(
                "TensorBoard not available. Install with: pip install tensorboard"
            )
            self._writer = None
    
    def on_train_end(self, trainer: Trainer) -> None:
        """Close TensorBoard writer."""
        if self._writer:
            self._writer.close()
    
    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """Log metrics to TensorBoard."""
        if not self._writer:
            return
        
        # Log metrics
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self._writer.add_scalar(f"epoch/{name}", value, epoch)
        
        # Log learning rate
        lr = trainer.optimizer.param_groups[0]["lr"]
        self._writer.add_scalar("epoch/learning_rate", lr, epoch)
        
        # Log histograms
        if self.log_histograms and epoch % 10 == 0:
            for name, param in trainer.model.named_parameters():
                self._writer.add_histogram(f"weights/{name}", param.data, epoch)
                if param.grad is not None:
                    self._writer.add_histogram(f"gradients/{name}", param.grad, epoch)
    
    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """Log step metrics to TensorBoard."""
        if not self._writer:
            return
        
        if step % self.log_frequency == 0:
            for name, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(f"step/{name}", value, step)
    
    def on_batch_end(
        self,
        trainer: Trainer,
        batch_idx: int,
        batch: Any,
        outputs: Dict[str, Any],
        **kwargs
    ) -> None:
        """Called after each batch."""
        if not self._writer:
            return
        
        # Log model graph once
        if self.log_graph and not self._logged_graph:
            try:
                # Try to log the model graph
                if isinstance(batch, dict):
                    sample_input = {k: v[:1] for k, v in batch.items()
                                  if isinstance(v, torch.Tensor)}
                    self._writer.add_graph(trainer.model, sample_input)
                self._logged_graph = True
            except Exception as e:
                self.logger.debug(f"Could not log graph: {e}")


class WandBCallback(LoggingCallback):
    """
    Callback for Weights & Biases logging.
    
    Integrates with W&B for experiment tracking and visualization.
    
    Example:
        >>> callback = WandBCallback(
        ...     project="fair-synthesis",
        ...     entity="my-team",
        ...     config=training_config
        ... )
    """
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        log_frequency: int = 100,
        log_model: bool = False,
        name: str = "wandb_callback"
    ):
        """
        Initialize W&B callback.
        
        Args:
            project: W&B project name
            entity: W&B entity/team name
            run_name: Name for this run
            config: Configuration to log
            log_frequency: Log every N steps
            log_model: Whether to log model checkpoints
            name: Callback name
        """
        super().__init__(
            log_frequency=log_frequency,
            log_to_file=False,
            log_to_console=False,
            name=name
        )
        
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.config = config
        self.log_model = log_model
        
        self._run = None
    
    def on_train_begin(self, trainer: Trainer) -> None:
        """Initialize W&B run."""
        try:
            import wandb
            
            # Initialize run
            self._run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.run_name,
                config=self.config or {},
                reinit=True
            )
            
            self.logger.info(f"W&B initialized: {self._run.url}")
            
        except ImportError:
            self.logger.warning(
                "Weights & Biases not available. Install with: pip install wandb"
            )
        except Exception as e:
            self.logger.warning(f"Could not initialize W&B: {e}")
    
    def on_train_end(self, trainer: Trainer) -> None:
        """Finish W&B run."""
        if self._run:
            import wandb
            wandb.finish()
    
    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """Log metrics to W&B."""
        if not self._run:
            return
        
        import wandb
        
        # Log metrics
        log_dict = {"epoch": epoch, **metrics}
        
        # Add learning rate
        lr = trainer.optimizer.param_groups[0]["lr"]
        log_dict["learning_rate"] = lr
        
        wandb.log(log_dict)
    
    def on_step_end(
        self,
        trainer: Trainer,
        step: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """Log step metrics to W&B."""
        if not self._run or step % self.log_frequency != 0:
            return
        
        import wandb
        
        wandb.log({"step": step, **metrics})


class MetricsLogger:
    """
    Utility class for aggregating and tracking metrics.
    
    Provides efficient metric aggregation with rolling averages
    and statistics computation.
    
    Example:
        >>> logger = MetricsLogger()
        >>> for batch in dataloader:
        ...     loss = train_step(batch)
        ...     logger.update("loss", loss.item())
        >>> 
        >>> avg_loss = logger.get_average("loss")
    """
    
    def __init__(
        self,
        window_size: int = 100,
        precision: int = 6
    ):
        """
        Initialize metrics logger.
        
        Args:
            window_size: Size of rolling window for averages
            precision: Decimal precision for display
        """
        self.window_size = window_size
        self.precision = precision
        
        self._metrics: Dict[str, List[float]] = {}
        self._counts: Dict[str, int] = {}
    
    def update(
        self,
        name: str,
        value: float
    ) -> None:
        """
        Update a metric with a new value.
        
        Args:
            name: Metric name
            value: New value
        """
        if name not in self._metrics:
            self._metrics[name] = []
            self._counts[name] = 0
        
        self._metrics[name].append(value)
        self._counts[name] += 1
        
        # Keep only window_size values
        if len(self._metrics[name]) > self.window_size:
            self._metrics[name] = self._metrics[name][-self.window_size:]
    
    def get_average(self, name: str) -> float:
        """Get rolling average for a metric."""
        if name not in self._metrics or not self._metrics[name]:
            return 0.0
        return sum(self._metrics[name]) / len(self._metrics[name])
    
    def get_last(self, name: str) -> float:
        """Get last value for a metric."""
        if name not in self._metrics or not self._metrics[name]:
            return 0.0
        return self._metrics[name][-1]
    
    def get_std(self, name: str) -> float:
        """Get standard deviation for a metric."""
        if name not in self._metrics or len(self._metrics[name]) < 2:
            return 0.0
        
        import math
        values = self._metrics[name]
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def get_all_averages(self) -> Dict[str, float]:
        """Get rolling averages for all metrics."""
        return {name: self.get_average(name) for name in self._metrics}
    
    def reset(self, name: Optional[str] = None) -> None:
        """Reset metrics."""
        if name:
            self._metrics[name] = []
            self._counts[name] = 0
        else:
            self._metrics = {}
            self._counts = {}
    
    def format_metric(self, name: str) -> str:
        """Format metric for display."""
        avg = self.get_average(name)
        std = self.get_std(name)
        return f"{name}: {avg:.{self.precision}f} ± {std:.{self.precision}f}"
