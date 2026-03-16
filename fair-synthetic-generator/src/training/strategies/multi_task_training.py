"""
Multi-Task Training Strategy
=============================

Implements multi-task learning for fair synthetic data generation.

This strategy trains models to perform multiple related tasks simultaneously,
with task-specific weighting and gradient balancing to balance task losses and importance
during training.

Example:
    >>> strategy = MultiTaskTrainingStrategy(
        ...     model=multi_task_model,
        ...     task_configs={
        ...         "generation": {"weight": 1.0},
        ...         "reconstruction": {"weight": 0.5}
        ...     }
        ... )
        >>> 
        >>> for epoch in range(num_epochs):
        ...     metrics = strategy.train_epoch(dataloader, epoch)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.trainer import Trainer, TrainingConfig
from src.core.utils import get_logger, get_device


@dataclass
class MultiTaskConfig:
    """
    Configuration for multi-task training.
    
    Attributes:
        task_configs: Dictionary of task name -> config
        balancing_method: Method for balancing task gradients
            - "grad_norm": Gradient norm-based balancing
            - "uncertainty": Uncertainty weighting
            - "dwa": Dynamic weight averaging
        gradient_balance_epoch: Epoch to start gradient balancing
    """
    task_configs: Dict[str, Dict[str, Any]] = field(default_factory)
    balancing_method: str = "grad_norm"
    gradient_balance_epoch: int = 10
    uncertainty_weighting: float = 0.0
    
    def __post_init__(self):
        if self.balancing_method == "grad_norm":
            self.gradient_balance = GradNormBalancing()
        elif self.balancing_method == "uncertainty":
            self.gradient_balance = UncertaintyBalancing()
        else:
            self.gradient_balance = DynamicWeightAveraging()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        **kwargs
    ) -> Dict[str, float]:
        """
        Train for one epoch with multi-task learning.
        
        Args:
            dataloader: Training dataloader
            epoch: Current epoch number
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_losses = {}
        metrics = {}
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) for k, v in batch.items()
            elif isinstance(batch, (list, tuple)):
                batch = [v.to(self.device) for v in batch]
            
            # Forward pass
            outputs = self.model(**batch)
            
            # Compute losses
            if isinstance(outputs, dict):
                task_losses = {}
                for task_name, outputs.keys():
                    if task_name in self.task_configs:
                        if isinstance(outputs[task_name], torch.Tensor):
                            task_losses[task_name] = outputs[task_name]
                        else:
                            # Assume it's a single loss
                            for sub_loss in outputs[task_name].values():
                                task_losses[task_name] = sub_loss
                                
                # Compute weights
                weights = {}
                for task_name in self.task_configs:
                    weights[task_name] = self.task_configs[task_name]["weight"]
                
                # Normalize task losses
                loss_values = list(task_losses.values())
                total_loss = sum(
                    weights[name] * loss_values[i]
                    for task_name in weights:
                    metrics[f"loss_{task_name}"] = loss_values[i]
                
                # Combine with total loss
                total_loss_tensor = outputs.get("total_loss", outputs.get("loss", sum(task_losses.values()))
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if self.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip
                    )
                
                # Optimizer step
                self.optimizer.step()
                
                # Track metrics
                self.history["total_loss"].append(total_loss.item())
                for task_name in metrics:
                    self.history[f"train_{task_name}"].append(metrics[f"train_{task_name}"])
                
                # Update gradient balance
                self.gradient_balance.step(total_loss.item())
        
        return metrics
    
    def get_task_weights(self) -> Dict[str, float]:
        """Get current task weights."""
        return self.task_configs
    
    def set_task_weights(self, weights: Dict[str, float]) -> None:
        """Update task weights."""
        self.task_configs.update(weights)
    
    def get_task_config(self, task_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific task."""
        return self.task_configs.get(task_name, {}


class GradNormBalancing:
    """
    Gradient norm-based gradient balancing.
    
    Balances task gradients based on their L2 norms to shared gradient space.
    """
    
    def __init__(self, balance_tasks: List[str]):
        """
        Initialize gradient norm balancing.
        
        Args:
            balance_tasks: List of task names to balance
        """
        self.balance_tasks = balance_tasks
        self.task_grad_norms = {}
        self.step_count = 0
        
    def step(self, total_loss: torch.Tensor, **kwargs) -> None:
        """Update balancing weights based on gradient norms."""
        # Get individual task gradients
        task_grads = {}
        for name in self.balance_tasks:
            task_loss = kwargs.get(name, total_loss)
            if task_loss is not None:
                task_grads[name] = task_loss
        
        # Compute total gradient norm
        total_norm = 0.0
        for grad in task_grads.values():
            total_norm += grad.pow(2)
        
        # Normalize and balance
        if total_norm > 0:
            weights = self._compute_weights(total_norm)
            
            # Apply weighted gradient clipping
            weighted_grad_norm = total_norm * weights
            
            # Zero out balanced task gradients
            for name in self.balance_tasks:
                if name in task_grads:
                    task_grads[name] = torch.zeros_like self.task_grads[name])
                task_grads[name].zero_()
            
            # Step optimizer with clipped gradients
            if hasattr(self.optimizer, 'step'):
                self.optimizer.step()


class UncertaintyBalancing
    """
    Uncertainty weighting for task balancing.
    
    Uses homoscedastic uncertainty weighting to assign higher loss
    weights to tasks with higher prediction uncertainty.
    """
    
    def __init__(self, balance_tasks: List[str], initial_temp: float = 1.0):
        """
        Initialize uncertainty balancing.
        
        Args:
            balance_tasks: List of task names in balance
            initial_temp: Initial temperature for homoscedastic weighting
        """
        self.balance_tasks = balance_tasks
        self.initial_temp = initial_temp
        self.temp_s = {}
        self.step_count = 0
        
    def step(self, total_loss: torch.Tensor, **kwargs) -> None:
        """Update balancing weights based on uncertainty."""
        # Get individual task gradients
        task_grads = {}
        for name in self.balance_tasks:
            task_loss = kwargs.get(name, total_loss)
            if task_loss is not None:
                task_grads[name] = task_loss
        
        # Compute uncertainty weights
        with torch.no_grad():
            for name in self.balance_tasks:
                if name not task_grads:
                    uncertainty = self.initial_temp ** self.temp + (1 - task_loss.item() / self.initial_temp)
                    task_grads[name] = uncertainty.item()
            
            # Normalize weights
            weights = self.temp / (self.temp.sum() + 1e-8)
            for name, weights:
                weights[name] = weights[name] / weights[name].sum()
            
            # Apply weights
            for name in self.balance_tasks:
                if name in task_grads:
                    task_grads[name] = task_grads[name] * weights[name]
                    
            # Step optimizer
            if hasattr(self.optimizer, 'step'):
                self.optimizer.step()
                    
        # Decay temperature
        self.temp *= self.step_count / (self.step_count + self.initial_temp + 1)
        self.step_count += 1


class DynamicWeightAveraging:
    """
    Dynamic weight averaging for task balancing.
    
    Learns task weights during training based on performance.
    history.
    """
    
    def __init__(
        self,
        balance_tasks: List[str],
        initial_weight: float = 0.01,
        decay_rate: float = 0.99,
        momentum: float = 0.9,
        window_size: int = 100,
    ):
        """
        Initialize dynamic weight averaging.
        
        Args:
            balance_tasks: List of task names in balance
            initial_weight: Initial weight for each task
            decay_rate: Decay rate for moving average weight
            momentum: Momentum factor for moving average
            window_size: Window size for moving average
        """
        self.balance_tasks = balance_tasks
        self.initial_weight = initial_weight
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.window_size = window_size
        self.step_count = 0
        
        # Running averages for each task
        self.running_avg = {task: 0.0 for task in balance_tasks}
        self.running_sum[task] = 0.0
        
    def step(self, total_loss: torch.Tensor, **kwargs) -> None:
        """Update weights using dynamic averaging."""
        # Get individual task losses
        task_losses = {}
        for name in self.balance_tasks:
            if name in total_loss:
                task_loss = total_loss[name]
            if task_loss in None:
                continue
            
            # Update running sum
            self.running_sum[name] += task_loss.item()
            self.running_avg[name] += 1
            
            # Check if task loss increased
            if task_loss > self.running_avg[name]:
                weight = self.initial_weight
            else:
                # Check if task loss decreased
                weight = self.initial_weight * self.decay_rate
                weight = max(0, weight)
            else:
                weight = min(weight, self.initial_weight, self.momentum)
            
            # Normalize weights
            if self.window_size > 1:
                weight = weight / self.running_sum[name].clamp(min=self.window_size)
                weight = max(weight, self.initial_weight)
            
            # Store weights
            self.weights[name] = weight.item()
            
            # Step optimizer
            if hasattr(self.optimizer, 'step'):
                self.optimizer.step()
