"""
Multi-Objective Fairness Loss
=============================

Multi-objective optimization for fairness-aware training.

This module provides:
- Scalarization methods for multi-objective optimization
- Pareto frontier tracking
- Dynamic weight adjustment strategies
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScalarizationMethod(Enum):
    """Methods for scalarizing multi-objective losses."""
    WEIGHTED_SUM = "weighted_sum"
    TCHEBYCHEFF = "tchebycheff"
    WEIGHTED_TCHEBYCHEFF = "weighted_tchebycheff"
    AUGMENTED_TCHEBYCHEFF = "augmented_tchebycheff"
    PENALTY_BOUNDARY = "penalty_boundary"
    EPSILON_CONSTRAINT = "epsilon_constraint"


class WeightUpdateStrategy(Enum):
    """Strategies for updating fairness weights."""
    FIXED = "fixed"
    GRADIENT_BASED = "gradient_based"
    PERFORMANCE_DRIVEN = "performance_driven"
    ADAPTIVE = "adaptive"
    SCHEDULED = "scheduled"


class MultiObjectiveLoss(nn.Module):
    """
    Multi-Objective Loss for Fairness-Aware Training.
    
    Combines multiple objectives (utility + fairness) using various
    scalarization methods and adaptive weight strategies.
    
    Mathematical Formulation:
        minimize: [L_task, L_fairness_1, L_fairness_2, ...]
        
    Scalarization Methods:
        - Weighted Sum: Σ w_i * L_i
        - Tchebycheff: max_i(w_i * |L_i - z_i|)
        - Augmented Tchebycheff: max_i(...) + ρ * Σ w_i * |L_i - z_i|
    
    Example:
        >>> loss_fn = MultiObjectiveLoss(
        ...     objectives={
        ...         "task": task_loss_fn,
        ...         "demographic_parity": dp_constraint,
        ...         "equalized_odds": eo_constraint
        ...     },
        ...     weights={"task": 1.0, "demographic_parity": 0.5, "equalized_odds": 0.5},
        ...     method="weighted_sum"
        ... )
        >>> 
        >>> losses = loss_fn(**inputs)
        >>> losses["total"].backward()
    """
    
    def __init__(
        self,
        objectives: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
        method: str = "weighted_sum",
        weight_update: str = "fixed",
        reference_point: Optional[Dict[str, float]] = None,
        rho: float = 0.01,
        initial_fairness_weight: float = 0.1,
        max_fairness_weight: float = 10.0,
        fairness_weight_step: float = 0.01,
        name: str = "multi_objective_loss"
    ):
        """
        Initialize multi-objective loss.
        
        Args:
            objectives: Dictionary of objective name -> loss function
            weights: Initial weights for each objective
            method: Scalarization method
            weight_update: Weight update strategy
            reference_point: Reference point for Tchebycheff methods (z*)
            rho: Augmentation parameter for augmented Tchebycheff
            initial_fairness_weight: Starting weight for fairness objectives
            max_fairness_weight: Maximum fairness weight
            fairness_weight_step: Step size for weight updates
            name: Loss name
        """
        super().__init__()
        
        self.objectives = nn.ModuleDict(objectives)
        self.method = ScalarizationMethod(method)
        self.weight_update = WeightUpdateStrategy(weight_update)
        self.rho = rho
        self.name = name
        
        # Initialize weights
        if weights is not None:
            self.weights = weights
        else:
            # Default: task weight = 1, fairness weights = initial_fairness_weight
            self.weights = {}
            for name in objectives:
                if "fairness" in name.lower() or "parity" in name.lower() or "odds" in name.lower():
                    self.weights[name] = initial_fairness_weight
                else:
                    self.weights[name] = 1.0
        
        # Reference point for Tchebycheff
        if reference_point is not None:
            self.reference_point = reference_point
        else:
            self.reference_point = {name: 0.0 for name in objectives}
        
        # Adaptive weight parameters
        self.max_fairness_weight = max_fairness_weight
        self.fairness_weight_step = fairness_weight_step
        self._step_count = 0
        
        # Track history for Pareto frontier
        self._loss_history = []
        self._weight_history = []
    
    def forward(
        self,
        predictions: torch.Tensor,
        groups: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective loss.
        
        Args:
            predictions: Model predictions
            groups: Group membership tensor
            labels: Ground truth labels
            features: Feature tensor for individual fairness
            **kwargs: Additional arguments for specific objectives
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # Compute individual objective losses
        losses = {}
        
        for name, objective in self.objectives.items():
            try:
                if hasattr(objective, "loss"):
                    # Constraint-style objective
                    loss = objective.loss(predictions, groups, labels, **kwargs)
                else:
                    # Loss function-style objective
                    if labels is not None:
                        loss = objective(predictions, labels)
                    else:
                        loss = objective(predictions)
                
                if isinstance(loss, dict):
                    losses[name] = loss.get("total", loss.get("loss", loss.get(name, 0)))
                else:
                    losses[name] = loss
            except Exception as e:
                # Handle objectives that need different inputs
                losses[name] = torch.tensor(0.0, device=predictions.device)
        
        # Compute total loss using scalarization
        total = self._scalarize(losses)
        losses["total"] = total
        
        # Store history
        with torch.no_grad():
            loss_values = {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
            self._loss_history.append(loss_values)
            self._weight_history.append(self.weights.copy())
        
        return losses
    
    def _scalarize(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Scalarize multi-objective losses.
        
        Args:
            losses: Dictionary of objective losses
            
        Returns:
            Scalarized total loss
        """
        if self.method == ScalarizationMethod.WEIGHTED_SUM:
            # Simple weighted sum: Σ w_i * L_i
            total = torch.tensor(0.0, device=next(iter(losses.values())).device)
            for name, loss in losses.items():
                if name in self.weights and torch.is_tensor(loss):
                    total = total + self.weights[name] * loss
            return total
        
        elif self.method == ScalarizationMethod.TCHEBYCHEFF:
            # Tchebycheff: max_i(w_i * |L_i - z_i|)
            weighted_losses = []
            device = next(iter(losses.values())).device
            
            for name, loss in losses.items():
                if name in self.weights and torch.is_tensor(loss):
                    ref = self.reference_point.get(name, 0.0)
                    weighted = self.weights[name] * (loss - ref).abs()
                    weighted_losses.append(weighted)
            
            if not weighted_losses:
                return torch.tensor(0.0, device=device)
            
            stacked = torch.stack(weighted_losses)
            return stacked.max()
        
        elif self.method == ScalarizationMethod.WEIGHTED_TCHEBYCHEFF:
            # Weighted Tchebycheff with normalization
            device = next(iter(losses.values())).device
            weighted_losses = []
            
            for name, loss in losses.items():
                if name in self.weights and torch.is_tensor(loss):
                    ref = self.reference_point.get(name, 0.0)
                    normalized = (loss - ref).abs() / (loss.detach() + 1e-8)
                    weighted = self.weights[name] * normalized
                    weighted_losses.append(weighted)
            
            if not weighted_losses:
                return torch.tensor(0.0, device=device)
            
            stacked = torch.stack(weighted_losses)
            return stacked.max()
        
        elif self.method == ScalarizationMethod.AUGMENTED_TCHEBYCHEFF:
            # Augmented Tchebycheff: max_i(...) + ρ * Σ w_i * |L_i - z_i|
            device = next(iter(losses.values())).device
            weighted_losses = []
            sum_term = torch.tensor(0.0, device=device)
            
            for name, loss in losses.items():
                if name in self.weights and torch.is_tensor(loss):
                    ref = self.reference_point.get(name, 0.0)
                    weighted = self.weights[name] * (loss - ref).abs()
                    weighted_losses.append(weighted)
                    sum_term = sum_term + weighted
            
            if not weighted_losses:
                return torch.tensor(0.0, device=device)
            
            stacked = torch.stack(weighted_losses)
            max_term = stacked.max()
            
            return max_term + self.rho * sum_term
        
        elif self.method == ScalarizationMethod.PENALTY_BOUNDARY:
            # Penalty Boundary Intersection
            device = next(iter(losses.values())).device
            
            # Reference direction
            total = torch.tensor(0.0, device=device)
            penalty = torch.tensor(0.0, device=device)
            
            for name, loss in losses.items():
                if name in self.weights and torch.is_tensor(loss):
                    total = total + self.weights[name] * loss
            
            # Penalty for being far from the ideal point
            for name, loss in losses.items():
                if name in self.weights and torch.is_tensor(loss):
                    ref = self.reference_point.get(name, 0.0)
                    penalty = penalty + (loss - ref).pow(2)
            
            return total + self.rho * penalty
        
        else:
            # Default to weighted sum
            device = next(iter(losses.values())).device
            total = torch.tensor(0.0, device=device)
            for name, loss in losses.items():
                if name in self.weights and torch.is_tensor(loss):
                    total = total + self.weights[name] * loss
            return total
    
    def update_weights(
        self,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update weights based on strategy.
        
        Args:
            performance_metrics: Current performance metrics (accuracy, fairness, etc.)
        """
        self._step_count += 1
        
        if self.weight_update == WeightUpdateStrategy.FIXED:
            return
        
        elif self.weight_update == WeightUpdateStrategy.SCHEDULED:
            # Increase fairness weights over time
            for name in self.weights:
                if "fairness" in name.lower() or "parity" in name.lower() or "odds" in name.lower():
                    new_weight = min(
                        self.weights[name] + self.fairness_weight_step,
                        self.max_fairness_weight
                    )
                    self.weights[name] = new_weight
        
        elif self.weight_update == WeightUpdateStrategy.PERFORMANCE_DRIVEN:
            if performance_metrics is not None:
                # Increase fairness weight if fairness is poor
                fairness_violation = performance_metrics.get("fairness_violation", 0)
                if fairness_violation > 0.05:
                    for name in self.weights:
                        if "fairness" in name.lower() or "parity" in name.lower():
                            self.weights[name] = min(
                                self.weights[name] * 1.1,
                                self.max_fairness_weight
                            )
        
        elif self.weight_update == WeightUpdateStrategy.ADAPTIVE:
            # Adaptive weight adjustment based on gradient magnitudes
            if len(self._loss_history) > 1:
                recent_losses = self._loss_history[-10:]  # Last 10 losses
                
                for name in self.weights:
                    if name == "task":
                        continue
                    
                    # Check if loss is increasing
                    values = [h.get(name, 0) for h in recent_losses if name in h]
                    if len(values) >= 2:
                        trend = values[-1] - values[0]
                        if trend > 0:  # Loss increasing, increase weight
                            self.weights[name] = min(
                                self.weights[name] * 1.05,
                                self.max_fairness_weight
                            )
                        elif trend < 0:  # Loss decreasing, can reduce weight
                            self.weights[name] = max(
                                self.weights[name] * 0.99,
                                0.1
                            )
    
    def get_pareto_frontier(
        self,
        objective_names: Optional[List[str]] = None
    ) -> List[Dict[str, float]]:
        """
        Get Pareto frontier from loss history.
        
        Args:
            objective_names: Names of objectives to include
            
        Returns:
            List of non-dominated points
        """
        if objective_names is None:
            objective_names = list(self.objectives.keys())
        
        # Extract relevant losses
        points = []
        for h in self._loss_history:
            point = {name: h.get(name, float('inf')) for name in objective_names}
            points.append(point)
        
        # Find non-dominated points
        pareto_frontier = []
        
        for i, point in enumerate(points):
            dominated = False
            
            for j, other in enumerate(points):
                if i == j:
                    continue
                
                # Check if other dominates point
                dominates = all(
                    other.get(name, float('inf')) <= point.get(name, float('inf'))
                    for name in objective_names
                )
                strictly_better = any(
                    other.get(name, float('inf')) < point.get(name, float('inf'))
                    for name in objective_names
                )
                
                if dominates and strictly_better:
                    dominated = True
                    break
            
            if not dominated:
                pareto_frontier.append(point)
        
        return pareto_frontier
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set weights."""
        self.weights.update(weights)
    
    def get_loss_history(self) -> List[Dict[str, float]]:
        """Get loss history."""
        return self._loss_history.copy()
    
    def reset_history(self) -> None:
        """Reset loss and weight history."""
        self._loss_history = []
        self._weight_history = []
        self._step_count = 0


class DynamicFairnessWeightScheduler:
    """
    Dynamic scheduler for fairness weights.
    
    Adjusts fairness weights during training based on various strategies:
        - Linear warmup
        - Cosine annealing
        - Performance-based adjustment
    
    Example:
        >>> scheduler = DynamicFairnessWeightScheduler(
        ...     initial_weight=0.1,
        ...     target_weight=1.0,
        ...     warmup_epochs=10
        ... )
        >>> 
        >>> for epoch in range(num_epochs):
        ...     weight = scheduler.get_weight(epoch)
        ...     loss_fn.set_weights({"fairness": weight})
    """
    
    def __init__(
        self,
        initial_weight: float = 0.1,
        target_weight: float = 1.0,
        warmup_epochs: int = 10,
        schedule_type: str = "linear",
        min_weight: float = 0.01,
        max_weight: float = 10.0,
        cycle_epochs: Optional[int] = None
    ):
        """
        Initialize weight scheduler.
        
        Args:
            initial_weight: Starting weight
            target_weight: Target weight after warmup
            warmup_epochs: Number of warmup epochs
            schedule_type: Schedule type ("linear", "cosine", "step", "cycle")
            min_weight: Minimum weight
            max_weight: Maximum weight
            cycle_epochs: Cycle length for cyclical schedule
        """
        self.initial_weight = initial_weight
        self.target_weight = target_weight
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.cycle_epochs = cycle_epochs
    
    def get_weight(self, epoch: int) -> float:
        """
        Get weight for current epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Weight value
        """
        if self.schedule_type == "linear":
            if epoch < self.warmup_epochs:
                progress = epoch / self.warmup_epochs
                weight = self.initial_weight + progress * (self.target_weight - self.initial_weight)
            else:
                weight = self.target_weight
        
        elif self.schedule_type == "cosine":
            if epoch < self.warmup_epochs:
                progress = epoch / self.warmup_epochs
                weight = self.initial_weight + (1 - math.cos(math.pi * progress)) / 2 * \
                         (self.target_weight - self.initial_weight)
            else:
                # Optional decay after warmup
                weight = self.target_weight
        
        elif self.schedule_type == "step":
            step_size = self.warmup_epochs
            num_steps = epoch // step_size
            weight = min(
                self.initial_weight * (self.target_weight / self.initial_weight) ** num_steps,
                self.target_weight
            )
        
        elif self.schedule_type == "cycle" and self.cycle_epochs is not None:
            cycle_pos = epoch % self.cycle_epochs
            progress = cycle_pos / self.cycle_epochs
            weight = self.min_weight + (1 - math.cos(math.pi * progress)) / 2 * \
                     (self.max_weight - self.min_weight)
        
        else:
            weight = self.target_weight if epoch >= self.warmup_epochs else self.initial_weight
        
        return max(self.min_weight, min(self.max_weight, weight))
    
    def get_weights_dict(
        self,
        epoch: int,
        fairness_objective_names: List[str]
    ) -> Dict[str, float]:
        """
        Get weights dictionary for all fairness objectives.
        
        Args:
            epoch: Current epoch
            fairness_objective_names: Names of fairness objectives
            
        Returns:
            Dictionary of weights
        """
        weight = self.get_weight(epoch)
        return {name: weight for name in fairness_objective_names}


class EpsilonConstraintHandler:
    """
    Handler for epsilon-constraint method.
    
    Instead of scalarizing, uses constraints on all but one objective:
        minimize: L_task
        subject to: L_fairness_i ≤ ε_i for all i
    
    Example:
        >>> handler = EpsilonConstraintHandler(
        ...     epsilon={"demographic_parity": 0.05, "equalized_odds": 0.05}
        ... )
        >>> 
        >>> # Check if constraints satisfied
        >>> if handler.is_feasible(losses):
        ...     loss = losses["task"]
        ... else:
        ...     loss = handler.penalty_loss(losses)
    """
    
    def __init__(
        self,
        epsilon: Dict[str, float],
        penalty_weight: float = 100.0,
        relaxation: str = "soft"
    ):
        """
        Initialize epsilon constraint handler.
        
        Args:
            epsilon: Dictionary of constraint bounds
            penalty_weight: Weight for constraint violations
            relaxation: Constraint relaxation type ("soft", "hard")
        """
        self.epsilon = epsilon
        self.penalty_weight = penalty_weight
        self.relaxation = relaxation
    
    def is_feasible(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> bool:
        """
        Check if all constraints are satisfied.
        
        Args:
            losses: Dictionary of losses
            
        Returns:
            True if all constraints satisfied
        """
        for name, eps in self.epsilon.items():
            if name in losses:
                value = losses[name]
                if torch.is_tensor(value):
                    value = value.item()
                if value > eps:
                    return False
        return True
    
    def penalty_loss(
        self,
        losses: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute penalty for constraint violations.
        
        Args:
            losses: Dictionary of losses
            
        Returns:
            Penalty loss tensor
        """
        device = next(v for v in losses.values() if torch.is_tensor(v)).device
        penalty = torch.tensor(0.0, device=device)
        
        for name, eps in self.epsilon.items():
            if name in losses:
                loss = losses[name]
                violation = torch.relu(loss - eps)
                
                if self.relaxation == "soft":
                    penalty = penalty + self.penalty_weight * violation.pow(2)
                else:
                    penalty = penalty + self.penalty_weight * violation
        
        return penalty
    
    def update_epsilon(
        self,
        new_epsilon: Dict[str, float]
    ) -> None:
        """Update constraint bounds."""
        self.epsilon.update(new_epsilon)
    
    def tighten_constraints(
        self,
        factor: float = 0.9
    ) -> None:
        """
        Tighten all constraints by a factor.
        
        Args:
            factor: Multiplication factor (< 1 to tighten)
        """
        for name in self.epsilon:
            self.epsilon[name] *= factor
