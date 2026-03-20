"""
Multi-Objective Optimizer
=============================

Implements optimization strategies for multiple training objectives.

This module provides:
- Gradient-based multi-objective optimization (MGDA, - Pareto descent methods
- Constraint-based optimization
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math

import torch
import torch.nn as nn
from torch.optim import Optimizer


class MGDAOptimizer:
    """
    Multiple Gradient Descent Algorithm (MGDA) Optimizer.
    
    Implements the Multi-Task Learning using MGDA for finding
    Pareto optimal solutions across all tasks.
    
    Reference:
        Sener and Koltun, "Multi-Task Learning Using MGDA"
    
    Example:
        >>> optimizer = MGDAOptimizer(
        ...     model.parameters(),
        ...     num_tasks=3,
        ...     lr=1e-4
        ... )
        >>> optimizer.step([loss1, loss2, loss3])
    """
    
    def __init__(
        self,
        params: List[nn.Parameter],
        num_tasks: int,
        lr: float = 1e-4,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        frank_wolfe: bool = False,
        eps: float = 1e-8,
    ):
        """
        Initialize MGDA optimizer.
        
        Args:
            params: Model parameters
            num_tasks: Number of tasks
            lr: Learning rate
            momentum: Momentum coefficient
            weight_decay: Weight decay (L2 regularization)
            frank_wolfe: Whether to use Frank-Wolfe direction
            eps: Small epsilon for numerical stability
        """
        self.params = list(params)
        self.num_tasks = num_tasks
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.frank_wolfe = frank_wolfe
        self.eps = eps
        
        # Initialize parameter state
        self.state = {}
        for p in self.params:
            self.state[p] = torch.zeros_like(p.data)
        
        # Store gradients for each task
        self.task_grads = []
    
    def zero_grad(self) -> None:
        """Zero gradients for all parameters."""
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def step(self, task_losses: List[torch.Tensor]) -> None:
        """
        Perform one optimization step with MGDA.
        
        Args:
            task_losses: List of task losses
        """
        if len(task_losses) != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} task losses, "
                f"got {len(task_losses)}"
            )
        
        # Compute gradients for each task
        self.task_grads = []
        for loss in task_losses:
            self.zero_grad()
            loss.backward(retain_graph=True)
            grads = []
            for p in self.params:
                if p.grad is not None:
                    grads.append(p.grad.clone())
            self.task_grads.append(grads)
        
        # Find Pareto optimal direction
        alphas = self._solve_pareto_direction()
        
        # Update parameters
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is not None:
                    # Compute combined gradient
                    combined_grad = torch.zeros_like(p.grad)
                    for task_idx in range(self.num_tasks):
                        combined_grad += alphas[task_idx] * self.task_grads[task_idx][i]
                    
                    # Apply momentum
                    self.state[p] = self.momentum * self.state[p] + combined_grad
                    
                    # Apply weight decay
                    if self.weight_decay > 0:
                        p.data.add_(self.weight_decay * p.data)
                    
                    # Update parameters
                    p.add_(self.lr * self.state[p])
    
    def _solve_pareto_direction(self) -> List[float]:
        """
        Solve for Pareto optimal gradient direction.
        
        Uses Frank-Wolfe or exact algorithm depending on configuration.
        
        Returns:
            List of task weights (alphas)
        """
        # Stack gradients for each task
        grads_stacked = []
        for task_idx in range(self.num_tasks):
            flat_grads = []
            for layer_grads in self.task_grads[task_idx]:
                flat_grads.append(layer_grads.flatten())
            grads_stacked.append(torch.cat(flat_grads))
        
        grads_matrix = torch.stack(grads_stacked)  # [num_tasks, total_params]
        num_tasks, total_params = grads_matrix.shape
        
        if self.frank_wolfe:
            return self._frank_wolfe_solve(grads_matrix)
        else:
            return self._exact_solve(grads_matrix)
    
    def _frank_wolfe_solve(
        self,
        grads_matrix: torch.Tensor
    ) -> List[float]:
        """
        Solve using Frank-Wolfe algorithm.
        
        Args:
            grads_matrix: Matrix of gradients [num_tasks, total_params]
            
        Returns:
            List of task weights
        """
        num_tasks = grads_matrix.shape[0]
        
        # Initialize with uniform weights
        alphas = [1.0 / num_tasks] * num_tasks
        
        for _ in range(100):  # Frank-Wolfe iterations
            # Compute Frank-Wolfe direction
            g = grads_matrix.t() @ grads_matrix @ alphas
            
            # Find task with steepest descent
            descent = grads_matrix @ alphas
            
            # Find best task for gradient direction
            min_idx = torch.argmax(descent).item()
            
            # Update alphas
            if descent > 0:
                # Move weight towards task with smallest gradient
                alphas[min_idx] = min(1.0, alphas[min_idx] + 0.01)
        
        # Normalize
        total = sum(alphas)
        alphas = [a / total for a in alphas]
        
        return alphas
    
    def _exact_solve(
        self,
        grads_matrix: torch.Tensor
    ) -> List[float]:
        """
        Solve exactly using QP formulation.
        
        Args:
            grads_matrix: Matrix of gradients [num_tasks, total_params]
            
        Returns:
            List of task weights
        """
        # This is a simplified exact solve
        # For full solution, use quadratic programming
        
        num_tasks = grads_matrix.shape[0]
        
        # Initialize with uniform weights
        alphas = torch.ones(num_tasks) / num_tasks
        
        # Simple iteration to find approximate solution
        for _ in range(50):
            # Compute gradient of squared norm
            norm_sq = (grads_matrix @ alphas).pow(2).sum()
            
            # Compute gradient
            grad = 2 * grads_matrix.t() @ grads_matrix @ alphas
            
            # Gradient descent step
            alphas = alphas - 0.01 * grad
            alphas = torch.clamp(alphas, 0.0, 1.0)
            alphas = alphas / alphas.sum()
        
        return alphas.tolist()


class ParetoOptimizer:
    """
    Pareto Optimizer for Multi-Objective Optimization.
    
    Implements various methods for finding Pareto optimal solutions:
    - Weighted sum scalarization
    - Tchebycheff scalarization
    - Epsilon-constraint method
    
    Example:
        >>> optimizer = ParetoOptimizer(
        ...     model.parameters(),
        ...     objectives=[obj1, obj2, obj3],
        ...     method="weighted_sum"
        ... )
        >>> optimizer.step()
    """
    
    def __init__(
        self,
        params: List[nn.Parameter],
        objectives: List[nn.Module],
        method: str = "weighted_sum",
        weights: Optional[List[float]] = None,
        lr: float = 1e-4,
    ):
        """
        Initialize Pareto optimizer.
        
        Args:
            params: Model parameters
            objectives: List of objective functions
            method: Scalarization method
            weights: Initial weights for objectives
            lr: Learning rate
        """
        self.params = list(params)
        self.objectives = objectives
        self.method = method
        self.lr = lr
        
        # Initialize weights
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [1.0 / len(objectives)] * len(objectives)
        
        # Base optimizer
        self.base_optimizer = torch.optim.Adam(params, lr=lr)
        
        # Pareto front history
        self.pareto_front = []
    
    def step(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform one optimization step.
        
        Args:
            inputs: Input tensor
            labels: Optional labels tensor
            
        Returns:
            Dictionary of objective values
        """
        # Compute all objectives
        objective_values = []
        for obj in self.objectives:
            value = obj(inputs, labels)
            objective_values.append(value)
        
        # Compute combined loss based on method
        if self.method == "weighted_sum":
            loss = sum(w * v for w, v in zip(self.weights, objective_values))
        
        elif self.method == "tchebycheff":
            # Reference point (ideal point)
            z = torch.zeros(len(self.objectives))
            loss = max(w * abs(v - z) for w, v in zip(self.weights, objective_values))
        
        elif self.method == "epsilon_constraint":
            # Optimize first objective, subject to constraints on others
            loss = objective_values[0]
            for i, (w, v) in enumerate(zip(self.weights[1:], objective_values[1:])):
                loss = loss + w * F.relu(v - 0.1)
        else:
            loss = sum(objective_values)
        
        # Backward pass
        self.base_optimizer.zero_grad()
        loss.backward()
        self.base_optimizer.step()
        
        # Record Pareto front
        self.pareto_front.append([v.item() for v in objective_values])
        
        return {
            f"objective_{i}": v.item()
            for i, v in enumerate(objective_values)
        }
    
    def get_pareto_front(self) -> List[List[float]]:
        """Get recorded Pareto front."""
        return self.pareto_front
    
    def update_weights(self, new_weights: List[float]) -> None:
        """Update objective weights."""
        if len(new_weights) != len(self.weights):
            raise ValueError("Number of weights must match objectives")
        self.weights = new_weights


class ConstraintOptimizer:
    """
    Constraint-Based Multi-Objective Optimizer.
    
    Implements constrained optimization with penalties for fairness constraints.
    """
    
    def __init__(
        self,
        params: List[nn.Parameter],
        primary_objective: nn.Module,
        constraints: List[nn.Module],
        constraint_weights: Optional[List[float]] = None,
        lr: float = 1e-4,
        penalty_factor: float = 10.0,
        penalty_update_rate: float = 1.1,
    ):
        """
        Initialize constraint optimizer.
        
        Args:
            params: Model parameters
            primary_objective: Primary objective to minimize
            constraints: List of constraint functions
            constraint_weights: Weights for constraint violations
            lr: Learning rate
            penalty_factor: Initial penalty factor
            penalty_update_rate: Rate for increasing penalty
        """
        self.params = list(params)
        self.primary_objective = primary_objective
        self.constraints = constraints
        self.constraint_weights = constraint_weights or [1.0] * len(constraints)
        self.penalty_factor = penalty_factor
        self.penalty_update_rate = penalty_update_rate
        
        # Base optimizer
        self.base_optimizer = torch.optim.Adam(params, lr=lr)
        
        # Constraint violation history
        self.violation_history = []
    
    def step(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Perform one optimization step.
        
        Args:
            inputs: Input tensor
            labels: Optional labels tensor
            
        Returns:
            Dictionary of objective values
        """
        # Compute primary objective
        primary_value = self.primary_objective(inputs, labels)
        
        # Compute constraint violations
        constraint_values = []
        for c in self.constraints:
            value = c(inputs, labels)
            constraint_values.append(value)
        
        # Compute penalty
        penalty = 0.0
        for i, (c_val, w) in enumerate(zip(constraint_values, self.constraint_weights)):
            violation = F.relu(c_val)  # Positive violation
            penalty = penalty + w * self.penalty_factor * violation
        
        # Total loss
        total_loss = primary_value + penalty
        
        # Backward Pass
        self.base_optimizer.zero_grad()
        total_loss.backward()
        self.base_optimizer.step()
        
        # Record violations
        violations = [v.item() for v in constraint_values]
        self.violation_history.append(violations)
        
        # Update penalty factor if violations persist
        if len(self.violation_history) > 10:
            recent_violations = self.violation_history[-10:]
            avg_violation = sum(recent_violations) / len(recent_violations)
            if avg_violation > 0.05:
                self.penalty_factor *= self.penalty_update_rate
        
        return {
            "primary_loss": primary_value.item(),
            "penalty": penalty.item(),
            **{f"constraint_{i}": v.item() for i, v in enumerate(constraint_values)}
        }
    
    def get_violation_history(self) -> List[List[float]]:
        """Get constraint violation history."""
        return self.violation_history

# Alias for compatibility
MultiObjectiveOptimizer = MGDAOptimizer
