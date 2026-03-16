"""
Base Fairness Constraint
========================

Abstract base class for all fairness constraints.

This module provides the foundation for implementing various fairness
paradigms including group fairness, individual fairness, and counterfactual fairness.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import torch
import torch.nn as nn
import numpy as np


class ConstraintType(Enum):
    """Enumeration of constraint types."""
    GROUP = "group"
    INDIVIDUAL = "individual"
    COUNTERFACTUAL = "counterfactual"


class RelaxationType(Enum):
    """Constraint relaxation types."""
    HARD = "hard"      # Hard constraint (exact satisfaction)
    SOFT = "soft"      # Soft constraint (penalty-based)
    BARRIER = "barrier"  # Barrier function constraint
    AUGMENTED = "augmented"  # Augmented Lagrangian


class BaseFairnessConstraint(ABC, nn.Module):
    """
    Abstract base class for fairness constraints.
    
    All fairness constraints should inherit from this class and implement
    the compute() and loss() methods.
    
    A fairness constraint defines what it means for a model or predictions
    to be "fair" with respect to protected/sensitive attributes.
    
    Key Concepts:
        - Constraint Type: Group, Individual, or Counterfactual
        - Relaxation: How strictly to enforce the constraint
        - Threshold: Maximum allowed violation
        - Aggregation: How to aggregate violations across groups/individuals
    
    Subclasses must implement:
        - compute(): Calculate the fairness metric
        - loss(): Calculate the differentiable loss for optimization
    
    Example:
        >>> class MyConstraint(BaseFairnessConstraint):
        ...     def compute(self, predictions, groups, labels=None):
        ...         # Compute fairness metric
        ...         return metric_value
        ...     
        ...     def loss(self, predictions, groups, labels=None):
        ...         # Compute differentiable loss
        ...         return loss_tensor
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        relaxation: str = "soft",
        weight: float = 1.0,
        name: Optional[str] = None
    ):
        """
        Initialize the fairness constraint.
        
        Args:
            threshold: Maximum allowed violation threshold.
                       Lower values = stricter constraints.
            relaxation: Constraint relaxation type:
                - "hard": Exact constraint satisfaction (difficult to optimize)
                - "soft": Penalty-based relaxation (default)
                - "barrier": Barrier function approach
                - "augmented": Augmented Lagrangian method
            weight: Weight for the constraint in multi-constraint settings
            name: Optional name for the constraint
        """
        super().__init__()
        
        self.threshold = threshold
        self.relaxation = RelaxationType(relaxation)
        self.weight = weight
        self.name = name or self.__class__.__name__
        
        # Constraint type to be set by subclasses
        self._constraint_type = None
        
        # Training state
        self._lambda = 1.0  # Lagrange multiplier for augmented Lagrangian
        self._rho = 1.0     # Penalty parameter for augmented Lagrangian
        
    @property
    def constraint_type(self) -> ConstraintType:
        """Return the constraint type."""
        return self._constraint_type
    
    @abstractmethod
    def compute(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute the fairness metric.
        
        This method computes a scalar metric that measures the degree
        of fairness violation. Lower values indicate more fairness.
        
        Args:
            predictions: Model predictions (batch_size,) or (batch_size, num_classes)
            groups: Group membership tensor (batch_size,)
            labels: Ground truth labels if applicable (batch_size,)
            **kwargs: Additional constraint-specific arguments
            
        Returns:
            Scalar metric value (lower is more fair)
        """
        pass
    
    @abstractmethod
    def loss(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute the constraint loss for optimization.
        
        This method computes a differentiable loss that can be used
        for gradient-based optimization.
        
        Args:
            predictions: Model predictions (batch_size,) or (batch_size, num_classes)
            groups: Group membership tensor (batch_size,)
            labels: Ground truth labels if applicable (batch_size,)
            **kwargs: Additional constraint-specific arguments
            
        Returns:
            Scalar loss tensor (to be minimized)
        """
        pass
    
    def is_satisfied(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> bool:
        """
        Check if the constraint is satisfied.
        
        Args:
            predictions: Model predictions
            groups: Group membership tensor
            labels: Ground truth labels if applicable
            **kwargs: Additional arguments
            
        Returns:
            True if constraint is satisfied (metric <= threshold)
        """
        metric = self.compute(predictions, groups, labels, **kwargs)
        return metric <= self.threshold
    
    def get_violation(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Get the constraint violation amount.
        
        Args:
            predictions: Model predictions
            groups: Group membership tensor
            labels: Ground truth labels if applicable
            
        Returns:
            Violation amount (0 if satisfied, positive otherwise)
        """
        metric = self.compute(predictions, groups, labels, **kwargs)
        return max(0, metric - self.threshold)
    
    def soft_loss(
        self,
        violation: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute soft constraint loss.
        
        Args:
            violation: Constraint violation tensor
            
        Returns:
            Soft penalty loss
        """
        return self.weight * violation.pow(2)
    
    def hard_loss(
        self,
        violation: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hard constraint loss.
        
        Args:
            violation: Constraint violation tensor
            
        Returns:
            Hinge-like penalty loss
        """
        return self.weight * torch.relu(violation)
    
    def barrier_loss(
        self,
        violation: torch.Tensor,
        t: float = 1.0
    ) -> torch.Tensor:
        """
        Compute barrier function loss.
        
        Args:
            violation: Constraint violation tensor
            t: Barrier parameter (smaller = stronger barrier)
            
        Returns:
            Barrier penalty loss
        """
        # Log barrier: -t * log(-violation) when violation < 0
        # Becomes infinite as violation approaches 0 from below
        safe_violation = violation.clamp(max=-1e-8)
        return -t * torch.log(-safe_violation)
    
    def augmented_lagrangian_loss(
        self,
        violation: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute augmented Lagrangian loss.
        
        Combines Lagrangian and quadratic penalty terms.
        
        Args:
            violation: Constraint violation tensor
            
        Returns:
            Augmented Lagrangian loss
        """
        # L(x, λ, ρ) = λ * violation + (ρ/2) * violation^2
        return self._lambda * violation + (self._rho / 2) * violation.pow(2)
    
    def update_lagrange_multiplier(
        self,
        violation: float,
        step_size: float = 0.1
    ) -> None:
        """
        Update Lagrange multiplier for augmented Lagrangian.
        
        Args:
            violation: Current constraint violation
            step_size: Step size for multiplier update
        """
        self._lambda = self._lambda + step_size * violation
        # Keep lambda non-negative for inequality constraints
        self._lambda = max(0, self._lambda)
    
    def update_penalty_parameter(
        self,
        violation: float,
        growth_factor: float = 2.0,
        max_rho: float = 1000.0
    ) -> None:
        """
        Update penalty parameter for augmented Lagrangian.
        
        Args:
            violation: Current constraint violation
            growth_factor: Factor to increase penalty
            max_rho: Maximum penalty parameter
        """
        if violation > self.threshold:
            self._rho = min(self._rho * growth_factor, max_rho)
    
    def reset_optimization_state(self) -> None:
        """Reset Lagrangian optimization state."""
        self._lambda = 1.0
        self._rho = 1.0
    
    def apply_relaxation(
        self,
        violation: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Apply the selected relaxation to the violation.
        
        Args:
            violation: Constraint violation tensor
            **kwargs: Relaxation-specific parameters
            
        Returns:
            Relaxed loss tensor
        """
        if self.relaxation == RelaxationType.SOFT:
            return self.soft_loss(violation)
        elif self.relaxation == RelaxationType.HARD:
            return self.hard_loss(violation)
        elif self.relaxation == RelaxationType.BARRIER:
            t = kwargs.get("barrier_t", 1.0)
            return self.barrier_loss(violation, t)
        elif self.relaxation == RelaxationType.AUGMENTED:
            return self.augmented_lagrangian_loss(violation)
        else:
            return self.soft_loss(violation)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert constraint to dictionary representation.
        
        Returns:
            Dictionary with constraint configuration
        """
        return {
            "name": self.name,
            "type": self._constraint_type.value if self._constraint_type else None,
            "threshold": self.threshold,
            "relaxation": self.relaxation.value,
            "weight": self.weight,
            "lambda": self._lambda,
            "rho": self._rho
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BaseFairnessConstraint":
        """
        Create constraint from dictionary configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Constraint instance
        """
        raise NotImplementedError("Subclasses must implement from_dict()")
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold}, "
            f"relaxation={self.relaxation.value}, "
            f"weight={self.weight})"
        )


class ConstraintCombiner(nn.Module):
    """
    Combines multiple fairness constraints into a single loss.
    
    Supports different combination strategies:
        - Sum: Weighted sum of all constraint losses
        - Max: Maximum violation across all constraints
        - Product: Product of constraint satisfaction probabilities
    
    Example:
        >>> combiner = ConstraintCombiner([
        ...     DemographicParity(threshold=0.05),
        ...     EqualizedOdds(threshold=0.05)
        ... ], combination="sum")
        >>> total_loss = combiner(predictions, groups, labels)
    """
    
    def __init__(
        self,
        constraints: List[BaseFairnessConstraint],
        combination: str = "sum",
        weights: Optional[List[float]] = None
    ):
        """
        Initialize constraint combiner.
        
        Args:
            constraints: List of fairness constraints
            combination: Combination strategy ("sum", "max", "product")
            weights: Optional weights for each constraint
        """
        super().__init__()
        
        self.constraints = nn.ModuleList(constraints)
        self.combination = combination
        
        if weights is not None:
            assert len(weights) == len(constraints)
            self.weights = weights
        else:
            self.weights = [1.0] * len(constraints)
    
    def forward(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined constraint loss.
        
        Args:
            predictions: Model predictions
            groups: Group membership tensor
            labels: Ground truth labels if applicable
            
        Returns:
            Tuple of (total_loss, individual_metrics)
        """
        losses = []
        metrics = {}
        
        for i, constraint in enumerate(self.constraints):
            loss = constraint.loss(predictions, groups, labels, **kwargs)
            weighted_loss = self.weights[i] * loss
            losses.append(weighted_loss)
            
            # Compute metric
            metric = constraint.compute(predictions, groups, labels, **kwargs)
            metrics[constraint.name] = metric
        
        if self.combination == "sum":
            total_loss = sum(losses)
        elif self.combination == "max":
            total_loss = max(losses)
        elif self.combination == "product":
            # Convert losses to probabilities and multiply
            probs = [torch.exp(-loss) for loss in losses]
            total_loss = -torch.log(torch.stack(probs).prod())
        else:
            total_loss = sum(losses)
        
        return total_loss, metrics
    
    def all_satisfied(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> bool:
        """Check if all constraints are satisfied."""
        for constraint in self.constraints:
            if not constraint.is_satisfied(predictions, groups, labels, **kwargs):
                return False
        return True
    
    def get_violations(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Get violation amounts for all constraints."""
        violations = {}
        for constraint in self.constraints:
            violations[constraint.name] = constraint.get_violation(
                predictions, groups, labels, **kwargs
            )
        return violations
