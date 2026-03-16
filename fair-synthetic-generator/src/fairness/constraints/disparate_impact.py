"""
Disparate Impact Constraint
===========================

Implementation of disparate impact fairness constraint.

Disparate impact focuses on the ratio of positive outcomes between groups:
    0.8 ≤ P(Ŷ=1|A=unprivileged) / P(Ŷ=1|A=privileged) ≤ 1.25

Based on the "80% rule" from US employment law, which states that the selection
rate for a protected group should be at least 80% of the selection rate for
the group with the highest rate.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.fairness.constraints.base_constraint import (
    BaseFairnessConstraint,
    ConstraintType,
    RelaxationType
)


class DisparateImpact(BaseFairnessConstraint):
    """
    Disparate Impact constraint.
    
    Ensures that the ratio of positive predictions between groups
    is within acceptable bounds:
        min_ratio ≤ P(Ŷ=1|A=unprivileged) / P(Ŷ=1|A=privileged) ≤ max_ratio
    
    The "80% rule" from US employment law suggests:
        - min_ratio = 0.8 (protected group should have at least 80% selection rate)
        - max_ratio = 1.25 (inverse of 0.8)
    
    Mathematical Definition:
        Disparate Impact Ratio = P(Ŷ=1|A=unprivileged) / P(Ŷ=1|A=privileged)
    
    Properties:
        - Legal grounding in employment discrimination law
        - Focuses on outcomes rather than process
        - Does not require labeled data
        - Simple to compute and interpret
    
    Example:
        >>> constraint = DisparateImpact(min_ratio=0.8, max_ratio=1.25)
        >>> 
        >>> # Check if satisfied
        >>> ratio = constraint.compute(predictions, groups, privileged_group=0)
        >>> satisfied = constraint.is_satisfied(predictions, groups)
    """
    
    def __init__(
        self,
        min_ratio: float = 0.8,
        max_ratio: float = 1.25,
        relaxation: str = "soft",
        privileged_group: int = 0,
        weight: float = 1.0,
        use_logits: bool = True,
        temperature: float = 1.0,
        name: str = "disparate_impact"
    ):
        """
        Initialize disparate impact constraint.
        
        Args:
            min_ratio: Minimum acceptable ratio (default: 0.8 for 80% rule)
            max_ratio: Maximum acceptable ratio (default: 1.25)
            relaxation: Constraint relaxation type
            privileged_group: Index of the privileged group
            weight: Constraint weight
            use_logits: Whether inputs are logits
            temperature: Temperature for softmax
            name: Constraint name
        """
        # Use min_ratio as threshold for base class
        super().__init__(min_ratio, relaxation, weight, name)
        
        self._constraint_type = ConstraintType.GROUP
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.privileged_group = privileged_group
        self.use_logits = use_logits
        self.temperature = temperature
    
    def _get_probabilities(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """Convert predictions to probabilities."""
        if self.use_logits:
            if predictions.dim() == 1:
                return torch.sigmoid(predictions / self.temperature)
            else:
                return F.softmax(predictions / self.temperature, dim=-1)[..., 1]
        else:
            return predictions.clamp(0, 1)
    
    def compute(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        privileged_group: Optional[int] = None,
        **kwargs
    ) -> float:
        """
        Compute disparate impact ratio.
        
        Args:
            predictions: Model predictions (batch_size,)
            groups: Group indices (batch_size,)
            labels: Ignored for disparate impact
            privileged_group: Override for privileged group index
            
        Returns:
            Disparate impact ratio (should be between min_ratio and max_ratio)
        """
        if privileged_group is None:
            privileged_group = self.privileged_group
        
        probs = self._get_probabilities(predictions)
        
        privileged_mask = (groups == privileged_group)
        unprivileged_mask = (groups != privileged_group)
        
        if privileged_mask.sum() == 0 or unprivileged_mask.sum() == 0:
            return 1.0  # No disparate impact if only one group
        
        privileged_rate = probs[privileged_mask].mean().item()
        unprivileged_rate = probs[unprivileged_mask].mean().item()
        
        # Handle edge cases
        if privileged_rate < 1e-10:
            if unprivileged_rate < 1e-10:
                return 1.0  # Both groups have zero rate
            else:
                return float('inf')  # Privileged has zero rate
        
        return unprivileged_rate / privileged_rate
    
    def compute_ratio(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        **kwargs
    ) -> float:
        """
        Compute disparate impact ratio (alias for compute).
        
        Returns:
            Ratio of positive rates (unprivileged/privileged)
        """
        return self.compute(predictions, groups, **kwargs)
    
    def loss(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        privileged_group: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute disparate impact loss.
        
        Penalizes ratios outside the acceptable range.
        
        Args:
            predictions: Model predictions
            groups: Group indices
            labels: Ignored
            privileged_group: Override for privileged group index
            
        Returns:
            Loss tensor
        """
        if privileged_group is None:
            privileged_group = self.privileged_group
        
        probs = self._get_probabilities(predictions)
        
        privileged_mask = (groups == privileged_group)
        unprivileged_mask = (groups != privileged_group)
        
        if privileged_mask.sum() == 0 or unprivileged_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        privileged_rate = probs[privileged_mask].mean()
        unprivileged_rate = probs[unprivileged_mask].mean()
        
        # Add small epsilon to prevent division by zero
        eps = 1e-7
        privileged_rate = torch.clamp(privileged_rate, min=eps)
        
        ratio = unprivileged_rate / privileged_rate
        
        # Penalize if ratio is outside acceptable range
        violation = torch.tensor(0.0, device=predictions.device)
        
        if ratio < self.min_ratio:
            violation = (self.min_ratio - ratio) ** 2
        elif ratio > self.max_ratio:
            violation = (ratio - self.max_ratio) ** 2
        
        return self.apply_relaxation(violation, **kwargs)
    
    def is_satisfied(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        privileged_group: Optional[int] = None,
        **kwargs
    ) -> bool:
        """
        Check if disparate impact constraint is satisfied.
        
        Args:
            predictions: Model predictions
            groups: Group indices
            labels: Ignored
            privileged_group: Override for privileged group index
            
        Returns:
            True if ratio is within acceptable bounds
        """
        ratio = self.compute(predictions, groups, labels, privileged_group, **kwargs)
        
        if ratio == float('inf') or ratio == float('-inf'):
            return False
        
        return self.min_ratio <= ratio <= self.max_ratio
    
    def get_selection_rates(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        privileged_group: Optional[int] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Get selection rates for privileged and unprivileged groups.
        
        Args:
            predictions: Model predictions
            groups: Group indices
            privileged_group: Override for privileged group index
            
        Returns:
            Dictionary with selection rates
        """
        if privileged_group is None:
            privileged_group = self.privileged_group
        
        probs = self._get_probabilities(predictions)
        
        privileged_mask = (groups == privileged_group)
        unprivileged_mask = (groups != privileged_group)
        
        rates = {}
        
        if privileged_mask.sum() > 0:
            rates["privileged_rate"] = probs[privileged_mask].mean().item()
        
        if unprivileged_mask.sum() > 0:
            rates["unprivileged_rate"] = probs[unprivileged_mask].mean().item()
        
        rates["ratio"] = self.compute_ratio(predictions, groups, privileged_group)
        
        return rates
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        d = super().to_dict()
        d.update({
            "min_ratio": self.min_ratio,
            "max_ratio": self.max_ratio,
            "privileged_group": self.privileged_group,
            "use_logits": self.use_logits,
            "temperature": self.temperature
        })
        return d
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DisparateImpact":
        """Create from dictionary configuration."""
        return cls(
            min_ratio=config.get("min_ratio", 0.8),
            max_ratio=config.get("max_ratio", 1.25),
            relaxation=config.get("relaxation", "soft"),
            privileged_group=config.get("privileged_group", 0),
            weight=config.get("weight", 1.0),
            use_logits=config.get("use_logits", True),
            temperature=config.get("temperature", 1.0),
            name=config.get("name", "disparate_impact")
        )
    
    def __repr__(self) -> str:
        return (
            f"DisparateImpact("
            f"min_ratio={self.min_ratio}, "
            f"max_ratio={self.max_ratio}, "
            f"privileged_group={self.privileged_group})"
        )


class FourFifthsRule(DisparateImpact):
    """
    Four-Fifths Rule constraint.
    
    A specific case of disparate impact that uses the exact "80% rule"
    from US employment law:
        P(Ŷ=1|A=unprivileged) ≥ 0.8 * P(Ŷ=1|A=privileged)
    
    This is a legally recognized standard for assessing adverse impact
    in employment selection procedures.
    
    Example:
        >>> constraint = FourFifthsRule()
        >>> satisfied = constraint.is_satisfied(predictions, groups)
    """
    
    def __init__(
        self,
        relaxation: str = "soft",
        privileged_group: int = 0,
        **kwargs
    ):
        """
        Initialize Four-Fifths Rule constraint.
        
        Args:
            relaxation: Constraint relaxation type
            privileged_group: Index of the privileged group
        """
        super().__init__(
            min_ratio=0.8,
            max_ratio=1.25,
            relaxation=relaxation,
            privileged_group=privileged_group,
            name="four_fifths_rule",
            **kwargs
        )


class StatisticalParityDifference(BaseFairnessConstraint):
    """
    Statistical Parity Difference constraint.
    
    Measures the difference in positive prediction rates:
        SPD = P(Ŷ=1|A=unprivileged) - P(Ŷ=1|A=privileged)
    
    Values:
        - SPD = 0: Perfect fairness
        - SPD < 0: Unprivileged group disadvantaged
        - SPD > 0: Privileged group disadvantaged
    
    Example:
        >>> constraint = StatisticalParityDifference(threshold=0.1)
        >>> spd = constraint.compute(predictions, groups)
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        relaxation: str = "soft",
        privileged_group: int = 0,
        use_logits: bool = True,
        temperature: float = 1.0,
        name: str = "statistical_parity_difference"
    ):
        """
        Initialize statistical parity difference constraint.
        
        Args:
            threshold: Maximum allowed absolute difference
            relaxation: Constraint relaxation type
            privileged_group: Index of the privileged group
            use_logits: Whether inputs are logits
            temperature: Temperature for softmax
            name: Constraint name
        """
        super().__init__(threshold, relaxation, name=name)
        
        self._constraint_type = ConstraintType.GROUP
        self.privileged_group = privileged_group
        self.use_logits = use_logits
        self.temperature = temperature
    
    def _get_probabilities(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """Convert predictions to probabilities."""
        if self.use_logits:
            if predictions.dim() == 1:
                return torch.sigmoid(predictions / self.temperature)
            else:
                return F.softmax(predictions / self.temperature, dim=-1)[..., 1]
        else:
            return predictions.clamp(0, 1)
    
    def compute(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        privileged_group: Optional[int] = None,
        **kwargs
    ) -> float:
        """
        Compute statistical parity difference.
        
        Returns:
            SPD value (should be close to 0)
        """
        if privileged_group is None:
            privileged_group = self.privileged_group
        
        probs = self._get_probabilities(predictions)
        
        privileged_mask = (groups == privileged_group)
        unprivileged_mask = (groups != privileged_group)
        
        if privileged_mask.sum() == 0 or unprivileged_mask.sum() == 0:
            return 0.0
        
        unprivileged_rate = probs[unprivileged_mask].mean().item()
        privileged_rate = probs[privileged_mask].mean().item()
        
        return unprivileged_rate - privileged_rate
    
    def loss(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        privileged_group: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute SPD loss."""
        if privileged_group is None:
            privileged_group = self.privileged_group
        
        probs = self._get_probabilities(predictions)
        
        privileged_mask = (groups == privileged_group)
        unprivileged_mask = (groups != privileged_group)
        
        if privileged_mask.sum() == 0 or unprivileged_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        unprivileged_rate = probs[unprivileged_mask].mean()
        privileged_rate = probs[privileged_mask].mean()
        
        # Absolute difference from zero
        violation = (unprivileged_rate - privileged_rate).abs()
        
        return self.apply_relaxation(violation, **kwargs)
    
    def is_satisfied(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        privileged_group: Optional[int] = None,
        **kwargs
    ) -> bool:
        """Check if SPD is within threshold."""
        spd = self.compute(predictions, groups, labels, privileged_group, **kwargs)
        return abs(spd) <= self.threshold


class CalibrationDifference(BaseFairnessConstraint):
    """
    Calibration Difference constraint.
    
    Measures whether predictions are calibrated equally across groups:
        P(Y=1|Ŷ=p, A=0) = P(Y=1|Ŷ=p, A=1) for all p
    
    This ensures that when the model predicts probability p,
    the actual positive rate is p regardless of group.
    
    Example:
        >>> constraint = CalibrationDifference(threshold=0.05)
        >>> cal_diff = constraint.compute(predictions, groups, labels)
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        relaxation: str = "soft",
        num_bins: int = 10,
        name: str = "calibration_difference"
    ):
        """
        Initialize calibration difference constraint.
        
        Args:
            threshold: Maximum allowed calibration difference
            relaxation: Constraint relaxation type
            num_bins: Number of bins for computing calibration
            name: Constraint name
        """
        super().__init__(threshold, relaxation, name=name)
        
        self._constraint_type = ConstraintType.GROUP
        self.num_bins = num_bins
    
    def compute(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> float:
        """
        Compute maximum calibration difference across groups.
        
        Args:
            predictions: Predicted probabilities (batch_size,)
            groups: Group indices (batch_size,)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            Maximum calibration difference
        """
        probs = predictions if predictions.max() <= 1 else torch.sigmoid(predictions)
        
        unique_groups = torch.unique(groups)
        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=probs.device)
        
        max_diff = 0.0
        
        for bin_idx in range(self.num_bins):
            bin_mask = (probs >= bin_edges[bin_idx]) & (probs < bin_edges[bin_idx + 1])
            
            if bin_mask.sum() == 0:
                continue
            
            group_calibrations = {}
            
            for g in unique_groups:
                mask = bin_mask & (groups == g)
                if mask.sum() > 0:
                    # Calibration: actual positive rate in this bin
                    calibration = labels[mask].float().mean().item()
                    group_calibrations[g.item()] = calibration
            
            if len(group_calibrations) >= 2:
                diff = max(group_calibrations.values()) - min(group_calibrations.values())
                max_diff = max(max_diff, diff)
        
        return max_diff
    
    def loss(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute calibration loss."""
        # Use expected calibration error approach
        probs = predictions if predictions.max() <= 1 else torch.sigmoid(predictions)
        
        unique_groups = torch.unique(groups)
        bin_edges = torch.linspace(0, 1, self.num_bins + 1, device=probs.device)
        
        total_loss = torch.tensor(0.0, device=predictions.device)
        count = 0
        
        for bin_idx in range(self.num_bins):
            bin_mask = (probs >= bin_edges[bin_idx]) & (probs < bin_edges[bin_idx + 1])
            
            if bin_mask.sum() == 0:
                continue
            
            bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            
            group_errors = []
            
            for g in unique_groups:
                mask = bin_mask & (groups == g)
                if mask.sum() > 0:
                    actual_rate = labels[mask].float().mean()
                    error = (actual_rate - bin_center).abs()
                    group_errors.append(error)
            
            if len(group_errors) >= 2:
                errors = torch.stack(group_errors)
                # Variance of calibration errors
                total_loss = total_loss + torch.var(errors)
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return self.apply_relaxation(total_loss, **kwargs)
