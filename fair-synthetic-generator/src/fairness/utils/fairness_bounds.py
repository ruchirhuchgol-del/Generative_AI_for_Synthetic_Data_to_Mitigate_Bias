"""
Fairness Bounds Computation
===========================

Utilities for computing and managing fairness bounds and thresholds.

This module provides:
- FairnessBounds: Static fairness bounds
- AdaptiveFairnessBounds: Adaptive bounds based on data
- FairnessThresholdScheduler: Scheduled threshold adjustments
- Statistical bounds computation utilities
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import math

import torch
import torch.nn as nn
import numpy as np


class BoundType(Enum):
    """Types of fairness bounds."""
    ABSOLUTE = "absolute"        # Absolute difference threshold
    RATIO = "ratio"              # Ratio-based threshold
    PERCENTILE = "percentile"    # Percentile-based threshold
    STATISTICAL = "statistical"  # Statistical significance bound
    ADAPTIVE = "adaptive"        # Data-adaptive bound


class FairnessBounds:
    """
    Static Fairness Bounds Container.
    
    Defines acceptable fairness violation bounds for different
    fairness metrics.
    
    Example:
        >>> bounds = FairnessBounds({
        ...     "demographic_parity": 0.05,
        ...     "equalized_odds": 0.05,
        ...     "disparate_impact": (0.8, 1.25)
        ... })
        >>> 
        >>> # Check if violation is within bounds
        >>> is_fair = bounds.is_satisfied("demographic_parity", 0.03)
    """
    
    def __init__(
        self,
        bounds: Dict[str, Union[float, Tuple[float, float]]],
        bound_type: str = "absolute"
    ):
        """
        Initialize fairness bounds.
        
        Args:
            bounds: Dictionary of metric name -> threshold or (min, max) tuple
            bound_type: Type of bounds
        """
        self.bounds = {}
        self.bound_type = BoundType(bound_type)
        
        for name, value in bounds.items():
            if isinstance(value, tuple):
                self.bounds[name] = value
            else:
                # Single threshold: symmetric around 0
                self.bounds[name] = (-abs(value), abs(value))
    
    def is_satisfied(
        self,
        metric_name: str,
        value: float
    ) -> bool:
        """
        Check if metric value satisfies bounds.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            
        Returns:
            True if within bounds
        """
        if metric_name not in self.bounds:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        min_val, max_val = self.bounds[metric_name]
        return min_val <= value <= max_val
    
    def get_violation(
        self,
        metric_name: str,
        value: float
    ) -> float:
        """
        Get violation amount (0 if satisfied).
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            
        Returns:
            Violation amount
        """
        if metric_name not in self.bounds:
            raise ValueError(f"Unknown metric: {metric_name}")
        
        min_val, max_val = self.bounds[metric_name]
        
        if value < min_val:
            return min_val - value
        elif value > max_val:
            return value - max_val
        else:
            return 0.0
    
    def get_bounds(
        self,
        metric_name: str
    ) -> Tuple[float, float]:
        """Get bounds for a metric."""
        return self.bounds.get(metric_name, (float('-inf'), float('inf')))
    
    def get_threshold(
        self,
        metric_name: str
    ) -> float:
        """Get absolute threshold for a metric."""
        min_val, max_val = self.bounds.get(metric_name, (0, 0))
        return max(abs(min_val), abs(max_val))
    
    def update_bounds(
        self,
        new_bounds: Dict[str, Union[float, Tuple[float, float]]]
    ) -> None:
        """Update bounds."""
        for name, value in new_bounds.items():
            if isinstance(value, tuple):
                self.bounds[name] = value
            else:
                self.bounds[name] = (-abs(value), abs(value))
    
    def tighten(
        self,
        factor: float = 0.9
    ) -> None:
        """
        Tighten all bounds by a factor.
        
        Args:
            factor: Multiplication factor (< 1 to tighten)
        """
        for name in self.bounds:
            min_val, max_val = self.bounds[name]
            self.bounds[name] = (min_val * factor, max_val * factor)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "bounds": self.bounds,
            "bound_type": self.bound_type.value
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "FairnessBounds":
        """Create from dictionary configuration."""
        return cls(
            bounds=config["bounds"],
            bound_type=config.get("bound_type", "absolute")
        )
    
    def __repr__(self) -> str:
        return f"FairnessBounds({self.bounds})"


class AdaptiveFairnessBounds(nn.Module):
    """
    Adaptive Fairness Bounds.
    
    Computes and adjusts fairness bounds based on data distribution
    and model performance.
    
    Features:
        - Statistical significance-based bounds
        - Performance-fairness tradeoff handling
        - Automatic bound tightening/loosening
    
    Example:
        >>> bounds = AdaptiveFairnessBounds(
        ...     base_threshold=0.05,
        ...     adaptation_rate=0.1,
        ...     min_threshold=0.01
        ... )
        >>> 
        >>> # Update bounds based on data
        >>> bounds.update(predictions, groups, labels)
        >>> 
        >>> # Get current threshold
        >>> threshold = bounds.get_threshold("demographic_parity")
    """
    
    def __init__(
        self,
        base_threshold: float = 0.05,
        adaptation_rate: float = 0.1,
        min_threshold: float = 0.01,
        max_threshold: float = 0.2,
        confidence_level: float = 0.95,
        use_statistical_bounds: bool = True,
        name: str = "adaptive_fairness_bounds"
    ):
        """
        Initialize adaptive bounds.
        
        Args:
            base_threshold: Starting threshold
            adaptation_rate: Rate of adaptation (0-1)
            min_threshold: Minimum allowed threshold
            max_threshold: Maximum allowed threshold
            confidence_level: Confidence level for statistical bounds
            use_statistical_bounds: Whether to use statistical bounds
            name: Bounds name
        """
        super().__init__()
        
        self.base_threshold = base_threshold
        self.adaptation_rate = adaptation_rate
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.confidence_level = confidence_level
        self.use_statistical_bounds = use_statistical_bounds
        self.name = name
        
        # Current thresholds per metric
        self._thresholds = {}
        
        # History for adaptation
        self._violation_history = []
        
        # Statistical bounds cache
        self._statistical_bounds = {}
    
    def get_threshold(
        self,
        metric_name: str
    ) -> float:
        """Get current threshold for a metric."""
        if metric_name not in self._thresholds:
            self._thresholds[metric_name] = self.base_threshold
        return self._thresholds[metric_name]
    
    def get_bounds(
        self,
        metric_name: str
    ) -> Tuple[float, float]:
        """Get bounds for a metric."""
        threshold = self.get_threshold(metric_name)
        return (-threshold, threshold)
    
    def update(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        violations: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Update bounds based on current state.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            labels: Ground truth labels
            violations: Optional pre-computed violations
        """
        if violations is None:
            violations = self._compute_violations(predictions, groups, labels)
        
        self._violation_history.append(violations)
        
        # Adapt thresholds
        for metric_name, violation in violations.items():
            current = self.get_threshold(metric_name)
            
            # Statistical bounds adjustment
            if self.use_statistical_bounds:
                stat_bound = self._compute_statistical_bound(
                    predictions, groups, metric_name
                )
                current = max(current, stat_bound)
            
            # Adapt based on violation history
            if len(self._violation_history) >= 5:
                recent_violations = [
                    h.get(metric_name, 0) 
                    for h in self._violation_history[-5:]
                ]
                avg_violation = np.mean(recent_violations)
                
                if avg_violation > current:
                    # Violations too high, loosen threshold
                    new_threshold = min(
                        current * (1 + self.adaptation_rate),
                        self.max_threshold
                    )
                else:
                    # Violations acceptable, tighten threshold
                    new_threshold = max(
                        current * (1 - self.adaptation_rate),
                        self.min_threshold
                    )
                
                self._thresholds[metric_name] = new_threshold
    
    def _compute_violations(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute current violations."""
        violations = {}
        
        # Demographic parity violation
        unique_groups = torch.unique(groups)
        rates = []
        
        for g in unique_groups:
            mask = (groups == g)
            if mask.sum() > 0:
                rate = predictions[mask].mean().item()
                rates.append(rate)
        
        if len(rates) >= 2:
            violations["demographic_parity"] = max(rates) - min(rates)
        
        # Equalized odds violation
        if labels is not None:
            tprs = []
            fprs = []
            
            for g in unique_groups:
                mask = (groups == g)
                pos_mask = mask & (labels == 1)
                neg_mask = mask & (labels == 0)
                
                if pos_mask.sum() > 0:
                    tpr = predictions[pos_mask].mean().item()
                    tprs.append(tpr)
                
                if neg_mask.sum() > 0:
                    fpr = predictions[neg_mask].mean().item()
                    fprs.append(fpr)
            
            if len(tprs) >= 2:
                violations["equalized_odds_tpr"] = max(tprs) - min(tprs)
            
            if len(fprs) >= 2:
                violations["equalized_odds_fpr"] = max(fprs) - min(fprs)
        
        return violations
    
    def _compute_statistical_bound(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        metric_name: str
    ) -> float:
        """
        Compute statistically significant bound.
        
        Uses bootstrap or analytical methods to determine
        bounds that are statistically significant.
        """
        if metric_name == "demographic_parity":
            return self._dp_statistical_bound(predictions, groups)
        else:
            return self.base_threshold
    
    def _dp_statistical_bound(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor
    ) -> float:
        """Statistical bound for demographic parity."""
        unique_groups = torch.unique(groups)
        
        # Compute group sizes and variances
        group_stats = []
        
        for g in unique_groups:
            mask = (groups == g)
            n = mask.sum().item()
            
            if n > 0:
                preds = predictions[mask]
                mean = preds.mean().item()
                var = preds.var().item() if n > 1 else 0
                group_stats.append((n, mean, var))
        
        if len(group_stats) < 2:
            return self.base_threshold
        
        # Compute standard error of difference
        # SE = sqrt(var1/n1 + var2/n2)
        n1, mean1, var1 = group_stats[0]
        n2, mean2, var2 = group_stats[1]
        
        se = np.sqrt(var1/n1 + var2/n2)
        
        # Confidence interval
        z = 1.96  # 95% CI
        ci = z * se
        
        return ci
    
    def reset(self) -> None:
        """Reset to base thresholds."""
        self._thresholds = {}
        self._violation_history = []
        self._statistical_bounds = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "base_threshold": self.base_threshold,
            "adaptation_rate": self.adaptation_rate,
            "min_threshold": self.min_threshold,
            "max_threshold": self.max_threshold,
            "current_thresholds": self._thresholds.copy()
        }


class FairnessThresholdScheduler:
    """
    Scheduler for fairness thresholds during training.
    
    Implements various scheduling strategies:
        - Linear warmup/decay
        - Cosine annealing
        - Step-wise adjustment
        - Performance-based adjustment
    
    Example:
        >>> scheduler = FairnessThresholdScheduler(
        ...     initial_threshold=0.1,
        ...     target_threshold=0.01,
        ...     schedule_type="cosine",
        ...     warmup_epochs=50
        ... )
        >>> 
        >>> for epoch in range(num_epochs):
        ...     threshold = scheduler.get_threshold(epoch)
        ...     # Use threshold in fairness constraint
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.1,
        target_threshold: float = 0.01,
        schedule_type: str = "linear",
        warmup_epochs: int = 50,
        cooldown_epochs: int = 0,
        min_threshold: float = 0.001,
        max_threshold: float = 0.5,
        performance_adjustment: bool = False,
        adjustment_rate: float = 0.1
    ):
        """
        Initialize threshold scheduler.
        
        Args:
            initial_threshold: Starting threshold
            target_threshold: Final target threshold
            schedule_type: Schedule type ("linear", "cosine", "step", "exponential")
            warmup_epochs: Epochs to reach target
            cooldown_epochs: Epochs at target before cooldown
            min_threshold: Minimum threshold
            max_threshold: Maximum threshold
            performance_adjustment: Whether to adjust based on performance
            adjustment_rate: Rate of performance-based adjustment
        """
        self.initial_threshold = initial_threshold
        self.target_threshold = target_threshold
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        self.cooldown_epochs = cooldown_epochs
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.performance_adjustment = performance_adjustment
        self.adjustment_rate = adjustment_rate
        
        # Performance history
        self._performance_history = []
        self._current_adjustment = 0.0
    
    def get_threshold(
        self,
        epoch: int,
        performance: Optional[float] = None
    ) -> float:
        """
        Get threshold for current epoch.
        
        Args:
            epoch: Current epoch number
            performance: Optional performance metric for adjustment
            
        Returns:
            Threshold value
        """
        base_threshold = self._compute_base_threshold(epoch)
        
        # Performance-based adjustment
        if self.performance_adjustment and performance is not None:
            self._performance_history.append(performance)
            adjustment = self._compute_performance_adjustment()
            base_threshold = base_threshold * (1 + adjustment)
        
        return max(self.min_threshold, min(self.max_threshold, base_threshold))
    
    def _compute_base_threshold(self, epoch: int) -> float:
        """Compute base threshold from schedule."""
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            
            if self.schedule_type == "linear":
                threshold = self.initial_threshold + progress * \
                           (self.target_threshold - self.initial_threshold)
            
            elif self.schedule_type == "cosine":
                cos_progress = (1 - math.cos(math.pi * progress)) / 2
                threshold = self.initial_threshold + cos_progress * \
                           (self.target_threshold - self.initial_threshold)
            
            elif self.schedule_type == "exponential":
                decay_rate = (self.target_threshold / self.initial_threshold) ** \
                            (1 / self.warmup_epochs)
                threshold = self.initial_threshold * (decay_rate ** epoch)
            
            elif self.schedule_type == "step":
                num_steps = 5  # Number of steps
                step_size = self.warmup_epochs // num_steps
                step = epoch // step_size
                threshold = self.initial_threshold - step * \
                           (self.initial_threshold - self.target_threshold) / num_steps
            
            else:
                threshold = self.initial_threshold + progress * \
                           (self.target_threshold - self.initial_threshold)
        
        elif epoch < self.warmup_epochs + self.cooldown_epochs:
            threshold = self.target_threshold
        
        else:
            # Cooldown phase - gradually increase threshold
            cooldown_progress = (epoch - self.warmup_epochs - self.cooldown_epochs) / \
                               max(1, self.warmup_epochs)
            cooldown_progress = min(1, cooldown_progress)
            
            threshold = self.target_threshold + cooldown_progress * \
                       (self.initial_threshold - self.target_threshold)
        
        return threshold
    
    def _compute_performance_adjustment(self) -> float:
        """Compute adjustment based on performance history."""
        if len(self._performance_history) < 5:
            return 0.0
        
        recent = self._performance_history[-5:]
        
        # Check if performance is declining
        if len(recent) >= 2:
            trend = recent[-1] - recent[0]
            
            if trend < -0.05:  # Performance declining
                # Loosen constraints (increase threshold)
                self._current_adjustment = min(
                    self._current_adjustment + self.adjustment_rate,
                    0.5  # Max 50% increase
                )
            elif trend > 0.02:  # Performance improving
                # Can tighten constraints
                self._current_adjustment = max(
                    self._current_adjustment - self.adjustment_rate,
                    -0.3  # Max 30% decrease
                )
        
        return self._current_adjustment
    
    def reset(self) -> None:
        """Reset scheduler state."""
        self._performance_history = []
        self._current_adjustment = 0.0


def compute_fairness_bounds(
    predictions: torch.Tensor,
    groups: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    confidence_level: float = 0.95,
    method: str = "bootstrap"
) -> Dict[str, Tuple[float, float]]:
    """
    Compute fairness bounds from data.
    
    Args:
        predictions: Model predictions
        groups: Group membership
        labels: Ground truth labels
        confidence_level: Confidence level for bounds
        method: Computation method ("bootstrap", "analytical")
        
    Returns:
        Dictionary of metric name -> (lower_bound, upper_bound)
    """
    bounds = {}
    
    if method == "bootstrap":
        bounds.update(_bootstrap_fairness_bounds(
            predictions, groups, labels, confidence_level
        ))
    else:
        bounds.update(_analytical_fairness_bounds(
            predictions, groups, labels, confidence_level
        ))
    
    return bounds


def compute_statistical_bounds(
    predictions: torch.Tensor,
    groups: torch.Tensor,
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    Compute statistical bounds for group rate differences.
    
    Uses normal approximation for the difference of proportions.
    
    Args:
        predictions: Model predictions
        groups: Group membership
        confidence_level: Confidence level
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    unique_groups = torch.unique(groups)
    
    if len(unique_groups) != 2:
        # Return wide bounds for more than 2 groups
        return (-0.5, 0.5)
    
    # Group 0
    mask0 = (groups == unique_groups[0])
    n0 = mask0.sum().item()
    p0 = predictions[mask0].mean().item()
    var0 = predictions[mask0].var().item() if n0 > 1 else 0
    
    # Group 1
    mask1 = (groups == unique_groups[1])
    n1 = mask1.sum().item()
    p1 = predictions[mask1].mean().item()
    var1 = predictions[mask1].var().item() if n1 > 1 else 0
    
    # Standard error of difference
    se = np.sqrt(var0/n0 + var1/n1)
    
    # Z-score for confidence level
    from scipy import stats
    z = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Difference and bounds
    diff = p1 - p0
    margin = z * se
    
    return (diff - margin, diff + margin)


def _bootstrap_fairness_bounds(
    predictions: torch.Tensor,
    groups: torch.Tensor,
    labels: Optional[torch.Tensor],
    confidence_level: float,
    n_bootstrap: int = 1000
) -> Dict[str, Tuple[float, float]]:
    """Compute bootstrap-based fairness bounds."""
    bounds = {}
    n = len(predictions)
    
    # Bootstrap for demographic parity
    dp_diffs = []
    
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        pred_sample = predictions[idx]
        group_sample = groups[idx]
        
        unique = torch.unique(group_sample)
        rates = []
        for g in unique:
            mask = (group_sample == g)
            if mask.sum() > 0:
                rates.append(pred_sample[mask].mean().item())
        
        if len(rates) >= 2:
            dp_diffs.append(max(rates) - min(rates))
    
    if dp_diffs:
        lower = np.percentile(dp_diffs, (1 - confidence_level) / 2 * 100)
        upper = np.percentile(dp_diffs, (1 + confidence_level) / 2 * 100)
        bounds["demographic_parity"] = (lower, upper)
    
    return bounds


def _analytical_fairness_bounds(
    predictions: torch.Tensor,
    groups: torch.Tensor,
    labels: Optional[torch.Tensor],
    confidence_level: float
) -> Dict[str, Tuple[float, float]]:
    """Compute analytical fairness bounds."""
    bounds = {}
    
    # Demographic parity bounds
    dp_bounds = compute_statistical_bounds(predictions, groups, confidence_level)
    bounds["demographic_parity"] = dp_bounds
    
    return bounds
