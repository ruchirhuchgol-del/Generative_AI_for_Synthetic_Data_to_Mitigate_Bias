"""
Demographic Parity Constraint
=============================

Implementation of demographic parity (statistical parity) fairness constraint.

Demographic parity requires that the prediction rate be equal across all groups:
    P(Ŷ=1|A=0) = P(Ŷ=1|A=1)

Also known as:
- Statistical Parity
- Group Fairness
- Equal Acceptance Rate
- Fairness Through Unawareness (related concept)
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


class AggregationType(Enum):
    """Aggregation types for multi-group settings."""
    MAX = "max"          # Maximum difference between any two groups
    MEAN = "mean"        # Mean of all pairwise differences
    STD = "std"          # Standard deviation of group rates
    PAIRWISE = "pairwise"  # All pairwise differences


class DemographicParity(BaseFairnessConstraint):
    """
    Demographic Parity constraint.
    
    Ensures that the prediction rate is equal across groups:
        P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
    
    Mathematical Definition:
        Demographic Parity Difference = max |P(Ŷ=1|A=a) - P(Ŷ=1|A=b)|
        
    For continuous predictions, we can use:
        E[Ŷ|A=a] = E[Ŷ|A=b]
    
    Properties:
        - Interventions: Requires changing the decision threshold per group
        - Trade-offs: May reduce accuracy for well-calibrated predictions
        - Use Cases: Hiring, lending, admissions
    
    Example:
        >>> constraint = DemographicParity(threshold=0.05)
        >>> 
        >>> # Binary predictions
        >>> diff = constraint.compute(predictions, groups)
        >>> 
        >>> # Training loss
        >>> loss = constraint.loss(logits, groups)
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        relaxation: str = "soft",
        aggregation: str = "max",
        weight: float = 1.0,
        use_logits: bool = True,
        temperature: float = 1.0,
        name: str = "demographic_parity"
    ):
        """
        Initialize demographic parity constraint.
        
        Args:
            threshold: Maximum allowed difference in prediction rates
            relaxation: Constraint relaxation type ("soft", "hard", "barrier")
            aggregation: How to aggregate across multiple groups:
                - "max": Maximum difference between any two groups
                - "mean": Mean of pairwise differences
                - "std": Standard deviation of rates
                - "pairwise": All pairwise differences (for logging)
            weight: Constraint weight in multi-constraint settings
            use_logits: Whether inputs are logits (vs probabilities)
            temperature: Temperature for softmax if using logits
            name: Constraint name
        """
        super().__init__(threshold, relaxation, weight, name)
        
        self._constraint_type = ConstraintType.GROUP
        self.aggregation = AggregationType(aggregation)
        self.use_logits = use_logits
        self.temperature = temperature
        
    def _get_probabilities(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert predictions to probabilities.
        
        Args:
            predictions: Raw predictions or logits
            
        Returns:
            Probability tensor
        """
        if self.use_logits:
            if predictions.dim() == 1:
                # Binary classification with single output
                return torch.sigmoid(predictions / self.temperature)
            else:
                # Multi-class: take positive class probability
                return F.softmax(predictions / self.temperature, dim=-1)[..., 1]
        else:
            # Already probabilities
            return predictions.clamp(0, 1)
    
    def compute(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute demographic parity difference.
        
        Args:
            predictions: Model predictions (batch_size,)
            groups: Group indices (batch_size,)
            labels: Ignored for demographic parity
            
        Returns:
            Demographic parity difference (lower is more fair)
        """
        probs = self._get_probabilities(predictions)
        
        unique_groups = torch.unique(groups)
        group_rates = {}
        
        for g in unique_groups:
            mask = (groups == g)
            if mask.sum() > 0:
                rate = probs[mask].mean().item()
                group_rates[g.item()] = rate
        
        if len(group_rates) < 2:
            return 0.0
        
        rates = list(group_rates.values())
        
        if self.aggregation == AggregationType.MAX:
            return float(max(rates) - min(rates))
        elif self.aggregation == AggregationType.MEAN:
            # Mean of all pairwise differences
            diffs = []
            for i, r1 in enumerate(rates):
                for r2 in rates[i+1:]:
                    diffs.append(abs(r1 - r2))
            return float(np.mean(diffs))
        elif self.aggregation == AggregationType.STD:
            return float(np.std(rates))
        else:
            return float(max(rates) - min(rates))
    
    def loss(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute demographic parity loss.
        
        Uses differentiable aggregation for gradient-based optimization.
        
        Args:
            predictions: Model predictions (batch_size,)
            groups: Group indices (batch_size,)
            labels: Ignored for demographic parity
            
        Returns:
            Loss tensor to minimize
        """
        probs = self._get_probabilities(predictions)
        
        unique_groups = torch.unique(groups)
        group_rates = []
        group_counts = []
        
        for g in unique_groups:
            mask = (groups == g)
            count = mask.sum().float()
            if count > 0:
                rate = probs[mask].mean()
                group_rates.append(rate)
                group_counts.append(count)
        
        if len(group_rates) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        rates = torch.stack(group_rates)
        counts = torch.stack(group_counts)
        
        # Compute violation based on aggregation
        if self.aggregation == AggregationType.MAX:
            # Approximate max difference using soft-max
            violation = rates.max() - rates.min()
        elif self.aggregation == AggregationType.STD:
            # Variance-based soft constraint
            violation = torch.var(rates)
        else:
            # Default: variance
            violation = torch.var(rates)
        
        # Apply relaxation
        return self.apply_relaxation(violation, **kwargs)
    
    def get_group_rates(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor
    ) -> Dict[int, float]:
        """
        Get prediction rates per group.
        
        Args:
            predictions: Model predictions
            groups: Group indices
            
        Returns:
            Dictionary mapping group ID to prediction rate
        """
        probs = self._get_probabilities(predictions)
        
        unique_groups = torch.unique(groups)
        group_rates = {}
        
        for g in unique_groups:
            mask = (groups == g)
            if mask.sum() > 0:
                rate = probs[mask].mean().item()
                group_rates[g.item()] = rate
        
        return group_rates
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        d = super().to_dict()
        d.update({
            "aggregation": self.aggregation.value,
            "use_logits": self.use_logits,
            "temperature": self.temperature
        })
        return d
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "DemographicParity":
        """Create from dictionary configuration."""
        return cls(
            threshold=config.get("threshold", 0.05),
            relaxation=config.get("relaxation", "soft"),
            aggregation=config.get("aggregation", "max"),
            weight=config.get("weight", 1.0),
            use_logits=config.get("use_logits", True),
            temperature=config.get("temperature", 1.0),
            name=config.get("name", "demographic_parity")
        )
    
    def __repr__(self) -> str:
        return (
            f"DemographicParity("
            f"threshold={self.threshold}, "
            f"aggregation={self.aggregation.value}, "
            f"relaxation={self.relaxation.value})"
        )


class ConditionalDemographicParity(DemographicParity):
    """
    Conditional Demographic Parity constraint.
    
    Extends demographic parity to allow for legitimate features:
        P(Ŷ=1|A=a, L=l) = P(Ŷ=1|A=b, L=l)
    
    where L is a set of legitimate (non-sensitive) features.
    
    This relaxes demographic parity when there are legitimate reasons
    for different prediction rates across groups.
    
    Example:
        >>> constraint = ConditionalDemographicParity(
        ...     legitimate_features=["credit_score"]
        ... )
    """
    
    def __init__(
        self,
        legitimate_feature_indices: List[int],
        num_bins: int = 10,
        **kwargs
    ):
        """
        Initialize conditional demographic parity.
        
        Args:
            legitimate_feature_indices: Indices of legitimate features
            num_bins: Number of bins for discretizing continuous features
            **kwargs: Arguments passed to DemographicParity
        """
        super().__init__(**kwargs)
        self.legitimate_feature_indices = legitimate_feature_indices
        self.num_bins = num_bins
    
    def compute(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute conditional demographic parity.
        
        Args:
            predictions: Model predictions
            groups: Group indices
            features: Feature tensor for computing conditions
            labels: Ignored
            
        Returns:
            Conditional demographic parity difference
        """
        probs = self._get_probabilities(predictions)
        
        # Discretize legitimate features
        legitimate_feats = features[:, self.legitimate_feature_indices]
        
        # Bin continuous features
        bins = torch.linspace(0, 1, self.num_bins + 1, device=features.device)
        binned = torch.bucketize(legitimate_feats, bins) - 1
        binned = binned.clamp(0, self.num_bins - 1)
        
        # Compute parity within each bin
        max_diff = 0.0
        
        for bin_idx in range(self.num_bins):
            bin_mask = (binned == bin_idx).all(dim=1)
            
            if bin_mask.sum() == 0:
                continue
            
            unique_groups = torch.unique(groups[bin_mask])
            
            if len(unique_groups) < 2:
                continue
            
            group_rates = []
            for g in unique_groups:
                mask = bin_mask & (groups == g)
                if mask.sum() > 0:
                    rate = probs[mask].mean().item()
                    group_rates.append(rate)
            
            if len(group_rates) >= 2:
                diff = max(group_rates) - min(group_rates)
                max_diff = max(max_diff, diff)
        
        return max_diff
    
    def loss(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute conditional demographic parity loss."""
        probs = self._get_probabilities(predictions)
        
        legitimate_feats = features[:, self.legitimate_feature_indices]
        bins = torch.linspace(0, 1, self.num_bins + 1, device=features.device)
        binned = torch.bucketize(legitimate_feats, bins) - 1
        binned = binned.clamp(0, self.num_bins - 1)
        
        total_loss = torch.tensor(0.0, device=predictions.device)
        count = 0
        
        for bin_idx in range(self.num_bins):
            bin_mask = (binned == bin_idx).all(dim=1)
            
            if bin_mask.sum() == 0:
                continue
            
            unique_groups = torch.unique(groups[bin_mask])
            
            if len(unique_groups) < 2:
                continue
            
            group_rates = []
            for g in unique_groups:
                mask = bin_mask & (groups == g)
                if mask.sum() > 0:
                    rate = probs[mask].mean()
                    group_rates.append(rate)
            
            if len(group_rates) >= 2:
                rates = torch.stack(group_rates)
                total_loss = total_loss + torch.var(rates)
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return self.apply_relaxation(total_loss, **kwargs)
