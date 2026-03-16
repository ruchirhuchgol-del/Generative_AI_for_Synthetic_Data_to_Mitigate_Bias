"""
Equalized Odds Constraint
=========================

Implementation of equalized odds fairness constraint.

Equalized odds requires equal true positive rates (TPR) and false positive rates (FPR)
across all groups, conditional on the true label:
    P(Ŷ=1|Y=1, A=0) = P(Ŷ=1|Y=1, A=1)  (TPR parity)
    P(Ŷ=1|Y=0, A=0) = P(Ŷ=1|Y=0, A=1)  (FPR parity)

Also known as:
- Conditional Procedure Accuracy Equality
- Disparate Mistreatment
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


class EqualizedOddsType(Enum):
    """Types of equalized odds constraints."""
    FULL = "full"            # Both TPR and FPR parity
    TPR_ONLY = "tpr_only"    # Only true positive rate parity (equal opportunity)
    FPR_ONLY = "fpr_only"    # Only false positive rate parity
    BALANCED = "balanced"    # Weighted combination of TPR and FPR


class EqualizedOdds(BaseFairnessConstraint):
    """
    Equalized Odds constraint.
    
    Ensures equal true positive rates and false positive rates across groups:
        TPR_a = TPR_b for all groups a, b
        FPR_a = FPR_b for all groups a, b
    
    Mathematical Definition:
        Equalized Odds Difference = max(
            |TPR_a - TPR_b|,
            |FPR_a - FPR_b|
        )
    
    Properties:
        - Stronger than demographic parity (considers accuracy)
        - Allows for different base rates across groups
        - Requires labeled data for training
        - Compatible with perfect accuracy
    
    Subsumes:
        - Equal Opportunity: Only TPR parity
        - Predictive Equality: Only FPR parity
    
    Example:
        >>> constraint = EqualizedOdds(threshold=0.05)
        >>> 
        >>> # Compute violation
        >>> diff = constraint.compute(predictions, groups, labels)
        >>> 
        >>> # Training loss
        >>> loss = constraint.loss(logits, groups, labels)
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        relaxation: str = "soft",
        constraint_type: str = "full",
        tpr_weight: float = 1.0,
        fpr_weight: float = 1.0,
        weight: float = 1.0,
        use_logits: bool = True,
        temperature: float = 1.0,
        name: str = "equalized_odds"
    ):
        """
        Initialize equalized odds constraint.
        
        Args:
            threshold: Maximum allowed difference in rates
            relaxation: Constraint relaxation type
            constraint_type: Type of equalized odds:
                - "full": Both TPR and FPR parity
                - "tpr_only": Only TPR parity (Equal Opportunity)
                - "fpr_only": Only FPR parity (Predictive Equality)
                - "balanced": Weighted combination
            tpr_weight: Weight for TPR component (for "balanced" type)
            fpr_weight: Weight for FPR component (for "balanced" type)
            weight: Overall constraint weight
            use_logits: Whether inputs are logits
            temperature: Temperature for softmax
            name: Constraint name
        """
        super().__init__(threshold, relaxation, weight, name)
        
        self._constraint_type = ConstraintType.GROUP
        self.constraint_type = EqualizedOddsType(constraint_type)
        self.tpr_weight = tpr_weight
        self.fpr_weight = fpr_weight
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
        labels: torch.Tensor,
        **kwargs
    ) -> float:
        """
        Compute equalized odds difference.
        
        Args:
            predictions: Model predictions (batch_size,)
            groups: Group indices (batch_size,)
            labels: Ground truth labels (batch_size,)
            
        Returns:
            Equalized odds difference (lower is more fair)
        """
        probs = self._get_probabilities(predictions)
        
        unique_groups = torch.unique(groups)
        
        tprs = {}
        fprs = {}
        
        for g in unique_groups:
            group_mask = (groups == g)
            positive_mask = group_mask & (labels == 1)
            negative_mask = group_mask & (labels == 0)
            
            # TPR: True Positive Rate (recall)
            if positive_mask.sum() > 0:
                tpr = probs[positive_mask].mean().item()
                tprs[g.item()] = tpr
            
            # FPR: False Positive Rate
            if negative_mask.sum() > 0:
                fpr = probs[negative_mask].mean().item()
                fprs[g.item()] = fpr
        
        max_diff = 0.0
        
        # TPR parity
        if self.constraint_type in [EqualizedOddsType.FULL, EqualizedOddsType.TPR_ONLY, EqualizedOddsType.BALANCED]:
            if len(tprs) >= 2:
                tpr_diff = max(tprs.values()) - min(tprs.values())
                max_diff = max(max_diff, tpr_diff)
        
        # FPR parity
        if self.constraint_type in [EqualizedOddsType.FULL, EqualizedOddsType.FPR_ONLY, EqualizedOddsType.BALANCED]:
            if len(fprs) >= 2:
                fpr_diff = max(fprs.values()) - min(fprs.values())
                max_diff = max(max_diff, fpr_diff)
        
        return max_diff
    
    def loss(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute equalized odds loss.
        
        Uses variance-based soft constraint for differentiability.
        
        Args:
            predictions: Model predictions
            groups: Group indices
            labels: Ground truth labels
            
        Returns:
            Loss tensor
        """
        probs = self._get_probabilities(predictions)
        
        unique_groups = torch.unique(groups)
        
        tpr_losses = []
        fpr_losses = []
        
        for g in unique_groups:
            group_mask = (groups == g)
            positive_mask = group_mask & (labels == 1)
            negative_mask = group_mask & (labels == 0)
            
            if positive_mask.sum() > 0:
                tpr = probs[positive_mask].mean()
                tpr_losses.append(tpr)
            
            if negative_mask.sum() > 0:
                fpr = probs[negative_mask].mean()
                fpr_losses.append(fpr)
        
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        # TPR loss
        if self.constraint_type in [EqualizedOddsType.FULL, EqualizedOddsType.TPR_ONLY, EqualizedOddsType.BALANCED]:
            if len(tpr_losses) >= 2:
                tprs = torch.stack(tpr_losses)
                tpr_loss = torch.var(tprs)
                
                if self.constraint_type == EqualizedOddsType.BALANCED:
                    tpr_loss = self.tpr_weight * tpr_loss
                
                total_loss = total_loss + tpr_loss
        
        # FPR loss
        if self.constraint_type in [EqualizedOddsType.FULL, EqualizedOddsType.FPR_ONLY, EqualizedOddsType.BALANCED]:
            if len(fpr_losses) >= 2:
                fprs = torch.stack(fpr_losses)
                fpr_loss = torch.var(fprs)
                
                if self.constraint_type == EqualizedOddsType.BALANCED:
                    fpr_loss = self.fpr_weight * fpr_loss
                
                total_loss = total_loss + fpr_loss
        
        return self.apply_relaxation(total_loss, **kwargs)
    
    def get_rates(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, Dict[int, float]]:
        """
        Get TPR and FPR per group.
        
        Args:
            predictions: Model predictions
            groups: Group indices
            labels: Ground truth labels
            
        Returns:
            Dictionary with 'tpr' and 'fpr' sub-dictionaries
        """
        probs = self._get_probabilities(predictions)
        
        unique_groups = torch.unique(groups)
        
        tprs = {}
        fprs = {}
        
        for g in unique_groups:
            group_mask = (groups == g)
            positive_mask = group_mask & (labels == 1)
            negative_mask = group_mask & (labels == 0)
            
            if positive_mask.sum() > 0:
                tprs[g.item()] = probs[positive_mask].mean().item()
            
            if negative_mask.sum() > 0:
                fprs[g.item()] = probs[negative_mask].mean().item()
        
        return {"tpr": tprs, "fpr": fprs}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        d = super().to_dict()
        d.update({
            "constraint_type": self.constraint_type.value,
            "tpr_weight": self.tpr_weight,
            "fpr_weight": self.fpr_weight,
            "use_logits": self.use_logits,
            "temperature": self.temperature
        })
        return d
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "EqualizedOdds":
        """Create from dictionary configuration."""
        return cls(
            threshold=config.get("threshold", 0.05),
            relaxation=config.get("relaxation", "soft"),
            constraint_type=config.get("constraint_type", "full"),
            tpr_weight=config.get("tpr_weight", 1.0),
            fpr_weight=config.get("fpr_weight", 1.0),
            weight=config.get("weight", 1.0),
            use_logits=config.get("use_logits", True),
            temperature=config.get("temperature", 1.0),
            name=config.get("name", "equalized_odds")
        )
    
    def __repr__(self) -> str:
        return (
            f"EqualizedOdds("
            f"threshold={self.threshold}, "
            f"type={self.constraint_type.value}, "
            f"relaxation={self.relaxation.value})"
        )


class EqualOpportunity(EqualizedOdds):
    """
    Equal Opportunity constraint.
    
    A special case of equalized odds that only enforces TPR parity:
        P(Ŷ=1|Y=1, A=0) = P(Ŷ=1|Y=1, A=1)
    
    This ensures that qualified individuals have equal opportunity
    regardless of group membership.
    
    Example:
        >>> constraint = EqualOpportunity(threshold=0.05)
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        relaxation: str = "soft",
        **kwargs
    ):
        super().__init__(
            threshold=threshold,
            relaxation=relaxation,
            constraint_type="tpr_only",
            name="equal_opportunity",
            **kwargs
        )


class PredictiveEquality(EqualizedOdds):
    """
    Predictive Equality constraint.
    
    A special case of equalized odds that only enforces FPR parity:
        P(Ŷ=1|Y=0, A=0) = P(Ŷ=1|Y=0, A=1)
    
    This ensures that unqualified individuals are equally likely
    to be incorrectly selected across groups.
    
    Example:
        >>> constraint = PredictiveEquality(threshold=0.05)
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        relaxation: str = "soft",
        **kwargs
    ):
        super().__init__(
            threshold=threshold,
            relaxation=relaxation,
            constraint_type="fpr_only",
            name="predictive_equality",
            **kwargs
        )


class AccuracyParity(BaseFairnessConstraint):
    """
    Accuracy Parity constraint.
    
    Ensures equal accuracy across groups:
        P(Ŷ=Y|A=0) = P(Ŷ=Y|A=1)
    
    This is another form of fairness that focuses on overall
    prediction correctness rather than specific error types.
    
    Example:
        >>> constraint = AccuracyParity(threshold=0.05)
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        relaxation: str = "soft",
        use_logits: bool = True,
        temperature: float = 1.0,
        name: str = "accuracy_parity"
    ):
        super().__init__(threshold, relaxation, name=name)
        self._constraint_type = ConstraintType.GROUP
        self.use_logits = use_logits
        self.temperature = temperature
    
    def _get_predictions(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """Convert predictions to binary."""
        if self.use_logits:
            return (predictions > 0).float()
        else:
            return (predictions > 0.5).float()
    
    def compute(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> float:
        """Compute accuracy parity difference."""
        preds = self._get_predictions(predictions)
        correct = (preds == labels).float()
        
        unique_groups = torch.unique(groups)
        accuracies = {}
        
        for g in unique_groups:
            mask = (groups == g)
            if mask.sum() > 0:
                acc = correct[mask].mean().item()
                accuracies[g.item()] = acc
        
        if len(accuracies) < 2:
            return 0.0
        
        return max(accuracies.values()) - min(accuracies.values())
    
    def loss(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Compute accuracy parity loss."""
        probs = torch.sigmoid(predictions) if self.use_logits else predictions
        correct_prob = torch.where(labels == 1, probs, 1 - probs)
        
        unique_groups = torch.unique(groups)
        accs = []
        
        for g in unique_groups:
            mask = (groups == g)
            if mask.sum() > 0:
                acc = correct_prob[mask].mean()
                accs.append(acc)
        
        if len(accs) < 2:
            return torch.tensor(0.0, device=predictions.device)
        
        accs = torch.stack(accs)
        violation = torch.var(accs)
        
        return self.apply_relaxation(violation, **kwargs)
