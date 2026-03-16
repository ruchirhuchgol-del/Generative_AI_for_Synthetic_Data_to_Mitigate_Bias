"""
Fairness Loss Functions
=======================

Loss functions for fairness-aware training.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FairnessLoss(nn.Module):
    """
    Base class for fairness losses.
    
    Provides a unified interface for computing fairness-related losses.
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize the loss.
        
        Args:
            weight: Weight for this loss in total loss computation
        """
        super().__init__()
        self.weight = weight
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Compute the loss."""
        raise NotImplementedError


class MultiObjectiveFairnessLoss(nn.Module):
    """
    Multi-objective fairness loss combining multiple fairness constraints.
    
    Combines:
    - Group fairness (demographic parity, equalized odds)
    - Individual fairness (Lipschitz constraint)
    - Counterfactual fairness
    
    Loss = α * L_group + β * L_individual + γ * L_counterfactual
    """
    
    def __init__(
        self,
        group_weight: float = 1.0,
        individual_weight: float = 0.5,
        counterfactual_weight: float = 0.5,
        group_metrics: List[str] = ["demographic_parity"],
        threshold: float = 0.05
    ):
        """
        Initialize multi-objective fairness loss.
        
        Args:
            group_weight: Weight for group fairness loss
            individual_weight: Weight for individual fairness loss
            counterfactual_weight: Weight for counterfactual fairness loss
            group_metrics: List of group fairness metrics to use
            threshold: Fairness threshold for constraints
        """
        super().__init__()
        
        self.group_weight = group_weight
        self.individual_weight = individual_weight
        self.counterfactual_weight = counterfactual_weight
        self.group_metrics = group_metrics
        self.threshold = threshold
        
        from src.fairness.constraints import (
            DemographicParity,
            EqualizedOdds,
            LipschitzConstraint,
            CounterfactualFairness
        )
        
        # Initialize constraints
        self.group_constraints = nn.ModuleDict()
        
        if "demographic_parity" in group_metrics:
            self.group_constraints["demographic_parity"] = DemographicParity(threshold)
        
        if "equalized_odds" in group_metrics:
            self.group_constraints["equalized_odds"] = EqualizedOdds(threshold)
        
        self.individual_constraint = LipschitzConstraint(lambda_lipschitz=0.1)
        self.counterfactual_constraint = CounterfactualFairness(threshold)
    
    def forward(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        counterfactual_predictions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-objective fairness loss.
        
        Args:
            predictions: Model predictions
            groups: Group membership tensor
            features: Feature tensor for individual fairness
            labels: Ground truth labels for equalized odds
            counterfactual_predictions: Counterfactual predictions
            
        Returns:
            Dictionary of individual and total losses
        """
        losses = {}
        
        # Group fairness losses
        group_loss = torch.tensor(0.0, device=predictions.device)
        
        for name, constraint in self.group_constraints.items():
            if name == "demographic_parity":
                loss = constraint.loss(predictions, groups)
            elif name == "equalized_odds":
                if labels is not None:
                    loss = constraint.loss(predictions, groups, labels)
                else:
                    loss = torch.tensor(0.0, device=predictions.device)
            else:
                loss = torch.tensor(0.0, device=predictions.device)
            
            losses[f"group_{name}"] = loss
            group_loss = group_loss + loss
        
        losses["group_total"] = self.group_weight * group_loss
        
        # Individual fairness loss
        if features is not None:
            ind_loss = self.individual_constraint.loss(predictions, features)
            losses["individual"] = self.individual_weight * ind_loss
        else:
            losses["individual"] = torch.tensor(0.0, device=predictions.device)
        
        # Counterfactual fairness loss
        if counterfactual_predictions is not None:
            cf_loss = self.counterfactual_constraint.loss(
                predictions, counterfactual_predictions
            )
            losses["counterfactual"] = self.counterfactual_weight * cf_loss
        else:
            losses["counterfactual"] = torch.tensor(0.0, device=predictions.device)
        
        # Total fairness loss
        losses["total"] = (
            losses["group_total"] +
            losses["individual"] +
            losses["counterfactual"]
        )
        
        return losses


class AdversarialDebiasingLoss(nn.Module):
    """
    Loss for adversarial debiasing.
    
    Trains an adversary to predict sensitive attributes from latent representations,
    while the main model tries to fool the adversary.
    """
    
    def __init__(
        self,
        num_sensitive_groups: int,
        adversary_hidden_dims: List[int] = [256, 128],
        dropout: float = 0.3
    ):
        """
        Initialize adversarial debiasing loss.
        
        Args:
            num_sensitive_groups: Number of sensitive attribute groups
            adversary_hidden_dims: Hidden dimensions for adversary network
            dropout: Dropout rate for adversary
        """
        super().__init__()
        
        self.num_sensitive_groups = num_sensitive_groups
        
        # Build adversary network
        layers = []
        prev_dim = None  # Will be set at runtime
        
        for dim in adversary_hidden_dims:
            if prev_dim is not None:
                layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout)
                ])
            prev_dim = dim
        
        self.hidden_layers = nn.Sequential(*layers) if layers else None
        self.output_layer = nn.Linear(prev_dim or 512, num_sensitive_groups)
    
    def forward(
        self,
        latent: torch.Tensor,
        sensitive_attrs: torch.Tensor,
        lambda_grl: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adversarial debiasing loss.
        
        Args:
            latent: Latent representation tensor
            sensitive_attrs: Sensitive attribute labels
            lambda_grl: Gradient reversal strength
            
        Returns:
            Dictionary containing adversary loss and predictions
        """
        # Forward through adversary
        if self.hidden_layers is not None:
            hidden = self.hidden_layers(latent)
        else:
            hidden = latent
        
        logits = self.output_layer(hidden)
        
        # Compute loss
        loss = F.cross_entropy(logits, sensitive_attrs)
        
        return {
            "adversary_loss": loss,
            "adversary_logits": logits,
            "adversary_predictions": torch.argmax(logits, dim=-1)
        }
