"""
Fairness Regularizer
====================

Regularization module for enforcing fairness constraints during training.

This module provides:
- FairnessRegularizer: Main regularizer class
- ConstraintBasedRegularizer: Constraint-based regularization
- AdversarialRegularizer: Adversarial debiasing regularizer
- MultiFairnessRegularizer: Combined fairness regularizer
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fairness.constraints.base_constraint import BaseFairnessConstraint, ConstraintCombiner
from src.fairness.constraints.demographic_parity import DemographicParity
from src.fairness.constraints.equalized_odds import EqualizedOdds, EqualOpportunity
from src.fairness.constraints.disparate_impact import DisparateImpact
from src.fairness.individual_fairness.lipschitz_constraint import LipschitzConstraint
from src.fairness.individual_fairness.consistency_constraint import ConsistencyConstraint
from src.fairness.losses.adversarial_loss import AdversarialDebiasingLoss
from src.fairness.modules.gradient_reversal import ScheduledGradientReversalLayer


class RegularizationType(Enum):
    """Types of fairness regularization."""
    PENALTY = "penalty"              # Add penalty to loss
    CONSTRAINT = "constraint"        # Hard constraint satisfaction
    ADVERSARIAL = "adversarial"      # Adversarial debiasing
    REWEIGHING = "reweighing"        # Sample reweighing
    AUGMENTATION = "augmentation"    # Data augmentation


class FairnessRegularizer(nn.Module):
    """
    Main Fairness Regularizer.
    
    Combines multiple fairness constraints and regularization methods
    into a unified interface for training fair models.
    
    Features:
        - Multiple fairness paradigms (group, individual, counterfactual)
        - Various regularization methods
        - Adaptive weight scheduling
        - Constraint satisfaction monitoring
    
    Example:
        >>> regularizer = FairnessRegularizer(
        ...     constraints=[
        ...         DemographicParity(threshold=0.05),
        ...         EqualizedOdds(threshold=0.05)
        ...     ],
        ...     regularization_type="penalty",
        ...     initial_weight=0.1
        ... )
        >>> 
        >>> # During training
        >>> reg_loss = regularizer(predictions, groups, labels)
        >>> total_loss = task_loss + reg_loss
    """
    
    def __init__(
        self,
        constraints: List[BaseFairnessConstraint],
        regularization_type: str = "penalty",
        initial_weight: float = 1.0,
        max_weight: float = 10.0,
        weight_schedule: str = "constant",
        warmup_steps: int = 1000,
        combination: str = "sum",
        track_metrics: bool = True,
        name: str = "fairness_regularizer"
    ):
        """
        Initialize fairness regularizer.
        
        Args:
            constraints: List of fairness constraints
            regularization_type: Type of regularization
            initial_weight: Initial regularization weight
            max_weight: Maximum weight for scheduled increase
            weight_schedule: Weight schedule ("constant", "linear", "cosine")
            warmup_steps: Steps to reach max weight
            combination: How to combine constraints ("sum", "max")
            track_metrics: Whether to track fairness metrics
            name: Regularizer name
        """
        super().__init__()
        
        self.constraints = nn.ModuleList(constraints)
        self.regularization_type = RegularizationType(regularization_type)
        self.initial_weight = initial_weight
        self.max_weight = max_weight
        self.weight_schedule = weight_schedule
        self.warmup_steps = warmup_steps
        self.combination = combination
        self.track_metrics = track_metrics
        self.name = name
        
        self._step_count = 0
        self._metrics_history = []
    
    def get_current_weight(self) -> float:
        """Get current regularization weight based on schedule."""
        if self.weight_schedule == "constant":
            return self.initial_weight
        
        elif self.weight_schedule == "linear":
            progress = min(self._step_count / self.warmup_steps, 1.0)
            return self.initial_weight + progress * (self.max_weight - self.initial_weight)
        
        elif self.weight_schedule == "cosine":
            progress = min(self._step_count / self.warmup_steps, 1.0)
            cos_progress = (1 - math.cos(math.pi * progress)) / 2
            return self.initial_weight + cos_progress * (self.max_weight - self.initial_weight)
        
        else:
            return self.initial_weight
    
    def forward(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute regularization loss.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            labels: Ground truth labels
            features: Features for individual fairness
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with regularization loss and metrics
        """
        self._step_count += 1
        weight = self.get_current_weight()
        
        losses = {}
        metrics = {}
        
        for constraint in self.constraints:
            loss = constraint.loss(predictions, groups, labels, features=features, **kwargs)
            losses[constraint.name] = loss
            
            if self.track_metrics:
                metric = constraint.compute(predictions, groups, labels, features=features, **kwargs)
                metrics[constraint.name] = metric
        
        # Combine losses
        if self.combination == "sum":
            total_loss = sum(losses.values())
        elif self.combination == "max":
            total_loss = max(losses.values())
        else:
            total_loss = sum(losses.values())
        
        # Apply weight
        total_loss = weight * total_loss
        
        result = {
            "regularization_loss": total_loss,
            "weight": weight,
            **{f"loss_{k}": v for k, v in losses.items()}
        }
        
        if self.track_metrics:
            result.update({f"metric_{k}": v for k, v in metrics.items()})
            self._metrics_history.append(metrics)
        
        return result
    
    def get_metrics_summary(
        self,
        last_n: int = 100
    ) -> Dict[str, Dict[str, float]]:
        """
        Get summary of tracked metrics.
        
        Args:
            last_n: Number of recent steps to summarize
            
        Returns:
            Dictionary with mean, std, min, max for each metric
        """
        if not self._metrics_history:
            return {}
        
        recent = self._metrics_history[-last_n:]
        
        summary = {}
        metric_names = recent[0].keys() if recent else []
        
        for name in metric_names:
            values = [h[name] for h in recent if name in h]
            if values:
                summary[name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "last": values[-1]
                }
        
        return summary
    
    def reset(self) -> None:
        """Reset regularizer state."""
        self._step_count = 0
        self._metrics_history = []


class AdversarialRegularizer(nn.Module):
    """
    Adversarial Debiasing Regularizer.
    
    Uses adversarial training to learn fair representations.
    The adversary tries to predict sensitive attributes from
    latent representations, while the main model tries to prevent this.
    
    Example:
        >>> regularizer = AdversarialRegularizer(
        ...     latent_dim=512,
        ...     num_sensitive_groups=2,
        ...     adversary_hidden=[256, 128]
        ... )
        >>> 
        >>> # Main model training
        >>> result = regularizer.main_model_loss(latent, sensitive_attrs)
        >>> result["loss"].backward()
        >>> 
        >>> # Adversary training
        >>> result = regularizer.adversary_loss(latent, sensitive_attrs)
        >>> result["loss"].backward()
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_sensitive_groups: int,
        adversary_hidden: List[int] = [256, 128],
        fairness_weight: float = 1.0,
        grl_lambda: float = 1.0,
        grl_schedule: str = "linear",
        grl_warmup: int = 10,
        dropout: float = 0.3,
        name: str = "adversarial_regularizer"
    ):
        """
        Initialize adversarial regularizer.
        
        Args:
            latent_dim: Dimension of latent representations
            num_sensitive_groups: Number of sensitive attribute groups
            adversary_hidden: Hidden dimensions for adversary
            fairness_weight: Weight for fairness loss
            grl_lambda: Gradient reversal lambda
            grl_schedule: GRL schedule type
            grl_warmup: Warmup epochs for GRL
            dropout: Dropout rate
            name: Regularizer name
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_sensitive_groups = num_sensitive_groups
        self.fairness_weight = fairness_weight
        self.name = name
        
        # Build adversary network
        layers = []
        prev_dim = latent_dim
        
        for dim in adversary_hidden:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, num_sensitive_groups))
        
        self.adversary = nn.Sequential(*layers)
        
        # Gradient reversal
        self.grl = ScheduledGradientReversalLayer(
            lambda_start=0.0,
            lambda_end=grl_lambda,
            warmup_epochs=grl_warmup,
            schedule_type=grl_schedule
        )
        
        self._current_epoch = 0
    
    def forward(
        self,
        latent: torch.Tensor,
        sensitive_attrs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass (for main model training with GRL).
        
        Args:
            latent: Latent representation
            sensitive_attrs: Sensitive attribute labels
            
        Returns:
            Dictionary with loss
        """
        return self.main_model_loss(latent, sensitive_attrs)
    
    def main_model_loss(
        self,
        latent: torch.Tensor,
        sensitive_attrs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for main model training.
        
        Uses gradient reversal to make latent fair.
        """
        # Apply GRL
        latent_reversed = self.grl(latent)
        
        # Adversary prediction
        logits = self.adversary(latent_reversed)
        
        # Loss (main model wants to minimize this, GRL reverses gradient)
        loss = F.cross_entropy(logits, sensitive_attrs)
        
        return {
            "loss": self.fairness_weight * loss,
            "adversary_accuracy": (logits.argmax(dim=-1) == sensitive_attrs).float().mean().item()
        }
    
    def adversary_loss(
        self,
        latent: torch.Tensor,
        sensitive_attrs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for adversary training.
        
        No GRL - adversary tries to predict sensitive attributes.
        """
        # Direct prediction (no GRL)
        logits = self.adversary(latent)
        
        # Loss for adversary
        loss = F.cross_entropy(logits, sensitive_attrs)
        
        return {
            "loss": loss,
            "accuracy": (logits.argmax(dim=-1) == sensitive_attrs).float().mean().item()
        }
    
    def step_epoch(self) -> None:
        """Advance epoch counter."""
        self._current_epoch += 1
        self.grl.step()
    
    def get_grl_lambda(self) -> float:
        """Get current GRL lambda."""
        return self.grl.lambda_


class ReweighingRegularizer(nn.Module):
    """
    Reweighing-based Fairness Regularizer.
    
    Computes instance weights to achieve demographic parity:
        W(x, y, a) = P(a) * P(y) / P(a, y)
    
    These weights can be used in the loss function to debias training.
    
    Example:
        >>> regularizer = ReweighingRegularizer()
        >>> 
        >>> # Compute weights
        >>> weights = regularizer.compute_weights(groups, labels)
        >>> 
        >>> # Use in loss
        >>> loss = F.binary_cross_entropy(predictions, labels, weight=weights)
    """
    
    def __init__(
        self,
        smoothing: float = 0.0,
        name: str = "reweighing_regularizer"
    ):
        """
        Initialize reweighing regularizer.
        
        Args:
            smoothing: Label smoothing for weight computation
            name: Regularizer name
        """
        super().__init__()
        
        self.smoothing = smoothing
        self.name = name
        
        # Store computed weights
        self._weights_cache = None
    
    def compute_weights(
        self,
        groups: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute instance weights for fairness.
        
        Args:
            groups: Group membership tensor
            labels: Ground truth labels
            
        Returns:
            Instance weights tensor
        """
        batch_size = groups.size(0)
        device = groups.device
        
        # Convert to numpy for easier computation
        groups_np = groups.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        
        unique_groups = np.unique(groups_np)
        unique_labels = np.unique(labels_np)
        
        # Compute marginal probabilities
        n_total = len(groups_np)
        
        # P(A=a)
        p_a = {a: np.mean(groups_np == a) for a in unique_groups}
        
        # P(Y=y)
        p_y = {y: np.mean(labels_np == y) for y in unique_labels}
        
        # P(A=a, Y=y)
        p_ay = {}
        for a in unique_groups:
            for y in unique_labels:
                mask = (groups_np == a) & (labels_np == y)
                p_ay[(a, y)] = np.mean(mask)
        
        # Compute weights
        weights = np.ones(batch_size)
        
        for i in range(batch_size):
            a = groups_np[i]
            y = labels_np[i]
            
            if p_ay[(a, y)] > 0:
                # W = P(A) * P(Y) / P(A, Y)
                weights[i] = (p_a[a] * p_y[y]) / p_ay[(a, y)]
        
        # Normalize weights
        weights = weights / weights.mean()
        
        # Apply smoothing
        if self.smoothing > 0:
            weights = (1 - self.smoothing) * weights + self.smoothing * 1.0
        
        weights_tensor = torch.from_numpy(weights).float().to(device)
        self._weights_cache = weights_tensor
        
        return weights_tensor
    
    def forward(
        self,
        groups: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute and return weights."""
        return self.compute_weights(groups, labels)


class CombinedFairnessRegularizer(nn.Module):
    """
    Combined Fairness Regularizer.
    
    Combines multiple regularization methods:
        - Constraint-based penalty
        - Adversarial debiasing
        - Reweighing
    
    Example:
        >>> regularizer = CombinedFairnessRegularizer(
        ...     constraints=[DemographicParity(), EqualizedOdds()],
        ...     use_adversarial=True,
        ...     use_reweighing=True,
        ...     latent_dim=512
        ... )
    """
    
    def __init__(
        self,
        constraints: List[BaseFairnessConstraint],
        use_adversarial: bool = False,
        use_reweighing: bool = False,
        latent_dim: Optional[int] = None,
        num_sensitive_groups: int = 2,
        constraint_weight: float = 1.0,
        adversarial_weight: float = 1.0,
        name: str = "combined_fairness_regularizer"
    ):
        """
        Initialize combined regularizer.
        
        Args:
            constraints: List of fairness constraints
            use_adversarial: Whether to use adversarial debiasing
            use_reweighing: Whether to use reweighing
            latent_dim: Dimension for adversarial (required if use_adversarial)
            num_sensitive_groups: Number of sensitive groups
            constraint_weight: Weight for constraint losses
            adversarial_weight: Weight for adversarial loss
            name: Regularizer name
        """
        super().__init__()
        
        self.name = name
        self.use_adversarial = use_adversarial
        self.use_reweighing = use_reweighing
        
        # Constraint regularizer
        self.constraint_regularizer = FairnessRegularizer(
            constraints=constraints,
            initial_weight=constraint_weight
        )
        
        # Adversarial regularizer
        if use_adversarial:
            if latent_dim is None:
                raise ValueError("latent_dim required for adversarial regularizer")
            self.adversarial_regularizer = AdversarialRegularizer(
                latent_dim=latent_dim,
                num_sensitive_groups=num_sensitive_groups,
                fairness_weight=adversarial_weight
            )
        else:
            self.adversarial_regularizer = None
        
        # Reweighing regularizer
        if use_reweighing:
            self.reweighing_regularizer = ReweighingRegularizer()
        else:
            self.reweighing_regularizer = None
    
    def forward(
        self,
        predictions: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        latent: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined regularization loss.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            labels: Ground truth labels
            latent: Latent representation (for adversarial)
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with combined loss
        """
        result = {}
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        # Constraint losses
        constraint_result = self.constraint_regularizer(
            predictions, groups, labels, **kwargs
        )
        result.update(constraint_result)
        total_loss = total_loss + constraint_result.get("regularization_loss", 0)
        
        # Adversarial loss
        if self.use_adversarial and latent is not None and labels is not None:
            adv_result = self.adversarial_regularizer(latent, groups)
            result["adversarial_loss"] = adv_result["loss"]
            total_loss = total_loss + adv_result["loss"]
        
        # Reweighing weights
        if self.use_reweighing and labels is not None:
            weights = self.reweighing_regularizer(groups, labels)
            result["sample_weights"] = weights
        
        result["total_loss"] = total_loss
        
        return result
    
    def step_epoch(self) -> None:
        """Advance epoch for schedulers."""
        self.constraint_regularizer._step_count = 0  # Reset step count if needed
        if self.adversarial_regularizer is not None:
            self.adversarial_regularizer.step_epoch()


def create_fairness_regularizer(
    fairness_config: Dict[str, Any]
) -> Union[FairnessRegularizer, AdversarialRegularizer, CombinedFairnessRegularizer]:
    """
    Factory function to create fairness regularizer from config.
    
    Args:
        fairness_config: Configuration dictionary
        
    Returns:
        Appropriate regularizer instance
    """
    reg_type = fairness_config.get("type", "penalty")
    
    # Parse constraints
    constraints = []
    constraint_configs = fairness_config.get("constraints", [])
    
    for cc in constraint_configs:
        constraint_type = cc.get("type", "")
        threshold = cc.get("threshold", 0.05)
        
        if constraint_type == "demographic_parity":
            constraints.append(DemographicParity(threshold=threshold))
        elif constraint_type == "equalized_odds":
            constraints.append(EqualizedOdds(threshold=threshold))
        elif constraint_type == "equal_opportunity":
            constraints.append(EqualOpportunity(threshold=threshold))
        elif constraint_type == "disparate_impact":
            constraints.append(DisparateImpact(min_ratio=threshold))
        elif constraint_type == "lipschitz":
            constraints.append(LipschitzConstraint(
                lambda_lipschitz=cc.get("lambda", 0.1)
            ))
        elif constraint_type == "consistency":
            constraints.append(ConsistencyConstraint(
                k_neighbors=cc.get("k_neighbors", 10)
            ))
    
    if reg_type == "adversarial":
        return AdversarialRegularizer(
            latent_dim=fairness_config.get("latent_dim", 512),
            num_sensitive_groups=fairness_config.get("num_sensitive_groups", 2),
            fairness_weight=fairness_config.get("weight", 1.0)
        )
    elif reg_type == "combined":
        return CombinedFairnessRegularizer(
            constraints=constraints,
            use_adversarial=fairness_config.get("use_adversarial", False),
            use_reweighing=fairness_config.get("use_reweighing", False),
            latent_dim=fairness_config.get("latent_dim"),
            num_sensitive_groups=fairness_config.get("num_sensitive_groups", 2)
        )
    else:
        return FairnessRegularizer(
            constraints=constraints,
            regularization_type=reg_type,
            initial_weight=fairness_config.get("weight", 1.0)
        )
