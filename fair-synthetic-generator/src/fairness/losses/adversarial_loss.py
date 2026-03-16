"""
Adversarial Loss for Fairness Debiasing
=======================================

Loss functions for adversarial training and fairness debiasing.

This module provides:
- AdversarialDebiasingLoss: Main adversarial loss for debiasing
- GradientReversalLoss: Loss with gradient reversal layer
- MultiAdversaryLoss: Loss for multiple sensitive attributes
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.fairness.modules.gradient_reversal import (
    GradientReversalLayer,
    ScheduledGradientReversalLayer
)


class AdversarialDebiasingLoss(nn.Module):
    """
    Adversarial Debiasing Loss.
    
    Implements adversarial training for fairness by training an adversary
    to predict sensitive attributes from latent representations, while
    the main model tries to prevent this.
    
    Architecture:
        Main Model: x -> z -> y_pred
        Adversary: z -> s_pred (sensitive attribute prediction)
    
    Training Objectives:
        Main Model: Minimize task loss + λ * adversary success
        Adversary: Maximize sensitive attribute prediction accuracy
    
    The gradient reversal layer (GRL) reverses gradients from the adversary
    to the main model, causing the main model to learn representations
    that are independent of sensitive attributes.
    
    Example:
        >>> loss_fn = AdversarialDebiasingLoss(
        ...     adversary=adversary_network,
        ...     fairness_weight=1.0
        ... )
        >>> 
        >>> # During training
        >>> loss_dict = loss_fn(latent, sensitive_attrs)
        >>> loss_dict["total"].backward()
    """
    
    def __init__(
        self,
        adversary: nn.Module,
        fairness_weight: float = 1.0,
        use_grl: bool = True,
        grl_lambda: float = 1.0,
        grl_schedule: str = "constant",
        grl_warmup_epochs: int = 10,
        adversary_loss_type: str = "cross_entropy",
        label_smoothing: float = 0.0,
        name: str = "adversarial_debiasing_loss"
    ):
        """
        Initialize adversarial debiasing loss.
        
        Args:
            adversary: Network that predicts sensitive attributes from latent
            fairness_weight: Weight for fairness/adversarial loss component
            use_grl: Whether to use gradient reversal layer
            grl_lambda: Gradient reversal strength
            grl_schedule: GRL schedule type ("constant", "linear", "cosine")
            grl_warmup_epochs: Epochs for GRL warmup
            adversary_loss_type: Loss type for adversary ("cross_entropy", "bce", "mse")
            label_smoothing: Label smoothing for cross entropy
            name: Loss name
        """
        super().__init__()
        
        self.adversary = adversary
        self.fairness_weight = fairness_weight
        self.use_grl = use_grl
        self.adversary_loss_type = adversary_loss_type
        self.label_smoothing = label_smoothing
        self.name = name
        
        # Gradient reversal layer
        if use_grl:
            if grl_schedule == "constant":
                self.grl = GradientReversalLayer(lambda_=grl_lambda)
            else:
                self.grl = ScheduledGradientReversalLayer(
                    lambda_start=0.0,
                    lambda_end=grl_lambda,
                    warmup_epochs=grl_warmup_epochs,
                    schedule_type=grl_schedule
                )
        else:
            self.grl = None
        
        self._current_epoch = 0
    
    def forward(
        self,
        latent: torch.Tensor,
        sensitive_attrs: torch.Tensor,
        apply_grl: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute adversarial debiasing loss.
        
        Args:
            latent: Latent representation tensor
            sensitive_attrs: Sensitive attribute labels
            apply_grl: Whether to apply gradient reversal
            
        Returns:
            Dictionary containing:
                - "adversary_loss": Loss for training adversary
                - "fairness_loss": Loss for training main model
                - "total": Combined loss
                - "adversary_accuracy": Accuracy of adversary
        """
        # Apply gradient reversal for main model training
        if self.use_grl and apply_grl and self.grl is not None:
            latent_reversed = self.grl(latent)
        else:
            latent_reversed = latent
        
        # Get adversary predictions
        adversary_logits = self.adversary(latent_reversed)
        
        # Compute adversary loss
        if self.adversary_loss_type == "cross_entropy":
            if adversary_logits.dim() == 1:
                # Binary classification
                adversary_loss = F.binary_cross_entropy_with_logits(
                    adversary_logits,
                    sensitive_attrs.float(),
                    reduction="mean"
                )
            else:
                # Multi-class
                adversary_loss = F.cross_entropy(
                    adversary_logits,
                    sensitive_attrs,
                    label_smoothing=self.label_smoothing,
                    reduction="mean"
                )
        elif self.adversary_loss_type == "bce":
            adversary_loss = F.binary_cross_entropy_with_logits(
                adversary_logits,
                sensitive_attrs.float(),
                reduction="mean"
            )
        elif self.adversary_loss_type == "mse":
            if adversary_logits.dim() > 1:
                # One-hot encode targets
                num_classes = adversary_logits.size(-1)
                targets = F.one_hot(sensitive_attrs, num_classes).float()
            else:
                targets = sensitive_attrs.float()
            adversary_loss = F.mse_loss(
                torch.sigmoid(adversary_logits),
                targets,
                reduction="mean"
            )
        else:
            adversary_loss = F.cross_entropy(
                adversary_logits,
                sensitive_attrs,
                reduction="mean"
            )
        
        # Compute accuracy
        with torch.no_grad():
            if adversary_logits.dim() == 1:
                preds = (adversary_logits > 0).long()
            else:
                preds = adversary_logits.argmax(dim=-1)
            accuracy = (preds == sensitive_attrs).float().mean().item()
        
        # Fairness loss is adversary loss for main model
        # (GRL reverses gradient, so main model minimizes adversary's success)
        fairness_loss = self.fairness_weight * adversary_loss
        
        return {
            "adversary_loss": adversary_loss,
            "fairness_loss": fairness_loss,
            "total": fairness_loss,  # For backward pass on main model
            "adversary_accuracy": accuracy
        }
    
    def adversary_train_step(
        self,
        latent: torch.Tensor,
        sensitive_attrs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training the adversary (without GRL).
        
        Args:
            latent: Latent representation
            sensitive_attrs: Sensitive attribute labels
            
        Returns:
            Dictionary with loss for adversary training
        """
        # No GRL for adversary training
        return self.forward(latent, sensitive_attrs, apply_grl=False)
    
    def main_model_train_step(
        self,
        latent: torch.Tensor,
        sensitive_attrs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss for training main model (with GRL).
        
        Args:
            latent: Latent representation
            sensitive_attrs: Sensitive attribute labels
            
        Returns:
            Dictionary with loss for main model training
        """
        return self.forward(latent, sensitive_attrs, apply_grl=True)
    
    def step_epoch(self) -> None:
        """Advance epoch counter (for scheduled GRL)."""
        self._current_epoch += 1
        if self.grl is not None and hasattr(self.grl, "step"):
            self.grl.step()
    
    def get_grl_lambda(self) -> float:
        """Get current GRL lambda value."""
        if self.grl is None:
            return 0.0
        return self.grl.lambda_
    
    def extra_repr(self) -> str:
        return (
            f"fairness_weight={self.fairness_weight}, "
            f"use_grl={self.use_grl}, "
            f"adversary_loss_type={self.adversary_loss_type}"
        )


class MultiAdversaryLoss(nn.Module):
    """
    Multi-Adversary Loss for Multiple Sensitive Attributes.
    
    Trains separate adversaries for each sensitive attribute
    and combines their losses.
    
    Example:
        >>> loss_fn = MultiAdversaryLoss(
        ...     adversaries={
        ...         "gender": gender_adversary,
        ...         "race": race_adversary
        ...     },
        ...     weights={"gender": 1.0, "race": 0.5}
        ... )
    """
    
    def __init__(
        self,
        adversaries: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None,
        use_grl: bool = True,
        grl_lambda: float = 1.0,
        name: str = "multi_adversary_loss"
    ):
        """
        Initialize multi-adversary loss.
        
        Args:
            adversaries: Dictionary of adversary networks
            weights: Optional weights for each adversary loss
            use_grl: Whether to use gradient reversal
            grl_lambda: GRL strength
            name: Loss name
        """
        super().__init__()
        
        self.adversaries = nn.ModuleDict(adversaries)
        self.weights = weights or {k: 1.0 for k in adversaries}
        self.use_grl = use_grl
        self.name = name
        
        if use_grl:
            self.grl = GradientReversalLayer(lambda_=grl_lambda)
        else:
            self.grl = None
    
    def forward(
        self,
        latent: torch.Tensor,
        sensitive_attrs: Dict[str, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-adversary loss.
        
        Args:
            latent: Latent representation
            sensitive_attrs: Dictionary of sensitive attribute tensors
            
        Returns:
            Dictionary with total loss and per-attribute losses
        """
        if self.use_grl and self.grl is not None:
            latent_reversed = self.grl(latent)
        else:
            latent_reversed = latent
        
        losses = {}
        accuracies = {}
        
        for name, adversary in self.adversaries.items():
            if name not in sensitive_attrs:
                continue
            
            logits = adversary(latent_reversed)
            targets = sensitive_attrs[name]
            
            if logits.dim() == 1:
                loss = F.binary_cross_entropy_with_logits(
                    logits, targets.float()
                )
                acc = ((logits > 0).long() == targets).float().mean().item()
            else:
                loss = F.cross_entropy(logits, targets)
                acc = (logits.argmax(dim=-1) == targets).float().mean().item()
            
            losses[name] = self.weights.get(name, 1.0) * loss
            accuracies[name] = acc
        
        total_loss = sum(losses.values())
        
        return {
            "total": total_loss,
            "per_attribute_losses": losses,
            "per_attribute_accuracies": accuracies
        }


class ContrastiveFairnessLoss(nn.Module):
    """
    Contrastive Loss for Individual Fairness.
    
    Uses contrastive learning to encourage similar individuals
    to have similar latent representations:
        L = Σ (1 - y) * d² + y * max(0, m - d)²
    where y=1 for similar pairs, y=0 for dissimilar pairs.
    
    Example:
        >>> loss_fn = ContrastiveFairnessLoss(margin=1.0)
        >>> loss = loss_fn(latent1, latent2, is_similar)
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        similarity_weight: float = 1.0,
        dissimilarity_weight: float = 0.5,
        hard_negative_mining: bool = True,
        name: str = "contrastive_fairness_loss"
    ):
        """
        Initialize contrastive fairness loss.
        
        Args:
            margin: Margin for dissimilar pairs
            similarity_weight: Weight for similar pairs loss
            dissimilarity_weight: Weight for dissimilar pairs loss
            hard_negative_mining: Whether to use hard negative mining
            name: Loss name
        """
        super().__init__()
        
        self.margin = margin
        self.similarity_weight = similarity_weight
        self.dissimilarity_weight = dissimilarity_weight
        self.hard_negative_mining = hard_negative_mining
        self.name = name
    
    def forward(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
        is_similar: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            latent1: First set of latent representations
            latent2: Second set of latent representations
            is_similar: Binary tensor (1 for similar, 0 for dissimilar)
            
        Returns:
            Contrastive loss tensor
        """
        # Compute Euclidean distance
        dist = F.pairwise_distance(latent1, latent2, p=2)
        
        # Similar pairs: minimize distance
        similar_loss = is_similar.float() * dist.pow(2)
        
        # Dissimilar pairs: maximize distance (up to margin)
        dissimilar_loss = (1 - is_similar.float()) * F.relu(self.margin - dist).pow(2)
        
        if self.hard_negative_mining:
            # Weight harder negatives more
            weights = F.softmax(dist * (1 - is_similar.float()), dim=0)
            dissimilar_loss = dissimilar_loss * weights * len(weights)
        
        # Combine
        loss = (
            self.similarity_weight * similar_loss.mean() +
            self.dissimilarity_weight * dissimilar_loss.mean()
        )
        
        return loss


class MutualInformationLoss(nn.Module):
    """
    Mutual Information Minimization Loss.
    
    Minimizes mutual information between latent representations
    and sensitive attributes for fairness.
    
    Uses the CLUB (Contrastive Log-ratio Upper Bound) estimator:
        MI(X,Y) ≤ E[log q(y|x)] - E[log q(y)]
    
    Example:
        >>> loss_fn = MutualInformationLoss(latent_dim=512, num_classes=2)
        >>> loss = loss_fn(latent, sensitive_attrs)
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        name: str = "mutual_information_loss"
    ):
        """
        Initialize MI loss.
        
        Args:
            latent_dim: Dimension of latent representations
            num_classes: Number of sensitive attribute classes
            hidden_dim: Hidden dimension for conditional estimator
            name: Loss name
        """
        super().__init__()
        
        self.name = name
        
        # Conditional distribution estimator q(y|x)
        self.conditional_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Marginal distribution q(y)
        self.marginal_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
    
    def forward(
        self,
        latent: torch.Tensor,
        sensitive_attrs: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute mutual information upper bound.
        
        Args:
            latent: Latent representations
            sensitive_attrs: Sensitive attribute labels
            
        Returns:
            MI upper bound loss
        """
        batch_size = latent.size(0)
        
        # Conditional log-likelihood: log q(y|x)
        conditional_logits = self.conditional_net(latent)
        conditional_log_prob = F.log_softmax(conditional_logits, dim=-1)
        log_q_y_given_x = conditional_log_prob.gather(
            1, sensitive_attrs.unsqueeze(1)
        ).squeeze(1)
        
        # Marginal log-likelihood: log q(y)
        # Shuffle latents to get independent samples
        shuffled_indices = torch.randperm(batch_size, device=latent.device)
        shuffled_latent = latent[shuffled_indices]
        
        marginal_logits = self.marginal_net(shuffled_latent)
        marginal_log_prob = F.log_softmax(marginal_logits, dim=-1)
        log_q_y = marginal_log_prob.gather(
            1, sensitive_attrs.unsqueeze(1)
        ).squeeze(1)
        
        # CLUB upper bound
        mi_upper_bound = (log_q_y_given_x - log_q_y).mean()
        
        return mi_upper_bound
    
    def get_conditional_distribution(
        self,
        latent: torch.Tensor
    ) -> torch.Tensor:
        """Get conditional distribution q(y|x)."""
        logits = self.conditional_net(latent)
        return F.softmax(logits, dim=-1)