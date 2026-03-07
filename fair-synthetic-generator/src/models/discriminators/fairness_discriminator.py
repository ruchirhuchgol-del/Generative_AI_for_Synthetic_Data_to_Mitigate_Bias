"""
Fairness Discriminator
======================

Discriminators for fairness-aware adversarial training.
Supports adversarial debiasing for group, individual, and counterfactual fairness.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.discriminators.base_discriminator import (
    BaseDiscriminator,
    BinaryDiscriminator,
    MultiClassDiscriminator
)


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) function.
    
    During forward pass: identity
    During backward pass: -λ * gradient
    
    This allows adversarial training by maximizing the adversary loss
    while minimizing the main model's prediction of sensitive attributes.
    """
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output * (-ctx.lambda_), None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module.
    
    Wraps the GRL function for use in nn.Sequential.
    
    Example:
        >>> grl = GradientReversalLayer(lambda_=1.0)
        >>> x_reversed = grl(x)  # Forward: identity, Backward: -λ * grad
    """
    
    def __init__(self, lambda_: float = 1.0):
        """
        Initialize GRL.
        
        Args:
            lambda_: Gradient reversal strength
        """
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal."""
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_: float) -> None:
        """Update lambda value."""
        self.lambda_ = lambda_


class SensitiveAttributeDiscriminator(BaseDiscriminator):
    """
    Discriminator for predicting sensitive attributes from representations.
    
    Used in adversarial debiasing to ensure representations are
    invariant to sensitive attributes.
    
    Architecture:
        Representation → GRL → MLP → Sensitive Attribute Prediction
        
    Training Objective:
        Main model minimizes adversary accuracy (via GRL)
        Adversary maximizes prediction accuracy
        
    Example:
        >>> adversary = SensitiveAttributeDiscriminator(
        ...     input_dim=512,
        ...     sensitive_attribute="gender",
        ...     num_classes=2
        ... )
        >>> # During training:
        >>> pred = adversary(z)  # z from main model
        >>> loss = F.cross_entropy(pred, sensitive_labels)
    """
    
    def __init__(
        self,
        input_dim: int,
        sensitive_attribute: str,
        num_classes: int = 2,
        hidden_dims: List[int] = [256, 128],
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        use_gradient_reversal: bool = True,
        grl_lambda: float = 1.0,
        spectral_norm: bool = False,
        name: str = "sensitive_attribute_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize sensitive attribute discriminator.
        
        Args:
            input_dim: Input representation dimension
            sensitive_attribute: Name of sensitive attribute
            num_classes: Number of classes (2 for binary)
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            dropout: Dropout rate
            use_gradient_reversal: Apply gradient reversal layer
            grl_lambda: GRL lambda value
            spectral_norm: Apply spectral normalization
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.input_dim = input_dim
        self.sensitive_attribute = sensitive_attribute
        self.num_classes = num_classes
        self.use_gradient_reversal = use_gradient_reversal
        
        # Gradient reversal layer
        if use_gradient_reversal:
            self.grl = GradientReversalLayer(grl_lambda)
        else:
            self.grl = None
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            linear = nn.Linear(prev_dim, dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LayerNorm(dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.layers = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(prev_dim, num_classes)
        if spectral_norm:
            self.output = nn.utils.spectral_norm(self.output)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute sensitive attribute prediction.
        
        Args:
            x: Representation tensor (batch, input_dim)
            
        Returns:
            Class logits (batch, num_classes)
        """
        if self.grl is not None:
            x = self.grl(x)
        
        h = self.layers(x)
        return self.output(h)
    
    def compute_loss(
        self,
        z: torch.Tensor,
        sensitive_labels: torch.Tensor,
        return_accuracy: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        """
        Compute adversarial loss.
        
        Args:
            z: Representation tensor
            sensitive_labels: Ground truth sensitive attribute labels
            return_accuracy: Also return prediction accuracy
            
        Returns:
            Loss (and accuracy if requested)
        """
        logits = self.discriminate(z)
        loss = F.cross_entropy(logits, sensitive_labels)
        
        if return_accuracy:
            with torch.no_grad():
                predictions = logits.argmax(dim=-1)
                accuracy = (predictions == sensitive_labels).float().mean().item()
            return loss, accuracy
        
        return loss
    
    def set_grl_lambda(self, lambda_: float) -> None:
        """Update gradient reversal lambda."""
        if self.grl is not None:
            self.grl.set_lambda(lambda_)


class MultiSensitiveDiscriminator(BaseDiscriminator):
    """
    Discriminator for multiple sensitive attributes simultaneously.
    
    Predicts all sensitive attributes from a shared representation.
    Useful when multiple fairness constraints need to be enforced.
    
    Example:
        >>> adversary = MultiSensitiveDiscriminator(
        ...     input_dim=512,
        ...     sensitive_attributes={
        ...         "gender": 2,
        ...         "race": 5,
        ...         "age_group": 4
        ...     }
        ... )
        >>> preds = adversary(z)
        >>> # preds = {"gender": ..., "race": ..., "age_group": ...}
    """
    
    def __init__(
        self,
        input_dim: int,
        sensitive_attributes: Dict[str, int],
        shared_dims: List[int] = [256],
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        use_gradient_reversal: bool = True,
        grl_lambda: float = 1.0,
        spectral_norm: bool = False,
        name: str = "multi_sensitive_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize multi-sensitive discriminator.
        
        Args:
            input_dim: Input dimension
            sensitive_attributes: Dict mapping attribute names to num_classes
            shared_dims: Shared hidden dimensions
            activation: Activation function
            dropout: Dropout rate
            use_gradient_reversal: Use GRL
            grl_lambda: GRL lambda
            spectral_norm: Apply spectral normalization
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.input_dim = input_dim
        self.sensitive_attributes = sensitive_attributes
        
        # Gradient reversal
        if use_gradient_reversal:
            self.grl = GradientReversalLayer(grl_lambda)
        else:
            self.grl = None
        
        # Shared layers
        layers = []
        prev_dim = input_dim
        
        for dim in shared_dims:
            linear = nn.Linear(prev_dim, dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LayerNorm(dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Attribute-specific heads
        self.heads = nn.ModuleDict()
        for attr_name, num_classes in sensitive_attributes.items():
            head = nn.Sequential(
                nn.Linear(prev_dim, prev_dim // 2),
                nn.LayerNorm(prev_dim // 2),
                self._get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(prev_dim // 2, num_classes)
            )
            if spectral_norm:
                head[-1] = nn.utils.spectral_norm(head[-1])
            self.heads[attr_name] = head
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def discriminate(
        self, 
        x: torch.Tensor,
        attributes: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute predictions for sensitive attributes.
        
        Args:
            x: Representation tensor
            attributes: Subset of attributes to predict (default: all)
            
        Returns:
            Dictionary of logits per attribute
        """
        if self.grl is not None:
            x = self.grl(x)
        
        h = self.shared_layers(x)
        
        if attributes is None:
            attributes = list(self.heads.keys())
        
        outputs = {}
        for attr in attributes:
            if attr in self.heads:
                outputs[attr] = self.heads[attr](h)
        
        return outputs
    
    def compute_loss(
        self,
        z: torch.Tensor,
        sensitive_labels: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-attribute adversarial loss.
        
        Args:
            z: Representation tensor
            sensitive_labels: Dict of ground truth labels per attribute
            weights: Optional per-attribute weights
            
        Returns:
            Tuple of (total_loss, accuracy_dict)
        """
        predictions = self.discriminate(z)
        
        total_loss = 0.0
        accuracies = {}
        
        if weights is None:
            weights = {attr: 1.0 for attr in self.sensitive_attributes}
        
        for attr, logits in predictions.items():
            if attr in sensitive_labels:
                labels = sensitive_labels[attr]
                loss = F.cross_entropy(logits, labels)
                total_loss = total_loss + weights.get(attr, 1.0) * loss
                
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    accuracies[attr] = (preds == labels).float().mean().item()
        
        return total_loss, accuracies


class FairnessAdversary(BaseDiscriminator):
    """
    Complete fairness adversary module for adversarial debiasing.
    
    Combines:
    - Gradient reversal for training
    - Multiple sensitive attribute prediction
    - Fairness-aware loss computation
    
    Training Process:
        1. Main model forward: x → z → y_pred
        2. Adversary forward: z → sensitive_pred
        3. Main model backward: -λ * ∂L_adversary/∂z
        4. Adversary backward: +∂L_adversary/∂θ_adv
        
    Example:
        >>> adversary = FairnessAdversary(
        ...     input_dim=512,
        ...     sensitive_configs=[
        ...         {"name": "gender", "num_classes": 2},
        ...         {"name": "race", "num_classes": 5}
        ...     ]
        ... )
        >>> loss, metrics = adversary.compute_fairness_loss(z, labels)
    """
    
    def __init__(
        self,
        input_dim: int,
        sensitive_configs: List[Dict[str, Any]],
        hidden_dims: List[int] = [256, 128],
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        grl_lambda: float = 1.0,
        lambda_schedule: str = "constant",  # constant, linear, cyclic
        max_iterations: int = 10000,
        spectral_norm: bool = False,
        name: str = "fairness_adversary",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize fairness adversary.
        
        Args:
            input_dim: Input dimension
            sensitive_configs: List of sensitive attribute configs
            hidden_dims: Hidden dimensions
            activation: Activation function
            dropout: Dropout rate
            grl_lambda: Base GRL lambda
            lambda_schedule: Lambda scheduling strategy
            max_iterations: Max training iterations (for scheduling)
            spectral_norm: Apply spectral normalization
            name: Adversary name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.input_dim = input_dim
        self.sensitive_configs = sensitive_configs
        self.lambda_schedule = lambda_schedule
        self.max_iterations = max_iterations
        self.base_lambda = grl_lambda
        
        # Create attribute dict
        self.sensitive_attrs = {
            cfg["name"]: cfg["num_classes"]
            for cfg in sensitive_configs
        }
        
        # GRL
        self.grl = GradientReversalLayer(grl_lambda)
        
        # Shared feature extractor
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            linear = nn.Linear(prev_dim, dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.LayerNorm(dim))
            layers.append(self._get_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.shared = nn.Sequential(*layers)
        
        # Attribute heads
        self.heads = nn.ModuleDict()
        for cfg in sensitive_configs:
            self.heads[cfg["name"]] = nn.Linear(prev_dim, cfg["num_classes"])
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def update_lambda(self, iteration: int) -> float:
        """
        Update GRL lambda based on schedule.
        
        Args:
            iteration: Current training iteration
            
        Returns:
            Updated lambda value
        """
        if self.lambda_schedule == "constant":
            lambda_ = self.base_lambda
        
        elif self.lambda_schedule == "linear":
            progress = min(1.0, iteration / self.max_iterations)
            lambda_ = self.base_lambda * (2.0 / (1.0 + math.exp(-10 * progress)) - 1.0)
        
        elif self.lambda_schedule == "cyclic":
            cycle_length = self.max_iterations // 4
            cycle_pos = iteration % cycle_length
            lambda_ = self.base_lambda * (cycle_pos / cycle_length)
        
        else:
            lambda_ = self.base_lambda
        
        self.grl.set_lambda(lambda_)
        return lambda_
    
    def discriminate(
        self,
        x: torch.Tensor,
        attributes: Optional[List[str]] = None,
        apply_grl: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute sensitive attribute predictions.
        
        Args:
            x: Representation tensor
            attributes: Subset of attributes to predict
            apply_grl: Whether to apply GRL
            
        Returns:
            Dict of logits per attribute
        """
        if apply_grl:
            x = self.grl(x)
        
        h = self.shared(x)
        
        if attributes is None:
            attributes = list(self.heads.keys())
        
        outputs = {}
        for attr in attributes:
            if attr in self.heads:
                outputs[attr] = self.heads[attr](h)
        
        return outputs
    
    def compute_fairness_loss(
        self,
        z: torch.Tensor,
        sensitive_labels: Dict[str, torch.Tensor],
        iteration: Optional[int] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute complete fairness loss with metrics.
        
        Args:
            z: Representation tensor
            sensitive_labels: Ground truth labels per attribute
            iteration: Current iteration (for lambda scheduling)
            weights: Per-attribute weights
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        # Update lambda if needed
        if iteration is not None:
            current_lambda = self.update_lambda(iteration)
        else:
            current_lambda = self.grl.lambda_
        
        # Get predictions
        predictions = self.discriminate(z)
        
        # Compute loss and metrics
        total_loss = 0.0
        metrics = {
            "adversary_loss": 0.0,
            "current_lambda": current_lambda,
        }
        
        if weights is None:
            weights = {attr: 1.0 for attr in self.sensitive_attrs}
        
        for attr, logits in predictions.items():
            if attr in sensitive_labels:
                labels = sensitive_labels[attr]
                loss = F.cross_entropy(logits, labels)
                total_loss = total_loss + weights.get(attr, 1.0) * loss
                
                # Metrics
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == labels).float().mean().item()
                    metrics[f"{attr}_accuracy"] = acc
                    metrics[f"{attr}_loss"] = loss.item()
        
        metrics["adversary_loss"] = total_loss.item()
        
        return total_loss, metrics


class ContrastiveFairnessDiscriminator(BaseDiscriminator):
    """
    Contrastive discriminator for individual fairness.
    
    Ensures similar individuals receive similar outcomes by
    learning to distinguish positive pairs (similar) from negative pairs.
    
    Loss: InfoNCE / Contrastive loss
    
    Example:
        >>> discriminator = ContrastiveFairnessDiscriminator(
        ...     input_dim=512,
        ...     projection_dim=128
        ... )
        >>> loss = discriminator.compute_loss(z_i, z_j, is_similar)
    """
    
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        temperature: float = 0.1,
        name: str = "contrastive_fairness_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize contrastive discriminator.
        
        Args:
            input_dim: Input dimension
            projection_dim: Projection dimension for contrastive space
            hidden_dim: Hidden dimension
            temperature: Temperature for contrastive loss
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.temperature = temperature
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim)
        )
    
    def discriminate(
        self, 
        x: torch.Tensor,
        return_normalized: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Project to contrastive space.
        
        Args:
            x: Input tensor
            return_normalized: Return L2-normalized projection
            
        Returns:
            Projected tensor
        """
        z = self.projection(x)
        if return_normalized:
            z = F.normalize(z, dim=-1)
        return z
    
    def compute_loss(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
        is_similar: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute contrastive fairness loss.
        
        Args:
            z_i: First set of representations
            z_j: Second set of representations
            is_similar: Binary tensor (1=similar pair, 0=dissimilar)
            
        Returns:
            Contrastive loss
        """
        # Project
        h_i = self.discriminate(z_i)
        h_j = self.discriminate(z_j)
        
        batch_size = h_i.shape[0]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(h_i, h_j.T) / self.temperature
        
        if is_similar is not None:
            # Supervised contrastive loss
            # Positive pairs
            pos_mask = is_similar.float()
            # Negative pairs
            neg_mask = 1.0 - pos_mask
            
            # InfoNCE-style loss
            pos_sim = (sim_matrix * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
            neg_sim = torch.logsumexp(sim_matrix + torch.log(neg_mask + 1e-8), dim=1)
            
            loss = -pos_sim + neg_sim
            loss = loss[pos_mask.sum(dim=1) > 0].mean() if (pos_mask.sum(dim=1) > 0).any() else loss.mean()
        else:
            # Standard contrastive loss (SimCLR-style)
            # Positive pairs are diagonal
            labels = torch.arange(batch_size, device=z_i.device)
            loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def compute_similarity(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity scores between representations.
        
        Args:
            z_i: First representations
            z_j: Second representations
            
        Returns:
            Similarity scores
        """
        h_i = self.discriminate(z_i)
        h_j = self.discriminate(z_j)
        return torch.mm(h_i, h_j.T)


class CounterfactualDiscriminator(BaseDiscriminator):
    """
    Discriminator for counterfactual fairness.
    
    Ensures that changing sensitive attributes does not change predictions
    for non-sensitive outcomes.
    
    Architecture:
        z_counterfactual → Predict original sensitive attribute
        If prediction is random, counterfactual fairness is achieved.
    
    Example:
        >>> discriminator = CounterfactualDiscriminator(
        ...     input_dim=512,
        ...     num_sensitive_classes=2
        ... )
        >>> loss = discriminator.compute_counterfactual_loss(z_original, z_counterfactual)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_sensitive_classes: int = 2,
        hidden_dims: List[int] = [256, 128],
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        name: str = "counterfactual_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize counterfactual discriminator.
        
        Args:
            input_dim: Input dimension
            num_sensitive_classes: Number of sensitive attribute classes
            hidden_dims: Hidden dimensions
            activation: Activation function
            dropout: Dropout rate
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, num_sensitive_classes)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Predict sensitive attribute from representation.
        
        Args:
            x: Representation tensor
            
        Returns:
            Class logits
        """
        h = self.layers(x)
        return self.output(h)
    
    def compute_counterfactual_loss(
        self,
        z_original: torch.Tensor,
        z_counterfactual: torch.Tensor,
        sensitive_original: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute counterfactual fairness loss.
        
        Loss encourages predictions on counterfactual representations
        to match original sensitive attribute (should be independent).
        
        Args:
            z_original: Original representations
            z_counterfactual: Counterfactual representations
            sensitive_original: Original sensitive attribute values
            
        Returns:
            Tuple of (loss, metrics)
        """
        # Predictions on original
        logits_orig = self.discriminate(z_original)
        loss_orig = F.cross_entropy(logits_orig, sensitive_original)
        
        # Predictions on counterfactual (should be random/uniform)
        logits_cf = self.discriminate(z_counterfactual)
        
        # Encourage uniform distribution on counterfactual
        uniform_target = torch.ones_like(logits_cf) / logits_cf.shape[-1]
        loss_cf = F.kl_div(
            F.log_softmax(logits_cf, dim=-1),
            uniform_target,
            reduction="batchmean"
        )
        
        # Total loss
        total_loss = loss_orig + loss_cf
        
        # Metrics
        with torch.no_grad():
            acc_orig = (logits_orig.argmax(dim=-1) == sensitive_original).float().mean().item()
            entropy_cf = -(
                F.softmax(logits_cf, dim=-1) * F.log_softmax(logits_cf, dim=-1)
            ).sum(dim=-1).mean().item()
            max_entropy = math.log(logits_cf.shape[-1])
        
        metrics = {
            "original_accuracy": acc_orig,
            "counterfactual_entropy": entropy_cf,
            "max_entropy": max_entropy,
            "counterfactual_uniformity": entropy_cf / max_entropy
        }
        
        return total_loss, metrics


class DomainDiscriminator(BaseDiscriminator):
    """
    Domain discriminator for domain adaptation in fairness.
    
    Used to ensure representations are invariant across different
    domains/datasets while maintaining fairness constraints.
    
    Architecture:
        Representation → GRL → Domain Classifier
        
    Example:
        >>> discriminator = DomainDiscriminator(
        ...     input_dim=512,
        ...     num_domains=3
        ... )
        >>> loss = discriminator.compute_domain_loss(z, domain_labels)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_domains: int = 2,
        hidden_dims: List[int] = [256, 128],
        activation: str = "leaky_relu",
        dropout: float = 0.1,
        grl_lambda: float = 1.0,
        name: str = "domain_discriminator",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize domain discriminator.
        
        Args:
            input_dim: Input dimension
            num_domains: Number of domains
            hidden_dims: Hidden dimensions
            activation: Activation function
            dropout: Dropout rate
            grl_lambda: GRL lambda
            name: Discriminator name
            config: Optional configuration
        """
        super().__init__(name=name, config=config)
        
        self.num_domains = num_domains
        
        # Gradient reversal
        self.grl = GradientReversalLayer(grl_lambda)
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                self._get_activation(activation),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, num_domains)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "gelu": nn.GELU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))
    
    def discriminate(self, x: torch.Tensor, apply_grl: bool = True, **kwargs) -> torch.Tensor:
        """
        Predict domain from representation.
        
        Args:
            x: Representation tensor
            apply_grl: Apply gradient reversal
            
        Returns:
            Domain logits
        """
        if apply_grl:
            x = self.grl(x)
        h = self.layers(x)
        return self.output(h)
    
    def compute_domain_loss(
        self,
        z: torch.Tensor,
        domain_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute domain classification loss.
        
        Args:
            z: Representation tensor
            domain_labels: Domain labels
            
        Returns:
            Tuple of (loss, accuracy)
        """
        logits = self.discriminate(z)
        loss = F.cross_entropy(logits, domain_labels)
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == domain_labels).float().mean().item()
        
        return loss, accuracy
