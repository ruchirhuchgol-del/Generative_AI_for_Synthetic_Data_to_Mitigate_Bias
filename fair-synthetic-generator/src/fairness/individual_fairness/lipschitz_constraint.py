"""
Lipschitz Constraint for Individual Fairness
=============================================

Implementation of Lipschitz-based individual fairness constraints.

Lipschitz continuity ensures that similar individuals receive similar predictions:
    |f(x_i) - f(x_j)| ≤ λ * d(x_i, x_j)

where λ is the Lipschitz constant and d is a distance metric.
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


class DistanceMetric(Enum):
    """Distance metrics for comparing individuals."""
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    COSINE = "cosine"
    L2 = "l2"
    L1 = "l1"
    MAHALANOBIS = "mahalanobis"


class LipschitzConstraint(BaseFairnessConstraint):
    """
    Lipschitz-based individual fairness constraint.
    
    Enforces that the prediction function is Lipschitz continuous
    with respect to the input features:
        |f(x_i) - f(x_j)| ≤ λ * ||x_i - x_j||
    
    This ensures that similar individuals (by feature distance)
    receive similar predictions.
    
    Mathematical Definition:
        Violation = max_{i,j} [ |f(x_i) - f(x_j)| - λ * d(x_i, x_j) ]
    
    Properties:
        - Uses pairwise comparisons between individuals
        - Requires defining a meaningful similarity metric
        - Can be computationally expensive for large batches
        - Stronger constraint than group fairness
    
    Example:
        >>> constraint = LipschitzConstraint(
        ...     lambda_lipschitz=0.1,
        ...     distance_metric="euclidean"
        ... )
        >>> 
        >>> # Compute violation
        >>> violation = constraint.compute(predictions, features)
        >>> 
        >>> # Training loss
        >>> loss = constraint.loss(predictions, features)
    """
    
    def __init__(
        self,
        lambda_lipschitz: float = 0.1,
        distance_metric: str = "euclidean",
        normalization: str = "min_max",
        relaxation: str = "soft",
        weight: float = 1.0,
        sample_pairs: int = 0,
        use_approximation: bool = True,
        name: str = "lipschitz_constraint"
    ):
        """
        Initialize Lipschitz constraint.
        
        Args:
            lambda_lipschitz: Lipschitz constant bound.
                              Lower values = stricter constraint.
            distance_metric: Distance metric for feature comparison:
                - "euclidean": L2 distance
                - "manhattan": L1 distance
                - "cosine": Cosine distance
                - "mahalanobis": Mahalanobis distance
            normalization: Feature normalization method:
                - "min_max": Min-max normalization
                "standard": Z-score normalization
                - "none": No normalization
            relaxation: Constraint relaxation type
            weight: Constraint weight
            sample_pairs: Number of pairs to sample (0 = all pairs)
            use_approximation: Whether to use efficient approximation
            name: Constraint name
        """
        super().__init__(lambda_lipschitz, relaxation, weight, name)
        
        self._constraint_type = ConstraintType.INDIVIDUAL
        self.lambda_lipschitz = lambda_lipschitz
        self.distance_metric = DistanceMetric(distance_metric)
        self.normalization = normalization
        self.sample_pairs = sample_pairs
        self.use_approximation = use_approximation
        
        # For Mahalanobis distance
        self._cov_inv = None
    
    def _normalize_features(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Normalize features for distance computation."""
        if self.normalization == "none":
            return features
        elif self.normalization == "min_max":
            min_val = features.min(dim=0, keepdim=True)[0]
            max_val = features.max(dim=0, keepdim=True)[0]
            range_val = max_val - min_val + 1e-8
            return (features - min_val) / range_val
        elif self.normalization == "standard":
            mean = features.mean(dim=0, keepdim=True)
            std = features.std(dim=0, keepdim=True) + 1e-8
            return (features - mean) / std
        else:
            return features
    
    def compute_distances(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pairwise distances between features.
        
        Args:
            features: Feature tensor (batch_size, feature_dim)
            
        Returns:
            Distance matrix (batch_size, batch_size)
        """
        normalized = self._normalize_features(features)
        
        if self.distance_metric == DistanceMetric.EUCLIDEAN:
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
            sq_norm = (normalized ** 2).sum(dim=1, keepdim=True)
            dist_sq = sq_norm + sq_norm.t() - 2 * normalized @ normalized.t()
            distances = torch.sqrt(torch.clamp(dist_sq, min=0))
            
        elif self.distance_metric == DistanceMetric.MANHATTAN:
            # |x - y|_1
            diff = normalized.unsqueeze(1) - normalized.unsqueeze(0)
            distances = diff.abs().sum(dim=-1)
            
        elif self.distance_metric == DistanceMetric.COSINE:
            # 1 - cos(x, y)
            normalized = F.normalize(normalized, p=2, dim=1)
            similarity = normalized @ normalized.t()
            distances = 1 - similarity
            
        elif self.distance_metric == DistanceMetric.L2:
            sq_norm = (normalized ** 2).sum(dim=1, keepdim=True)
            dist_sq = sq_norm + sq_norm.t() - 2 * normalized @ normalized.t()
            distances = torch.sqrt(torch.clamp(dist_sq, min=0))
            
        elif self.distance_metric == DistanceMetric.L1:
            diff = normalized.unsqueeze(1) - normalized.unsqueeze(0)
            distances = diff.abs().sum(dim=-1)
            
        elif self.distance_metric == DistanceMetric.MAHALANOBIS:
            if self._cov_inv is None:
                # Compute inverse covariance
                cov = torch.cov(normalized.t())
                self._cov_inv = torch.inverse(cov + 1e-6 * torch.eye(cov.size(0), device=cov.device))
            
            diff = normalized.unsqueeze(1) - normalized.unsqueeze(0)
            distances = torch.sqrt((diff @ self._cov_inv) * diff).sum(dim=-1)
        else:
            # Default to Euclidean
            sq_norm = (normalized ** 2).sum(dim=1, keepdim=True)
            dist_sq = sq_norm + sq_norm.t() - 2 * normalized @ normalized.t()
            distances = torch.sqrt(torch.clamp(dist_sq, min=0))
        
        return distances
    
    def compute(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor,
        groups: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute Lipschitz violation fraction.
        
        Args:
            predictions: Model predictions (batch_size,)
            features: Feature tensor (batch_size, feature_dim)
            groups: Ignored for individual fairness
            labels: Ignored for individual fairness
            
        Returns:
            Fraction of pairs that violate the constraint
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        
        batch_size = predictions.size(0)
        
        # Compute prediction differences
        pred_diff = (predictions.unsqueeze(1) - predictions.unsqueeze(0)).abs()
        
        # Compute feature distances
        feature_dist = self.compute_distances(features)
        
        # Compute violations
        violations = pred_diff - self.lambda_lipschitz * feature_dist
        
        # Mask out diagonal (self-comparisons)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=predictions.device)
        
        # Count violations
        violation_count = (violations[mask] > 0).float().mean().item()
        
        return violation_count
    
    def loss(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor,
        groups: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute Lipschitz constraint loss.
        
        Penalizes pairs where prediction difference exceeds
        λ times the feature distance.
        
        Args:
            predictions: Model predictions
            features: Feature tensor
            groups: Ignored
            labels: Ignored
            
        Returns:
            Loss tensor
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        
        batch_size = predictions.size(0)
        
        if self.use_approximation and batch_size > 32:
            # Use sampled pairs for efficiency
            return self._approximate_loss(predictions, features)
        
        # Compute prediction differences
        pred_diff = (predictions.unsqueeze(1) - predictions.unsqueeze(0)).abs()
        
        # Compute feature distances
        feature_dist = self.compute_distances(features)
        
        # Compute violations (hinge loss)
        violations = torch.relu(pred_diff - self.lambda_lipschitz * feature_dist)
        
        # Weight by feature similarity (closer pairs matter more)
        weights = torch.exp(-feature_dist)
        
        # Mask out diagonal
        mask = 1 - torch.eye(batch_size, device=predictions.device)
        
        # Weighted average violation
        loss = (violations * weights * mask).sum() / (mask.sum() + 1e-8)
        
        return self.apply_relaxation(loss, **kwargs)
    
    def _approximate_loss(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute approximate loss using sampled pairs."""
        batch_size = predictions.size(0)
        n_pairs = self.sample_pairs if self.sample_pairs > 0 else min(1000, batch_size * batch_size)
        
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        for _ in range(n_pairs):
            # Sample random pair
            i = torch.randint(0, batch_size, (1,), device=predictions.device)
            j = torch.randint(0, batch_size, (1,), device=predictions.device)
            
            if i == j:
                continue
            
            pred_diff = (predictions[i] - predictions[j]).abs()
            
            # Compute distance for this pair
            feat_i = self._normalize_features(features[i:i+1])
            feat_j = self._normalize_features(features[j:j+1])
            
            if self.distance_metric in [DistanceMetric.EUCLIDEAN, DistanceMetric.L2]:
                feature_dist = (feat_i - feat_j).norm()
            elif self.distance_metric in [DistanceMetric.MANHATTAN, DistanceMetric.L1]:
                feature_dist = (feat_i - feat_j).abs().sum()
            else:
                feature_dist = (feat_i - feat_j).norm()
            
            violation = torch.relu(pred_diff - self.lambda_lipschitz * feature_dist)
            total_loss = total_loss + violation
        
        return total_loss / n_pairs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        d = super().to_dict()
        d.update({
            "lambda_lipschitz": self.lambda_lipschitz,
            "distance_metric": self.distance_metric.value,
            "normalization": self.normalization,
            "sample_pairs": self.sample_pairs,
            "use_approximation": self.use_approximation
        })
        return d
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LipschitzConstraint":
        """Create from dictionary configuration."""
        return cls(
            lambda_lipschitz=config.get("lambda_lipschitz", 0.1),
            distance_metric=config.get("distance_metric", "euclidean"),
            normalization=config.get("normalization", "min_max"),
            relaxation=config.get("relaxation", "soft"),
            weight=config.get("weight", 1.0),
            sample_pairs=config.get("sample_pairs", 0),
            use_approximation=config.get("use_approximation", True),
            name=config.get("name", "lipschitz_constraint")
        )
    
    def __repr__(self) -> str:
        return (
            f"LipschitzConstraint("
            f"lambda={self.lambda_lipschitz}, "
            f"metric={self.distance_metric.value}, "
            f"relaxation={self.relaxation.value})"
        )


class AdaptiveLipschitzConstraint(LipschitzConstraint):
    """
    Adaptive Lipschitz constraint with learned distance metric.
    
    Instead of using a fixed distance metric, learns a feature
    transformation that defines similarity:
        d(x_i, x_j) = ||W * x_i - W * x_j||
    
    This allows the constraint to learn task-relevant similarities.
    
    Example:
        >>> constraint = AdaptiveLipschitzConstraint(
        ...     feature_dim=100,
        ...     lambda_lipschitz=0.1
        ... )
    """
    
    def __init__(
        self,
        feature_dim: int,
        lambda_lipschitz: float = 0.1,
        hidden_dim: int = 64,
        relaxation: str = "soft",
        name: str = "adaptive_lipschitz_constraint",
        **kwargs
    ):
        """
        Initialize adaptive Lipschitz constraint.
        
        Args:
            feature_dim: Input feature dimension
            lambda_lipschitz: Lipschitz constant
            hidden_dim: Hidden dimension for transformation
            relaxation: Constraint relaxation type
            name: Constraint name
        """
        super().__init__(
            lambda_lipschitz=lambda_lipschitz,
            relaxation=relaxation,
            name=name,
            **kwargs
        )
        
        # Learnable distance transformation
        self.distance_transform = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Initialize to identity-like transformation
        nn.init.eye_(self.distance_transform[-1].weight)
    
    def compute_distances(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute distances using learned transformation."""
        transformed = self.distance_transform(features)
        
        # Euclidean distance in transformed space
        sq_norm = (transformed ** 2).sum(dim=1, keepdim=True)
        dist_sq = sq_norm + sq_norm.t() - 2 * transformed @ transformed.t()
        distances = torch.sqrt(torch.clamp(dist_sq, min=0))
        
        return distances


class FeatureWeightedLipschitzConstraint(LipschitzConstraint):
    """
    Feature-weighted Lipschitz constraint.
    
    Applies different weights to different features when computing
    distances, allowing domain knowledge to guide similarity:
        d(x_i, x_j) = sqrt(Σ w_k * (x_ik - x_jk)^2)
    
    Example:
        >>> constraint = FeatureWeightedLipschitzConstraint(
        ...     feature_weights=[1.0, 0.5, 0.1, ...],
        ...     lambda_lipschitz=0.1
        ... )
    """
    
    def __init__(
        self,
        feature_weights: List[float],
        lambda_lipschitz: float = 0.1,
        learnable: bool = False,
        relaxation: str = "soft",
        name: str = "feature_weighted_lipschitz",
        **kwargs
    ):
        """
        Initialize feature-weighted Lipschitz constraint.
        
        Args:
            feature_weights: Weights for each feature dimension
            lambda_lipschitz: Lipschitz constant
            learnable: Whether weights are learnable
            relaxation: Constraint relaxation type
            name: Constraint name
        """
        super().__init__(
            lambda_lipschitz=lambda_lipschitz,
            relaxation=relaxation,
            name=name,
            **kwargs
        )
        
        weights_tensor = torch.tensor(feature_weights)
        
        if learnable:
            self.feature_weights = nn.Parameter(weights_tensor)
        else:
            self.register_buffer("feature_weights", weights_tensor)
    
    def compute_distances(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted distances."""
        # Apply feature weights
        weighted_features = features * self.feature_weights.to(features.device)
        
        # Euclidean distance with weighted features
        sq_norm = (weighted_features ** 2).sum(dim=1, keepdim=True)
        dist_sq = sq_norm + sq_norm.t() - 2 * weighted_features @ weighted_features.t()
        distances = torch.sqrt(torch.clamp(dist_sq, min=0))
        
        return distances
    
    def get_feature_importance(self) -> Dict[int, float]:
        """Get feature importance from weights."""
        weights = self.feature_weights.detach().cpu().numpy()
        return {i: float(w) for i, w in enumerate(weights)}
