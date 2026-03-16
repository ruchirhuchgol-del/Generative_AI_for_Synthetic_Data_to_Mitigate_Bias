"""
Consistency Constraint for Individual Fairness
==============================================

Implementation of consistency-based individual fairness constraints.

Consistency measures how consistent predictions are for similar individuals,
typically measured using k-nearest neighbors or local neighborhoods.
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


class ConsistencyConstraint(BaseFairnessConstraint):
    """
    Consistency-based individual fairness constraint.
    
    Measures how consistent predictions are for similar individuals:
        Consistency = 1 - (1/n) * Σ|f(x_i) - f(N_k(x_i))|
    
    where N_k(x_i) is the average of k nearest neighbors' predictions.
    
    Mathematical Definition:
        For each sample i with k nearest neighbors:
        consistency_i = 1 - |f(x_i) - mean(f(x_j) for j in neighbors(i))|
    
    Higher consistency indicates better individual fairness.
    
    Properties:
        - Based on local neighborhood consistency
        - Does not require explicit distance metric
        - Computationally efficient with k-NN
        - Captures local smoothness of predictions
    
    Example:
        >>> constraint = ConsistencyConstraint(k_neighbors=10)
        >>> 
        >>> # Compute consistency
        >>> consistency = constraint.compute(predictions, features)
        >>> 
        >>> # Training loss
        >>> loss = constraint.loss(predictions, features)
    """
    
    def __init__(
        self,
        k_neighbors: int = 10,
        similarity_threshold: float = 0.9,
        relaxation: str = "soft",
        weight: float = 1.0,
        distance_metric: str = "euclidean",
        use_weights: bool = True,
        name: str = "consistency_constraint"
    ):
        """
        Initialize consistency constraint.
        
        Args:
            k_neighbors: Number of nearest neighbors to consider
            similarity_threshold: Threshold for considering individuals similar
            relaxation: Constraint relaxation type
            weight: Constraint weight
            distance_metric: Distance metric for finding neighbors
            use_weights: Whether to weight neighbors by similarity
            name: Constraint name
        """
        super().__init__(similarity_threshold, relaxation, weight, name)
        
        self._constraint_type = ConstraintType.INDIVIDUAL
        self.k_neighbors = k_neighbors
        self.similarity_threshold = similarity_threshold
        self.distance_metric = distance_metric
        self.use_weights = use_weights
    
    def _compute_distances(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        """Compute pairwise distances."""
        if self.distance_metric == "euclidean":
            sq_norm = (features ** 2).sum(dim=1, keepdim=True)
            dist_sq = sq_norm + sq_norm.t() - 2 * features @ features.t()
            return torch.sqrt(torch.clamp(dist_sq, min=0))
        elif self.distance_metric == "cosine":
            normalized = F.normalize(features, p=2, dim=1)
            similarity = normalized @ normalized.t()
            return 1 - similarity
        else:
            sq_norm = (features ** 2).sum(dim=1, keepdim=True)
            dist_sq = sq_norm + sq_norm.t() - 2 * features @ features.t()
            return torch.sqrt(torch.clamp(dist_sq, min=0))
    
    def compute(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor,
        groups: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute consistency score.
        
        Args:
            predictions: Model predictions (batch_size,)
            features: Feature tensor (batch_size, feature_dim)
            groups: Ignored for individual fairness
            labels: Ignored for individual fairness
            
        Returns:
            Consistency score between 0 and 1 (higher is better)
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        
        batch_size = predictions.size(0)
        k = min(self.k_neighbors, batch_size - 1)
        
        if k <= 0:
            return 1.0
        
        # Compute feature distances
        distances = self._compute_distances(features)
        
        # Normalize to similarities
        max_dist = distances.max() + 1e-8
        similarities = 1 - distances / max_dist
        
        # Get k nearest neighbors
        _, indices = torch.topk(similarities, k + 1, dim=1)
        indices = indices[:, 1:]  # Exclude self
        
        # Compute consistency
        total_diff = 0.0
        
        for i in range(batch_size):
            neighbor_preds = predictions[indices[i]]
            diff = torch.abs(predictions[i] - neighbor_preds).mean().item()
            total_diff += diff
        
        consistency = 1 - total_diff / batch_size
        
        return max(0, consistency)
    
    def loss(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor,
        groups: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute consistency loss.
        
        Penalizes inconsistent predictions for similar individuals.
        
        Args:
            predictions: Model predictions
            features: Feature tensor
            groups: Ignored
            labels: Ignored
            
        Returns:
            Loss tensor (to minimize for higher consistency)
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        
        batch_size = predictions.size(0)
        k = min(self.k_neighbors, batch_size - 1)
        
        if k <= 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute distances and similarities
        distances = self._compute_distances(features)
        max_dist = distances.max() + 1e-8
        similarities = 1 - distances / max_dist
        
        # Get k nearest neighbors
        _, indices = torch.topk(similarities, k + 1, dim=1)
        indices = indices[:, 1:]  # Exclude self
        
        # Compute weighted inconsistency
        total_loss = torch.tensor(0.0, device=predictions.device)
        
        for i in range(batch_size):
            neighbor_preds = predictions[indices[i]]
            neighbor_sims = similarities[i, indices[i]]
            
            # Prediction differences
            diffs = torch.abs(predictions[i] - neighbor_preds)
            
            if self.use_weights:
                # Weight by similarity
                weights = neighbor_sims / (neighbor_sims.sum() + 1e-8)
                weighted_diff = (diffs * weights).sum()
            else:
                weighted_diff = diffs.mean()
            
            total_loss = total_loss + weighted_diff
        
        # Average over batch
        total_loss = total_loss / batch_size
        
        return self.apply_relaxation(total_loss, **kwargs)
    
    def get_neighbor_consistency(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get per-sample consistency scores.
        
        Args:
            predictions: Model predictions
            features: Feature tensor
            
        Returns:
            Dictionary with consistency statistics
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        
        batch_size = predictions.size(0)
        k = min(self.k_neighbors, batch_size - 1)
        
        if k <= 0:
            return {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0}
        
        distances = self._compute_distances(features)
        max_dist = distances.max() + 1e-8
        similarities = 1 - distances / max_dist
        
        _, indices = torch.topk(similarities, k + 1, dim=1)
        indices = indices[:, 1:]
        
        consistencies = []
        
        for i in range(batch_size):
            neighbor_preds = predictions[indices[i]]
            diff = torch.abs(predictions[i] - neighbor_preds).mean().item()
            consistencies.append(1 - diff)
        
        consistencies = np.array(consistencies)
        
        return {
            "mean": float(consistencies.mean()),
            "std": float(consistencies.std()),
            "min": float(consistencies.min()),
            "max": float(consistencies.max())
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        d = super().to_dict()
        d.update({
            "k_neighbors": self.k_neighbors,
            "similarity_threshold": self.similarity_threshold,
            "distance_metric": self.distance_metric,
            "use_weights": self.use_weights
        })
        return d
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ConsistencyConstraint":
        """Create from dictionary configuration."""
        return cls(
            k_neighbors=config.get("k_neighbors", 10),
            similarity_threshold=config.get("similarity_threshold", 0.9),
            relaxation=config.get("relaxation", "soft"),
            weight=config.get("weight", 1.0),
            distance_metric=config.get("distance_metric", "euclidean"),
            use_weights=config.get("use_weights", True),
            name=config.get("name", "consistency_constraint")
        )
    
    def __repr__(self) -> str:
        return (
            f"ConsistencyConstraint("
            f"k_neighbors={self.k_neighbors}, "
            f"threshold={self.similarity_threshold})"
        )


class LocalFairnessConstraint(BaseFairnessConstraint):
    """
    Local Fairness constraint.
    
    Combines group fairness with local neighborhood structure.
    Ensures fairness within local neighborhoods:
        For each neighborhood N:
            P(Ŷ=1|A=a, x∈N) ≈ P(Ŷ=1|A=b, x∈N)
    
    This is a relaxation of global group fairness that allows
    for local variations.
    
    Example:
        >>> constraint = LocalFairnessConstraint(
        ...     k_neighbors=20,
        ...     base_constraint=DemographicParity()
        ... )
    """
    
    def __init__(
        self,
        k_neighbors: int = 20,
        base_constraint: Optional[BaseFairnessConstraint] = None,
        relaxation: str = "soft",
        weight: float = 1.0,
        name: str = "local_fairness_constraint"
    ):
        """
        Initialize local fairness constraint.
        
        Args:
            k_neighbors: Size of local neighborhood
            base_constraint: Base group fairness constraint to apply locally
            relaxation: Constraint relaxation type
            weight: Constraint weight
            name: Constraint name
        """
        super().__init__(0.05, relaxation, weight, name)
        
        self._constraint_type = ConstraintType.INDIVIDUAL
        self.k_neighbors = k_neighbors
        self.base_constraint = base_constraint
    
    def compute(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute local fairness violation.
        
        Returns average of base constraint violations within neighborhoods.
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        
        batch_size = predictions.size(0)
        k = min(self.k_neighbors, batch_size - 1)
        
        if k <= 0:
            return 0.0
        
        # Compute distances
        sq_norm = (features ** 2).sum(dim=1, keepdim=True)
        dist_sq = sq_norm + sq_norm.t() - 2 * features @ features.t()
        distances = torch.sqrt(torch.clamp(dist_sq, min=0))
        
        # Get neighborhoods
        _, indices = torch.topk(-distances, k + 1, dim=1)
        indices = indices[:, 1:]  # Exclude self
        
        # Compute local fairness violations
        violations = []
        
        for i in range(batch_size):
            neighborhood_idx = indices[i].tolist() + [i]
            
            local_preds = predictions[neighborhood_idx]
            local_groups = groups[neighborhood_idx]
            local_labels = labels[neighborhood_idx] if labels is not None else None
            
            if self.base_constraint is not None:
                violation = self.base_constraint.compute(
                    local_preds, local_groups, local_labels
                )
            else:
                # Default: compute demographic parity in neighborhood
                unique_groups = torch.unique(local_groups)
                if len(unique_groups) < 2:
                    continue
                
                rates = []
                for g in unique_groups:
                    mask = local_groups == g
                    if mask.sum() > 0:
                        rates.append(local_preds[mask].mean().item())
                
                if len(rates) >= 2:
                    violations.append(max(rates) - min(rates))
        
        if len(violations) == 0:
            return 0.0
        
        return float(np.mean(violations))
    
    def loss(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor,
        groups: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Compute local fairness loss."""
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        
        batch_size = predictions.size(0)
        k = min(self.k_neighbors, batch_size - 1)
        
        if k <= 0:
            return torch.tensor(0.0, device=predictions.device)
        
        # Compute distances
        sq_norm = (features ** 2).sum(dim=1, keepdim=True)
        dist_sq = sq_norm + sq_norm.t() - 2 * features @ features.t()
        distances = torch.sqrt(torch.clamp(dist_sq, min=0))
        
        # Get neighborhoods
        _, indices = torch.topk(-distances, k + 1, dim=1)
        indices = indices[:, 1:]
        
        # Compute local losses
        total_loss = torch.tensor(0.0, device=predictions.device)
        count = 0
        
        for i in range(batch_size):
            neighborhood_idx = indices[i].tolist() + [i]
            
            local_preds = predictions[neighborhood_idx]
            local_groups = groups[neighborhood_idx]
            local_labels = labels[neighborhood_idx] if labels is not None else None
            
            if self.base_constraint is not None:
                local_loss = self.base_constraint.loss(
                    local_preds, local_groups, local_labels
                )
                total_loss = total_loss + local_loss
                count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        return self.apply_relaxation(total_loss, **kwargs)


class SmoothedConsistencyConstraint(ConsistencyConstraint):
    """
    Smoothed Consistency constraint.
    
    Uses kernel density estimation for smoother neighbor weighting:
        Loss = Σ_i Σ_j K(x_i, x_j) * |f(x_i) - f(x_j)|
    
    where K is a smooth kernel function (e.g., Gaussian RBF).
    
    Example:
        >>> constraint = SmoothedConsistencyConstraint(
        ...     bandwidth=1.0,
        ...     kernel="rbf"
        ... )
    """
    
    def __init__(
        self,
        bandwidth: float = 1.0,
        kernel: str = "rbf",
        k_neighbors: int = 10,
        relaxation: str = "soft",
        name: str = "smoothed_consistency",
        **kwargs
    ):
        """
        Initialize smoothed consistency constraint.
        
        Args:
            bandwidth: Kernel bandwidth (sigma for RBF)
            kernel: Kernel type ("rbf", "laplacian", "uniform")
            k_neighbors: Number of neighbors to consider
            relaxation: Constraint relaxation type
            name: Constraint name
        """
        super().__init__(
            k_neighbors=k_neighbors,
            relaxation=relaxation,
            name=name,
            **kwargs
        )
        
        self.bandwidth = bandwidth
        self.kernel = kernel
    
    def _compute_kernel_weights(
        self,
        distances: torch.Tensor
    ) -> torch.Tensor:
        """Compute kernel weights from distances."""
        if self.kernel == "rbf":
            # Gaussian RBF kernel
            weights = torch.exp(-distances.pow(2) / (2 * self.bandwidth ** 2))
        elif self.kernel == "laplacian":
            # Laplacian kernel
            weights = torch.exp(-distances / self.bandwidth)
        elif self.kernel == "uniform":
            # Uniform kernel within bandwidth
            weights = (distances < self.bandwidth).float()
        else:
            # Default to RBF
            weights = torch.exp(-distances.pow(2) / (2 * self.bandwidth ** 2))
        
        return weights
    
    def loss(
        self,
        predictions: torch.Tensor,
        features: torch.Tensor,
        groups: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute smoothed consistency loss.
        
        Uses kernel-weighted prediction differences.
        """
        if predictions.dim() > 1:
            predictions = predictions.squeeze(-1)
        
        batch_size = predictions.size(0)
        
        # Compute distances
        sq_norm = (features ** 2).sum(dim=1, keepdim=True)
        dist_sq = sq_norm + sq_norm.t() - 2 * features @ features.t()
        distances = torch.sqrt(torch.clamp(dist_sq, min=0))
        
        # Compute kernel weights
        weights = self._compute_kernel_weights(distances)
        
        # Compute prediction differences
        pred_diff = (predictions.unsqueeze(1) - predictions.unsqueeze(0)).abs()
        
        # Mask out diagonal
        mask = 1 - torch.eye(batch_size, device=predictions.device)
        
        # Weighted consistency loss
        loss = (weights * pred_diff * mask).sum() / (weights * mask).sum().clamp(min=1e-8)
        
        return self.apply_relaxation(loss, **kwargs)
    
    def adaptive_bandwidth(
        self,
        features: torch.Tensor,
        quantile: float = 0.3
    ) -> None:
        """
        Set adaptive bandwidth based on data distribution.
        
        Args:
            features: Feature tensor
            quantile: Quantile of distances to use as bandwidth
        """
        with torch.no_grad():
            sq_norm = (features ** 2).sum(dim=1, keepdim=True)
            dist_sq = sq_norm + sq_norm.t() - 2 * features @ features.t()
            distances = torch.sqrt(torch.clamp(dist_sq, min=0))
            
            # Get non-zero distances
            mask = distances > 0
            non_zero_distances = distances[mask]
            
            if len(non_zero_distances) > 0:
                self.bandwidth = torch.quantile(non_zero_distances, quantile).item()
