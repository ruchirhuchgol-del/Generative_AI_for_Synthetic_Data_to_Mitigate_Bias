"""
Individual Fairness Metrics
===========================

Comprehensive evaluation metrics for individual fairness including:
- Consistency Score
- Lipschitz Estimation
- Local Fairness
- Individual Discrimination
- Similarity-based Fairness
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cdist, cosine
from scipy.stats import rankdata
import warnings


class IndividualFairnessMetric:
    """Base class for individual fairness metrics."""
    
    def compute(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> float:
        """Compute the metric value."""
        raise NotImplementedError
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute detailed metric report."""
        raise NotImplementedError


class ConsistencyScore(IndividualFairnessMetric):
    """
    Consistency score for individual fairness.
    
    Measures how similar predictions are for similar individuals.
    Higher scores indicate better individual fairness.
    
    Score = 1 - (1/n) * Σ|f(x_i) - avg(f(x_j))| for k nearest neighbors
    
    Originally proposed in "Fairness Through Awareness" (Dwork et al., 2012).
    """
    
    def __init__(self, k_neighbors: int = 10, distance_metric: str = "euclidean"):
        """
        Initialize consistency score.
        
        Args:
            k_neighbors: Number of nearest neighbors to consider
            distance_metric: Distance metric for finding neighbors
        """
        self.k_neighbors = k_neighbors
        self.distance_metric = distance_metric
    
    def compute(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute consistency score.
        
        Args:
            predictions: Model predictions
            features: Feature matrix
            groups: Ignored for individual fairness
            
        Returns:
            Consistency score between 0 and 1
        """
        n_samples = predictions.shape[0]
        k = min(self.k_neighbors, n_samples - 1)
        
        if k <= 0:
            return 1.0
        
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # Compute pairwise distances
        distances = cdist(features, features, metric=self.distance_metric)
        
        # Find k nearest neighbors for each sample (excluding self)
        total_diff = 0.0
        
        for i in range(n_samples):
            # Get distances to other samples
            dist_i = distances[i].copy()
            dist_i[i] = float('inf')  # Exclude self
            
            # Find k nearest neighbors
            neighbor_indices = np.argsort(dist_i)[:k]
            
            # Compute average prediction difference
            neighbor_preds = predictions[neighbor_indices]
            diff = np.abs(predictions[i] - neighbor_preds).mean()
            total_diff += diff
        
        consistency = 1 - total_diff / n_samples
        
        return float(max(0, min(1, consistency)))
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute detailed consistency report."""
        score = self.compute(predictions, features)
        
        # Compute per-sample consistency
        n_samples = predictions.shape[0]
        k = min(self.k_neighbors, n_samples - 1)
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        distances = cdist(features, features, metric=self.distance_metric)
        sample_consistency = []
        
        for i in range(n_samples):
            dist_i = distances[i].copy()
            dist_i[i] = float('inf')
            neighbor_indices = np.argsort(dist_i)[:k]
            neighbor_preds = predictions[neighbor_indices]
            sample_consistency.append(1 - np.abs(predictions[i] - neighbor_preds).mean())
        
        return {
            "metric": "consistency",
            "score": score,
            "k_neighbors": self.k_neighbors,
            "is_fair": score >= 0.9,
            "per_sample_mean": float(np.mean(sample_consistency)),
            "per_sample_std": float(np.std(sample_consistency)),
            "min_sample_consistency": float(min(sample_consistency)),
            "max_sample_consistency": float(max(sample_consistency)),
        }
    
    def compute_by_group(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute consistency score per group.
        
        Args:
            predictions: Model predictions
            features: Feature matrix
            groups: Group membership
            
        Returns:
            Dictionary with consistency per group
        """
        unique_groups = np.unique(groups)
        results = {}
        
        for g in unique_groups:
            mask = groups == g
            results[f"group_{g}"] = self.compute(
                predictions[mask],
                features[mask]
            )
        
        results["overall"] = self.compute(predictions, features)
        
        return results


class LipschitzEstimator(IndividualFairnessMetric):
    """
    Lipschitz constant estimator for individual fairness.
    
    Estimates the Lipschitz constant of the prediction function
    with respect to the feature space.
    
    A function f is L-Lipschitz if:
    |f(x) - f(y)| <= L * ||x - y||
    
    Lower values indicate better individual fairness.
    """
    
    def __init__(
        self,
        distance_metric: str = "euclidean",
        percentile: float = 95.0,
        sample_size: Optional[int] = None
    ):
        """
        Initialize Lipschitz estimator.
        
        Args:
            distance_metric: Distance metric to use
            percentile: Percentile for robust estimation
            sample_size: Optional sample size for large datasets
        """
        self.distance_metric = distance_metric
        self.percentile = percentile
        self.sample_size = sample_size
    
    def compute(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> float:
        """
        Estimate Lipschitz constant.
        
        Args:
            predictions: Model predictions
            features: Feature matrix
            groups: Ignored for individual fairness
            
        Returns:
            Estimated Lipschitz constant
        """
        n_samples = predictions.shape[0]
        
        if n_samples < 2:
            return 0.0
        
        # Ensure 2D
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # Sample for large datasets
        if self.sample_size is not None and n_samples > self.sample_size:
            indices = np.random.choice(n_samples, self.sample_size, replace=False)
            features = features[indices]
            predictions = predictions[indices]
            n_samples = self.sample_size
        
        # Compute pairwise distances
        distances = cdist(features, features, metric=self.distance_metric)
        
        # Compute prediction differences
        pred_diff = np.abs(predictions[:, None] - predictions[None, :])
        
        # Estimate Lipschitz constant as max ratio
        distances_safe = np.maximum(distances, 1e-7)
        
        # Exclude diagonal
        mask = ~np.eye(n_samples, dtype=bool)
        
        ratios = pred_diff[mask] / distances_safe[mask]
        
        return float(np.percentile(ratios, self.percentile))
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute detailed Lipschitz estimation report."""
        n_samples = predictions.shape[0]
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        distances = cdist(features, features, metric=self.distance_metric)
        pred_diff = np.abs(predictions[:, None] - predictions[None, :])
        distances_safe = np.maximum(distances, 1e-7)
        mask = ~np.eye(n_samples, dtype=bool)
        ratios = pred_diff[mask] / distances_safe[mask]
        
        return {
            "metric": "lipschitz",
            "lipschitz_constant": float(np.percentile(ratios, self.percentile)),
            "lipschitz_max": float(np.max(ratios)),
            "lipschitz_mean": float(np.mean(ratios)),
            "lipschitz_median": float(np.median(ratios)),
            "percentile_used": self.percentile,
            "is_fair": float(np.percentile(ratios, self.percentile)) < 1.0,
        }


class LocalFairnessMetric(IndividualFairnessMetric):
    """
    Local Fairness metric.
    
    Measures fairness within local neighborhoods, ensuring
    that predictions are fair for similar individuals.
    """
    
    def __init__(
        self,
        k_neighbors: int = 10,
        fairness_threshold: float = 0.1
    ):
        """
        Initialize local fairness metric.
        
        Args:
            k_neighbors: Size of local neighborhood
            fairness_threshold: Maximum acceptable prediction difference
        """
        self.k_neighbors = k_neighbors
        self.fairness_threshold = fairness_threshold
    
    def compute(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute local fairness score.
        
        Args:
            predictions: Model predictions
            features: Feature matrix
            groups: Optional group membership
            
        Returns:
            Proportion of locally fair predictions
        """
        n_samples = predictions.shape[0]
        k = min(self.k_neighbors, n_samples - 1)
        
        if k <= 0:
            return 1.0
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        distances = cdist(features, features, metric='euclidean')
        
        locally_fair = 0
        
        for i in range(n_samples):
            dist_i = distances[i].copy()
            dist_i[i] = float('inf')
            neighbor_indices = np.argsort(dist_i)[:k]
            
            # Check if prediction is within threshold of neighbors
            neighbor_preds = predictions[neighbor_indices]
            max_diff = np.abs(predictions[i] - neighbor_preds).max()
            
            if max_diff <= self.fairness_threshold:
                locally_fair += 1
        
        return float(locally_fair / n_samples)
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute detailed local fairness report."""
        score = self.compute(predictions, features)
        
        return {
            "metric": "local_fairness",
            "score": score,
            "k_neighbors": self.k_neighbors,
            "threshold": self.fairness_threshold,
            "is_fair": score >= 0.9,
        }


class IndividualDiscriminationMetric(IndividualFairnessMetric):
    """
    Individual Discrimination metric.
    
    Measures the rate at which similar individuals receive
    different predictions (discriminatory decisions).
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.1,
        prediction_threshold: float = 0.05
    ):
        """
        Initialize discrimination metric.
        
        Args:
            similarity_threshold: Threshold for considering individuals similar
            prediction_threshold: Threshold for considering predictions different
        """
        self.similarity_threshold = similarity_threshold
        self.prediction_threshold = prediction_threshold
    
    def compute(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute discrimination rate.
        
        Args:
            predictions: Model predictions
            features: Feature matrix (normalized)
            groups: Optional group membership
            
        Returns:
            Discrimination rate (lower is better)
        """
        n_samples = predictions.shape[0]
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # Normalize features
        features_norm = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-8)
        
        distances = cdist(features_norm, features_norm, metric='euclidean')
        
        discriminatory_pairs = 0
        total_similar_pairs = 0
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if distances[i, j] <= self.similarity_threshold:
                    total_similar_pairs += 1
                    if np.abs(predictions[i] - predictions[j]) > self.prediction_threshold:
                        discriminatory_pairs += 1
        
        if total_similar_pairs == 0:
            return 0.0
        
        return float(discriminatory_pairs / total_similar_pairs)
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute detailed discrimination report."""
        discrimination_rate = self.compute(predictions, features)
        
        return {
            "metric": "individual_discrimination",
            "discrimination_rate": discrimination_rate,
            "similarity_threshold": self.similarity_threshold,
            "prediction_threshold": self.prediction_threshold,
            "is_fair": discrimination_rate <= 0.1,
        }


class SimilarityBasedFairness(IndividualFairnessMetric):
    """
    Similarity-based Fairness metric.
    
    Uses a similarity function to weight fairness violations,
    giving more weight to more similar individuals.
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        kernel_width: Optional[float] = None
    ):
        """
        Initialize similarity-based fairness.
        
        Args:
            kernel: Kernel type ('rbf', 'linear')
            kernel_width: Width for RBF kernel
        """
        self.kernel = kernel
        self.kernel_width = kernel_width
    
    def compute(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute similarity-weighted fairness violation.
        
        Args:
            predictions: Model predictions
            features: Feature matrix
            groups: Optional group membership
            
        Returns:
            Weighted fairness violation score (lower is better)
        """
        n_samples = predictions.shape[0]
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # Compute similarity matrix
        if self.kernel == "rbf":
            distances = cdist(features, features, metric='euclidean')
            if self.kernel_width is None:
                self.kernel_width = np.median(distances[distances > 0])
            similarity = np.exp(-distances ** 2 / (2 * self.kernel_width ** 2))
        else:
            # Linear kernel
            similarity = np.dot(features, features.T)
            # Normalize
            norms = np.sqrt(np.diag(similarity))
            similarity = similarity / (norms[:, None] * norms[None, :])
        
        # Compute weighted fairness violation
        pred_diff = np.abs(predictions[:, None] - predictions[None, :])
        
        # Exclude diagonal
        mask = ~np.eye(n_samples, dtype=bool)
        
        # Weight by similarity
        weighted_violation = (similarity * pred_diff)[mask].sum()
        total_similarity = similarity[mask].sum()
        
        return float(weighted_violation / total_similarity) if total_similarity > 0 else 0.0
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute detailed similarity-based fairness report."""
        violation = self.compute(predictions, features)
        
        return {
            "metric": "similarity_based_fairness",
            "violation_score": violation,
            "kernel": self.kernel,
            "kernel_width": self.kernel_width,
            "is_fair": violation <= 0.1,
        }


class SmoothnessMetric(IndividualFairnessMetric):
    """
    Smoothness metric for individual fairness.
    
    Measures the smoothness of the prediction function
    with respect to the feature space.
    """
    
    def __init__(
        self,
        sigma: float = 1.0,
        normalize: bool = True
    ):
        """
        Initialize smoothness metric.
        
        Args:
            sigma: Scale parameter for smoothness computation
            normalize: Whether to normalize features
        """
        self.sigma = sigma
        self.normalize = normalize
    
    def compute(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute smoothness score.
        
        Args:
            predictions: Model predictions
            features: Feature matrix
            groups: Optional group membership
            
        Returns:
            Smoothness score (higher is better)
        """
        n_samples = predictions.shape[0]
        
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        
        # Normalize features
        if self.normalize:
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        distances = cdist(features, features, metric='euclidean')
        
        # Compute smoothness using Laplacian
        weights = np.exp(-distances ** 2 / (2 * self.sigma ** 2))
        
        # Degree matrix
        degree = weights.sum(axis=1)
        degree_inv_sqrt = 1.0 / np.sqrt(degree + 1e-8)
        
        # Normalized Laplacian
        L = np.eye(n_samples) - degree_inv_sqrt[:, None] * weights * degree_inv_sqrt[None, :]
        
        # Smoothness: y^T L y
        smoothness = predictions @ L @ predictions
        
        # Normalize by prediction variance
        pred_var = predictions.var()
        
        if pred_var > 0:
            return float(max(0, 1 - smoothness / (pred_var * n_samples)))
        return 1.0
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Compute detailed smoothness report."""
        score = self.compute(predictions, features)
        
        return {
            "metric": "smoothness",
            "score": score,
            "sigma": self.sigma,
            "is_fair": score >= 0.8,
        }


class IndividualFairnessEvaluator:
    """
    Comprehensive individual fairness evaluator.
    
    Combines multiple individual fairness metrics for thorough evaluation.
    """
    
    def __init__(
        self,
        k_neighbors: int = 10,
        consistency_threshold: float = 0.9,
        lipschitz_threshold: float = 1.0
    ):
        """
        Initialize individual fairness evaluator.
        
        Args:
            k_neighbors: Number of neighbors for consistency
            consistency_threshold: Threshold for consistency score
            lipschitz_threshold: Threshold for Lipschitz constant
        """
        self.k_neighbors = k_neighbors
        self.consistency_threshold = consistency_threshold
        self.lipschitz_threshold = lipschitz_threshold
        
        self.metrics = {
            "consistency": ConsistencyScore(k_neighbors=k_neighbors),
            "lipschitz": LipschitzEstimator(),
            "local_fairness": LocalFairnessMetric(k_neighbors=k_neighbors),
            "discrimination": IndividualDiscriminationMetric(),
            "similarity_fairness": SimilarityBasedFairness(),
            "smoothness": SmoothnessMetric(),
        }
    
    def evaluate(
        self,
        predictions: np.ndarray,
        features: np.ndarray,
        groups: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive individual fairness evaluation.
        
        Args:
            predictions: Model predictions
            features: Feature matrix
            groups: Optional group membership
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            "metrics": {},
            "detailed": {},
            "overall": {},
        }
        
        # Compute all metrics
        for name, metric in self.metrics.items():
            results["metrics"][name] = metric.compute(predictions, features, groups)
            results["detailed"][name] = metric.compute_detailed(predictions, features, groups)
        
        # Consistency by group if groups provided
        if groups is not None:
            results["consistency_by_group"] = self.metrics["consistency"].compute_by_group(
                predictions, features, groups
            )
        
        # Overall assessment
        fairness_scores = []
        
        # Consistency (higher is better)
        if results["metrics"]["consistency"] >= self.consistency_threshold:
            fairness_scores.append(1.0)
        else:
            fairness_scores.append(results["metrics"]["consistency"])
        
        # Lipschitz (lower is better)
        if results["metrics"]["lipschitz"] <= self.lipschitz_threshold:
            fairness_scores.append(1.0)
        else:
            fairness_scores.append(self.lipschitz_threshold / results["metrics"]["lipschitz"])
        
        # Local fairness
        fairness_scores.append(results["metrics"]["local_fairness"])
        
        # Discrimination (lower is better)
        fairness_scores.append(1 - results["metrics"]["discrimination"])
        
        # Similarity fairness (lower violation is better)
        fairness_scores.append(1 - results["metrics"]["similarity_fairness"])
        
        # Smoothness
        fairness_scores.append(results["metrics"]["smoothness"])
        
        results["overall"] = {
            "individual_fairness_score": float(np.mean(fairness_scores)),
            "is_fair": np.mean(fairness_scores) >= 0.8,
            "n_metrics": len(fairness_scores),
        }
        
        return results
