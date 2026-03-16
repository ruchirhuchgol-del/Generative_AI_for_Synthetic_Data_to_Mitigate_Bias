"""
Distribution Metrics for Synthetic Data Fidelity
==================================================

Comprehensive metrics for comparing distributions between
real and synthetic data across various statistical measures.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
import warnings


class BaseDistributionMetric(ABC):
    """Abstract base class for distribution comparison metrics."""
    
    @abstractmethod
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute the distribution metric.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Metric value
        """
        pass
    
    def compute_per_feature(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> Dict[int, float]:
        """
        Compute metric per feature for multivariate data.
        
        Args:
            real: Real data (n_samples, n_features)
            synthetic: Synthetic data (n_samples, n_features)
            
        Returns:
            Dictionary mapping feature index to metric value
        """
        if real.ndim == 1:
            return {0: self.compute(real, synthetic)}
        
        results = {}
        for i in range(real.shape[1]):
            try:
                results[i] = self.compute(real[:, i], synthetic[:, i])
            except Exception as e:
                warnings.warn(f"Failed to compute metric for feature {i}: {e}")
                results[i] = np.nan
        
        return results


class KolmogorovSmirnovTest(BaseDistributionMetric):
    """
    Kolmogorov-Smirnov test for distribution comparison.
    
    Tests whether two samples are drawn from the same continuous distribution.
    Returns the test statistic (D) and p-value.
    
    Lower D values indicate more similar distributions.
    """
    
    def __init__(self, alternative: str = "two-sided"):
        """
        Initialize KS test.
        
        Args:
            alternative: Type of test ('two-sided', 'less', 'greater')
        """
        self.alternative = alternative
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute KS statistic.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            KS statistic (D value)
        """
        statistic, _ = stats.ks_2samp(real, synthetic, alternative=self.alternative)
        return statistic
    
    def compute_with_pvalue(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute KS test with p-value.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Tuple of (statistic, p-value)
        """
        return stats.ks_2samp(real, synthetic, alternative=self.alternative)


class AndersonDarlingTest(BaseDistributionMetric):
    """
    Anderson-Darling test for distribution comparison.
    
    A modification of the KS test that gives more weight to the tails.
    More sensitive to differences in the tails of distributions.
    """
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute Anderson-Darling statistic.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Anderson-Darling statistic
        """
        result = stats.anderson_ksamp([real, synthetic])
        return result.statistic


class CramervonMisesTest(BaseDistributionMetric):
    """
    Cramér-von Mises test for distribution comparison.
    
    Tests whether two samples come from the same distribution.
    More sensitive to differences near the center of the distribution.
    """
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute Cramér-von Mises statistic.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Cramér-von Mises statistic
        """
        result = stats.cramervonmises_2samp(real, synthetic)
        return result.statistic


class MaximumMeanDiscrepancy(BaseDistributionMetric):
    """
    Maximum Mean Discrepancy (MMD) for distribution comparison.
    
    A kernel-based test for comparing distributions that can detect
    differences in higher moments and is sensitive to any type of difference.
    
    MMD² = E[k(X,X')] - 2E[k(X,Y)] + E[k(Y,Y')]
    where X, X' ~ P (real), Y, Y' ~ Q (synthetic)
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        sigma: Optional[float] = None,
        n_permutations: int = 100
    ):
        """
        Initialize MMD.
        
        Args:
            kernel: Kernel type ('rbf', 'linear', 'polynomial')
            sigma: Bandwidth for RBF kernel (auto-computed if None)
            n_permutations: Number of permutations for p-value estimation
        """
        self.kernel = kernel
        self.sigma = sigma
        self.n_permutations = n_permutations
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute MMD statistic.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            MMD value (0 indicates identical distributions)
        """
        # Ensure 2D
        if real.ndim == 1:
            real = real.reshape(-1, 1)
        if synthetic.ndim == 1:
            synthetic = synthetic.reshape(-1, 1)
        
        # Compute kernel matrices
        K_xx = self._compute_kernel_matrix(real, real)
        K_yy = self._compute_kernel_matrix(synthetic, synthetic)
        K_xy = self._compute_kernel_matrix(real, synthetic)
        
        n = len(real)
        m = len(synthetic)
        
        # Compute MMD
        mmd = (
            K_xx.sum() / (n * n) -
            2 * K_xy.sum() / (n * m) +
            K_yy.sum() / (m * m)
        )
        
        return float(np.sqrt(max(0, mmd)))
    
    def _compute_kernel_matrix(
        self,
        X: np.ndarray,
        Y: np.ndarray
    ) -> np.ndarray:
        """Compute kernel matrix between X and Y."""
        if self.kernel == "rbf":
            # Compute pairwise squared distances
            XX = np.sum(X ** 2, axis=1, keepdims=True)
            YY = np.sum(Y ** 2, axis=1, keepdims=True)
            distances = XX + YY.T - 2 * np.dot(X, Y.T)
            
            # Auto-compute sigma if not provided
            if self.sigma is None:
                self.sigma = np.median(np.sqrt(distances[distances > 0]))
            
            return np.exp(-distances / (2 * self.sigma ** 2))
        
        elif self.kernel == "linear":
            return np.dot(X, Y.T)
        
        elif self.kernel == "polynomial":
            return (1 + np.dot(X, Y.T)) ** 2
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def compute_with_threshold(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute MMD with permutation-based threshold.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Dictionary with MMD, threshold, and p-value
        """
        mmd_observed = self.compute(real, synthetic)
        
        # Permutation test
        combined = np.vstack([real, synthetic])
        n_real = len(real)
        
        mmd_permuted = []
        for _ in range(self.n_permutations):
            np.random.shuffle(combined)
            perm_real = combined[:n_real]
            perm_synth = combined[n_real:]
            mmd_permuted.append(self.compute(perm_real, perm_synth))
        
        threshold = np.percentile(mmd_permuted, 95)
        p_value = np.mean(np.array(mmd_permuted) >= mmd_observed)
        
        return {
            "mmd": mmd_observed,
            "threshold": threshold,
            "p_value": p_value,
            "is_similar": mmd_observed <= threshold
        }


class EnergyDistance(BaseDistributionMetric):
    """
    Energy Distance for distribution comparison.
    
    A metric that satisfies all properties of a metric on probability distributions.
    Related to Cramér's distance and can be seen as a generalization.
    
    E(P, Q) = 2E||X - Y|| - E||X - X'|| - E||Y - Y'||
    """
    
    def __init__(self, exponent: float = 1.0):
        """
        Initialize energy distance.
        
        Args:
            exponent: Exponent for the distance (1 for Euclidean)
        """
        self.exponent = exponent
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute energy distance.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Energy distance
        """
        # Ensure 2D
        if real.ndim == 1:
            real = real.reshape(-1, 1)
        if synthetic.ndim == 1:
            synthetic = synthetic.reshape(-1, 1)
        
        n = len(real)
        m = len(synthetic)
        
        # Compute pairwise distances
        from scipy.spatial.distance import cdist
        
        d_xy = cdist(real, synthetic, metric='minkowski', p=self.exponent)
        d_xx = cdist(real, real, metric='minkowski', p=self.exponent)
        d_yy = cdist(synthetic, synthetic, metric='minkowski', p=self.exponent)
        
        energy = (
            2 * d_xy.mean() -
            d_xx.sum() / (n * (n - 1)) -
            d_yy.sum() / (m * (m - 1))
        )
        
        return float(max(0, energy))


class TotalVariationDistance(BaseDistributionMetric):
    """
    Total Variation Distance for discrete distributions.
    
    Measures the largest possible difference between probabilities
    assigned to the same event by two distributions.
    
    TV(P, Q) = 0.5 * Σ |P(x) - Q(x)|
    """
    
    def __init__(self, n_bins: int = 50):
        """
        Initialize TV distance.
        
        Args:
            n_bins: Number of bins for discretizing continuous data
        """
        self.n_bins = n_bins
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute total variation distance.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            TV distance in [0, 1]
        """
        # Create common bins
        all_data = np.concatenate([real, synthetic])
        bins = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)
        
        # Compute histograms
        p, _ = np.histogram(real, bins=bins, density=True)
        q, _ = np.histogram(synthetic, bins=bins, density=True)
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Total variation distance
        tv = 0.5 * np.abs(p - q).sum()
        
        return float(tv)


class HistogramIntersection(BaseDistributionMetric):
    """
    Histogram Intersection similarity measure.
    
    Measures the overlap between two distributions.
    Higher values indicate more similar distributions.
    """
    
    def __init__(self, n_bins: int = 50):
        """
        Initialize histogram intersection.
        
        Args:
            n_bins: Number of bins for histograms
        """
        self.n_bins = n_bins
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute histogram intersection.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Intersection score in [0, 1]
        """
        # Create common bins
        all_data = np.concatenate([real, synthetic])
        bins = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)
        
        # Compute histograms
        p, _ = np.histogram(real, bins=bins, density=True)
        q, _ = np.histogram(synthetic, bins=bins, density=True)
        
        # Normalize
        p = p / p.sum()
        q = q / q.sum()
        
        # Histogram intersection
        intersection = np.minimum(p, q).sum()
        
        return float(intersection)


class ChiSquaredTest(BaseDistributionMetric):
    """
    Chi-squared test for discrete distributions.
    
    Tests whether two samples come from the same discrete distribution.
    """
    
    def __init__(self, n_bins: int = 20):
        """
        Initialize chi-squared test.
        
        Args:
            n_bins: Number of bins for discretization
        """
        self.n_bins = n_bins
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute chi-squared statistic.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Chi-squared statistic (normalized)
        """
        # Create common bins
        all_data = np.concatenate([real, synthetic])
        bins = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)
        
        # Compute counts
        observed, _ = np.histogram(real, bins=bins)
        expected, _ = np.histogram(synthetic, bins=bins)
        
        # Normalize to same total
        expected = expected * (observed.sum() / max(expected.sum(), 1))
        
        # Add small value to avoid division by zero
        expected = expected + 0.5
        observed = observed + 0.5
        
        # Chi-squared statistic
        chi2 = np.sum((observed - expected) ** 2 / expected)
        
        # Normalize by degrees of freedom
        dof = len(observed) - 1
        return float(chi2 / max(dof, 1))
    
    def compute_with_pvalue(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute chi-squared test with p-value.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Tuple of (statistic, p-value)
        """
        # Create common bins
        all_data = np.concatenate([real, synthetic])
        bins = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)
        
        # Compute counts
        observed, _ = np.histogram(real, bins=bins)
        expected, _ = np.histogram(synthetic, bins=bins)
        
        # Normalize
        expected = expected * (observed.sum() / max(expected.sum(), 1))
        
        # Add small value
        expected = expected + 0.5
        observed = observed + 0.5
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        return chi2, p_value


class DistributionComparator:
    """
    Comprehensive distribution comparison tool.
    
    Combines multiple metrics for thorough distribution evaluation.
    """
    
    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        n_bins: int = 50
    ):
        """
        Initialize comparator.
        
        Args:
            metrics: List of metrics to compute
            n_bins: Number of bins for discretization
        """
        self.metrics = metrics or [
            "ks", "js", "wasserstein", "mmd", "energy", "tv", "histogram"
        ]
        self.n_bins = n_bins
        
        self._metric_instances = {
            "ks": KolmogorovSmirnovTest(),
            "anderson": AndersonDarlingTest(),
            "cramer": CramervonMisesTest(),
            "mmd": MaximumMeanDiscrepancy(),
            "energy": EnergyDistance(),
            "tv": TotalVariationDistance(n_bins),
            "histogram": HistogramIntersection(n_bins),
            "chi2": ChiSquaredTest(n_bins),
        }
    
    def compare(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> Dict[str, Any]:
        """
        Comprehensive distribution comparison.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Dictionary with all metric values
        """
        results = {
            "n_real": len(real),
            "n_synthetic": len(synthetic),
            "metrics": {},
            "per_feature": {},
        }
        
        # Compute each metric
        for metric_name in self.metrics:
            if metric_name in self._metric_instances:
                try:
                    value = self._metric_instances[metric_name].compute(real, synthetic)
                    results["metrics"][metric_name] = value
                except Exception as e:
                    results["metrics"][metric_name] = f"Error: {str(e)}"
        
        # Compute per-feature metrics for multivariate data
        if real.ndim > 1 and real.shape[1] > 1:
            results["per_feature"] = self._compute_per_feature(real, synthetic)
        
        # Summary statistics
        results["summary"] = self._compute_summary(results["metrics"])
        
        return results
    
    def _compute_per_feature(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> Dict[str, Dict[int, float]]:
        """Compute metrics per feature."""
        per_feature = {}
        
        for metric_name in ["ks", "js", "wasserstein"]:
            if metric_name in self._metric_instances:
                per_feature[metric_name] = (
                    self._metric_instances[metric_name]
                    .compute_per_feature(real, synthetic)
                )
        
        return per_feature
    
    def _compute_summary(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Compute summary statistics."""
        numeric_values = [v for v in metrics.values() if isinstance(v, (int, float))]
        
        if not numeric_values:
            return {"average_metric": float('nan')}
        
        # Invert histogram intersection (higher is better)
        if "histogram" in metrics and isinstance(metrics["histogram"], (int, float)):
            inverted_values = [
                v if k != "histogram" else 1 - v
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            ]
        else:
            inverted_values = numeric_values
        
        return {
            "average_metric": float(np.mean(inverted_values)),
            "max_metric": float(np.max(inverted_values)),
            "min_metric": float(np.min(inverted_values)),
        }
