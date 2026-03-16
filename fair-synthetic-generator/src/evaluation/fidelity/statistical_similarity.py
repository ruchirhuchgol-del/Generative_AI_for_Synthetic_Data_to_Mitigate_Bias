"""
Statistical Similarity Metrics
==============================

Comprehensive metrics for measuring statistical similarity between
real and synthetic data including:
- Jensen-Shannon Divergence
- Wasserstein Distance
- Correlation Preservation
- Mutual Information
- Kolmogorov-Smirnov Statistics
- Moment Matching
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.special import entr
import warnings


class FidelityMetric:
    """Base class for fidelity metrics."""
    
    def compute(self, real: np.ndarray, synthetic: np.ndarray) -> float:
        """Compute the metric value."""
        raise NotImplementedError
    
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


class JensenShannonDivergence(FidelityMetric):
    """
    Jensen-Shannon Divergence between real and synthetic distributions.
    
    A symmetric, smoothed version of KL divergence that is bounded
    between 0 and 1. Lower values indicate more similar distributions.
    
    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)
    """
    
    def __init__(self, n_bins: int = 100):
        """
        Initialize JSD metric.
        
        Args:
            n_bins: Number of bins for histogram estimation
        """
        self.n_bins = n_bins
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute Jensen-Shannon divergence.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            JSD value between 0 and 1
        """
        # Handle multidimensional data
        if real.ndim > 1:
            jsds = []
            for i in range(real.shape[1]):
                jsd_i = self._compute_1d(real[:, i], synthetic[:, i])
                if not np.isnan(jsd_i):
                    jsds.append(jsd_i)
            return float(np.mean(jsds)) if jsds else 0.0
        
        return self._compute_1d(real, synthetic)
    
    def _compute_1d(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """Compute JSD for 1D arrays."""
        # Remove NaN values
        real = real[~np.isnan(real)]
        synthetic = synthetic[~np.isnan(synthetic)]
        
        if len(real) == 0 or len(synthetic) == 0:
            return np.nan
        
        # Create common bins
        all_data = np.concatenate([real, synthetic])
        bins = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)
        
        # Compute histograms (probability distributions)
        p, _ = np.histogram(real, bins=bins, density=True)
        q, _ = np.histogram(synthetic, bins=bins, density=True)
        
        # Normalize
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        
        # Use scipy's jensenshannon which is more stable
        return float(jensenshannon(p, q))
    
    def compute_detailed(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute detailed JSD report.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            feature_names: Optional feature names
            
        Returns:
            Detailed report dictionary
        """
        overall_jsd = self.compute(real, synthetic)
        
        if real.ndim > 1:
            per_feature = self.compute_per_feature(real, synthetic)
            
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(real.shape[1])]
            
            per_feature_named = {
                feature_names[i]: v 
                for i, v in per_feature.items() 
                if not np.isnan(v)
            }
            
            return {
                "overall_jsd": overall_jsd,
                "per_feature_jsd": per_feature_named,
                "max_jsd_feature": max(per_feature_named.keys(), key=lambda k: per_feature_named[k]),
                "min_jsd_feature": min(per_feature_named.keys(), key=lambda k: per_feature_named[k]),
                "mean_jsd": float(np.mean(list(per_feature_named.values()))),
            }
        
        return {
            "overall_jsd": overall_jsd,
            "per_feature_jsd": {"feature_0": overall_jsd},
        }


class WassersteinDistance(FidelityMetric):
    """
    Wasserstein (Earth Mover's) Distance between distributions.
    
    Measures the minimum "work" required to transform one distribution
    into another. Lower values indicate more similar distributions.
    
    Also known as:
    - Earth Mover's Distance (EMD)
    - 1st Wasserstein distance
    """
    
    def __init__(self, p: int = 1):
        """
        Initialize Wasserstein distance metric.
        
        Args:
            p: Order of the Wasserstein distance (1 or 2)
        """
        self.p = p
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute Wasserstein distance.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Wasserstein distance
        """
        # Handle multidimensional data
        if real.ndim > 1:
            distances = []
            for i in range(real.shape[1]):
                d = stats.wasserstein_distance(real[:, i], synthetic[:, i])
                if not np.isnan(d):
                    distances.append(d)
            return float(np.mean(distances)) if distances else 0.0
        
        return float(stats.wasserstein_distance(real, synthetic))
    
    def compute_normalized(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute normalized Wasserstein distance.
        
        Normalized by the range of the real data.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Normalized Wasserstein distance (0-1 range)
        """
        if real.ndim > 1:
            # Compute per feature and average
            distances = []
            for i in range(real.shape[1]):
                d = stats.wasserstein_distance(real[:, i], synthetic[:, i])
                data_range = real[:, i].max() - real[:, i].min()
                if data_range > 0:
                    distances.append(d / data_range)
            return float(np.mean(distances)) if distances else 0.0
        
        d = stats.wasserstein_distance(real, synthetic)
        data_range = real.max() - real.min()
        return float(d / data_range) if data_range > 0 else 0.0


class CorrelationPreservation(FidelityMetric):
    """
    Measures how well synthetic data preserves correlations from real data.
    
    Computes the difference between correlation matrices of real and synthetic data.
    Lower values indicate better correlation preservation.
    """
    
    def __init__(self, method: str = "pearson"):
        """
        Initialize correlation preservation metric.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        self.method = method
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute correlation preservation score.
        
        Args:
            real: Real data samples (n_samples, n_features)
            synthetic: Synthetic data samples (n_samples, n_features)
            
        Returns:
            Mean absolute difference in correlations
        """
        if real.ndim == 1:
            return 0.0  # No correlation for single feature
        
        # Compute correlation matrices
        real_corr = self._compute_correlation(real)
        synthetic_corr = self._compute_correlation(synthetic)
        
        # Compute difference (upper triangle only, excluding diagonal)
        mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
        diff = np.abs(real_corr[mask] - synthetic_corr[mask])
        
        return float(diff.mean())
    
    def _compute_correlation(self, data: np.ndarray) -> np.ndarray:
        """Compute correlation matrix."""
        if self.method == "pearson":
            return np.corrcoef(data.T)
        elif self.method == "spearman":
            from scipy.stats import spearmanr
            result = spearmanr(data)
            return result.correlation if hasattr(result, 'correlation') else result[0]
        elif self.method == "kendall":
            from scipy.stats import kendalltau
            n = data.shape[1]
            corr = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    corr[i, j] = kendalltau(data[:, i], data[:, j]).correlation
            return corr
        else:
            return np.corrcoef(data.T)
    
    def compute_detailed(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute detailed correlation preservation report.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            feature_names: Optional feature names
            
        Returns:
            Dictionary with detailed metrics
        """
        real_corr = self._compute_correlation(real)
        synthetic_corr = self._compute_correlation(synthetic)
        
        diff = real_corr - synthetic_corr
        
        # Find most preserved and most changed correlations
        mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
        
        abs_diff = np.abs(diff)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(real.shape[1])]
        
        # Find top changed correlations
        flat_diffs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                flat_diffs.append({
                    "pair": f"{feature_names[i]}_{feature_names[j]}",
                    "difference": float(diff[i, j]),
                    "abs_difference": float(abs_diff[i, j]),
                    "real_corr": float(real_corr[i, j]),
                    "synthetic_corr": float(synthetic_corr[i, j]),
                })
        
        # Sort by absolute difference
        flat_diffs.sort(key=lambda x: x["abs_difference"], reverse=True)
        
        return {
            "mean_absolute_error": self.compute(real, synthetic),
            "max_absolute_error": float(abs_diff[mask].max()),
            "std_absolute_error": float(abs_diff[mask].std()),
            "real_correlation_matrix": real_corr.tolist(),
            "synthetic_correlation_matrix": synthetic_corr.tolist(),
            "correlation_difference": diff.tolist(),
            "top_changed_pairs": flat_diffs[:10],
        }


class MutualInformationMetric(FidelityMetric):
    """
    Mutual Information metric for distribution comparison.
    
    Measures the amount of information shared between real and
    synthetic data distributions.
    """
    
    def __init__(self, n_bins: int = 20, normalized: bool = True):
        """
        Initialize mutual information metric.
        
        Args:
            n_bins: Number of bins for discretization
            normalized: Whether to use normalized mutual information
        """
        self.n_bins = n_bins
        self.normalized = normalized
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute mutual information.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Mutual information (or normalized) score
        """
        if real.ndim > 1:
            mis = []
            for i in range(real.shape[1]):
                mi = self._compute_1d(real[:, i], synthetic[:, i])
                if not np.isnan(mi):
                    mis.append(mi)
            return float(np.mean(mis)) if mis else 0.0
        
        return self._compute_1d(real, synthetic)
    
    def _compute_1d(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """Compute mutual information for 1D arrays."""
        # Discretize continuous data
        all_data = np.concatenate([real, synthetic])
        bins = np.linspace(all_data.min(), all_data.max(), self.n_bins + 1)
        
        real_binned = np.digitize(real, bins[1:-1])
        synthetic_binned = np.digitize(synthetic, bins[1:-1])
        
        # Compute joint histogram
        joint_hist, _, _ = np.histogram2d(
            real_binned, synthetic_binned,
            bins=self.n_bins
        )
        
        # Normalize to probabilities
        joint_prob = joint_hist / joint_hist.sum()
        
        # Marginal probabilities
        p_real = joint_prob.sum(axis=1)
        p_synth = joint_prob.sum(axis=0)
        
        # Compute mutual information
        mi = 0.0
        for i in range(len(p_real)):
            for j in range(len(p_synth)):
                if joint_prob[i, j] > 0 and p_real[i] > 0 and p_synth[j] > 0:
                    mi += joint_prob[i, j] * np.log(
                        joint_prob[i, j] / (p_real[i] * p_synth[j])
                    )
        
        if self.normalized:
            # Normalize by geometric mean of entropies
            h_real = -np.sum(p_real[p_real > 0] * np.log(p_real[p_real > 0]))
            h_synth = -np.sum(p_synth[p_synth > 0] * np.log(p_synth[p_synth > 0]))
            
            if h_real > 0 and h_synth > 0:
                return float(mi / np.sqrt(h_real * h_synth))
            return 0.0
        
        return float(mi)


class KolmogorovSmirnovStatistic(FidelityMetric):
    """
    Kolmogorov-Smirnov statistic for distribution comparison.
    
    Tests whether two samples are drawn from the same continuous distribution.
    Returns the test statistic and p-value.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize KS statistic.
        
        Args:
            alpha: Significance level for p-value threshold
        """
        self.alpha = alpha
    
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
        if real.ndim > 1:
            stats_list = []
            for i in range(real.shape[1]):
                d, _ = stats.ks_2samp(real[:, i], synthetic[:, i])
                stats_list.append(d)
            return float(np.mean(stats_list))
        
        d, _ = stats.ks_2samp(real, synthetic)
        return float(d)
    
    def compute_with_pvalue(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute KS test with p-value.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Dictionary with statistic and p-value
        """
        if real.ndim > 1:
            results = {}
            for i in range(real.shape[1]):
                d, p = stats.ks_2samp(real[:, i], synthetic[:, i])
                results[f"feature_{i}"] = {"statistic": d, "p_value": p}
            
            # Summary
            all_d = [r["statistic"] for r in results.values()]
            all_p = [r["p_value"] for r in results.values()]
            
            results["summary"] = {
                "mean_statistic": float(np.mean(all_d)),
                "mean_p_value": float(np.mean(all_p)),
                "n_significant": sum(1 for p in all_p if p < self.alpha),
            }
            
            return results
        
        d, p = stats.ks_2samp(real, synthetic)
        return {
            "statistic": float(d),
            "p_value": float(p),
            "is_similar": p >= self.alpha,
        }


class MomentMatchingMetric(FidelityMetric):
    """
    Moment Matching metric for distribution comparison.
    
    Compares statistical moments (mean, variance, skewness, kurtosis)
    between real and synthetic data.
    """
    
    def __init__(self, max_order: int = 4):
        """
        Initialize moment matching metric.
        
        Args:
            max_order: Maximum moment order to compare (1-4)
        """
        self.max_order = max_order
    
    def compute(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """
        Compute moment matching score.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            
        Returns:
            Average absolute difference in moments
        """
        if real.ndim > 1:
            scores = []
            for i in range(real.shape[1]):
                scores.append(self._compute_1d(real[:, i], synthetic[:, i]))
            return float(np.mean(scores))
        
        return self._compute_1d(real, synthetic)
    
    def _compute_1d(
        self,
        real: np.ndarray,
        synthetic: np.ndarray
    ) -> float:
        """Compute moment differences for 1D arrays."""
        differences = []
        
        # Mean (1st moment)
        if self.max_order >= 1:
            diff = abs(real.mean() - synthetic.mean())
            if real.std() > 0:
                diff /= real.std()
            differences.append(diff)
        
        # Variance (2nd moment)
        if self.max_order >= 2:
            diff = abs(real.var() - synthetic.var())
            if real.var() > 0:
                diff /= real.var()
            differences.append(diff)
        
        # Skewness (3rd moment)
        if self.max_order >= 3:
            real_skew = stats.skew(real)
            synth_skew = stats.skew(synthetic)
            differences.append(abs(real_skew - synth_skew))
        
        # Kurtosis (4th moment)
        if self.max_order >= 4:
            real_kurt = stats.kurtosis(real)
            synth_kurt = stats.kurtosis(synthetic)
            differences.append(abs(real_kurt - synth_kurt))
        
        return float(np.mean(differences))
    
    def compute_detailed(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute detailed moment matching report.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            feature_names: Optional feature names
            
        Returns:
            Detailed report with moment comparisons
        """
        if real.ndim == 1:
            real = real.reshape(-1, 1)
            synthetic = synthetic.reshape(-1, 1)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(real.shape[1])]
        
        results = {
            "moments": {},
            "overall_score": self.compute(real, synthetic),
        }
        
        for i, name in enumerate(feature_names):
            real_col = real[:, i]
            synth_col = synthetic[:, i]
            
            results["moments"][name] = {
                "mean": {
                    "real": float(real_col.mean()),
                    "synthetic": float(synth_col.mean()),
                    "difference": float(abs(real_col.mean() - synth_col.mean())),
                },
                "std": {
                    "real": float(real_col.std()),
                    "synthetic": float(synth_col.std()),
                    "difference": float(abs(real_col.std() - synth_col.std())),
                },
                "skewness": {
                    "real": float(stats.skew(real_col)),
                    "synthetic": float(stats.skew(synth_col)),
                    "difference": float(abs(stats.skew(real_col) - stats.skew(synth_col))),
                },
                "kurtosis": {
                    "real": float(stats.kurtosis(real_col)),
                    "synthetic": float(stats.kurtosis(synth_col)),
                    "difference": float(abs(stats.kurtosis(real_col) - stats.kurtosis(synth_col))),
                },
            }
        
        return results


class StatisticalSimilarityEvaluator:
    """
    Comprehensive statistical similarity evaluator.
    
    Combines multiple fidelity metrics for thorough evaluation.
    """
    
    def __init__(
        self,
        n_bins: int = 100,
        correlation_method: str = "pearson"
    ):
        """
        Initialize statistical similarity evaluator.
        
        Args:
            n_bins: Number of bins for histogram-based metrics
            correlation_method: Method for correlation computation
        """
        self.n_bins = n_bins
        self.correlation_method = correlation_method
        
        self.metrics = {
            "js_divergence": JensenShannonDivergence(n_bins),
            "wasserstein": WassersteinDistance(),
            "correlation": CorrelationPreservation(correlation_method),
            "mutual_information": MutualInformationMetric(n_bins),
            "ks_statistic": KolmogorovSmirnovStatistic(),
            "moment_matching": MomentMatchingMetric(),
        }
    
    def evaluate(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive statistical similarity evaluation.
        
        Args:
            real: Real data samples
            synthetic: Synthetic data samples
            feature_names: Optional feature names
            
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
            results["metrics"][name] = metric.compute(real, synthetic)
            
            if hasattr(metric, 'compute_detailed'):
                results["detailed"][name] = metric.compute_detailed(
                    real, synthetic, feature_names
                )
            else:
                results["detailed"][name] = {"value": results["metrics"][name]}
        
        # Compute overall fidelity score
        # JSD: lower is better, bounded [0, 1]
        jsd_score = 1 - results["metrics"]["js_divergence"]
        
        # Wasserstein: normalize and invert
        wass_norm = self.metrics["wasserstein"].compute_normalized(real, synthetic)
        wass_score = 1 - min(wass_norm, 1)
        
        # Correlation: lower is better, bounded [0, 2]
        corr_score = max(0, 1 - results["metrics"]["correlation"])
        
        # MI: higher is better, normalized [0, 1]
        mi_score = results["metrics"]["mutual_information"]
        
        # KS: lower is better, bounded [0, 1]
        ks_score = 1 - results["metrics"]["ks_statistic"]
        
        # Moments: lower is better
        moment_score = max(0, 1 - results["metrics"]["moment_matching"])
        
        results["overall"] = {
            "fidelity_score": float(np.mean([
                jsd_score, wass_score, corr_score, mi_score, ks_score, moment_score
            ])),
            "is_high_fidelity": np.mean([
                jsd_score, wass_score, corr_score, mi_score, ks_score, moment_score
            ]) >= 0.8,
            "component_scores": {
                "js_divergence_score": float(jsd_score),
                "wasserstein_score": float(wass_score),
                "correlation_score": float(corr_score),
                "mutual_information_score": float(mi_score),
                "ks_score": float(ks_score),
                "moment_score": float(moment_score),
            },
        }
        
        return results
