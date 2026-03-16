"""
Counterfactual Fairness Metrics
================================

Metrics for evaluating counterfactual fairness in synthetic data.
Counterfactual fairness ensures that decisions remain the same
in both the actual world and a counterfactual world where the
sensitive attribute had been different.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
import warnings


class CounterfactualFairnessMetric:
    """Base class for counterfactual fairness metrics."""
    
    def compute(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        sensitive_attribute: np.ndarray
    ) -> float:
        """
        Compute counterfactual fairness metric.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Predictions in counterfactual world
            sensitive_attribute: Sensitive attribute values
            
        Returns:
            Metric value
        """
        raise NotImplementedError


class CounterfactualInvariance(CounterfactualFairnessMetric):
    """
    Counterfactual Invariance metric.
    
    Measures the proportion of instances where the prediction
    remains unchanged under counterfactual intervention.
    
    A model is counterfactually fair if predictions are invariant
    under interventions on sensitive attributes.
    
    CF = (1/n) * Σ I(f(x) = f(x^(a→a')))
    where x^(a→a') is the counterfactual with sensitive attribute changed.
    """
    
    def __init__(self, tolerance: float = 0.01):
        """
        Initialize counterfactual invariance metric.
        
        Args:
            tolerance: Tolerance for considering predictions equal
        """
        self.tolerance = tolerance
    
    def compute(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        sensitive_attribute: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute counterfactual invariance.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Counterfactual predictions
            sensitive_attribute: Sensitive attribute (not used for basic metric)
            
        Returns:
            Proportion of invariant predictions
        """
        diff = np.abs(predictions - counterfactual_predictions)
        invariant = (diff <= self.tolerance).astype(float)
        
        return float(invariant.mean())
    
    def compute_per_group(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        sensitive_attribute: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute counterfactual invariance per sensitive group.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Counterfactual predictions
            sensitive_attribute: Sensitive attribute values
            
        Returns:
            Dictionary with invariance per group
        """
        unique_groups = np.unique(sensitive_attribute)
        results = {}
        
        for group in unique_groups:
            mask = sensitive_attribute == group
            invariance = self.compute(
                predictions[mask],
                counterfactual_predictions[mask]
            )
            results[f"group_{group}"] = invariance
        
        results["overall"] = self.compute(predictions, counterfactual_predictions)
        
        return results


class CounterfactualEffectSize(CounterfactualFairnessMetric):
    """
    Counterfactual Effect Size metric.
    
    Measures the average magnitude of the counterfactual effect.
    Smaller values indicate better counterfactual fairness.
    
    CES = (1/n) * Σ |f(x) - f(x^(a→a'))|
    """
    
    def __init__(self, normalize: bool = True):
        """
        Initialize counterfactual effect size metric.
        
        Args:
            normalize: Whether to normalize by prediction range
        """
        self.normalize = normalize
    
    def compute(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        sensitive_attribute: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute counterfactual effect size.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Counterfactual predictions
            sensitive_attribute: Sensitive attribute (not used)
            
        Returns:
            Average effect size
        """
        effects = np.abs(predictions - counterfactual_predictions)
        
        if self.normalize:
            pred_range = predictions.max() - predictions.min()
            if pred_range > 0:
                effects = effects / pred_range
        
        return float(effects.mean())
    
    def compute_distribution(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute distribution statistics of counterfactual effects.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Counterfactual predictions
            
        Returns:
            Dictionary with effect distribution statistics
        """
        effects = predictions - counterfactual_predictions
        
        return {
            "mean_effect": float(np.mean(effects)),
            "std_effect": float(np.std(effects)),
            "median_effect": float(np.median(effects)),
            "max_effect": float(np.max(np.abs(effects))),
            "positive_effect_ratio": float((effects > 0).mean()),
            "negative_effect_ratio": float((effects < 0).mean()),
        }


class CausalEffectRatio(CounterfactualFairnessMetric):
    """
    Causal Effect Ratio metric.
    
    Measures the ratio of total causal effect to direct causal effect.
    A ratio close to 1 indicates the effect is mainly direct,
    while larger values indicate indirect effects through mediators.
    """
    
    def compute(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        sensitive_attribute: np.ndarray
    ) -> float:
        """
        Compute causal effect ratio.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Counterfactual predictions
            sensitive_attribute: Sensitive attribute values
            
        Returns:
            Causal effect ratio
        """
        # Total effect
        total_effect = np.abs(
            predictions[sensitive_attribute == 1].mean() -
            predictions[sensitive_attribute == 0].mean()
        )
        
        # Direct effect (counterfactual)
        direct_effect = np.abs(
            counterfactual_predictions[sensitive_attribute == 1].mean() -
            predictions[sensitive_attribute == 1].mean()
        )
        
        if direct_effect < 1e-10:
            return 0.0 if total_effect < 1e-10 else float('inf')
        
        return float(total_effect / direct_effect)


class CounterfactualDemographicParity(CounterfactualFairnessMetric):
    """
    Counterfactual Demographic Parity metric.
    
    Extends demographic parity to the counterfactual setting.
    Ensures that the counterfactual predictions maintain
    demographic parity across groups.
    """
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize counterfactual demographic parity.
        
        Args:
            threshold: Threshold for fairness determination
        """
        self.threshold = threshold
    
    def compute(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        sensitive_attribute: np.ndarray
    ) -> float:
        """
        Compute counterfactual demographic parity.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Counterfactual predictions
            sensitive_attribute: Sensitive attribute values
            
        Returns:
            Counterfactual demographic parity difference
        """
        unique_groups = np.unique(sensitive_attribute)
        
        # Compute rates in counterfactual world
        counterfactual_rates = []
        actual_rates = []
        
        for group in unique_groups:
            mask = sensitive_attribute == group
            counterfactual_rates.append(counterfactual_predictions[mask].mean())
            actual_rates.append(predictions[mask].mean())
        
        # Difference between max and min rates in counterfactual world
        cf_dp = max(counterfactual_rates) - min(counterfactual_rates)
        
        return float(cf_dp)
    
    def is_fair(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        sensitive_attribute: np.ndarray
    ) -> bool:
        """
        Check if counterfactual demographic parity is satisfied.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Counterfactual predictions
            sensitive_attribute: Sensitive attribute values
            
        Returns:
            True if fairness is satisfied
        """
        dp = self.compute(predictions, counterfactual_predictions, sensitive_attribute)
        return dp <= self.threshold


class CounterfactualEqualizedOdds(CounterfactualFairnessMetric):
    """
    Counterfactual Equalized Odds metric.
    
    Extends equalized odds to counterfactual predictions,
    ensuring TPR and FPR parity in the counterfactual world.
    """
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize counterfactual equalized odds.
        
        Args:
            threshold: Threshold for fairness determination
        """
        self.threshold = threshold
    
    def compute(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        sensitive_attribute: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute counterfactual equalized odds.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Counterfactual predictions
            sensitive_attribute: Sensitive attribute values
            labels: Ground truth labels
            
        Returns:
            Counterfactual equalized odds difference
        """
        unique_groups = np.unique(sensitive_attribute)
        
        tprs = []
        fprs = []
        
        for group in unique_groups:
            mask = sensitive_attribute == group
            positive_mask = mask & (labels == 1)
            negative_mask = mask & (labels == 0)
            
            if positive_mask.sum() > 0:
                tpr = counterfactual_predictions[positive_mask].mean()
                tprs.append(tpr)
            
            if negative_mask.sum() > 0:
                fpr = counterfactual_predictions[negative_mask].mean()
                fprs.append(fpr)
        
        max_diff = 0.0
        
        if len(tprs) >= 2:
            max_diff = max(max_diff, max(tprs) - min(tprs))
        
        if len(fprs) >= 2:
            max_diff = max(max_diff, max(fprs) - min(fprs))
        
        return float(max_diff)


class IndividualTreatmentEffect(CounterfactualFairnessMetric):
    """
    Individual Treatment Effect (ITE) metric.
    
    Computes the distribution of individual treatment effects
    (differences between actual and counterfactual predictions).
    
    ITE_i = f(x_i) - f(x_i^(a→a'))
    """
    
    def compute(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        sensitive_attribute: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute individual treatment effect statistics.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Counterfactual predictions
            sensitive_attribute: Sensitive attribute values
            
        Returns:
            Dictionary with ITE statistics
        """
        ite = predictions - counterfactual_predictions
        
        results = {
            "ite_mean": float(np.mean(ite)),
            "ite_std": float(np.std(ite)),
            "ite_median": float(np.median(ite)),
            "ite_q25": float(np.percentile(ite, 25)),
            "ite_q75": float(np.percentile(ite, 75)),
            "ite_min": float(np.min(ite)),
            "ite_max": float(np.max(ite)),
            "ite_abs_mean": float(np.mean(np.abs(ite))),
            "ite_zero_ratio": float((np.abs(ite) < 1e-6).mean()),
        }
        
        if sensitive_attribute is not None:
            # ITE statistics per group
            unique_groups = np.unique(sensitive_attribute)
            for group in unique_groups:
                mask = sensitive_attribute == group
                group_ite = ite[mask]
                results[f"ite_mean_group_{group}"] = float(np.mean(group_ite))
                results[f"ite_std_group_{group}"] = float(np.std(group_ite))
        
        return results


class CounterfactualConsistency(CounterfactualFairnessMetric):
    """
    Counterfactual Consistency metric.
    
    Measures whether individuals who are similar in non-sensitive
    features receive similar counterfactual effects.
    """
    
    def __init__(self, k_neighbors: int = 5, distance_threshold: float = 0.1):
        """
        Initialize counterfactual consistency metric.
        
        Args:
            k_neighbors: Number of neighbors to consider
            distance_threshold: Distance threshold for similarity
        """
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
    
    def compute(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        features: np.ndarray,
        sensitive_attribute: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute counterfactual consistency.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Counterfactual predictions
            features: Non-sensitive feature matrix
            sensitive_attribute: Sensitive attribute values
            
        Returns:
            Consistency score
        """
        from scipy.spatial.distance import cdist
        
        n_samples = len(predictions)
        k = min(self.k_neighbors, n_samples - 1)
        
        if k <= 0:
            return 1.0
        
        # Compute counterfactual effects
        effects = np.abs(predictions - counterfactual_predictions)
        
        # Compute distances
        distances = cdist(features, features, metric='euclidean')
        
        consistency_scores = []
        
        for i in range(n_samples):
            # Find similar individuals
            dist_i = distances[i].copy()
            dist_i[i] = float('inf')
            
            similar_mask = dist_i <= self.distance_threshold
            
            if similar_mask.sum() > 0:
                # Compare effects with similar individuals
                similar_effects = effects[similar_mask]
                effect_variance = np.var(similar_effects)
                consistency_scores.append(1 / (1 + effect_variance))
        
        return float(np.mean(consistency_scores)) if consistency_scores else 1.0


class CounterfactualFairnessEvaluator:
    """
    Comprehensive counterfactual fairness evaluator.
    
    Combines multiple counterfactual fairness metrics for thorough evaluation.
    """
    
    def __init__(
        self,
        invariance_threshold: float = 0.95,
        effect_threshold: float = 0.1,
        dp_threshold: float = 0.05
    ):
        """
        Initialize counterfactual fairness evaluator.
        
        Args:
            invariance_threshold: Threshold for counterfactual invariance
            effect_threshold: Threshold for effect size
            dp_threshold: Threshold for demographic parity
        """
        self.invariance_threshold = invariance_threshold
        self.effect_threshold = effect_threshold
        self.dp_threshold = dp_threshold
        
        self.invariance = CounterfactualInvariance()
        self.effect_size = CounterfactualEffectSize()
        self.cf_dp = CounterfactualDemographicParity(dp_threshold)
        self.ite = IndividualTreatmentEffect()
        self.consistency = CounterfactualConsistency()
    
    def evaluate(
        self,
        predictions: np.ndarray,
        counterfactual_predictions: np.ndarray,
        sensitive_attribute: np.ndarray,
        features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive counterfactual fairness evaluation.
        
        Args:
            predictions: Actual predictions
            counterfactual_predictions: Counterfactual predictions
            sensitive_attribute: Sensitive attribute values
            features: Non-sensitive features (optional)
            labels: Ground truth labels (optional)
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            "invariance": {},
            "effect_size": {},
            "demographic_parity": {},
            "ite": {},
            "consistency": {},
            "overall": {},
        }
        
        # Invariance
        results["invariance"]["score"] = self.invariance.compute(
            predictions, counterfactual_predictions
        )
        results["invariance"]["per_group"] = self.invariance.compute_per_group(
            predictions, counterfactual_predictions, sensitive_attribute
        )
        
        # Effect size
        results["effect_size"]["score"] = self.effect_size.compute(
            predictions, counterfactual_predictions
        )
        results["effect_size"]["distribution"] = self.effect_size.compute_distribution(
            predictions, counterfactual_predictions
        )
        
        # Counterfactual demographic parity
        results["demographic_parity"]["difference"] = self.cf_dp.compute(
            predictions, counterfactual_predictions, sensitive_attribute
        )
        results["demographic_parity"]["is_fair"] = self.cf_dp.is_fair(
            predictions, counterfactual_predictions, sensitive_attribute
        )
        
        # Individual treatment effect
        results["ite"] = self.ite.compute(
            predictions, counterfactual_predictions, sensitive_attribute
        )
        
        # Consistency (if features provided)
        if features is not None:
            results["consistency"]["score"] = self.consistency.compute(
                predictions, counterfactual_predictions, features, sensitive_attribute
            )
        
        # Overall counterfactual fairness score
        results["overall"]["cf_fairness_score"] = self._compute_overall_score(results)
        
        return results
    
    def _compute_overall_score(self, results: Dict[str, Any]) -> float:
        """Compute overall counterfactual fairness score."""
        scores = []
        
        # Invariance score (higher is better)
        if "score" in results["invariance"]:
            scores.append(results["invariance"]["score"])
        
        # Effect size (convert to similarity: 1 - normalized_effect)
        if "score" in results["effect_size"]:
            effect_sim = max(0, 1 - results["effect_size"]["score"])
            scores.append(effect_sim)
        
        # Demographic parity (convert to similarity)
        if "difference" in results["demographic_parity"]:
            dp_sim = max(0, 1 - results["demographic_parity"]["difference"])
            scores.append(dp_sim)
        
        # Consistency
        if "score" in results.get("consistency", {}):
            scores.append(results["consistency"]["score"])
        
        return float(np.mean(scores)) if scores else 0.5
