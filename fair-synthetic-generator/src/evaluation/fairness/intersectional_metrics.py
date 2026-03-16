"""
Intersectional Fairness Metrics
================================

Metrics for evaluating fairness across intersectional groups.
Intersectional fairness considers combinations of multiple sensitive
attributes to identify disparities that may be hidden when examining
each attribute in isolation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from itertools import combinations, product
from collections import defaultdict
import warnings


class IntersectionalFairnessMetric:
    """Base class for intersectional fairness metrics."""
    
    def compute(
        self,
        predictions: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute intersectional fairness metric.
        
        Args:
            predictions: Model predictions
            sensitive_attributes: Dictionary mapping attribute names to values
            labels: Ground truth labels (optional)
            
        Returns:
            Metric value
        """
        raise NotImplementedError


class IntersectionalDemographicParity(IntersectionalFairnessMetric):
    """
    Intersectional Demographic Parity metric.
    
    Examines positive prediction rates across all combinations
    of sensitive attribute values.
    
    For example, with race (2 values) and gender (2 values),
    examines rates across 4 intersectional groups.
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        min_group_size: int = 10
    ):
        """
        Initialize intersectional demographic parity.
        
        Args:
            threshold: Maximum acceptable difference in rates
            min_group_size: Minimum group size to include
        """
        self.threshold = threshold
        self.min_group_size = min_group_size
    
    def compute(
        self,
        predictions: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute intersectional demographic parity.
        
        Args:
            predictions: Model predictions
            sensitive_attributes: Dictionary of sensitive attribute arrays
            labels: Ground truth labels (not used for DP)
            
        Returns:
            Maximum difference in positive rates across intersectional groups
        """
        group_rates = self._compute_group_rates(predictions, sensitive_attributes)
        
        if len(group_rates) < 2:
            return 0.0
        
        rates = list(group_rates.values())
        return float(max(rates) - min(rates))
    
    def _compute_group_rates(
        self,
        predictions: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray]
    ) -> Dict[Tuple, float]:
        """Compute positive rates per intersectional group."""
        # Create intersectional groups
        intersectional_groups = self._create_intersectional_groups(sensitive_attributes)
        
        group_rates = {}
        
        for group_key, mask in intersectional_groups.items():
            if mask.sum() >= self.min_group_size:
                group_rates[group_key] = float(predictions[mask].mean())
        
        return group_rates
    
    def _create_intersectional_groups(
        self,
        sensitive_attributes: Dict[str, np.ndarray]
    ) -> Dict[Tuple, np.ndarray]:
        """Create masks for each intersectional group."""
        attr_names = list(sensitive_attributes.keys())
        attr_values = [sensitive_attributes[name] for name in attr_names]
        
        # Get unique values for each attribute
        unique_values = [np.unique(values) for values in attr_values]
        
        groups = {}
        
        # Create all combinations
        for combo in product(*unique_values):
            mask = np.ones(len(attr_values[0]), dtype=bool)
            for i, val in enumerate(combo):
                mask &= (attr_values[i] == val)
            
            if mask.sum() > 0:
                groups[combo] = mask
        
        return groups
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
        attribute_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute detailed intersectional demographic parity report.
        
        Args:
            predictions: Model predictions
            sensitive_attributes: Dictionary of sensitive attribute arrays
            attribute_names: Names of attributes for reporting
            
        Returns:
            Detailed report with per-group metrics
        """
        group_rates = self._compute_group_rates(predictions, sensitive_attributes)
        
        if attribute_names is None:
            attribute_names = list(sensitive_attributes.keys())
        
        # Format results
        per_group = {}
        for group_key, rate in group_rates.items():
            group_name = "_".join(f"{attr_names[i]}_{val}" 
                                  for i, val in enumerate(group_key))
            group_mask = self._create_intersectional_groups(sensitive_attributes)[group_key]
            per_group[group_name] = {
                "rate": rate,
                "size": int(group_mask.sum()),
            }
        
        rates = list(group_rates.values())
        
        return {
            "max_difference": float(max(rates) - min(rates)) if rates else 0.0,
            "min_rate": float(min(rates)) if rates else 0.0,
            "max_rate": float(max(rates)) if rates else 0.0,
            "threshold": self.threshold,
            "is_fair": (max(rates) - min(rates)) <= self.threshold if rates else True,
            "per_group": per_group,
            "n_groups": len(group_rates),
        }


class IntersectionalEqualizedOdds(IntersectionalFairnessMetric):
    """
    Intersectional Equalized Odds metric.
    
    Examines TPR and FPR across all combinations of sensitive attributes.
    """
    
    def __init__(
        self,
        threshold: float = 0.1,
        min_group_size: int = 10
    ):
        """
        Initialize intersectional equalized odds.
        
        Args:
            threshold: Maximum acceptable difference
            min_group_size: Minimum group size
        """
        self.threshold = threshold
        self.min_group_size = min_group_size
    
    def compute(
        self,
        predictions: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
        labels: np.ndarray
    ) -> float:
        """
        Compute intersectional equalized odds.
        
        Args:
            predictions: Model predictions
            sensitive_attributes: Dictionary of sensitive attribute arrays
            labels: Ground truth labels
            
        Returns:
            Maximum of TPR difference and FPR difference
        """
        tprs, fprs = self._compute_group_rates(
            predictions, sensitive_attributes, labels
        )
        
        max_diff = 0.0
        
        if len(tprs) >= 2:
            max_diff = max(max_diff, max(tprs.values()) - min(tprs.values()))
        
        if len(fprs) >= 2:
            max_diff = max(max_diff, max(fprs.values()) - min(fprs.values()))
        
        return float(max_diff)
    
    def _compute_group_rates(
        self,
        predictions: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
        labels: np.ndarray
    ) -> Tuple[Dict[Tuple, float], Dict[Tuple, float]]:
        """Compute TPR and FPR per intersectional group."""
        # Create intersectional groups
        groups = self._create_intersectional_groups(sensitive_attributes)
        
        tprs = {}
        fprs = {}
        
        for group_key, mask in groups.items():
            positive_mask = mask & (labels == 1)
            negative_mask = mask & (labels == 0)
            
            if positive_mask.sum() >= self.min_group_size:
                tprs[group_key] = float(predictions[positive_mask].mean())
            
            if negative_mask.sum() >= self.min_group_size:
                fprs[group_key] = float(predictions[negative_mask].mean())
        
        return tprs, fprs
    
    def _create_intersectional_groups(
        self,
        sensitive_attributes: Dict[str, np.ndarray]
    ) -> Dict[Tuple, np.ndarray]:
        """Create masks for each intersectional group."""
        attr_names = list(sensitive_attributes.keys())
        attr_values = [sensitive_attributes[name] for name in attr_names]
        
        unique_values = [np.unique(values) for values in attr_values]
        
        groups = {}
        
        for combo in product(*unique_values):
            mask = np.ones(len(attr_values[0]), dtype=bool)
            for i, val in enumerate(combo):
                mask &= (attr_values[i] == val)
            
            if mask.sum() > 0:
                groups[combo] = mask
        
        return groups


class IntersectionalDisparateImpact(IntersectionalFairnessMetric):
    """
    Intersectional Disparate Impact metric.
    
    Examines the ratio of positive rates between intersectional groups.
    """
    
    def __init__(
        self,
        min_ratio: float = 0.8,
        max_ratio: float = 1.25,
        min_group_size: int = 10
    ):
        """
        Initialize intersectional disparate impact.
        
        Args:
            min_ratio: Minimum acceptable ratio
            max_ratio: Maximum acceptable ratio
            min_group_size: Minimum group size
        """
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_group_size = min_group_size
    
    def compute(
        self,
        predictions: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
        reference_group: Optional[Tuple] = None
    ) -> Dict[str, float]:
        """
        Compute intersectional disparate impact.
        
        Args:
            predictions: Model predictions
            sensitive_attributes: Dictionary of sensitive attribute arrays
            reference_group: Group to use as reference (if None, uses highest rate group)
            
        Returns:
            Dictionary with DI metrics
        """
        groups = self._create_intersectional_groups(sensitive_attributes)
        
        group_rates = {}
        for group_key, mask in groups.items():
            if mask.sum() >= self.min_group_size:
                group_rates[group_key] = float(predictions[mask].mean())
        
        if len(group_rates) < 2:
            return {"disparate_impact_ratio": 1.0, "is_fair": True}
        
        # Find reference group
        if reference_group is None:
            # Use group with highest rate as reference
            reference_group = max(group_rates.keys(), key=lambda k: group_rates[k])
        
        if reference_group not in group_rates:
            warnings.warn(f"Reference group {reference_group} not found")
            reference_group = max(group_rates.keys(), key=lambda k: group_rates[k])
        
        reference_rate = group_rates[reference_group]
        
        # Compute ratios
        ratios = {}
        for group_key, rate in group_rates.items():
            if reference_rate > 0:
                ratios[group_key] = rate / reference_rate
            else:
                ratios[group_key] = 1.0
        
        # Find most disadvantaged group
        min_ratio = min(ratios.values())
        max_ratio = max(ratios.values())
        
        return {
            "min_ratio": float(min_ratio),
            "max_ratio": float(max_ratio),
            "reference_group": str(reference_group),
            "is_fair": self.min_ratio <= min_ratio <= self.max_ratio,
        }
    
    def _create_intersectional_groups(
        self,
        sensitive_attributes: Dict[str, np.ndarray]
    ) -> Dict[Tuple, np.ndarray]:
        """Create masks for each intersectional group."""
        attr_names = list(sensitive_attributes.keys())
        attr_values = [sensitive_attributes[name] for name in attr_names]
        
        unique_values = [np.unique(values) for values in attr_values]
        
        groups = {}
        
        for combo in product(*unique_values):
            mask = np.ones(len(attr_values[0]), dtype=bool)
            for i, val in enumerate(combo):
                mask &= (attr_values[i] == val)
            
            if mask.sum() > 0:
                groups[combo] = mask
        
        return groups


class SubgroupAnalysis(IntersectionalFairnessMetric):
    """
    Subgroup Analysis for detailed intersectional examination.
    
    Identifies subgroups that are most disadvantaged
    and quantifies disparities between them.
    """
    
    def __init__(
        self,
        min_group_size: int = 10,
        percentile_threshold: float = 0.25
    ):
        """
        Initialize subgroup analysis.
        
        Args:
            min_group_size: Minimum group size
            percentile_threshold: Percentile for identifying worst-off groups
        """
        self.min_group_size = min_group_size
        self.percentile_threshold = percentile_threshold
    
    def compute(
        self,
        predictions: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform subgroup analysis.
        
        Args:
            predictions: Model predictions
            sensitive_attributes: Dictionary of sensitive attribute arrays
            labels: Ground truth labels (optional)
            
        Returns:
            Subgroup analysis results
        """
        groups = self._create_intersectional_groups(sensitive_attributes)
        
        group_stats = {}
        
        for group_key, mask in groups.items():
            if mask.sum() >= self.min_group_size:
                group_preds = predictions[mask]
                stats = {
                    "mean": float(group_preds.mean()),
                    "std": float(group_preds.std()),
                    "size": int(mask.sum()),
                    "rate": float(group_preds.mean()),  # For binary predictions
                }
                
                if labels is not None:
                    group_labels = labels[mask]
                    positive_mask = group_labels == 1
                    if positive_mask.sum() > 0:
                        stats["tpr"] = float(group_preds[positive_mask].mean())
                    negative_mask = group_labels == 0
                    if negative_mask.sum() > 0:
                        stats["fpr"] = float(group_preds[negative_mask].mean())
                
                group_stats[group_key] = stats
        
        if not group_stats:
            return {"error": "No valid groups found"}
        
        # Identify best and worst groups
        rates = {k: v["rate"] for k, v in group_stats.items()}
        sorted_groups = sorted(rates.keys(), key=lambda k: rates[k])
        
        n_worst = max(1, int(len(sorted_groups) * self.percentile_threshold))
        worst_groups = sorted_groups[:n_worst]
        best_groups = sorted_groups[-n_worst:]
        
        # Gap analysis
        worst_rate = rates[sorted_groups[0]]
        best_rate = rates[sorted_groups[-1]]
        
        return {
            "group_stats": {str(k): v for k, v in group_stats.items()},
            "worst_off_groups": [str(g) for g in worst_groups],
            "best_off_groups": [str(g) for g in best_groups],
            "min_rate": float(worst_rate),
            "max_rate": float(best_rate),
            "rate_gap": float(best_rate - worst_rate),
            "relative_gap": float((best_rate - worst_rate) / max(best_rate, 1e-10)),
            "n_groups": len(group_stats),
        }
    
    def _create_intersectional_groups(
        self,
        sensitive_attributes: Dict[str, np.ndarray]
    ) -> Dict[Tuple, np.ndarray]:
        """Create masks for each intersectional group."""
        attr_values = list(sensitive_attributes.values())
        unique_values = [np.unique(values) for values in attr_values]
        
        groups = {}
        
        for combo in product(*unique_values):
            mask = np.ones(len(attr_values[0]), dtype=bool)
            for i, val in enumerate(combo):
                mask &= (attr_values[i] == val)
            
            if mask.sum() > 0:
                groups[combo] = mask
        
        return groups


class AttributeInteractionEffect(IntersectionalFairnessMetric):
    """
    Attribute Interaction Effect metric.
    
    Measures whether the combined effect of multiple sensitive attributes
    differs from what would be expected from their individual effects.
    """
    
    def compute(
        self,
        predictions: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute attribute interaction effects.
        
        Args:
            predictions: Model predictions
            sensitive_attributes: Dictionary of sensitive attribute arrays
            
        Returns:
            Dictionary with interaction effect metrics
        """
        attr_names = list(sensitive_attributes.keys())
        
        if len(attr_names) < 2:
            return {"interaction_effect": 0.0, "n_pairs": 0}
        
        # Compute main effects
        main_effects = {}
        for attr in attr_names:
            values = sensitive_attributes[attr]
            unique_vals = np.unique(values)
            if len(unique_vals) >= 2:
                effect = abs(
                    predictions[values == unique_vals[0]].mean() -
                    predictions[values == unique_vals[1]].mean()
                )
                main_effects[attr] = effect
        
        # Compute pairwise interaction effects
        interaction_effects = {}
        
        for attr_a, attr_b in combinations(attr_names, 2):
            values_a = sensitive_attributes[attr_a]
            values_b = sensitive_attributes[attr_b]
            
            unique_a = np.unique(values_a)
            unique_b = np.unique(values_b)
            
            # Compute 2-way interaction
            cell_means = {}
            cell_sizes = {}
            
            for val_a in unique_a:
                for val_b in unique_b:
                    mask = (values_a == val_a) & (values_b == val_b)
                    if mask.sum() > 0:
                        cell_means[(val_a, val_b)] = predictions[mask].mean()
                        cell_sizes[(val_a, val_b)] = mask.sum()
            
            if len(cell_means) >= 4:  # Need all 4 cells for 2x2
                # Compute interaction effect
                means = list(cell_means.values())
                interaction = self._compute_interaction_effect(means)
                interaction_effects[f"{attr_a}_{attr_b}"] = interaction
        
        return {
            "main_effects": main_effects,
            "interaction_effects": interaction_effects,
            "max_interaction": max(interaction_effects.values()) if interaction_effects else 0.0,
            "avg_interaction": np.mean(list(interaction_effects.values())) if interaction_effects else 0.0,
        }
    
    def _compute_interaction_effect(self, means: List[float]) -> float:
        """
        Compute 2x2 interaction effect.
        
        For cells: a, b (row 1), c, d (row 2)
        Interaction = (a - b) - (c - d)
        """
        if len(means) < 4:
            return 0.0
        
        a, b, c, d = means[:4]
        return abs((a - b) - (c - d))


class IntersectionalFairnessEvaluator:
    """
    Comprehensive intersectional fairness evaluator.
    
    Combines multiple intersectional metrics for thorough evaluation.
    """
    
    def __init__(
        self,
        dp_threshold: float = 0.1,
        eo_threshold: float = 0.1,
        min_group_size: int = 10
    ):
        """
        Initialize intersectional fairness evaluator.
        
        Args:
            dp_threshold: Demographic parity threshold
            eo_threshold: Equalized odds threshold
            min_group_size: Minimum group size
        """
        self.dp_threshold = dp_threshold
        self.eo_threshold = eo_threshold
        self.min_group_size = min_group_size
        
        self.dp = IntersectionalDemographicParity(dp_threshold, min_group_size)
        self.eo = IntersectionalEqualizedOdds(eo_threshold, min_group_size)
        self.di = IntersectionalDisparateImpact(min_group_size=min_group_size)
        self.subgroup = SubgroupAnalysis(min_group_size)
        self.interaction = AttributeInteractionEffect()
    
    def evaluate(
        self,
        predictions: np.ndarray,
        sensitive_attributes: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive intersectional fairness evaluation.
        
        Args:
            predictions: Model predictions
            sensitive_attributes: Dictionary of sensitive attribute arrays
            labels: Ground truth labels (optional)
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            "demographic_parity": {},
            "equalized_odds": {},
            "disparate_impact": {},
            "subgroup_analysis": {},
            "interaction_effects": {},
            "overall": {},
        }
        
        # Demographic parity
        results["demographic_parity"]["difference"] = self.dp.compute(
            predictions, sensitive_attributes
        )
        results["demographic_parity"]["detailed"] = self.dp.compute_detailed(
            predictions, sensitive_attributes
        )
        
        # Equalized odds (if labels provided)
        if labels is not None:
            results["equalized_odds"]["difference"] = self.eo.compute(
                predictions, sensitive_attributes, labels
            )
        
        # Disparate impact
        results["disparate_impact"] = self.di.compute(
            predictions, sensitive_attributes
        )
        
        # Subgroup analysis
        results["subgroup_analysis"] = self.subgroup.compute(
            predictions, sensitive_attributes, labels
        )
        
        # Interaction effects
        results["interaction_effects"] = self.interaction.compute(
            predictions, sensitive_attributes
        )
        
        # Overall score
        results["overall"]["intersectional_fairness_score"] = self._compute_overall_score(results)
        
        return results
    
    def _compute_overall_score(self, results: Dict[str, Any]) -> float:
        """Compute overall intersectional fairness score."""
        scores = []
        
        # DP score
        dp_diff = results["demographic_parity"].get("difference", 0)
        scores.append(max(0, 1 - dp_diff))
        
        # EO score
        eo_diff = results["equalized_odds"].get("difference", 0)
        if eo_diff > 0:
            scores.append(max(0, 1 - eo_diff))
        
        # DI score
        di_result = results.get("disparate_impact", {})
        min_ratio = di_result.get("min_ratio", 1.0)
        if min_ratio < self.di.min_ratio:
            scores.append(min_ratio / self.di.min_ratio)
        elif min_ratio > self.di.max_ratio:
            scores.append(self.di.max_ratio / min_ratio)
        else:
            scores.append(1.0)
        
        # Interaction score
        max_interaction = results.get("interaction_effects", {}).get("max_interaction", 0)
        scores.append(max(0, 1 - max_interaction))
        
        return float(np.mean(scores)) if scores else 0.5
