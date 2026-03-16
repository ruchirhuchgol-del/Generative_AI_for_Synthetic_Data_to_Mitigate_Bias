"""
Group Fairness Metrics
======================

Comprehensive evaluation metrics for group fairness including:
- Demographic Parity (Statistical Parity)
- Equalized Odds
- Equal Opportunity
- Predictive Parity
- Calibration
- Disparate Impact
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats
import warnings


class GroupFairnessMetric:
    """Base class for group fairness metrics."""
    
    def __init__(self, threshold: float = 0.05):
        """
        Initialize metric.
        
        Args:
            threshold: Threshold for determining if fairness is satisfied
        """
        self.threshold = threshold
    
    def compute(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> float:
        """Compute the metric value."""
        raise NotImplementedError
    
    def is_fair(self, value: float) -> bool:
        """Check if metric value satisfies fairness threshold."""
        return value <= self.threshold


class DemographicParityMetric(GroupFairnessMetric):
    """
    Demographic Parity (Statistical Parity) metric.
    
    Measures the difference in positive prediction rates across groups.
    A classifier satisfies demographic parity if:
    P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
    
    The difference measures unfairness - lower is better.
    """
    
    def compute(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute demographic parity difference.
        
        Args:
            predictions: Model predictions (binary or probabilities)
            groups: Group membership (0, 1, 2, ...)
            
        Returns:
            Maximum absolute difference in positive rates
        """
        unique_groups = np.unique(groups)
        group_rates = []
        
        for g in unique_groups:
            mask = groups == g
            if mask.sum() > 0:
                rate = predictions[mask].mean()
                group_rates.append(rate)
        
        if len(group_rates) < 2:
            return 0.0
        
        return float(max(group_rates) - min(group_rates))
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        group_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute detailed demographic parity report.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            group_names: Optional group names
            
        Returns:
            Detailed report dictionary
        """
        unique_groups = np.unique(groups)
        
        if group_names is None:
            group_names = [f"Group_{g}" for g in unique_groups]
        
        group_rates = {}
        group_sizes = {}
        
        for g, name in zip(unique_groups, group_names):
            mask = groups == g
            if mask.sum() > 0:
                group_rates[name] = float(predictions[mask].mean())
                group_sizes[name] = int(mask.sum())
        
        rates = list(group_rates.values())
        
        return {
            "metric": "demographic_parity",
            "difference": self.compute(predictions, groups),
            "threshold": self.threshold,
            "is_fair": self.is_fair(max(rates) - min(rates)),
            "group_rates": group_rates,
            "group_sizes": group_sizes,
            "std_rates": float(np.std(rates)),
        }
    
    def compute_ratio(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        privileged_group: int = 0
    ) -> float:
        """
        Compute demographic parity ratio.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            privileged_group: Index of privileged group
            
        Returns:
            Ratio of minimum to maximum positive rates
        """
        unique_groups = np.unique(groups)
        group_rates = []
        
        for g in unique_groups:
            mask = groups == g
            if mask.sum() > 0:
                rate = predictions[mask].mean()
                group_rates.append(rate)
        
        if len(group_rates) < 2 or max(group_rates) == 0:
            return 1.0
        
        return float(min(group_rates) / max(group_rates))


class EqualizedOddsMetric(GroupFairnessMetric):
    """
    Equalized Odds metric.
    
    Requires equal TPR and FPR across groups.
    P(Ŷ=1|Y=1, A=0) = P(Ŷ=1|Y=1, A=1)  (TPR parity)
    P(Ŷ=1|Y=0, A=0) = P(Ŷ=1|Y=0, A=1)  (FPR parity)
    """
    
    def compute(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute equalized odds difference.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            labels: Ground truth labels
            
        Returns:
            Maximum of TPR difference and FPR difference
        """
        unique_groups = np.unique(groups)
        
        tprs = []
        fprs = []
        
        for g in unique_groups:
            mask = groups == g
            positive_mask = mask & (labels == 1)
            negative_mask = mask & (labels == 0)
            
            if positive_mask.sum() > 0:
                tpr = predictions[positive_mask].mean()
                tprs.append(tpr)
            
            if negative_mask.sum() > 0:
                fpr = predictions[negative_mask].mean()
                fprs.append(fpr)
        
        max_diff = 0.0
        
        if len(tprs) >= 2:
            max_diff = max(max_diff, max(tprs) - min(tprs))
        
        if len(fprs) >= 2:
            max_diff = max(max_diff, max(fprs) - min(fprs))
        
        return float(max_diff)
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray,
        group_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute detailed equalized odds report.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            labels: Ground truth labels
            group_names: Optional group names
            
        Returns:
            Detailed report dictionary
        """
        unique_groups = np.unique(groups)
        
        if group_names is None:
            group_names = [f"Group_{g}" for g in unique_groups]
        
        tprs = {}
        fprs = {}
        tnrs = {}
        fnrs = {}
        
        for g, name in zip(unique_groups, group_names):
            mask = groups == g
            positive_mask = mask & (labels == 1)
            negative_mask = mask & (labels == 0)
            
            if positive_mask.sum() > 0:
                tprs[name] = float(predictions[positive_mask].mean())
                fnrs[name] = float(1 - tprs[name])
            
            if negative_mask.sum() > 0:
                fprs[name] = float(predictions[negative_mask].mean())
                tnrs[name] = float(1 - fprs[name])
        
        tpr_diff = max(tprs.values()) - min(tprs.values()) if tprs else 0
        fpr_diff = max(fprs.values()) - min(fprs.values()) if fprs else 0
        
        return {
            "metric": "equalized_odds",
            "difference": self.compute(predictions, groups, labels),
            "threshold": self.threshold,
            "is_fair": self.is_fair(max(tpr_diff, fpr_diff)),
            "tpr_by_group": tprs,
            "fpr_by_group": fprs,
            "tnr_by_group": tnrs,
            "fnr_by_group": fnrs,
            "tpr_difference": float(tpr_diff),
            "fpr_difference": float(fpr_diff),
        }


class EqualOpportunityMetric(GroupFairnessMetric):
    """
    Equal Opportunity metric.
    
    Relaxed version of Equalized Odds - only requires TPR parity.
    P(Ŷ=1|Y=1, A=0) = P(Ŷ=1|Y=1, A=1)
    """
    
    def compute(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute equal opportunity difference.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            labels: Ground truth labels
            
        Returns:
            Difference in TPR across groups
        """
        unique_groups = np.unique(groups)
        tprs = []
        
        for g in unique_groups:
            mask = groups == g
            positive_mask = mask & (labels == 1)
            
            if positive_mask.sum() > 0:
                tpr = predictions[positive_mask].mean()
                tprs.append(tpr)
        
        if len(tprs) < 2:
            return 0.0
        
        return float(max(tprs) - min(tprs))
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray,
        group_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute detailed equal opportunity report."""
        unique_groups = np.unique(groups)
        
        if group_names is None:
            group_names = [f"Group_{g}" for g in unique_groups]
        
        tprs = {}
        
        for g, name in zip(unique_groups, group_names):
            mask = groups == g
            positive_mask = mask & (labels == 1)
            
            if positive_mask.sum() > 0:
                tprs[name] = float(predictions[positive_mask].mean())
        
        tpr_diff = max(tprs.values()) - min(tprs.values()) if tprs else 0
        
        return {
            "metric": "equal_opportunity",
            "difference": float(tpr_diff),
            "threshold": self.threshold,
            "is_fair": self.is_fair(tpr_diff),
            "tpr_by_group": tprs,
        }


class PredictiveParityMetric(GroupFairnessMetric):
    """
    Predictive Parity (Precision Parity) metric.
    
    Requires equal precision (PPV) across groups.
    P(Y=1|Ŷ=1, A=0) = P(Y=1|Ŷ=1, A=1)
    """
    
    def compute(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Compute predictive parity difference.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            labels: Ground truth labels
            threshold: Threshold for binary predictions
            
        Returns:
            Difference in precision across groups
        """
        binary_preds = (predictions >= threshold).astype(int)
        unique_groups = np.unique(groups)
        precisions = []
        
        for g in unique_groups:
            mask = groups == g
            positive_pred_mask = mask & (binary_preds == 1)
            
            if positive_pred_mask.sum() > 0:
                precision = labels[positive_pred_mask].mean()
                precisions.append(precision)
        
        if len(precisions) < 2:
            return 0.0
        
        return float(max(precisions) - min(precisions))


class CalibrationMetric(GroupFairnessMetric):
    """
    Calibration metric for group fairness.
    
    Measures whether predicted probabilities are calibrated
    equally across groups.
    """
    
    def __init__(self, n_bins: int = 10, threshold: float = 0.1):
        """
        Initialize calibration metric.
        
        Args:
            n_bins: Number of bins for calibration
            threshold: Maximum acceptable calibration difference
        """
        super().__init__(threshold)
        self.n_bins = n_bins
    
    def compute(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """
        Compute calibration difference across groups.
        
        Args:
            predictions: Model predictions (probabilities)
            groups: Group membership
            labels: Ground truth labels
            
        Returns:
            Maximum calibration error difference
        """
        unique_groups = np.unique(groups)
        calibration_errors = []
        
        for g in unique_groups:
            mask = groups == g
            group_preds = predictions[mask]
            group_labels = labels[mask]
            
            ce = self._compute_calibration_error(group_preds, group_labels)
            calibration_errors.append(ce)
        
        if len(calibration_errors) < 2:
            return 0.0
        
        return float(max(calibration_errors) - min(calibration_errors))
    
    def _compute_calibration_error(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        ece = 0.0
        
        for i in range(self.n_bins):
            in_bin = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
        
        return ece


class DisparateImpactMetric(GroupFairnessMetric):
    """
    Disparate Impact metric.
    
    Measures the ratio of positive prediction rates between groups.
    Based on the "80% rule" from US employment law.
    """
    
    def __init__(
        self,
        min_ratio: float = 0.8,
        max_ratio: float = 1.25
    ):
        """
        Initialize disparate impact metric.
        
        Args:
            min_ratio: Minimum acceptable ratio (typically 0.8 for 80% rule)
            max_ratio: Maximum acceptable ratio
        """
        super().__init__(min_ratio)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
    
    def compute(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        privileged_group: int = 0,
        labels: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute disparate impact ratio.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            privileged_group: Index of privileged group
            
        Returns:
            Ratio of positive rates (unprivileged/privileged)
        """
        privileged_mask = groups == privileged_group
        unprivileged_mask = groups != privileged_group
        
        if privileged_mask.sum() == 0 or unprivileged_mask.sum() == 0:
            return 1.0
        
        privileged_rate = predictions[privileged_mask].mean()
        unprivileged_rate = predictions[unprivileged_mask].mean()
        
        if privileged_rate == 0:
            return float('inf') if unprivileged_rate > 0 else 1.0
        
        return float(unprivileged_rate / privileged_rate)
    
    def is_fair(self, ratio: float) -> bool:
        """Check if ratio is within acceptable bounds."""
        return self.min_ratio <= ratio <= self.max_ratio
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        privileged_group: int = 0,
        group_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute detailed disparate impact report."""
        unique_groups = np.unique(groups)
        
        if group_names is None:
            group_names = [f"Group_{g}" for g in unique_groups]
        
        privileged_mask = groups == privileged_group
        privileged_rate = predictions[privileged_mask].mean() if privileged_mask.sum() > 0 else 0
        
        group_rates = {}
        group_ratios = {}
        
        for g, name in zip(unique_groups, group_names):
            mask = groups == g
            if mask.sum() > 0:
                rate = predictions[mask].mean()
                group_rates[name] = float(rate)
                if privileged_rate > 0:
                    group_ratios[name] = float(rate / privileged_rate)
                else:
                    group_ratios[name] = 1.0
        
        di_ratio = self.compute(predictions, groups, privileged_group)
        
        return {
            "metric": "disparate_impact",
            "ratio": di_ratio,
            "min_threshold": self.min_ratio,
            "max_threshold": self.max_ratio,
            "is_fair": self.is_fair(di_ratio),
            "group_rates": group_rates,
            "group_ratios": group_ratios,
            "privileged_group": group_names[privileged_group] if privileged_group < len(group_names) else f"Group_{privileged_group}",
        }


class AccuracyParityMetric(GroupFairnessMetric):
    """
    Accuracy Parity metric.
    
    Requires equal accuracy across groups.
    P(Ŷ=Y|A=0) = P(Ŷ=Y|A=1)
    """
    
    def compute(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        """
        Compute accuracy parity difference.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            labels: Ground truth labels
            threshold: Threshold for binary predictions
            
        Returns:
            Difference in accuracy across groups
        """
        binary_preds = (predictions >= threshold).astype(int)
        unique_groups = np.unique(groups)
        accuracies = []
        
        for g in unique_groups:
            mask = groups == g
            if mask.sum() > 0:
                accuracy = (binary_preds[mask] == labels[mask]).mean()
                accuracies.append(accuracy)
        
        if len(accuracies) < 2:
            return 0.0
        
        return float(max(accuracies) - min(accuracies))
    
    def compute_detailed(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray,
        threshold: float = 0.5,
        group_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute detailed accuracy parity report."""
        binary_preds = (predictions >= threshold).astype(int)
        unique_groups = np.unique(groups)
        
        if group_names is None:
            group_names = [f"Group_{g}" for g in unique_groups]
        
        accuracies = {}
        
        for g, name in zip(unique_groups, group_names):
            mask = groups == g
            if mask.sum() > 0:
                accuracies[name] = float((binary_preds[mask] == labels[mask]).mean())
        
        acc_diff = max(accuracies.values()) - min(accuracies.values()) if accuracies else 0
        
        return {
            "metric": "accuracy_parity",
            "difference": float(acc_diff),
            "threshold": self.threshold,
            "is_fair": self.is_fair(acc_diff),
            "accuracy_by_group": accuracies,
        }


class StatisticalParityDifference(GroupFairnessMetric):
    """
    Statistical Parity Difference metric.
    
    Alternative formulation: SP = P(Ŷ=1|A=unprivileged) - P(Ŷ=1|A=privileged)
    Values close to 0 indicate fairness.
    """
    
    def compute(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        privileged_group: int = 0,
        labels: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute statistical parity difference.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            privileged_group: Index of privileged group
            
        Returns:
            Difference in positive rates (unprivileged - privileged)
        """
        privileged_mask = groups == privileged_group
        unprivileged_mask = groups != privileged_group
        
        if privileged_mask.sum() == 0 or unprivileged_mask.sum() == 0:
            return 0.0
        
        privileged_rate = predictions[privileged_mask].mean()
        unprivileged_rate = predictions[unprivileged_mask].mean()
        
        return float(unprivileged_rate - privileged_rate)


class GroupFairnessEvaluator:
    """
    Comprehensive group fairness evaluator.
    
    Combines multiple group fairness metrics for thorough evaluation.
    """
    
    def __init__(
        self,
        fairness_threshold: float = 0.05,
        di_min_ratio: float = 0.8
    ):
        """
        Initialize group fairness evaluator.
        
        Args:
            fairness_threshold: Threshold for fairness metrics
            di_min_ratio: Minimum ratio for disparate impact
        """
        self.fairness_threshold = fairness_threshold
        
        self.metrics = {
            "demographic_parity": DemographicParityMetric(fairness_threshold),
            "equalized_odds": EqualizedOddsMetric(fairness_threshold),
            "equal_opportunity": EqualOpportunityMetric(fairness_threshold),
            "predictive_parity": PredictiveParityMetric(fairness_threshold),
            "calibration": CalibrationMetric(threshold=fairness_threshold),
            "disparate_impact": DisparateImpactMetric(min_ratio=di_min_ratio),
            "accuracy_parity": AccuracyParityMetric(fairness_threshold),
            "statistical_parity_difference": StatisticalParityDifference(fairness_threshold),
        }
    
    def evaluate(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: np.ndarray,
        group_names: Optional[List[str]] = None,
        privileged_group: int = 0
    ) -> Dict[str, Any]:
        """
        Comprehensive group fairness evaluation.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            labels: Ground truth labels
            group_names: Optional group names
            privileged_group: Index of privileged group
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            "metrics": {},
            "detailed": {},
            "overall": {},
        }
        
        # Demographic parity
        results["metrics"]["demographic_parity"] = self.metrics["demographic_parity"].compute(
            predictions, groups
        )
        results["detailed"]["demographic_parity"] = self.metrics["demographic_parity"].compute_detailed(
            predictions, groups, group_names
        )
        
        # Equalized odds
        results["metrics"]["equalized_odds"] = self.metrics["equalized_odds"].compute(
            predictions, groups, labels
        )
        results["detailed"]["equalized_odds"] = self.metrics["equalized_odds"].compute_detailed(
            predictions, groups, labels, group_names
        )
        
        # Equal opportunity
        results["metrics"]["equal_opportunity"] = self.metrics["equal_opportunity"].compute(
            predictions, groups, labels
        )
        results["detailed"]["equal_opportunity"] = self.metrics["equal_opportunity"].compute_detailed(
            predictions, groups, labels, group_names
        )
        
        # Disparate impact
        results["metrics"]["disparate_impact"] = self.metrics["disparate_impact"].compute(
            predictions, groups, privileged_group
        )
        results["detailed"]["disparate_impact"] = self.metrics["disparate_impact"].compute_detailed(
            predictions, groups, privileged_group, group_names
        )
        
        # Accuracy parity
        results["metrics"]["accuracy_parity"] = self.metrics["accuracy_parity"].compute(
            predictions, groups, labels
        )
        results["detailed"]["accuracy_parity"] = self.metrics["accuracy_parity"].compute_detailed(
            predictions, groups, labels, group_names=group_names
        )
        
        # Overall fairness assessment
        satisfied_count = sum(
            1 for metric, result in results["detailed"].items()
            if result.get("is_fair", True)
        )
        total_metrics = len(results["detailed"])
        
        results["overall"] = {
            "fairness_score": float(satisfied_count / total_metrics) if total_metrics > 0 else 1.0,
            "n_metrics_satisfied": satisfied_count,
            "n_metrics_total": total_metrics,
            "is_fair": satisfied_count == total_metrics,
        }
        
        return results
