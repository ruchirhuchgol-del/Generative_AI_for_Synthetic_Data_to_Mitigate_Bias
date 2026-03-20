"""
Fairness Auditor
================

Audits synthetic data for fairness violations and biases.
Provides comprehensive fairness assessment with actionable recommendations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import warnings


class FairnessAuditResult:
    """Container for fairness audit results."""
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.violations: List[Dict[str, Any]] = []
        self.recommendations: List[str] = []
        self.is_fair: bool = True
        self.details: Dict[str, Any] = {}
    
    def add_violation(
        self,
        metric_name: str,
        value: float,
        threshold: float,
        description: str
    ) -> None:
        """Add a fairness violation."""
        self.violations.append({
            "metric": metric_name,
            "value": value,
            "threshold": threshold,
            "description": description,
        })
        self.is_fair = False
    
    def add_recommendation(self, recommendation: str) -> None:
        """Add a recommendation."""
        self.recommendations.append(recommendation)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "violations": self.violations,
            "recommendations": self.recommendations,
            "is_fair": self.is_fair,
            "details": self.details,
        }


class RepresentationAuditor:
    """
    Audits representation balance in synthetic data.
    
    Checks whether all groups are adequately represented.
    """
    
    def __init__(
        self,
        min_representation_ratio: float = 0.1,
        max_representation_ratio: float = 10.0
    ):
        """
        Initialize representation auditor.
        
        Args:
            min_representation_ratio: Minimum ratio of smallest to largest group
            max_representation_ratio: Maximum ratio of largest to smallest group
        """
        self.min_ratio = min_representation_ratio
        self.max_ratio = max_representation_ratio
    
    def audit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        sensitive_columns: List[str],
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Audit representation balance.
        
        Args:
            data: Data to audit
            sensitive_columns: Sensitive attribute columns
            columns: Column names if data is numpy array
            
        Returns:
            Representation audit results
        """
        if isinstance(data, np.ndarray):
            if columns is None:
                columns = [f"col_{i}" for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=columns)
        
        results = {
            "per_attribute": {},
            "intersectional": {},
            "overall": {"is_balanced": True},
        }
        
        # Per-attribute analysis
        for col in sensitive_columns:
            if col not in data.columns:
                continue
            
            attr_result = self._audit_single_attribute(data, col)
            results["per_attribute"][col] = attr_result
            
            if not attr_result["is_balanced"]:
                results["overall"]["is_balanced"] = False
        
        # Intersectional analysis
        if len(sensitive_columns) >= 2:
            intersectional_result = self._audit_intersectional(data, sensitive_columns)
            results["intersectional"] = intersectional_result
            
            if not intersectional_result.get("is_balanced", True):
                results["overall"]["is_balanced"] = False
        
        return results
    
    def _audit_single_attribute(
        self,
        data: pd.DataFrame,
        column: str
    ) -> Dict[str, Any]:
        """Audit a single sensitive attribute."""
        value_counts = data[column].value_counts(normalize=True)
        
        if len(value_counts) < 2:
            return {
                "is_balanced": True,
                "ratio": 1.0,
                "distribution": value_counts.to_dict(),
                "n_groups": len(value_counts),
            }
        
        # Compute representation ratio
        max_prop = value_counts.max()
        min_prop = value_counts.min()
        
        ratio = max_prop / min_prop if min_prop > 0 else float('inf')
        
        is_balanced = ratio <= self.max_ratio
        
        return {
            "is_balanced": is_balanced,
            "ratio": float(ratio),
            "distribution": value_counts.to_dict(),
            "n_groups": len(value_counts),
            "max_proportion": float(max_prop),
            "min_proportion": float(min_prop),
            "expected_proportion": float(1.0 / len(value_counts)),
        }
    
    def _audit_intersectional(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> Dict[str, Any]:
        """Audit intersectional representation."""
        # Create intersectional groups
        data = data.copy()
        data['_intersectional'] = data[columns].apply(
            lambda row: '_'.join(str(v) for v in row), axis=1
        )
        
        value_counts = data['_intersectional'].value_counts(normalize=True)
        
        # Check for missing combinations
        expected_combinations = 1
        for col in columns:
            if col in data.columns:
                expected_combinations *= data[col].nunique()
        
        missing_combinations = expected_combinations - len(value_counts)
        
        # Compute metrics
        max_prop = value_counts.max()
        min_prop = value_counts.min()
        ratio = max_prop / min_prop if min_prop > 0 else float('inf')
        
        return {
            "is_balanced": ratio <= self.max_ratio,
            "ratio": float(ratio),
            "n_groups": len(value_counts),
            "expected_groups": expected_combinations,
            "missing_combinations": int(missing_combinations),
            "smallest_group_proportion": float(min_prop),
            "largest_group_proportion": float(max_prop),
        }


class DistributionAuditor:
    """
    Audits distribution consistency between groups.
    
    Checks whether feature distributions are similar across groups.
    """
    
    def __init__(
        self,
        ks_threshold: float = 0.1,
        correlation_threshold: float = 0.3
    ):
        """
        Initialize distribution auditor.
        
        Args:
            ks_threshold: Maximum acceptable KS statistic
            correlation_threshold: Maximum acceptable correlation with sensitive attr
        """
        self.ks_threshold = ks_threshold
        self.correlation_threshold = correlation_threshold
    
    def audit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        sensitive_columns: List[str],
        feature_columns: Optional[List[str]] = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Audit distribution consistency.
        
        Args:
            data: Data to audit
            sensitive_columns: Sensitive attribute columns
            feature_columns: Feature columns to check (None = all non-sensitive)
            columns: Column names if data is numpy array
            
        Returns:
            Distribution audit results
        """
        from scipy import stats
        
        if isinstance(data, np.ndarray):
            if columns is None:
                columns = [f"col_{i}" for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=columns)
        
        # Determine feature columns
        if feature_columns is None:
            feature_columns = [c for c in data.columns if c not in sensitive_columns]
        
        results = {
            "ks_tests": {},
            "correlations": {},
            "skewed_features": [],
            "overall": {"is_consistent": True},
        }
        
        for sens_col in sensitive_columns:
            if sens_col not in data.columns:
                continue
            
            # KS tests for each feature
            ks_results = {}
            for feat_col in feature_columns:
                if feat_col not in data.columns:
                    continue
                
                # Get unique groups
                groups = data[sens_col].unique()
                
                if len(groups) == 2:
                    group0 = data[data[sens_col] == groups[0]][feat_col].dropna()
                    group1 = data[data[sens_col] == groups[1]][feat_col].dropna()
                    
                    if len(group0) > 0 and len(group1) > 0:
                        ks_stat, p_value = stats.ks_2samp(group0, group1)
                        ks_results[feat_col] = {
                            "statistic": float(ks_stat),
                            "p_value": float(p_value),
                            "is_violation": ks_stat > self.ks_threshold,
                        }
                        
                        if ks_stat > self.ks_threshold:
                            results["skewed_features"].append({
                                "feature": feat_col,
                                "sensitive_attribute": sens_col,
                                "ks_statistic": float(ks_stat),
                            })
                            results["overall"]["is_consistent"] = False
            
            results["ks_tests"][sens_col] = ks_results
            
            # Correlation analysis
            if data[sens_col].dtype in [np.int64, np.float64]:
                correlations = {}
                for feat_col in feature_columns:
                    if feat_col in data.columns and data[feat_col].dtype in [np.int64, np.float64]:
                        corr = data[sens_col].corr(data[feat_col])
                        correlations[feat_col] = float(corr)
                
                results["correlations"][sens_col] = correlations
        
        return results


class CorrelationAuditor:
    """
    Audits problematic correlations in synthetic data.
    
    Identifies correlations that may lead to unfair outcomes.
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.5,
        protected_correlations: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize correlation auditor.
        
        Args:
            correlation_threshold: Maximum acceptable correlation
            protected_correlations: Dict mapping sensitive attrs to features that
                                   shouldn't be correlated with them
        """
        self.correlation_threshold = correlation_threshold
        self.protected_correlations = protected_correlations or {}
    
    def audit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        sensitive_columns: List[str],
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Audit correlations.
        
        Args:
            data: Data to audit
            sensitive_columns: Sensitive attribute columns
            columns: Column names if data is numpy array
            
        Returns:
            Correlation audit results
        """
        if isinstance(data, np.ndarray):
            if columns is None:
                columns = [f"col_{i}" for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=columns)
        
        results = {
            "high_correlations": [],
            "protected_violations": [],
            "correlation_matrix": {},
            "is_acceptable": True,
        }
        
        # Compute correlation matrix for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr()
            results["correlation_matrix"] = corr_matrix.to_dict()
            
            # Find high correlations with sensitive attributes
            for sens_col in sensitive_columns:
                if sens_col not in numeric_cols:
                    continue
                
                for other_col in numeric_cols:
                    if other_col == sens_col:
                        continue
                    
                    corr = corr_matrix.loc[sens_col, other_col]
                    
                    if abs(corr) > self.correlation_threshold:
                        results["high_correlations"].append({
                            "sensitive_attribute": sens_col,
                            "feature": other_col,
                            "correlation": float(corr),
                        })
                        
                        # Check if this is a protected correlation
                        if sens_col in self.protected_correlations:
                            if other_col in self.protected_correlations[sens_col]:
                                results["protected_violations"].append({
                                    "sensitive_attribute": sens_col,
                                    "feature": other_col,
                                    "correlation": float(corr),
                                })
                                results["is_acceptable"] = False
        
        return results


class ProxyAuditor:
    """
    Detects proxy variables that could leak sensitive information.
    
    Identifies features that can predict sensitive attributes.
    """
    
    def __init__(
        self,
        proxy_threshold: float = 0.7,
        min_samples: int = 100
    ):
        """
        Initialize proxy auditor.
        
        Args:
            proxy_threshold: R² threshold for proxy detection
            min_samples: Minimum samples for reliable detection
        """
        self.proxy_threshold = proxy_threshold
        self.min_samples = min_samples
    
    def audit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        sensitive_columns: List[str],
        columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Audit for proxy variables.
        
        Args:
            data: Data to audit
            sensitive_columns: Sensitive attribute columns
            columns: Column names if data is numpy array
            
        Returns:
            Proxy audit results
        """
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import LabelEncoder
        
        if isinstance(data, np.ndarray):
            if columns is None:
                columns = [f"col_{i}" for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=columns)
        
        results = {
            "proxy_features": {},
            "overall": {"has_proxies": False},
        }
        
        if len(data) < self.min_samples:
            results["warning"] = f"Insufficient samples ({len(data)}) for reliable proxy detection"
            return results
        
        for sens_col in sensitive_columns:
            if sens_col not in data.columns:
                continue
            
            feature_cols = [c for c in data.columns if c != sens_col]
            numeric_features = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_features) == 0:
                continue
            
            X = data[numeric_features].values
            y = data[sens_col].values
            
            # Remove rows with NaN
            valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < self.min_samples:
                continue
            
            # Encode target if categorical
            if y.dtype == object or len(np.unique(y)) <= 10:
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
                model = LogisticRegression(max_iter=1000, random_state=42)
            else:
                model = LinearRegression()
            
            try:
                # Cross-validated R² score
                scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                mean_r2 = scores.mean()
                
                # Identify top proxy features
                model.fit(X, y)
                feature_importance = dict(zip(numeric_features, model.coef_.flatten()))
                top_proxies = sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]
                
                results["proxy_features"][sens_col] = {
                    "r2_score": float(mean_r2),
                    "is_proxy": mean_r2 > self.proxy_threshold,
                    "top_predictive_features": [
                        {"feature": f, "coefficient": float(c)}
                        for f, c in top_proxies
                    ],
                }
                
                if mean_r2 > self.proxy_threshold:
                    results["overall"]["has_proxies"] = True
            
            except Exception as e:
                results["proxy_features"][sens_col] = {"error": str(e)}
        
        return results


class FairnessAuditor:
    """
    Comprehensive fairness auditor for synthetic data.
    
    Combines multiple auditing modules for thorough assessment.
    """
    
    def __init__(
        self,
        sensitive_columns: List[str],
        thresholds: Optional[Dict[str, float]] = None,
        check_proxies: bool = True,
        check_intersectional: bool = True
    ):
        """
        Initialize fairness auditor.
        
        Args:
            sensitive_columns: List of sensitive attribute columns
            thresholds: Fairness thresholds for each metric
            check_proxies: Whether to check for proxy variables
            check_intersectional: Whether to check intersectional fairness
        """
        self.sensitive_columns = sensitive_columns
        self.thresholds = thresholds or {
            "representation_ratio": 10.0,
            "ks_threshold": 0.1,
            "correlation_threshold": 0.5,
            "proxy_r2_threshold": 0.7,
        }
        self.check_proxies = check_proxies
        self.check_intersectional = check_intersectional
        
        # Initialize sub-auditors
        self.representation_auditor = RepresentationAuditor(
            max_representation_ratio=self.thresholds["representation_ratio"]
        )
        self.distribution_auditor = DistributionAuditor(
            ks_threshold=self.thresholds["ks_threshold"],
            correlation_threshold=self.thresholds.get("correlation_threshold", 0.5)
        )
        self.correlation_auditor = CorrelationAuditor(
            correlation_threshold=self.thresholds.get("correlation_threshold", 0.5)
        )
        
        if check_proxies:
            self.proxy_auditor = ProxyAuditor(
                proxy_threshold=self.thresholds.get("proxy_r2_threshold", 0.7)
            )
    
    def audit(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[str]] = None,
        feature_columns: Optional[List[str]] = None,
        reference_data: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive fairness audit.
        
        Args:
            data: Data to audit
            columns: Column names if data is numpy array
            feature_columns: Feature columns to check
            reference_data: Optional reference data for comparison
            
        Returns:
            Comprehensive audit report
        """
        result = FairnessAuditResult()
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            if columns is None:
                columns = [f"col_{i}" for i in range(data.shape[1])]
            data = pd.DataFrame(data, columns=columns)
        
        # 1. Representation audit
        rep_result = self.representation_auditor.audit(
            data, self.sensitive_columns
        )
        result.details["representation"] = rep_result
        
        for attr, attr_result in rep_result.get("per_attribute", {}).items():
            result.metrics[f"representation_ratio_{attr}"] = attr_result.get("ratio", 1.0)
            
            if not attr_result.get("is_balanced", True):
                result.add_violation(
                    f"representation_{attr}",
                    attr_result.get("ratio", float('inf')),
                    self.thresholds["representation_ratio"],
                    f"Unbalanced representation for {attr}"
                )
        
        # 2. Distribution audit
        dist_result = self.distribution_auditor.audit(
            data, self.sensitive_columns, feature_columns
        )
        result.details["distribution"] = dist_result
        
        for sens_attr, ks_results in dist_result.get("ks_tests", {}).items():
            for feat, ks_data in ks_results.items():
                if ks_data.get("is_violation", False):
                    result.add_violation(
                        f"distribution_{sens_attr}_{feat}",
                        ks_data.get("statistic", 0),
                        self.thresholds["ks_threshold"],
                        f"Distribution difference for {feat} across {sens_attr} groups"
                    )
        
        # 3. Correlation audit
        corr_result = self.correlation_auditor.audit(
            data, self.sensitive_columns
        )
        result.details["correlations"] = corr_result
        
        for high_corr in corr_result.get("high_correlations", []):
            result.add_recommendation(
                f"Consider decorrelating {high_corr['feature']} from "
                f"{high_corr['sensitive_attribute']} (correlation: {high_corr['correlation']:.3f})"
            )
        
        # 4. Proxy audit
        if self.check_proxies and hasattr(self, 'proxy_auditor'):
            proxy_result = self.proxy_auditor.audit(
                data, self.sensitive_columns
            )
            result.details["proxies"] = proxy_result
            
            for sens_attr, proxy_data in proxy_result.get("proxy_features", {}).items():
                if proxy_data.get("is_proxy", False):
                    result.add_violation(
                        f"proxy_{sens_attr}",
                        proxy_data.get("r2_score", 0),
                        self.thresholds.get("proxy_r2_threshold", 0.7),
                        f"Features can predict {sens_attr} (R²={proxy_data.get('r2_score', 0):.3f})"
                    )
        
        # 5. Comparison with reference data
        if reference_data is not None:
            comparison_result = self._compare_with_reference(data, reference_data)
            result.details["reference_comparison"] = comparison_result
        
        # Generate recommendations
        self._generate_recommendations(result)
        
        return result.to_dict()
    
    def _compare_with_reference(
        self,
        synthetic: pd.DataFrame,
        reference: pd.DataFrame
    ) -> Dict[str, Any]:
        """Compare synthetic data with reference data."""
        from scipy import stats
        
        comparison = {
            "distribution_shift": {},
            "representation_shift": {},
        }
        
        for col in self.sensitive_columns:
            if col not in synthetic.columns or col not in reference.columns:
                continue
            
            # Distribution shift
            if synthetic[col].dtype in [np.int64, np.float64]:
                ks_stat, _ = stats.ks_2samp(
                    synthetic[col].dropna(),
                    reference[col].dropna()
                )
                comparison["distribution_shift"][col] = float(ks_stat)
            
            # Representation shift
            synth_props = synthetic[col].value_counts(normalize=True)
            ref_props = reference[col].value_counts(normalize=True)
            
            # Total variation distance
            all_values = set(synth_props.index) | set(ref_props.index)
            tv_distance = sum(
                abs(synth_props.get(v, 0) - ref_props.get(v, 0))
                for v in all_values
            ) / 2
            
            comparison["representation_shift"][col] = float(tv_distance)
        
        return comparison
    
    def _generate_recommendations(self, result: FairnessAuditResult) -> None:
        """Generate actionable recommendations."""
        if not result.is_fair:
            result.add_recommendation(
                "Consider retraining with fairness constraints "
                "(e.g., adversarial debiasing, reweighing)"
            )
        
        if result.details.get("representation", {}).get("intersectional", {}).get("missing_combinations", 0) > 0:
            result.add_recommendation(
                "Some intersectional groups are missing. "
                "Consider oversampling underrepresented combinations."
            )
        
        if result.details.get("proxies", {}).get("overall", {}).get("has_proxies", False):
            result.add_recommendation(
                "Proxy variables detected. Consider removing or transforming "
                "features that can predict sensitive attributes."
            )
    
    def quick_check(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[str]] = None
    ) -> bool:
        """
        Quick fairness check.
        
        Args:
            data: Data to check
            columns: Column names if data is numpy array
            
        Returns:
            True if basic fairness criteria are met
        """
        result = self.audit(data, columns)
        return result["is_fair"]
    
    def get_fairness_score(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        columns: Optional[List[str]] = None
    ) -> float:
        """
        Compute overall fairness score.
        
        Args:
            data: Data to score
            columns: Column names if data is numpy array
            
        Returns:
            Fairness score between 0 and 1
        """
        result = self.audit(data, columns)
        
        # Start with perfect score
        score = 1.0
        
        # Penalize for violations
        for violation in result.get("violations", []):
            severity = abs(violation["value"] - violation["threshold"]) / violation["threshold"]
            score -= min(severity * 0.2, 0.3)  # Max 0.3 penalty per violation
        
        # Ensure score is in [0, 1]
        return max(0.0, min(1.0, score))
