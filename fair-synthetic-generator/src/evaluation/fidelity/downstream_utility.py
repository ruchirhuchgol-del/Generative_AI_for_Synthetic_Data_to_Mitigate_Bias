"""
Downstream Utility Metrics for Synthetic Data
==============================================

Metrics for evaluating how useful synthetic data is for
downstream machine learning tasks.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score, mean_absolute_error
)


class DownstreamUtilityMetric:
    """Base class for downstream utility metrics."""
    
    def evaluate(
        self,
        real_train: np.ndarray,
        real_test: np.ndarray,
        synthetic_train: np.ndarray,
        real_train_labels: np.ndarray,
        real_test_labels: np.ndarray,
        synthetic_train_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate downstream utility.
        
        Args:
            real_train: Real training data
            real_test: Real test data
            synthetic_train: Synthetic training data
            real_train_labels: Real training labels
            real_test_labels: Real test labels
            synthetic_train_labels: Synthetic training labels
            
        Returns:
            Dictionary with utility metrics
        """
        raise NotImplementedError


class TrainOnSyntheticTestOnReal(DownstreamUtilityMetric):
    """
    Train on Synthetic, Test on Real (TSTR) evaluation.
    
    Trains a model on synthetic data and evaluates on real data.
    Higher performance indicates better utility preservation.
    """
    
    def __init__(
        self,
        model_type: str = "auto",
        task_type: str = "auto",
        cv_folds: int = 5
    ):
        """
        Initialize TSTR evaluator.
        
        Args:
            model_type: Model to use ('lr', 'rf', 'auto')
            task_type: Task type ('classification', 'regression', 'auto')
            cv_folds: Number of cross-validation folds
        """
        self.model_type = model_type
        self.task_type = task_type
        self.cv_folds = cv_folds
    
    def _detect_task_type(self, labels: np.ndarray) -> str:
        """Detect task type from labels."""
        unique_values = np.unique(labels)
        if len(unique_values) <= 10 and labels.dtype in [np.int32, np.int64, np.float64]:
            if len(unique_values) == 2 or set(unique_values).issubset({0, 1, -1}):
                return "classification"
        
        # Check if labels appear to be continuous
        if len(unique_values) / len(labels) < 0.1:
            return "classification"
        
        return "regression"
    
    def _get_model(self, task_type: str):
        """Get model instance."""
        if self.model_type == "lr":
            if task_type == "classification":
                return LogisticRegression(max_iter=1000, random_state=42)
            else:
                return LinearRegression()
        elif self.model_type == "rf":
            if task_type == "classification":
                return RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                return RandomForestRegressor(n_estimators=100, random_state=42)
        else:  # auto
            if task_type == "classification":
                return LogisticRegression(max_iter=1000, random_state=42)
            else:
                return LinearRegression()
    
    def evaluate(
        self,
        real_train: np.ndarray,
        real_test: np.ndarray,
        synthetic_train: np.ndarray,
        real_train_labels: np.ndarray,
        real_test_labels: np.ndarray,
        synthetic_train_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate TSTR utility.
        
        Args:
            real_train: Real training data
            real_test: Real test data
            synthetic_train: Synthetic training data
            real_train_labels: Real training labels
            real_test_labels: Real test labels
            synthetic_train_labels: Synthetic training labels
            
        Returns:
            Dictionary with utility metrics
        """
        # Detect task type
        task_type = self.task_type
        if task_type == "auto":
            task_type = self._detect_task_type(real_test_labels)
        
        # Get model
        model = self._get_model(task_type)
        
        # Train on synthetic, test on real
        try:
            model.fit(synthetic_train, synthetic_train_labels)
            predictions = model.predict(real_test)
            
            if task_type == "classification":
                accuracy = accuracy_score(real_test_labels, predictions)
                
                try:
                    if len(np.unique(real_test_labels)) == 2:
                        proba = model.predict_proba(real_test)[:, 1]
                        auc = roc_auc_score(real_test_labels, proba)
                    else:
                        auc = 0.5
                except:
                    auc = 0.5
                
                f1 = f1_score(real_test_labels, predictions, average='weighted', zero_division=0)
                
                results = {
                    "tstr_accuracy": accuracy,
                    "tstr_f1": f1,
                    "tstr_auc": auc,
                }
            else:
                mse = mean_squared_error(real_test_labels, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(real_test_labels, predictions)
                
                try:
                    r2 = r2_score(real_test_labels, predictions)
                except:
                    r2 = 0.0
                
                results = {
                    "tstr_mse": mse,
                    "tstr_rmse": rmse,
                    "tstr_mae": mae,
                    "tstr_r2": r2,
                }
            
            return results
            
        except Exception as e:
            warnings.warn(f"TSTR evaluation failed: {e}")
            return {"error": str(e)}
    
    def evaluate_with_baseline(
        self,
        real_train: np.ndarray,
        real_test: np.ndarray,
        synthetic_train: np.ndarray,
        real_train_labels: np.ndarray,
        real_test_labels: np.ndarray,
        synthetic_train_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate TSTR with baseline (train on real, test on real).
        
        Args:
            real_train: Real training data
            real_test: Real test data
            synthetic_train: Synthetic training data
            real_train_labels: Real training labels
            real_test_labels: Real test labels
            synthetic_train_labels: Synthetic training labels
            
        Returns:
            Dictionary with comparison metrics
        """
        results = self.evaluate(
            real_train, real_test, synthetic_train,
            real_train_labels, real_test_labels, synthetic_train_labels
        )
        
        # Baseline: train on real, test on real
        task_type = self.task_type
        if task_type == "auto":
            task_type = self._detect_task_type(real_test_labels)
        
        model = self._get_model(task_type)
        
        try:
            model.fit(real_train, real_train_labels)
            predictions = model.predict(real_test)
            
            if task_type == "classification":
                baseline_accuracy = accuracy_score(real_test_labels, predictions)
                results["trtr_accuracy"] = baseline_accuracy
                results["utility_ratio"] = results.get("tstr_accuracy", 0) / max(baseline_accuracy, 0.01)
            else:
                baseline_mse = mean_squared_error(real_test_labels, predictions)
                results["trtr_mse"] = baseline_mse
                results["utility_ratio"] = baseline_mse / max(results.get("tstr_mse", 1), 0.01)
        
        except Exception as e:
            warnings.warn(f"Baseline evaluation failed: {e}")
        
        return results


class TrainOnRealTestOnSynthetic(DownstreamUtilityMetric):
    """
    Train on Real, Test on Synthetic (TRTS) evaluation.
    
    Trains a model on real data and evaluates on synthetic data.
    Measures how well synthetic data preserves decision boundaries.
    """
    
    def __init__(
        self,
        model_type: str = "auto",
        task_type: str = "auto"
    ):
        """
        Initialize TRTS evaluator.
        
        Args:
            model_type: Model to use ('lr', 'rf', 'auto')
            task_type: Task type ('classification', 'regression', 'auto')
        """
        self.model_type = model_type
        self.task_type = task_type
        self._tstr = TrainOnSyntheticTestOnReal(model_type, task_type)
    
    def evaluate(
        self,
        real_train: np.ndarray,
        synthetic_test: np.ndarray,
        real_train_labels: np.ndarray,
        synthetic_test_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate TRTS utility.
        
        Args:
            real_train: Real training data
            synthetic_test: Synthetic test data
            real_train_labels: Real training labels
            synthetic_test_labels: Synthetic test labels
            
        Returns:
            Dictionary with utility metrics
        """
        task_type = self.task_type
        if task_type == "auto":
            task_type = self._tstr._detect_task_type(synthetic_test_labels)
        
        model = self._tstr._get_model(task_type)
        
        try:
            model.fit(real_train, real_train_labels)
            predictions = model.predict(synthetic_test)
            
            if task_type == "classification":
                accuracy = accuracy_score(synthetic_test_labels, predictions)
                f1 = f1_score(synthetic_test_labels, predictions, average='weighted', zero_division=0)
                
                return {
                    "trts_accuracy": accuracy,
                    "trts_f1": f1,
                }
            else:
                mse = mean_squared_error(synthetic_test_labels, predictions)
                r2 = r2_score(synthetic_test_labels, predictions)
                
                return {
                    "trts_mse": mse,
                    "trts_r2": r2,
                }
        
        except Exception as e:
            warnings.warn(f"TRTS evaluation failed: {e}")
            return {"error": str(e)}


class CrossValidationUtility(DownstreamUtilityMetric):
    """
    Cross-validation based utility evaluation.
    
    Compares model performance using cross-validation on real vs synthetic data.
    """
    
    def __init__(
        self,
        model_type: str = "auto",
        task_type: str = "auto",
        cv_folds: int = 5
    ):
        """
        Initialize CV utility evaluator.
        
        Args:
            model_type: Model to use
            task_type: Task type
            cv_folds: Number of CV folds
        """
        self.model_type = model_type
        self.task_type = task_type
        self.cv_folds = cv_folds
        self._tstr = TrainOnSyntheticTestOnReal(model_type, task_type)
    
    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        real_labels: np.ndarray,
        synthetic_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate utility using cross-validation.
        
        Args:
            real_data: Real data
            synthetic_data: Synthetic data
            real_labels: Real labels
            synthetic_labels: Synthetic labels
            
        Returns:
            Dictionary with CV metrics
        """
        task_type = self.task_type
        if task_type == "auto":
            task_type = self._tstr._detect_task_type(real_labels)
        
        model = self._tstr._get_model(task_type)
        
        scoring = 'accuracy' if task_type == "classification" else 'r2'
        
        try:
            # CV on real data
            real_scores = cross_val_score(
                model, real_data, real_labels,
                cv=self.cv_folds, scoring=scoring
            )
            
            # CV on synthetic data
            synth_scores = cross_val_score(
                model, synthetic_data, synthetic_labels,
                cv=self.cv_folds, scoring=scoring
            )
            
            return {
                "real_cv_mean": float(real_scores.mean()),
                "real_cv_std": float(real_scores.std()),
                "synthetic_cv_mean": float(synth_scores.mean()),
                "synthetic_cv_std": float(synth_scores.std()),
                "cv_ratio": float(synth_scores.mean() / max(real_scores.mean(), 0.01)),
            }
        
        except Exception as e:
            warnings.warn(f"CV evaluation failed: {e}")
            return {"error": str(e)}


class FeatureImportancePreservation(DownstreamUtilityMetric):
    """
    Feature Importance Preservation evaluation.
    
    Measures how well synthetic data preserves feature importance rankings
    compared to real data.
    """
    
    def __init__(self, model_type: str = "rf"):
        """
        Initialize feature importance evaluator.
        
        Args:
            model_type: Model type for importance ('rf', 'lr')
        """
        self.model_type = model_type
    
    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        real_labels: np.ndarray,
        synthetic_labels: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate feature importance preservation.
        
        Args:
            real_data: Real data
            synthetic_data: Synthetic data
            real_labels: Real labels
            synthetic_labels: Synthetic labels
            feature_names: Optional feature names
            
        Returns:
            Dictionary with importance metrics
        """
        try:
            # Train models
            if self.model_type == "rf":
                real_model = RandomForestClassifier(n_estimators=100, random_state=42)
                synth_model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                real_model = LogisticRegression(max_iter=1000, random_state=42)
                synth_model = LogisticRegression(max_iter=1000, random_state=42)
            
            real_model.fit(real_data, real_labels)
            synth_model.fit(synthetic_data, synthetic_labels)
            
            # Get feature importances
            if self.model_type == "rf":
                real_importance = real_model.feature_importances_
                synth_importance = synth_model.feature_importances_
            else:
                real_importance = np.abs(real_model.coef_).flatten()
                synth_importance = np.abs(synth_model.coef_).flatten()
            
            # Compute correlation of importance rankings
            rank_correlation = np.corrcoef(
                np.argsort(real_importance),
                np.argsort(synth_importance)
            )[0, 1]
            
            # Compute Spearman correlation
            from scipy.stats import spearmanr
            spearman_corr, _ = spearmanr(real_importance, synth_importance)
            
            # Top-k overlap
            top_k_values = [5, 10, 20]
            top_k_overlap = {}
            
            for k in top_k_values:
                if k <= len(real_importance):
                    real_top_k = set(np.argsort(real_importance)[-k:])
                    synth_top_k = set(np.argsort(synth_importance)[-k:])
                    overlap = len(real_top_k & synth_top_k) / k
                    top_k_overlap[f"top_{k}_overlap"] = overlap
            
            # Build result
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(real_importance))]
            
            return {
                "rank_correlation": float(rank_correlation) if not np.isnan(rank_correlation) else 0.0,
                "spearman_correlation": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
                "importance_mse": float(np.mean((real_importance - synth_importance) ** 2)),
                "top_k_overlap": top_k_overlap,
                "real_top_features": [feature_names[i] for i in np.argsort(real_importance)[-5:]],
                "synthetic_top_features": [feature_names[i] for i in np.argsort(synth_importance)[-5:]],
            }
        
        except Exception as e:
            warnings.warn(f"Feature importance evaluation failed: {e}")
            return {"error": str(e)}


class QueryWorkloadUtility(DownstreamUtilityMetric):
    """
    Query Workload Utility evaluation.
    
    Measures how well synthetic data preserves answers to aggregate queries.
    """
    
    def __init__(
        self,
        query_types: Optional[List[str]] = None,
        tolerance: float = 0.1
    ):
        """
        Initialize query workload evaluator.
        
        Args:
            query_types: Types of queries to evaluate
            tolerance: Relative tolerance for query accuracy
        """
        self.query_types = query_types or ["mean", "std", "count", "sum", "median"]
        self.tolerance = tolerance
    
    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate query workload utility.
        
        Args:
            real_data: Real data
            synthetic_data: Synthetic data
            
        Returns:
            Dictionary with query metrics
        """
        results = {
            "per_query": {},
            "per_feature": {},
            "overall": {},
        }
        
        total_error = 0.0
        query_count = 0
        
        # Per-query evaluation
        for query_type in self.query_types:
            error = self._evaluate_query(real_data, synthetic_data, query_type)
            results["per_query"][query_type] = error
            total_error += error
            query_count += 1
        
        # Per-feature evaluation
        if real_data.ndim > 1:
            for feature_idx in range(real_data.shape[1]):
                feature_errors = {}
                for query_type in ["mean", "std", "median"]:
                    error = self._evaluate_query(
                        real_data[:, feature_idx],
                        synthetic_data[:, feature_idx],
                        query_type
                    )
                    feature_errors[query_type] = error
                results["per_feature"][f"feature_{feature_idx}"] = feature_errors
        
        # Overall metrics
        results["overall"]["average_error"] = total_error / max(query_count, 1)
        results["overall"]["within_tolerance"] = (
            sum(1 for e in results["per_query"].values() if e <= self.tolerance) /
            max(query_count, 1)
        )
        
        return results
    
    def _evaluate_query(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
        query_type: str
    ) -> float:
        """Evaluate a single query type."""
        if query_type == "mean":
            real_val = np.mean(real)
            synth_val = np.mean(synthetic)
        elif query_type == "std":
            real_val = np.std(real)
            synth_val = np.std(synthetic)
        elif query_type == "count":
            real_val = len(real)
            synth_val = len(synthetic)
        elif query_type == "sum":
            real_val = np.sum(real)
            synth_val = np.sum(synthetic)
        elif query_type == "median":
            real_val = np.median(real)
            synth_val = np.median(synthetic)
        elif query_type == "min":
            real_val = np.min(real)
            synth_val = np.min(synthetic)
        elif query_type == "max":
            real_val = np.max(real)
            synth_val = np.max(synthetic)
        else:
            return float('nan')
        
        # Compute relative error
        if abs(real_val) < 1e-10:
            return abs(synth_val - real_val)
        
        return abs(synth_val - real_val) / abs(real_val)


class DownstreamUtilityEvaluator:
    """
    Comprehensive downstream utility evaluator.
    
    Combines multiple utility metrics for thorough evaluation.
    """
    
    def __init__(
        self,
        include_baseline: bool = True,
        include_cv: bool = True,
        include_feature_importance: bool = True
    ):
        """
        Initialize comprehensive evaluator.
        
        Args:
            include_baseline: Include TRTR baseline
            include_cv: Include cross-validation
            include_feature_importance: Include feature importance
        """
        self.include_baseline = include_baseline
        self.include_cv = include_cv
        self.include_feature_importance = include_feature_importance
        
        self.tstr = TrainOnSyntheticTestOnReal()
        self.trts = TrainOnRealTestOnSynthetic()
        self.cv_utility = CrossValidationUtility()
        self.feature_importance = FeatureImportancePreservation()
        self.query_workload = QueryWorkloadUtility()
    
    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        real_labels: np.ndarray,
        synthetic_labels: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive downstream utility evaluation.
        
        Args:
            real_data: Real data
            synthetic_data: Synthetic data
            real_labels: Real labels
            synthetic_labels: Synthetic labels
            feature_names: Optional feature names
            
        Returns:
            Comprehensive evaluation results
        """
        # Split real data for TSTR
        real_train, real_test, label_train, label_test = train_test_split(
            real_data, real_labels, test_size=0.3, random_state=42
        )
        
        results = {
            "tstr": {},
            "trts": {},
            "cv": {},
            "feature_importance": {},
            "query_workload": {},
        }
        
        # TSTR evaluation
        tstr_results = self.tstr.evaluate_with_baseline(
            real_train, real_test, synthetic_data,
            label_train, label_test, synthetic_labels
        )
        results["tstr"] = tstr_results
        
        # TRTS evaluation
        trts_results = self.trts.evaluate(
            real_train, synthetic_data,
            label_train, synthetic_labels
        )
        results["trts"] = trts_results
        
        # Cross-validation
        if self.include_cv:
            cv_results = self.cv_utility.evaluate(
                real_data, synthetic_data,
                real_labels, synthetic_labels
            )
            results["cv"] = cv_results
        
        # Feature importance
        if self.include_feature_importance:
            fi_results = self.feature_importance.evaluate(
                real_data, synthetic_data,
                real_labels, synthetic_labels,
                feature_names
            )
            results["feature_importance"] = fi_results
        
        # Query workload
        qw_results = self.query_workload.evaluate(real_data, synthetic_data)
        results["query_workload"] = qw_results
        
        # Compute overall utility score
        results["overall"] = self._compute_overall_score(results)
        
        return results
    
    def _compute_overall_score(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Compute overall utility score."""
        scores = []
        
        # TSTR score (accuracy or R²)
        if "tstr_accuracy" in results.get("tstr", {}):
            scores.append(results["tstr"]["tstr_accuracy"])
        elif "tstr_r2" in results.get("tstr", {}):
            scores.append(max(0, results["tstr"]["tstr_r2"]))
        
        # Utility ratio
        if "utility_ratio" in results.get("tstr", {}):
            scores.append(min(results["tstr"]["utility_ratio"], 1.0))
        
        # Feature importance correlation
        if "spearman_correlation" in results.get("feature_importance", {}):
            scores.append(max(0, results["feature_importance"]["spearman_correlation"]))
        
        # Query workload accuracy
        if "within_tolerance" in results.get("query_workload", {}).get("overall", {}):
            scores.append(results["query_workload"]["overall"]["within_tolerance"])
        
        overall_score = np.mean(scores) if scores else 0.5
        
        return {
            "utility_score": float(overall_score),
            "n_components": len(scores),
        }
