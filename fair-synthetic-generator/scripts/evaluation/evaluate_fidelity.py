#!/usr/bin/env python
"""
Evaluate Fidelity Script
========================
Script for evaluating fidelity metrics of synthetic data.
Measures how well synthetic data preserves statistical properties of real data.

Usage:
    python evaluate_fidelity.py [OPTIONS]

Options:
    --real PATH           Path to real data file
    --synthetic PATH      Path to synthetic data file
    --output PATH         Output path for evaluation report
    --metrics LIST        Comma-separated list of metrics to compute
    --modality TYPE       Data modality: tabular, image, text, multimodal
    --verbose             Print detailed progress
    -h, --help            Show this help message
Examples:
    python evaluate_fidelity.py --real data/train.csv --synthetic data/synthetic.csv
    python evaluate_fidelity.py --real data/real.npy --synthetic data/fake.npy --modality image
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Fidelity of Synthetic Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--real",
        type=str,
        required=True,
        help="Path to real data file"
    )
    
    parser.add_argument(
        "--synthetic",
        type=str,
        required=True,
        help="Path to synthetic data file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation report"
    )
    
    parser.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated list of metrics to compute"
    )
    
    parser.add_argument(
        "--modality",
        type=str,
        choices=["tabular", "image", "text", "multimodal"],
        default="tabular",
        help="Data modality"
    )
    
    parser.add_argument(
        "--target-column",
        type=str,
        default=None,
        help="Target column for downstream utility evaluation"
    )
    
    parser.add_argument(
        "--sensitive-columns",
        type=str,
        default=None,
        help="Comma-separated list of sensitive columns"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Sample size for evaluation (for large datasets)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    
    return parser.parse_args()


class FidelityEvaluator:
    """
    Comprehensive fidelity evaluation for synthetic data.
    
    Evaluates multiple aspects:
    - Statistical similarity (distributions, correlations)
    - Distribution metrics (KL divergence, JS distance, Wasserstein)
    - Downstream utility (train on synthetic, test on real)
    - Feature-wise fidelity
    - Multivariate relationships
    """
    
    def __init__(
        self,
        real_data: Union[pd.DataFrame, np.ndarray],
        synthetic_data: Union[pd.DataFrame, np.ndarray],
        modality: str = "tabular",
        target_column: Optional[str] = None,
        sensitive_columns: Optional[List[str]] = None,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Initialize the fidelity evaluator.
        
        Args:
            real_data: Real data (DataFrame or array)
            synthetic_data: Synthetic data (DataFrame or array)
            modality: Data modality
            target_column: Target column for supervised evaluation
            sensitive_columns: Sensitive attribute columns
            seed: Random seed
            verbose: Print detailed progress
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.modality = modality
        self.target_column = target_column
        self.sensitive_columns = sensitive_columns or []
        self.seed = seed
        self.verbose = verbose
        
        np.random.seed(seed)
        
        # Convert to DataFrame if array
        if isinstance(real_data, np.ndarray):
            self.real_df = pd.DataFrame(real_data)
        else:
            self.real_df = real_data.copy()
        
        if isinstance(synthetic_data, np.ndarray):
            self.synthetic_df = pd.DataFrame(synthetic_data)
        else:
            self.synthetic_df = synthetic_data.copy()
        
        # Ensure columns match
        if not self.real_df.columns.equals(self.synthetic_df.columns):
            self.synthetic_df.columns = self.real_df.columns
    
    def log(self, msg: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[Fidelity] {msg}")
    
    def evaluate_all(self) -> Dict[str, Any]:
        """
        Run all fidelity evaluations.
        
        Returns:
            Dictionary of all metrics
        """
        self.log("Starting comprehensive fidelity evaluation...")
        
        results = {
            "metadata": {
                "n_real": len(self.real_df),
                "n_synthetic": len(self.synthetic_df),
                "n_features": len(self.real_df.columns),
                "modality": self.modality,
                "timestamp": datetime.now().isoformat()
            },
            "statistical_similarity": self.evaluate_statistical_similarity(),
            "distribution_metrics": self.evaluate_distribution_metrics(),
            "correlation_fidelity": self.evaluate_correlation_fidelity(),
            "feature_importance_preservation": self.evaluate_feature_importance(),
            "downstream_utility": self.evaluate_downstream_utility()
        }
        
        # Compute overall fidelity score
        results["overall_fidelity"] = self._compute_overall_score(results)
        
        self.log("Fidelity evaluation complete!")
        return results
    
    def evaluate_statistical_similarity(self) -> Dict[str, Any]:
        """
        Evaluate statistical similarity between real and synthetic data.
        
        Returns:
            Dictionary of statistical similarity metrics
        """
        self.log("Evaluating statistical similarity...")
        
        from scipy import stats
        
        results = {
            "column_metrics": {},
            "summary": {}
        }
        
        numerical_cols = self.real_df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.real_df.select_dtypes(include=['object', 'category']).columns
        
        # Numerical columns statistics
        numerical_stats = []
        
        for col in numerical_cols:
            real_col = self.real_df[col].dropna()
            synth_col = self.synthetic_df[col].dropna()
            
            # Basic statistics comparison
            col_stats = {
                "real_mean": float(real_col.mean()),
                "synth_mean": float(synth_col.mean()),
                "mean_diff": float(abs(real_col.mean() - synth_col.mean())),
                "real_std": float(real_col.std()),
                "synth_std": float(synth_col.std()),
                "std_diff": float(abs(real_col.std() - synth_col.std())),
                "real_min": float(real_col.min()),
                "synth_min": float(synth_col.min()),
                "real_max": float(real_col.max()),
                "synth_max": float(synth_col.max())
            }
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(real_col, synth_col)
            col_stats["ks_statistic"] = float(ks_stat)
            col_stats["ks_pvalue"] = float(ks_pvalue)
            
            # Wasserstein distance (Earth Mover's Distance)
            wasserstein = stats.wasserstein_distance(real_col, synth_col)
            col_stats["wasserstein_distance"] = float(wasserstein)
            
            results["column_metrics"][col] = col_stats
            
            if ks_stat < 0.1:  # Good similarity
                numerical_stats.append(1 - ks_stat)
        
        # Categorical columns statistics
        for col in categorical_cols:
            real_counts = self.real_df[col].value_counts(normalize=True)
            synth_counts = self.synthetic_df[col].value_counts(normalize=True)
            
            # Align categories
            all_categories = set(real_counts.index) | set(synth_counts.index)
            real_probs = [real_counts.get(cat, 0) for cat in all_categories]
            synth_probs = [synth_counts.get(cat, 0) for cat in all_categories]
            
            # Total Variation Distance
            tvd = 0.5 * sum(abs(r - s) for r, s in zip(real_probs, synth_probs))
            
            # Chi-square test
            real_obs = self.real_df[col].value_counts()
            synth_obs = self.synthetic_df[col].value_counts()
            
            # Align for chi-square
            all_cats = set(real_obs.index) | set(synth_obs.index)
            real_obs = [real_obs.get(cat, 0) for cat in all_cats]
            synth_obs = [synth_obs.get(cat, 0) for cat in all_cats]
            
            if min(real_obs) > 0 and min(synth_obs) > 0:
                chi2, chi2_pvalue = stats.chisquare(synth_obs, f_exp=real_obs)
            else:
                chi2, chi2_pvalue = None, None
            
            results["column_metrics"][col] = {
                "total_variation_distance": float(tvd),
                "chi2_statistic": float(chi2) if chi2 else None,
                "chi2_pvalue": float(chi2_pvalue) if chi2_pvalue else None,
                "real_categories": len(real_counts),
                "synth_categories": len(synth_counts)
            }
        
        # Summary statistics
        if numerical_stats:
            results["summary"]["avg_numerical_similarity"] = float(np.mean(numerical_stats))
        
        results["summary"]["n_numerical_columns"] = len(numerical_cols)
        results["summary"]["n_categorical_columns"] = len(categorical_cols)
        
        return results
    
    def evaluate_distribution_metrics(self) -> Dict[str, Any]:
        """
        Evaluate distribution-level fidelity metrics.
        
        Returns:
            Dictionary of distribution metrics
        """
        self.log("Evaluating distribution metrics...")
        
        from scipy import stats
        
        results = {}
        
        # Jensen-Shannon Divergence for each column
        js_divergences = []
        
        numerical_cols = self.real_df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            real_col = self.real_df[col].dropna()
            synth_col = self.synthetic_df[col].dropna()
            
            # Compute histograms
            bins = np.linspace(
                min(real_col.min(), synth_col.min()),
                max(real_col.max(), synth_col.max()),
                50
            )
            
            real_hist, _ = np.histogram(real_col, bins=bins, density=True)
            synth_hist, _ = np.histogram(synth_col, bins=bins, density=True)
            
            # Normalize
            real_hist = real_hist / real_hist.sum()
            synth_hist = synth_hist / synth_hist.sum()
            
            # Jensen-Shannon Divergence
            m = 0.5 * (real_hist + synth_hist)
            js_div = 0.5 * (
                stats.entropy(real_hist + 1e-10, m + 1e-10) +
                stats.entropy(synth_hist + 1e-10, m + 1e-10)
            )
            
            js_divergences.append(js_div)
        
        if js_divergences:
            results["avg_js_divergence"] = float(np.mean(js_divergences))
            results["max_js_divergence"] = float(np.max(js_divergences))
            results["min_js_divergence"] = float(np.min(js_divergences))
        
        # Maximum Mean Discrepancy (MMD)
        if len(numerical_cols) > 0:
            real_array = self.real_df[numerical_cols].dropna().values
            synth_array = self.synthetic_df[numerical_cols].dropna().values
            
            # Sample if too large
            max_samples = 5000
            if len(real_array) > max_samples:
                idx = np.random.choice(len(real_array), max_samples, replace=False)
                real_array = real_array[idx]
            if len(synth_array) > max_samples:
                idx = np.random.choice(len(synth_array), max_samples, replace=False)
                synth_array = synth_array[idx]
            
            mmd = self._compute_mmd(real_array, synth_array)
            results["maximum_mean_discrepancy"] = float(mmd)
        
        return results
    
    def _compute_mmd(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        kernel: str = "rbf",
        gamma: float = 1.0
    ) -> float:
        """
        Compute Maximum Mean Discrepancy.
        
        Args:
            X: First sample
            Y: Second sample
            kernel: Kernel type
            gamma: Kernel bandwidth
            
        Returns:
            MMD value
        """
        n_x = len(X)
        n_y = len(Y)
        
        if kernel == "rbf":
            # RBF kernel
            def rbf_kernel(A, B):
                from scipy.spatial.distance import cdist
                distances = cdist(A, B, 'sqeuclidean')
                return np.exp(-gamma * distances)
            
            K_xx = rbf_kernel(X, X)
            K_yy = rbf_kernel(Y, Y)
            K_xy = rbf_kernel(X, Y)
        
        # MMD estimate
        mmd = (
            K_xx.sum() / (n_x * (n_x - 1)) -
            2 * K_xy.sum() / (n_x * n_y) +
            K_yy.sum() / (n_y * (n_y - 1))
        )
        
        return float(np.sqrt(max(0, mmd)))
    
    def evaluate_correlation_fidelity(self) -> Dict[str, Any]:
        """
        Evaluate how well correlations are preserved.
        
        Returns:
            Dictionary of correlation fidelity metrics
        """
        self.log("Evaluating correlation fidelity...")
        
        results = {}
        
        numerical_cols = self.real_df.select_dtypes(include=[np.number]).columns
        
        if len(numerical_cols) < 2:
            return {"message": "Not enough numerical columns for correlation analysis"}
        
        # Correlation matrices
        real_corr = self.real_df[numerical_cols].corr()
        synth_corr = self.synthetic_df[numerical_cols].corr()
        
        # Correlation matrix difference
        corr_diff = (real_corr - synth_corr).abs()
        
        results["mean_correlation_diff"] = float(corr_diff.values[np.triu_indices(len(numerical_cols), k=1)].mean())
        results["max_correlation_diff"] = float(corr_diff.values[np.triu_indices(len(numerical_cols), k=1)].max())
        
        # Frobenius norm of difference
        results["frobenius_norm"] = float(np.linalg.norm(real_corr - synth_corr, 'fro'))
        
        # Spearman correlation of correlation matrices
        real_corr_flat = real_corr.values[np.triu_indices(len(numerical_cols), k=1)]
        synth_corr_flat = synth_corr.values[np.triu_indices(len(numerical_cols), k=1)]
        
        from scipy import stats
        spearman_corr, _ = stats.spearmanr(real_corr_flat, synth_corr_flat)
        results["spearman_correlation_of_correlations"] = float(spearman_corr)
        
        return results
    
    def evaluate_feature_importance(self) -> Dict[str, Any]:
        """
        Evaluate preservation of feature importance for prediction.
        
        Returns:
            Dictionary of feature importance metrics
        """
        self.log("Evaluating feature importance preservation...")
        
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        
        results = {}
        
        # Need target column for feature importance
        target = self.target_column
        if target is None:
            target = self.real_df.columns[-1]
        
        if target not in self.real_df.columns:
            return {"message": "Target column not found for feature importance evaluation"}
        
        # Prepare data
        feature_cols = [c for c in self.real_df.columns if c != target]
        
        # Encode categorical columns
        real_encoded = self.real_df.copy()
        synth_encoded = self.synthetic_df.copy()
        
        encoders = {}
        for col in feature_cols:
            if real_encoded[col].dtype == 'object':
                encoders[col] = LabelEncoder()
                real_encoded[col] = encoders[col].fit_transform(real_encoded[col].astype(str))
                # Handle unseen categories in synthetic
                synth_encoded[col] = synth_encoded[col].astype(str).map(
                    lambda x: encoders[col].transform([x])[0] if x in encoders[col].classes_ else -1
                )
        
        # Train models
        X_real = real_encoded[feature_cols].values
        y_real = real_encoded[target].values
        
        X_synth = synth_encoded[feature_cols].values
        y_synth = synth_encoded[target].values
        
        # Encode target if categorical
        if real_encoded[target].dtype == 'object':
            target_encoder = LabelEncoder()
            y_real = target_encoder.fit_transform(y_real.astype(str))
            y_synth = target_encoder.transform(y_synth.astype(str))
            is_classification = True
        else:
            is_classification = len(np.unique(y_real)) < 10 and y_real.dtype == 'int64'
        
        try:
            if is_classification:
                model_real = RandomForestClassifier(n_estimators=50, random_state=self.seed, n_jobs=-1)
                model_synth = RandomForestClassifier(n_estimators=50, random_state=self.seed, n_jobs=-1)
            else:
                model_real = RandomForestRegressor(n_estimators=50, random_state=self.seed, n_jobs=-1)
                model_synth = RandomForestRegressor(n_estimators=50, random_state=self.seed, n_jobs=-1)
            
            model_real.fit(X_real, y_real)
            model_synth.fit(X_synth, y_synth)
            
            # Compare feature importances
            real_importance = model_real.feature_importances_
            synth_importance = model_synth.feature_importances_
            
            # Spearman correlation of importances
            from scipy import stats
            spearman_corr, _ = stats.spearmanr(real_importance, synth_importance)
            
            results["importance_correlation"] = float(spearman_corr)
            
            # Top-k overlap
            k = min(5, len(feature_cols))
            real_top_k = set(np.argsort(real_importance)[-k:])
            synth_top_k = set(np.argsort(synth_importance)[-k:])
            results["top_{}_overlap".format(k)] = len(real_top_k & synth_top_k) / k
            
            # Feature-level importance
            results["feature_importances"] = {
                col: {"real": float(r), "synth": float(s), "diff": float(abs(r - s))}
                for col, r, s in zip(feature_cols, real_importance, synth_importance)
            }
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def evaluate_downstream_utility(self) -> Dict[str, Any]:
        """
        Evaluate downstream utility (train on synthetic, test on real).
        
        Returns:
            Dictionary of downstream utility metrics
        """
        self.log("Evaluating downstream utility...")
        
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        
        results = {}
        
        target = self.target_column
        if target is None:
            target = self.real_df.columns[-1]
        
        if target not in self.real_df.columns:
            return {"message": "Target column not found for downstream utility evaluation"}
        
        feature_cols = [c for c in self.real_df.columns if c != target]
        
        # Encode data
        real_encoded = self.real_df.copy()
        synth_encoded = self.synthetic_df.copy()
        
        for col in feature_cols:
            if real_encoded[col].dtype == 'object':
                le = LabelEncoder()
                real_encoded[col] = le.fit_transform(real_encoded[col].astype(str))
                synth_encoded[col] = synth_encoded[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        X_real = real_encoded[feature_cols].values
        y_real = real_encoded[target].values
        X_synth = synth_encoded[feature_cols].values
        y_synth = synth_encoded[target].values
        
        if real_encoded[target].dtype == 'object':
            le_target = LabelEncoder()
            y_real = le_target.fit_transform(y_real.astype(str))
            y_synth = le_target.transform(y_synth.astype(str))
            is_classification = True
        else:
            is_classification = len(np.unique(y_real)) < 10 and y_real.dtype == 'int64'
        
        # Scale features
        scaler = StandardScaler()
        X_real = scaler.fit_transform(X_real)
        X_synth = scaler.transform(X_synth)
        
        # Split real data
        from sklearn.model_selection import train_test_split
        X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(
            X_real, y_real, test_size=0.3, random_state=self.seed
        )
        
        # Train models
        if is_classification:
            # Train on real, test on real (upper bound)
            model_real = RandomForestClassifier(n_estimators=50, random_state=self.seed, n_jobs=-1)
            model_real.fit(X_real_train, y_real_train)
            real_pred = model_real.predict(X_real_test)
            real_accuracy = accuracy_score(y_real_test, real_pred)
            real_f1 = f1_score(y_real_test, real_pred, average='weighted')
            
            # Train on synthetic, test on real
            model_synth = RandomForestClassifier(n_estimators=50, random_state=self.seed, n_jobs=-1)
            model_synth.fit(X_synth, y_synth)
            synth_pred = model_synth.predict(X_real_test)
            synth_accuracy = accuracy_score(y_real_test, synth_pred)
            synth_f1 = f1_score(y_real_test, synth_pred, average='weighted')
            
            results["real_train_accuracy"] = float(real_accuracy)
            results["synth_train_accuracy"] = float(synth_accuracy)
            results["real_train_f1"] = float(real_f1)
            results["synth_train_f1"] = float(synth_f1)
            results["utility_ratio"] = float(synth_accuracy / real_accuracy) if real_accuracy > 0 else 0
            
        else:
            model_real = RandomForestRegressor(n_estimators=50, random_state=self.seed, n_jobs=-1)
            model_real.fit(X_real_train, y_real_train)
            real_pred = model_real.predict(X_real_test)
            real_r2 = r2_score(y_real_test, real_pred)
            real_rmse = np.sqrt(mean_squared_error(y_real_test, real_pred))
            
            model_synth = RandomForestRegressor(n_estimators=50, random_state=self.seed, n_jobs=-1)
            model_synth.fit(X_synth, y_synth)
            synth_pred = model_synth.predict(X_real_test)
            synth_r2 = r2_score(y_real_test, synth_pred)
            synth_rmse = np.sqrt(mean_squared_error(y_real_test, synth_pred))
            
            results["real_train_r2"] = float(real_r2)
            results["synth_train_r2"] = float(synth_r2)
            results["real_train_rmse"] = float(real_rmse)
            results["synth_train_rmse"] = float(synth_rmse)
            results["utility_ratio"] = float(synth_r2 / real_r2) if real_r2 > 0 else 0
        
        return results
    
    def _compute_overall_score(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute overall fidelity score.
        
        Args:
            results: All evaluation results
            
        Returns:
            Dictionary with overall score and component scores
        """
        scores = {}
        
        # Statistical similarity score
        stat_sim = results.get("statistical_similarity", {})
        if "summary" in stat_sim:
            scores["statistical_similarity_score"] = stat_sim["summary"].get("avg_numerical_similarity", 0.5)
        
        # Distribution score (inverse of divergence)
        dist_metrics = results.get("distribution_metrics", {})
        if "avg_js_divergence" in dist_metrics:
            scores["distribution_score"] = max(0, 1 - dist_metrics["avg_js_divergence"])
        
        # Correlation fidelity score
        corr_fid = results.get("correlation_fidelity", {})
        if "spearman_correlation_of_correlations" in corr_fid:
            scores["correlation_score"] = max(0, corr_fid["spearman_correlation_of_correlations"])
        
        # Downstream utility score
        downstream = results.get("downstream_utility", {})
        if "utility_ratio" in downstream:
            scores["utility_score"] = downstream["utility_ratio"]
        
        # Overall average
        if scores:
            scores["overall_fidelity"] = float(np.mean(list(scores.values())))
        
        return scores
    
    def generate_report(self) -> str:
        """
        Generate a human-readable report.
        
        Returns:
            Report string
        """
        results = self.evaluate_all()
        
        report = []
        report.append("=" * 60)
        report.append("Fidelity Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Metadata
        report.append("Dataset Summary:")
        report.append(f"  Real samples: {results['metadata']['n_real']}")
        report.append(f"  Synthetic samples: {results['metadata']['n_synthetic']}")
        report.append(f"  Features: {results['metadata']['n_features']}")
        report.append(f"  Modality: {results['metadata']['modality']}")
        report.append("")
        
        # Overall score
        overall = results.get("overall_fidelity", {})
        report.append("Overall Fidelity Scores:")
        for metric, score in overall.items():
            report.append(f"  {metric}: {score:.4f}")
        report.append("")
        
        # Statistical similarity
        stat_sim = results.get("statistical_similarity", {})
        if "summary" in stat_sim:
            report.append("Statistical Similarity:")
            for key, value in stat_sim["summary"].items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.4f}")
                else:
                    report.append(f"  {key}: {value}")
        report.append("")
        
        # Distribution metrics
        dist_metrics = results.get("distribution_metrics", {})
        report.append("Distribution Metrics:")
        for key, value in dist_metrics.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
        report.append("")
        
        # Correlation fidelity
        corr_fid = results.get("correlation_fidelity", {})
        report.append("Correlation Fidelity:")
        for key, value in corr_fid.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
        report.append("")
        
        # Downstream utility
        downstream = results.get("downstream_utility", {})
        report.append("Downstream Utility:")
        for key, value in downstream.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def load_data(path: str) -> Union[pd.DataFrame, np.ndarray]:
    """Load data from file."""
    path = Path(path)
    
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix in [".npy", ".npz"]:
        return np.load(path, allow_pickle=True)
    elif path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix in [".pkl", ".pickle"]:
        return pd.read_pickle(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Fair Synthetic Data Generator - Fidelity Evaluation")
    print("=" * 60 + "\n")
    
    # Load data
    print(f"Loading real data from {args.real}...")
    real_data = load_data(args.real)
    
    print(f"Loading synthetic data from {args.synthetic}...")
    synthetic_data = load_data(args.synthetic)
    
    # Parse sensitive columns
    sensitive_cols = None
    if args.sensitive_columns:
        sensitive_cols = [s.strip() for s in args.sensitive_columns.split(",")]
    
    # Sample if needed
    if args.sample_size:
        if len(real_data) > args.sample_size:
            real_data = real_data.sample(n=args.sample_size, random_state=args.seed)
        if len(synthetic_data) > args.sample_size:
            synthetic_data = synthetic_data.sample(n=args.sample_size, random_state=args.seed)
    
    # Create evaluator
    evaluator = FidelityEvaluator(
        real_data=real_data,
        synthetic_data=synthetic_data,
        modality=args.modality,
        target_column=args.target_column,
        sensitive_columns=sensitive_cols,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        output_dir = project_root / "artifacts" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"fidelity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Save results
    results = evaluator.evaluate_all()
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
