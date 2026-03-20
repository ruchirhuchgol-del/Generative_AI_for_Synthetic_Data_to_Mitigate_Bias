#!/usr/bin/env python
"""
Evaluate Fairness Script
========================

Script for evaluating fairness metrics of synthetic data.
Measures bias mitigation and fairness properties across different paradigms.

Usage:
    python evaluate_fairness.py [OPTIONS]

Options:
    --real PATH           Path to real data file
    --synthetic PATH      Path to synthetic data file
    --sensitive-cols LIST Comma-separated list of sensitive attributes
    --target-col NAME     Target column name
    --output PATH         Output path for evaluation report
    --paradigm TYPE       Fairness paradigm: group, individual, counterfactual, all
    --verbose             Print detailed progress
    -h, --help            Show this help message

Examples:
    python evaluate_fairness.py --real data/train.csv --synthetic data/synthetic.csv --sensitive-cols gender,race
    python evaluate_fairness.py --real data.csv --synthetic synth.csv --paradigm counterfactual
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
from sklearn.preprocessing import LabelEncoder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Fairness of Synthetic Data",
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
        "--sensitive-cols",
        type=str,
        required=True,
        help="Comma-separated list of sensitive attributes"
    )
    
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Target column name"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for evaluation report"
    )
    
    parser.add_argument(
        "--paradigm",
        type=str,
        choices=["group", "individual", "counterfactual", "intersectional", "all"],
        default="all",
        help="Fairness paradigm to evaluate"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Threshold for disparate impact ratio"
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


class FairnessEvaluator:
    """
    Comprehensive fairness evaluation for synthetic data.
    
    Supports multiple fairness paradigms:
    - Group fairness (demographic parity, equalized odds)
    - Individual fairness (similarity-based)
    - Counterfactual fairness (causal reasoning)
    - Intersectional fairness (multiple protected attributes)
    """
    
    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_columns: List[str],
        target_column: Optional[str] = None,
        threshold: float = 0.8,
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Initialize the fairness evaluator.
        
        Args:
            real_data: Real data DataFrame
            synthetic_data: Synthetic data DataFrame
            sensitive_columns: List of sensitive attribute columns
            target_column: Target column for classification fairness
            threshold: Threshold for fairness metrics
            seed: Random seed
            verbose: Print detailed progress
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.sensitive_columns = sensitive_columns
        self.target_column = target_column or real_data.columns[-1]
        self.threshold = threshold
        self.seed = seed
        self.verbose = verbose
        
        np.random.seed(seed)
        
        # Ensure columns match
        if not self.real_data.columns.equals(self.synthetic_data.columns):
            self.synthetic_data.columns = self.real_data.columns
        
        # Encode sensitive attributes for processing
        self.label_encoders = {}
        for col in sensitive_columns:
            if self.real_data[col].dtype == 'object':
                le = LabelEncoder()
                self.real_data[col] = le.fit_transform(self.real_data[col].astype(str))
                self.synthetic_data[col] = self.synthetic_data[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
                self.label_encoders[col] = le
        
        if self.target_column and self.real_data[self.target_column].dtype == 'object':
            le = LabelEncoder()
            self.real_data[self.target_column] = le.fit_transform(
                self.real_data[self.target_column].astype(str)
            )
            self.synthetic_data[self.target_column] = le.transform(
                self.synthetic_data[self.target_column].astype(str)
            )
            self.label_encoders[self.target_column] = le
    
    def log(self, msg: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[Fairness] {msg}")
    
    def evaluate_all(self) -> Dict[str, Any]:
        """
        Run all fairness evaluations.
        
        Returns:
            Dictionary of all fairness metrics
        """
        self.log("Starting comprehensive fairness evaluation...")
        
        results = {
            "metadata": {
                "n_real": len(self.real_data),
                "n_synthetic": len(self.synthetic_data),
                "sensitive_attributes": self.sensitive_columns,
                "target_column": self.target_column,
                "threshold": self.threshold,
                "timestamp": datetime.now().isoformat()
            },
            "group_fairness": self.evaluate_group_fairness(),
            "individual_fairness": self.evaluate_individual_fairness(),
            "counterfactual_fairness": self.evaluate_counterfactual_fairness(),
            "intersectional_fairness": self.evaluate_intersectional_fairness()
        }
        
        # Compute overall fairness score
        results["overall_fairness"] = self._compute_overall_score(results)
        
        self.log("Fairness evaluation complete!")
        return results
    
    def evaluate_group_fairness(self) -> Dict[str, Any]:
        """
        Evaluate group fairness metrics.
        
        Includes:
        - Demographic Parity (Statistical Parity)
        - Equalized Odds
        - Disparate Impact
        - Calibration
        
        Returns:
            Dictionary of group fairness metrics
        """
        self.log("Evaluating group fairness...")
        
        results = {
            "real_data": {},
            "synthetic_data": {},
            "comparison": {}
        }
        
        target = self.target_column
        
        for sensitive_attr in self.sensitive_columns:
            attr_results_real = {}
            attr_results_synth = {}
            attr_comparison = {}
            
            # Get unique groups
            groups_real = self.real_data[sensitive_attr].unique()
            groups_synth = self.synthetic_data[sensitive_attr].unique()
            
            # Demographic Parity
            dp_real = self._compute_demographic_parity(
                self.real_data, sensitive_attr, target
            )
            dp_synth = self._compute_demographic_parity(
                self.synthetic_data, sensitive_attr, target
            )
            
            attr_results_real["demographic_parity"] = dp_real
            attr_results_synth["demographic_parity"] = dp_synth
            attr_comparison["demographic_parity_improvement"] = dp_synth - dp_real
            
            # Disparate Impact Ratio
            dir_real = self._compute_disparate_impact_ratio(
                self.real_data, sensitive_attr, target
            )
            dir_synth = self._compute_disparate_impact_ratio(
                self.synthetic_data, sensitive_attr, target
            )
            
            attr_results_real["disparate_impact_ratio"] = dir_real
            attr_results_synth["disparate_impact_ratio"] = dir_synth
            attr_comparison["disparate_impact_improvement"] = abs(dir_synth - 1.0) - abs(dir_real - 1.0)
            
            # Equalized Odds Difference
            eod_real = self._compute_equalized_odds_difference(
                self.real_data, sensitive_attr, target
            )
            eod_synth = self._compute_equalized_odds_difference(
                self.synthetic_data, sensitive_attr, target
            )
            
            attr_results_real["equalized_odds_difference"] = eod_real
            attr_results_synth["equalized_odds_difference"] = eod_synth
            attr_comparison["equalized_odds_improvement"] = eod_real - eod_synth
            
            # Statistical Parity Difference
            spd_real = self._compute_statistical_parity_difference(
                self.real_data, sensitive_attr, target
            )
            spd_synth = self._compute_statistical_parity_difference(
                self.synthetic_data, sensitive_attr, target
            )
            
            attr_results_real["statistical_parity_difference"] = spd_real
            attr_results_synth["statistical_parity_difference"] = spd_synth
            
            results["real_data"][sensitive_attr] = attr_results_real
            results["synthetic_data"][sensitive_attr] = attr_results_synth
            results["comparison"][sensitive_attr] = attr_comparison
        
        # Overall group fairness score
        improvements = [
            v.get("demographic_parity_improvement", 0)
            for v in results["comparison"].values()
        ]
        results["overall_improvement"] = float(np.mean(improvements)) if improvements else 0.0
        
        return results
    
    def _compute_demographic_parity(
        self,
        data: pd.DataFrame,
        sensitive_attr: str,
        target: str
    ) -> float:
        """
        Compute demographic parity difference.
        
        P(Y=1|A=unprivileged) - P(Y=1|A=privileged)
        
        Lower is better (0 = perfect parity).
        """
        groups = data[sensitive_attr].unique()
        
        if len(groups) != 2:
            # For multi-class, compute max difference
            positive_rates = []
            for group in groups:
                group_data = data[data[sensitive_attr] == group]
                if len(group_data) > 0:
                    pos_rate = group_data[target].mean()
                    positive_rates.append(pos_rate)
            return float(max(positive_rates) - min(positive_rates)) if positive_rates else 0.0
        
        privileged, unprivileged = sorted(groups)
        
        priv_rate = data[data[sensitive_attr] == privileged][target].mean()
        unpriv_rate = data[data[sensitive_attr] == unprivileged][target].mean()
        
        return float(abs(priv_rate - unpriv_rate))
    
    def _compute_disparate_impact_ratio(
        self,
        data: pd.DataFrame,
        sensitive_attr: str,
        target: str
    ) -> float:
        """
        Compute disparate impact ratio.
        
        P(Y=1|A=unprivileged) / P(Y=1|A=privileged)
        
        Value of 1.0 indicates no bias.
        Legal threshold is typically 0.8.
        """
        groups = data[sensitive_attr].unique()
        
        if len(groups) != 2:
            # For multi-class, compute ratio of min/max
            positive_rates = []
            for group in groups:
                group_data = data[data[sensitive_attr] == group]
                if len(group_data) > 0:
                    pos_rate = group_data[target].mean()
                    positive_rates.append(pos_rate)
            if len(positive_rates) >= 2 and max(positive_rates) > 0:
                return float(min(positive_rates) / max(positive_rates))
            return 1.0
        
        privileged, unprivileged = sorted(groups)
        
        priv_rate = data[data[sensitive_attr] == privileged][target].mean()
        unpriv_rate = data[data[sensitive_attr] == unprivileged][target].mean()
        
        if priv_rate == 0:
            return 1.0 if unpriv_rate == 0 else float('inf')
        
        return float(unpriv_rate / priv_rate)
    
    def _compute_equalized_odds_difference(
        self,
        data: pd.DataFrame,
        sensitive_attr: str,
        target: str
    ) -> float:
        """
        Compute equalized odds difference.
        
        Measures difference in TPR and FPR across groups.
        Lower is better.
        """
        groups = data[sensitive_attr].unique()
        
        tprs = []
        fprs = []
        
        for group in groups:
            group_data = data[data[sensitive_attr] == group]
            
            # True positive rate
            positives = group_data[group_data[target] == 1]
            if len(positives) > 0:
                tpr = len(positives) / len(group_data[group_data[target] == 1]) if len(data[data[target] == 1]) > 0 else 0
                tprs.append(len(positives) / len(group_data))
            
            # False positive rate
            negatives = group_data[group_data[target] == 0]
            if len(negatives) > 0:
                fprs.append(len(negatives) / len(group_data))
        
        tpr_diff = max(tprs) - min(tprs) if len(tprs) >= 2 else 0.0
        fpr_diff = max(fprs) - min(fprs) if len(fprs) >= 2 else 0.0
        
        return float(max(tpr_diff, fpr_diff))
    
    def _compute_statistical_parity_difference(
        self,
        data: pd.DataFrame,
        sensitive_attr: str,
        target: str
    ) -> float:
        """Compute statistical parity difference (same as demographic parity)."""
        return self._compute_demographic_parity(data, sensitive_attr, target)
    
    def evaluate_individual_fairness(self) -> Dict[str, Any]:
        """
        Evaluate individual fairness metrics.
        
        Individual fairness requires that similar individuals receive similar outcomes.
        Uses Lipschitz continuity and consistency metrics.
        
        Returns:
            Dictionary of individual fairness metrics
        """
        self.log("Evaluating individual fairness...")
        
        results = {
            "real_data": {},
            "synthetic_data": {},
            "comparison": {}
        }
        
        # Get numerical features for similarity computation
        numerical_cols = self.real_data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numerical_cols if c not in self.sensitive_columns and c != self.target_column]
        
        if len(feature_cols) < 2:
            return {"message": "Not enough numerical features for individual fairness evaluation"}
        
        # Consistency score (similar individuals should have similar outcomes)
        consistency_real = self._compute_consistency_score(
            self.real_data, feature_cols, self.target_column
        )
        consistency_synth = self._compute_consistency_score(
            self.synthetic_data, feature_cols, self.target_column
        )
        
        results["real_data"]["consistency_score"] = consistency_real
        results["synthetic_data"]["consistency_score"] = consistency_synth
        results["comparison"]["consistency_improvement"] = consistency_synth - consistency_real
        
        # Lipschitz constant estimation
        lipschitz_real = self._estimate_lipschitz_constant(
            self.real_data, feature_cols, self.target_column
        )
        lipschitz_synth = self._estimate_lipschitz_constant(
            self.synthetic_data, feature_cols, self.target_column
        )
        
        results["real_data"]["lipschitz_estimate"] = lipschitz_real
        results["synthetic_data"]["lipschitz_estimate"] = lipschitz_synth
        results["comparison"]["lipschitz_improvement"] = lipschitz_real - lipschitz_synth
        
        return results
    
    def _compute_consistency_score(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target: str,
        k: int = 5
    ) -> float:
        """
        Compute consistency score.
        
        Measures how often similar individuals have similar outcomes.
        Higher is better (1.0 = perfect consistency).
        """
        from sklearn.neighbors import NearestNeighbors
        
        X = data[feature_cols].values
        y = data[target].values
        
        # Find k nearest neighbors for each sample
        n_samples = min(len(X), 1000)  # Limit for efficiency
        if len(X) > n_samples:
            idx = np.random.choice(len(X), n_samples, replace=False)
            X = X[idx]
            y = y[idx]
        
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Compute consistency
        consistency_scores = []
        for i in range(len(X)):
            neighbor_targets = y[indices[i, 1:]]  # Exclude self
            own_target = y[i]
            
            # Proportion of neighbors with same outcome
            same_outcome = np.mean(neighbor_targets == own_target)
            consistency_scores.append(same_outcome)
        
        return float(np.mean(consistency_scores))
    
    def _estimate_lipschitz_constant(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target: str,
        n_pairs: int = 1000
    ) -> float:
        """
        Estimate Lipschitz constant.
        
        L = max(|f(x) - f(y)| / |x - y|)
        Lower is better for fairness.
        """
        X = data[feature_cols].values
        y = data[target].values
        
        # Sample pairs
        n = len(X)
        if n < 2:
            return 0.0
        
        n_pairs = min(n_pairs, n * (n - 1) // 2)
        
        lipschitz_ratios = []
        for _ in range(n_pairs):
            i, j = np.random.choice(n, 2, replace=False)
            
            x_diff = np.linalg.norm(X[i] - X[j])
            y_diff = abs(y[i] - y[j])
            
            if x_diff > 1e-6:
                lipschitz_ratios.append(y_diff / x_diff)
        
        # Return 90th percentile (robust to outliers)
        return float(np.percentile(lipschitz_ratios, 90)) if lipschitz_ratios else 0.0
    
    def evaluate_counterfactual_fairness(self) -> Dict[str, Any]:
        """
        Evaluate counterfactual fairness.
        
        Counterfactual fairness: if the sensitive attribute were different,
        the outcome should not change.
        
        Returns:
            Dictionary of counterfactual fairness metrics
        """
        self.log("Evaluating counterfactual fairness...")
        
        results = {
            "real_data": {},
            "synthetic_data": {},
            "comparison": {}
        }
        
        # Counterfactual invariance test
        # We approximate this by measuring outcome stability when sensitive attribute changes
        
        for sensitive_attr in self.sensitive_columns:
            # Measure outcome distribution across groups
            groups = self.real_data[sensitive_attr].unique()
            
            if len(groups) < 2:
                continue
            
            # Counterfactual fairness approximation:
            # How much does the outcome distribution change across groups?
            
            real_outcome_by_group = []
            synth_outcome_by_group = []
            
            for group in groups:
                real_outcome_by_group.append(
                    self.real_data[self.real_data[sensitive_attr] == group][self.target_column].mean()
                )
                synth_outcome_by_group.append(
                    self.synthetic_data[self.synthetic_data[sensitive_attr] == group][self.target_column].mean()
                )
            
            # Counterfactual fairness score (variance of means across groups)
            cf_fairness_real = 1.0 - np.var(real_outcome_by_group) if real_outcome_by_group else 1.0
            cf_fairness_synth = 1.0 - np.var(synth_outcome_by_group) if synth_outcome_by_group else 1.0
            
            results["real_data"][f"{sensitive_attr}_cf_score"] = float(cf_fairness_real)
            results["synthetic_data"][f"{sensitive_attr}_cf_score"] = float(cf_fairness_synth)
            results["comparison"][f"{sensitive_attr}_improvement"] = float(cf_fairness_synth - cf_fairness_real)
        
        return results
    
    def evaluate_intersectional_fairness(self) -> Dict[str, Any]:
        """
        Evaluate intersectional fairness.
        
        Examines fairness across combinations of sensitive attributes.
        
        Returns:
            Dictionary of intersectional fairness metrics
        """
        self.log("Evaluating intersectional fairness...")
        
        results = {
            "real_data": {},
            "synthetic_data": {},
            "comparison": {}
        }
        
        if len(self.sensitive_columns) < 2:
            return {"message": "Need at least 2 sensitive attributes for intersectional analysis"}
        
        # Create intersection groups
        real_intersection = self.real_data[self.sensitive_columns].astype(str).agg('_'.join, axis=1)
        synth_intersection = self.synthetic_data[self.sensitive_columns].astype(str).agg('_'.join, axis=1)
        
        # Compute outcome distribution across intersection groups
        real_group_outcomes = self.real_data.groupby(real_intersection)[self.target_column].mean()
        synth_group_outcomes = self.synthetic_data.groupby(synth_intersection)[self.target_column].mean()
        
        # Intersectional disparity (max difference across intersection groups)
        real_disparity = real_group_outcomes.max() - real_group_outcomes.min()
        synth_disparity = synth_group_outcomes.max() - synth_group_outcomes.min()
        
        results["real_data"]["intersectional_disparity"] = float(real_disparity)
        results["synthetic_data"]["intersectional_disparity"] = float(synth_disparity)
        results["comparison"]["disparity_reduction"] = float(real_disparity - synth_disparity)
        
        # Number of intersection groups
        results["real_data"]["n_intersection_groups"] = int(len(real_group_outcomes))
        results["synthetic_data"]["n_intersection_groups"] = int(len(synth_group_outcomes))
        
        # Intersectional demographic parity
        results["real_data"]["intersectional_demographic_parity"] = float(real_disparity)
        results["synthetic_data"]["intersectional_demographic_parity"] = float(synth_disparity)
        
        return results
    
    def _compute_overall_score(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute overall fairness score.
        
        Args:
            results: All evaluation results
            
        Returns:
            Dictionary with overall score and component scores
        """
        scores = {}
        
        # Group fairness score
        group_fairness = results.get("group_fairness", {})
        if "overall_improvement" in group_fairness:
            scores["group_fairness_improvement"] = group_fairness["overall_improvement"]
        
        # Individual fairness score
        individual_fairness = results.get("individual_fairness", {})
        if "comparison" in individual_fairness:
            scores["individual_fairness_improvement"] = individual_fairness["comparison"].get(
                "consistency_improvement", 0
            )
        
        # Counterfactual fairness score
        counterfactual = results.get("counterfactual_fairness", {})
        if "comparison" in counterfactual:
            cf_improvements = [
                v for k, v in counterfactual["comparison"].items()
                if "improvement" in k
            ]
            if cf_improvements:
                scores["counterfactual_fairness_improvement"] = float(np.mean(cf_improvements))
        
        # Intersectional fairness score
        intersectional = results.get("intersectional_fairness", {})
        if "comparison" in intersectional:
            scores["intersectional_fairness_improvement"] = intersectional["comparison"].get(
                "disparity_reduction", 0
            )
        
        # Overall improvement
        if scores:
            scores["overall_fairness_improvement"] = float(np.mean(list(scores.values())))
        
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
        report.append("Fairness Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        # Metadata
        report.append("Dataset Summary:")
        report.append(f"  Real samples: {results['metadata']['n_real']}")
        report.append(f"  Synthetic samples: {results['metadata']['n_synthetic']}")
        report.append(f"  Sensitive attributes: {results['metadata']['sensitive_attributes']}")
        report.append(f"  Target column: {results['metadata']['target_column']}")
        report.append("")
        
        # Overall score
        overall = results.get("overall_fairness", {})
        report.append("Overall Fairness Improvements:")
        for metric, score in overall.items():
            report.append(f"  {metric}: {score:.4f}")
        report.append("")
        
        # Group fairness
        group_fairness = results.get("group_fairness", {})
        report.append("Group Fairness Metrics:")
        for attr in self.sensitive_columns:
            report.append(f"\n  Sensitive Attribute: {attr}")
            if attr in group_fairness.get("real_data", {}):
                real_metrics = group_fairness["real_data"][attr]
                synth_metrics = group_fairness["synthetic_data"][attr]
                comparison = group_fairness["comparison"].get(attr, {})
                
                report.append(f"    Demographic Parity:")
                report.append(f"      Real: {real_metrics.get('demographic_parity', 'N/A'):.4f}")
                report.append(f"      Synthetic: {synth_metrics.get('demographic_parity', 'N/A'):.4f}")
                
                report.append(f"    Disparate Impact Ratio:")
                report.append(f"      Real: {real_metrics.get('disparate_impact_ratio', 'N/A'):.4f}")
                report.append(f"      Synthetic: {synth_metrics.get('disparate_impact_ratio', 'N/A'):.4f}")
        report.append("")
        
        # Individual fairness
        individual = results.get("individual_fairness", {})
        report.append("Individual Fairness Metrics:")
        if "real_data" in individual:
            report.append(f"  Consistency Score:")
            report.append(f"    Real: {individual['real_data'].get('consistency_score', 'N/A'):.4f}")
            report.append(f"    Synthetic: {individual['synthetic_data'].get('consistency_score', 'N/A'):.4f}")
        report.append("")
        
        # Intersectional fairness
        intersectional = results.get("intersectional_fairness", {})
        report.append("Intersectional Fairness Metrics:")
        if "real_data" in intersectional:
            report.append(f"  Intersectional Disparity:")
            report.append(f"    Real: {intersectional['real_data'].get('intersectional_disparity', 'N/A'):.4f}")
            report.append(f"    Synthetic: {intersectional['synthetic_data'].get('intersectional_disparity', 'N/A'):.4f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def load_data(path: str) -> pd.DataFrame:
    """Load data from file."""
    path = Path(path)
    
    if path.suffix == ".csv":
        return pd.read_csv(path)
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
    print("Fair Synthetic Data Generator - Fairness Evaluation")
    print("=" * 60 + "\n")
    
    # Load data
    print(f"Loading real data from {args.real}...")
    real_data = load_data(args.real)
    
    print(f"Loading synthetic data from {args.synthetic}...")
    synthetic_data = load_data(args.synthetic)
    
    # Parse sensitive columns
    sensitive_cols = [s.strip() for s in args.sensitive_cols.split(",")]
    
    # Create evaluator
    evaluator = FairnessEvaluator(
        real_data=real_data,
        synthetic_data=synthetic_data,
        sensitive_columns=sensitive_cols,
        target_column=args.target_col,
        threshold=args.threshold,
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
        output_path = output_dir / f"fairness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Save results
    results = evaluator.evaluate_all()
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
