
"""
Generate Evaluation Report
==========================

Comprehensive script for generating evaluation reports for synthetic data.
Supports multiple output formats, visualizations, and combined fidelity/fairness analysis.

Usage:
    python generate_report.py [OPTIONS]

Options:
    --real PATH           Path to real data file
    --synthetic PATH      Path to synthetic data file
    --output PATH         Output path for report
    --format FMT          Output format: html, pdf, json, markdown
    --sensitive-attrs     Comma-separated sensitive attributes
    --target-col NAME     Target column name
    --include-viz         Include visualizations in report
    --compare-baseline    Compare with baseline metrics
    -h, --help            Show this help message

Examples:
    python generate_report.py --real data/train.csv --synthetic data/synth.csv --format html
    python generate_report.py --real data.csv --synthetic synth.csv --sensitive-attrs gender,race --include-viz
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
        description="Generate Comprehensive Evaluation Report",
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
        help="Output path for report (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["html", "pdf", "json", "markdown", "all"],
        default="html",
        help="Output format for the report"
    )
    
    parser.add_argument(
        "--sensitive-attrs",
        type=str,
        default=None,
        help="Comma-separated list of sensitive attributes"
    )
    
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Target column name"
    )
    
    parser.add_argument(
        "--include-viz",
        action="store_true",
        default=True,
        help="Include visualizations in report"
    )
    
    parser.add_argument(
        "--compare-baseline",
        type=str,
        default=None,
        help="Path to baseline report for comparison"
    )
    
    parser.add_argument(
        "--title",
        type=str,
        default="Synthetic Data Evaluation Report",
        help="Report title"
    )
    
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Report description"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    
    return parser.parse_args()


class ReportGenerator:
    """
    Comprehensive report generator for synthetic data evaluation.
    
    Features:
    - Combined fidelity and fairness analysis
    - Multiple output formats (HTML, PDF, JSON, Markdown)
    - Interactive visualizations
    - Baseline comparison
    - Executive summary generation
    """
    
    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        sensitive_attrs: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        title: str = "Synthetic Data Evaluation Report",
        description: str = "",
        seed: int = 42,
        verbose: bool = True
    ):
        """
        Initialize report generator.
        
        Args:
            real_data: Real data DataFrame
            synthetic_data: Synthetic data DataFrame
            sensitive_attrs: List of sensitive attributes
            target_col: Target column name
            title: Report title
            description: Report description
            seed: Random seed
            verbose: Print progress
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.sensitive_attrs = sensitive_attrs or []
        self.target_col = target_col or real_data.columns[-1]
        self.title = title
        self.description = description
        self.seed = seed
        self.verbose = verbose
        
        np.random.seed(seed)
        
        # Ensure column alignment
        if not self.real_data.columns.equals(self.synthetic_data.columns):
            self.synthetic_data.columns = self.real_data.columns
        
        # Metrics storage
        self.fidelity_metrics = {}
        self.fairness_metrics = {}
        self.visualizations = {}
    
    def log(self, msg: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[Report] {msg}")
    
    def compute_all_metrics(self) -> Dict[str, Any]:
        """
        Compute all evaluation metrics.
        
        Returns:
            Dictionary containing all metrics
        """
        self.log("Computing all evaluation metrics...")
        
        # Fidelity metrics
        self.log("  Computing fidelity metrics...")
        self.fidelity_metrics = self._compute_fidelity_metrics()
        
        # Fairness metrics
        if self.sensitive_attrs:
            self.log("  Computing fairness metrics...")
            self.fairness_metrics = self._compute_fairness_metrics()
        
        # Combined results
        results = {
            "metadata": {
                "title": self.title,
                "description": self.description,
                "timestamp": datetime.now().isoformat(),
                "n_real": len(self.real_data),
                "n_synthetic": len(self.synthetic_data),
                "n_features": len(self.real_data.columns),
                "sensitive_attributes": self.sensitive_attrs,
                "target_column": self.target_col
            },
            "fidelity": self.fidelity_metrics,
            "fairness": self.fairness_metrics,
            "overall": self._compute_overall_scores()
        }
        
        return results
    
    def _compute_fidelity_metrics(self) -> Dict[str, Any]:
        """Compute fidelity metrics."""
        from scipy import stats
        
        metrics = {
            "statistical_similarity": {},
            "distribution_metrics": {},
            "correlation_fidelity": {},
            "downstream_utility": {}
        }
        
        numerical_cols = self.real_data.select_dtypes(include=[np.number]).columns
        
        # Statistical similarity
        col_metrics = {}
        for col in numerical_cols:
            real_col = self.real_data[col].dropna()
            synth_col = self.synthetic_data[col].dropna()
            
            # KS test
            ks_stat, ks_pvalue = stats.ks_2samp(real_col, synth_col)
            
            # Wasserstein distance
            wasserstein = stats.wasserstein_distance(real_col, synth_col)
            
            col_metrics[col] = {
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pvalue),
                "wasserstein_distance": float(wasserstein),
                "mean_diff": float(abs(real_col.mean() - synth_col.mean())),
                "std_diff": float(abs(real_col.std() - synth_col.std()))
            }
        
        metrics["statistical_similarity"]["column_metrics"] = col_metrics
        metrics["statistical_similarity"]["avg_ks_statistic"] = float(
            np.mean([m["ks_statistic"] for m in col_metrics.values()])
        )
        
        # Correlation fidelity
        if len(numerical_cols) >= 2:
            real_corr = self.real_data[numerical_cols].corr()
            synth_corr = self.synthetic_data[numerical_cols].corr()
            corr_diff = (real_corr - synth_corr).abs()
            
            metrics["correlation_fidelity"] = {
                "mean_correlation_diff": float(corr_diff.values[np.triu_indices(len(numerical_cols), k=1)].mean()),
                "max_correlation_diff": float(corr_diff.values.max())
            }
        
        # Downstream utility
        metrics["downstream_utility"] = self._compute_downstream_utility()
        
        return metrics
    
    def _compute_downstream_utility(self) -> Dict[str, Any]:
        """Compute downstream utility metrics."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
        
        results = {}
        
        feature_cols = [c for c in self.real_data.columns if c != self.target_col]
        
        # Encode categorical columns
        real_encoded = self.real_data.copy()
        synth_encoded = self.synthetic_data.copy()
        
        for col in feature_cols:
            if real_encoded[col].dtype == 'object':
                le = LabelEncoder()
                real_encoded[col] = le.fit_transform(real_encoded[col].astype(str))
                synth_encoded[col] = synth_encoded[col].astype(str).map(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
        
        X_real = real_encoded[feature_cols].values
        y_real = real_encoded[self.target_col].values
        X_synth = synth_encoded[feature_cols].values
        y_synth = synth_encoded[self.target_col].values
        
        if real_encoded[self.target_col].dtype == 'object':
            le_target = LabelEncoder()
            y_real = le_target.fit_transform(y_real.astype(str))
            y_synth = le_target.transform(y_synth.astype(str))
            is_classification = True
        else:
            is_classification = len(np.unique(y_real)) < 10
        
        # Scale features
        scaler = StandardScaler()
        X_real = scaler.fit_transform(X_real)
        X_synth = scaler.transform(X_synth)
        
        # Split real data
        X_train_real, X_test, y_train_real, y_test = train_test_split(
            X_real, y_real, test_size=0.3, random_state=self.seed
        )
        
        try:
            if is_classification:
                # Train on real
                model_real = RandomForestClassifier(n_estimators=50, random_state=self.seed, n_jobs=-1)
                model_real.fit(X_train_real, y_train_real)
                real_pred = model_real.predict(X_test)
                real_acc = accuracy_score(y_test, real_pred)
                
                # Train on synthetic
                model_synth = RandomForestClassifier(n_estimators=50, random_state=self.seed, n_jobs=-1)
                model_synth.fit(X_synth, y_synth)
                synth_pred = model_synth.predict(X_test)
                synth_acc = accuracy_score(y_test, synth_pred)
                
                results["real_train_accuracy"] = float(real_acc)
                results["synth_train_accuracy"] = float(synth_acc)
                results["utility_ratio"] = float(synth_acc / real_acc) if real_acc > 0 else 0
            else:
                model_real = RandomForestRegressor(n_estimators=50, random_state=self.seed, n_jobs=-1)
                model_real.fit(X_train_real, y_train_real)
                real_pred = model_real.predict(X_test)
                real_r2 = r2_score(y_test, real_pred)
                
                model_synth = RandomForestRegressor(n_estimators=50, random_state=self.seed, n_jobs=-1)
                model_synth.fit(X_synth, y_synth)
                synth_pred = model_synth.predict(X_test)
                synth_r2 = r2_score(y_test, synth_pred)
                
                results["real_train_r2"] = float(real_r2)
                results["synth_train_r2"] = float(synth_r2)
                results["utility_ratio"] = float(synth_r2 / real_r2) if real_r2 > 0 else 0
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def _compute_fairness_metrics(self) -> Dict[str, Any]:
        """Compute fairness metrics."""
        metrics = {}
        
        for attr in self.sensitive_attrs:
            if attr not in self.real_data.columns:
                continue
            
            attr_metrics = {}
            
            # Encode if necessary
            if self.real_data[attr].dtype == 'object':
                le = LabelEncoder()
                real_attr = le.fit_transform(self.real_data[attr].astype(str))
                synth_attr = le.transform(self.synthetic_data[attr].astype(str))
            else:
                real_attr = self.real_data[attr].values
                synth_attr = self.synthetic_data[attr].values
            
            # Demographic parity
            real_dp = self._demographic_parity(real_attr, self.real_data[self.target_col].values)
            synth_dp = self._demographic_parity(synth_attr, self.synthetic_data[self.target_col].values)
            
            attr_metrics["demographic_parity"] = {
                "real": float(real_dp),
                "synthetic": float(synth_dp),
                "improvement": float(real_dp - synth_dp)
            }
            
            # Disparate impact ratio
            real_dir = self._disparate_impact_ratio(real_attr, self.real_data[self.target_col].values)
            synth_dir = self._disparate_impact_ratio(synth_attr, self.synthetic_data[self.target_col].values)
            
            attr_metrics["disparate_impact_ratio"] = {
                "real": float(real_dir),
                "synthetic": float(synth_dir),
                "improvement": float(abs(real_dir - 1.0) - abs(synth_dir - 1.0))
            }
            
            metrics[attr] = attr_metrics
        
        return metrics
    
    def _demographic_parity(self, sensitive: np.ndarray, target: np.ndarray) -> float:
        """Compute demographic parity difference."""
        groups = np.unique(sensitive)
        rates = [target[sensitive == g].mean() for g in groups if (sensitive == g).sum() > 0]
        return float(max(rates) - min(rates)) if rates else 0.0
    
    def _disparate_impact_ratio(self, sensitive: np.ndarray, target: np.ndarray) -> float:
        """Compute disparate impact ratio."""
        groups = np.unique(sensitive)
        rates = [target[sensitive == g].mean() for g in groups if (sensitive == g).sum() > 0]
        if len(rates) >= 2 and max(rates) > 0:
            return float(min(rates) / max(rates))
        return 1.0
    
    def _compute_overall_scores(self) -> Dict[str, float]:
        """Compute overall evaluation scores."""
        scores = {}
        
        # Fidelity score
        fidelity_scores = []
        if "avg_ks_statistic" in self.fidelity_metrics.get("statistical_similarity", {}):
            fidelity_scores.append(1 - self.fidelity_metrics["statistical_similarity"]["avg_ks_statistic"])
        if "downstream_utility" in self.fidelity_metrics:
            fidelity_scores.append(self.fidelity_metrics["downstream_utility"].get("utility_ratio", 0.5))
        
        if fidelity_scores:
            scores["fidelity_score"] = float(np.mean(fidelity_scores))
        
        # Fairness score
        if self.fairness_metrics:
            fairness_improvements = []
            for attr_metrics in self.fairness_metrics.values():
                if "demographic_parity" in attr_metrics:
                    fairness_improvements.append(attr_metrics["demographic_parity"]["improvement"])
            if fairness_improvements:
                scores["fairness_improvement"] = float(np.mean(fairness_improvements))
        
        # Overall score
        if scores:
            scores["overall_score"] = float(np.mean(list(scores.values())))
        
        return scores
    
    def generate_visualizations(self) -> Dict[str, Any]:
        """Generate visualizations for the report."""
        self.log("Generating visualizations...")
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        visualizations = {}
        
        # Distribution comparison plots
        numerical_cols = self.real_data.select_dtypes(include=[np.number]).columns[:6]  # Limit to 6
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(numerical_cols):
            ax = axes[idx]
            ax.hist(self.real_data[col].dropna(), bins=30, alpha=0.5, label='Real', density=True)
            ax.hist(self.synthetic_data[col].dropna(), bins=30, alpha=0.5, label='Synthetic', density=True)
            ax.set_title(col)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('/tmp/distribution_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        with open('/tmp/distribution_comparison.png', 'rb') as f:
            import base64
            visualizations['distribution_comparison'] = base64.b64encode(f.read()).decode()
        
        # Correlation matrix comparison
        if len(self.real_data.select_dtypes(include=[np.number]).columns) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            real_corr = self.real_data.select_dtypes(include=[np.number]).corr()
            synth_corr = self.synthetic_data.select_dtypes(include=[np.number]).corr()
            
            sns.heatmap(real_corr, ax=axes[0], cmap='coolwarm', center=0, cbar=False)
            axes[0].set_title('Real Data Correlations')
            
            sns.heatmap(synth_corr, ax=axes[1], cmap='coolwarm', center=0, cbar=False)
            axes[1].set_title('Synthetic Data Correlations')
            
            plt.tight_layout()
            plt.savefig('/tmp/correlation_comparison.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            with open('/tmp/correlation_comparison.png', 'rb') as f:
                visualizations['correlation_comparison'] = base64.b64encode(f.read()).decode()
        
        return visualizations
    
    def generate_html_report(
        self,
        results: Dict[str, Any],
        visualizations: Optional[Dict[str, Any]] = None,
        baseline: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate HTML report."""
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #f5f5f5; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px; text-align: center; border-radius: 10px; margin-bottom: 30px; }}
        .header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; font-size: 1.1em; }}
        .card {{ background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; overflow: hidden; }}
        .card-header {{ background: #f8f9fa; padding: 20px; border-bottom: 1px solid #e9ecef; }}
        .card-header h2 {{ color: #495057; font-size: 1.3em; }}
        .card-body {{ padding: 20px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; }}
        .metric-item {{ text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .metric-label {{ color: #6c757d; margin-top: 5px; }}
        .score-excellent {{ color: #28a745; }}
        .score-good {{ color: #17a2b8; }}
        .score-moderate {{ color: #ffc107; }}
        .score-poor {{ color: #dc3545; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #e9ecef; }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .progress-bar {{ background: #e9ecef; border-radius: 10px; height: 20px; overflow: hidden; }}
        .progress-fill {{ height: 100%; transition: width 0.3s ease; }}
        .timestamp {{ color: #6c757d; font-size: 0.9em; }}
        .viz-container {{ text-align: center; margin: 20px 0; }}
        .viz-container img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .footer {{ text-align: center; padding: 30px; color: #6c757d; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p>{description}</p>
            <p class="timestamp">Generated: {timestamp}</p>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h2>📊 Executive Summary</h2>
            </div>
            <div class="card-body">
                <div class="metrics-grid">
                    <div class="metric-item">
                        <div class="metric-value {fidelity_class}">{fidelity_score:.2%}</div>
                        <div class="metric-label">Fidelity Score</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value {fairness_class}">{fairness_improvement:.2%}</div>
                        <div class="metric-label">Fairness Improvement</div>
                    </div>
                    <div class="metric-item">
                        <div class="metric-value {overall_class}">{overall_score:.2%}</div>
                        <div class="metric-label">Overall Score</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h2>📈 Dataset Overview</h2>
            </div>
            <div class="card-body">
                <table>
                    <tr><th>Property</th><th>Real Data</th><th>Synthetic Data</th></tr>
                    <tr><td>Number of Samples</td><td>{n_real:,}</td><td>{n_synthetic:,}</td></tr>
                    <tr><td>Number of Features</td><td>{n_features}</td><td>{n_features}</td></tr>
                    <tr><td>Target Column</td><td colspan="2">{target_col}</td></tr>
                    <tr><td>Sensitive Attributes</td><td colspan="2">{sensitive_attrs}</td></tr>
                </table>
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h2>🎯 Fidelity Metrics</h2>
            </div>
            <div class="card-body">
                <h3>Statistical Similarity</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Average KS Statistic</td><td>{avg_ks:.4f}</td></tr>
                    <tr><td>Average Wasserstein Distance</td><td>{avg_wasserstein:.4f}</td></tr>
                </table>
                
                <h3 style="margin-top: 20px;">Downstream Utility</h3>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Train on Real Accuracy/R²</td><td>{real_train_metric:.4f}</td></tr>
                    <tr><td>Train on Synthetic Accuracy/R²</td><td>{synth_train_metric:.4f}</td></tr>
                    <tr><td>Utility Ratio</td><td>{utility_ratio:.2%}</td></tr>
                </table>
            </div>
        </div>
        
        {fairness_section}
        
        {visualization_section}
        
        <div class="footer">
            <p>Generated by Fair Synthetic Data Generator</p>
            <p>© {year} - All rights reserved</p>
        </div>
    </div>
</body>
</html>
"""
        
        # Determine score classes
        overall = results.get("overall", {})
        fidelity_score = overall.get("fidelity_score", 0.5)
        fairness_improvement = overall.get("fairness_improvement", 0)
        overall_score = overall.get("overall_score", 0.5)
        
        def get_score_class(score):
            if score >= 0.8: return "score-excellent"
            if score >= 0.6: return "score-good"
            if score >= 0.4: return "score-moderate"
            return "score-poor"
        
        # Build fairness section
        fairness_html = ""
        if results.get("fairness"):
            fairness_html = """
        <div class="card">
            <div class="card-header">
                <h2>⚖️ Fairness Metrics</h2>
            </div>
            <div class="card-body">
                <table>
                    <tr><th>Attribute</th><th>Metric</th><th>Real</th><th>Synthetic</th><th>Improvement</th></tr>
"""
            for attr, metrics in results["fairness"].items():
                for metric_name, values in metrics.items():
                    if isinstance(values, dict):
                        fairness_html += f"""
                    <tr>
                        <td>{attr}</td>
                        <td>{metric_name}</td>
                        <td>{values.get('real', 'N/A'):.4f}</td>
                        <td>{values.get('synthetic', 'N/A'):.4f}</td>
                        <td>{values.get('improvement', 'N/A'):.4f}</td>
                    </tr>
"""
            fairness_html += """
                </table>
            </div>
        </div>
"""
        
        # Build visualization section
        viz_html = ""
        if visualizations:
            viz_html = """
        <div class="card">
            <div class="card-header">
                <h2>📊 Visualizations</h2>
            </div>
            <div class="card-body">
"""
            if 'distribution_comparison' in visualizations:
                viz_html += """
                <h3>Distribution Comparison</h3>
                <div class="viz-container">
                    <img src="data:image/png;base64,{dist_viz}" alt="Distribution Comparison">
                </div>
""".format(dist_viz=visualizations['distribution_comparison'])
            
            if 'correlation_comparison' in visualizations:
                viz_html += """
                <h3>Correlation Matrix Comparison</h3>
                <div class="viz-container">
                    <img src="data:image/png;base64,{corr_viz}" alt="Correlation Comparison">
                </div>
""".format(corr_viz=visualizations['correlation_comparison'])
            
            viz_html += """
            </div>
        </div>
"""
        
        # Get downstream utility metrics
        downstream = results.get("fidelity", {}).get("downstream_utility", {})
        real_train_metric = downstream.get("real_train_accuracy", downstream.get("real_train_r2", 0))
        synth_train_metric = downstream.get("synth_train_accuracy", downstream.get("synth_train_r2", 0))
        utility_ratio = downstream.get("utility_ratio", 0)
        
        # Format HTML
        html_content = html_template.format(
            title=self.title,
            description=self.description or "Comprehensive evaluation of synthetic data quality",
            timestamp=results["metadata"]["timestamp"],
            fidelity_score=fidelity_score,
            fairness_improvement=fairness_improvement,
            overall_score=overall_score,
            fidelity_class=get_score_class(fidelity_score),
            fairness_class=get_score_class(max(0, fairness_improvement)),
            overall_class=get_score_class(overall_score),
            n_real=results["metadata"]["n_real"],
            n_synthetic=results["metadata"]["n_synthetic"],
            n_features=results["metadata"]["n_features"],
            target_col=results["metadata"]["target_column"],
            sensitive_attrs=", ".join(results["metadata"]["sensitive_attributes"]) or "None specified",
            avg_ks=results.get("fidelity", {}).get("statistical_similarity", {}).get("avg_ks_statistic", 0),
            avg_wasserstein=np.mean([
                m.get("wasserstein_distance", 0) 
                for m in results.get("fidelity", {}).get("statistical_similarity", {}).get("column_metrics", {}).values()
            ]) if results.get("fidelity", {}).get("statistical_similarity", {}).get("column_metrics") else 0,
            real_train_metric=real_train_metric,
            synth_train_metric=synth_train_metric,
            utility_ratio=utility_ratio,
            fairness_section=fairness_html,
            visualization_section=viz_html,
            year=datetime.now().year
        )
        
        return html_content
    
    def generate_json_report(self, results: Dict[str, Any]) -> str:
        """Generate JSON report."""
        return json.dumps(results, indent=2, default=str)
    
    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate Markdown report."""
        
        md = f"""# {self.title}

{self.description}

*Generated: {results['metadata']['timestamp']}*

## Executive Summary

| Metric | Score |
|--------|-------|
| Fidelity Score | {results['overall'].get('fidelity_score', 0):.2%} |
| Fairness Improvement | {results['overall'].get('fairness_improvement', 0):.2%} |
| Overall Score | {results['overall'].get('overall_score', 0):.2%} |

## Dataset Overview

| Property | Real Data | Synthetic Data |
|----------|-----------|----------------|
| Samples | {results['metadata']['n_real']:,} | {results['metadata']['n_synthetic']:,} |
| Features | {results['metadata']['n_features']} | {results['metadata']['n_features']} |
| Target Column | {results['metadata']['target_column']} | |
| Sensitive Attributes | {', '.join(results['metadata']['sensitive_attributes']) or 'None'} | |

## Fidelity Metrics

### Statistical Similarity

| Metric | Value |
|--------|-------|
| Average KS Statistic | {results.get('fidelity', {}).get('statistical_similarity', {}).get('avg_ks_statistic', 0):.4f} |

### Downstream Utility

| Metric | Value |
|--------|-------|
| Train on Real | {results.get('fidelity', {}).get('downstream_utility', {}).get('real_train_accuracy', results.get('fidelity', {}).get('downstream_utility', {}).get('real_train_r2', 0)):.4f} |
| Train on Synthetic | {results.get('fidelity', {}).get('downstream_utility', {}).get('synth_train_accuracy', results.get('fidelity', {}).get('downstream_utility', {}).get('synth_train_r2', 0)):.4f} |
| Utility Ratio | {results.get('fidelity', {}).get('downstream_utility', {}).get('utility_ratio', 0):.2%} |

## Fairness Metrics

"""
        for attr, metrics in results.get("fairness", {}).items():
            md += f"### {attr}\n\n| Metric | Real | Synthetic | Improvement |\n|--------|------|-----------|-------------|\n"
            for metric_name, values in metrics.items():
                if isinstance(values, dict):
                    md += f"| {metric_name} | {values.get('real', 0):.4f} | {values.get('synthetic', 0):.4f} | {values.get('improvement', 0):.4f} |\n"
            md += "\n"
        
        return md
    
    def save_report(
        self,
        results: Dict[str, Any],
        output_path: str,
        format: str = "html",
        include_viz: bool = True
    ):
        """Save report to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "html":
            visualizations = self.generate_visualizations() if include_viz else None
            content = self.generate_html_report(results, visualizations)
        elif format == "json":
            content = self.generate_json_report(results)
        elif format == "markdown":
            content = self.generate_markdown_report(results)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.log(f"Report saved to {output_file}")


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
    print("Fair Synthetic Data Generator - Report Generation")
    print("=" * 60 + "\n")
    
    # Load data
    print(f"Loading real data from {args.real}...")
    real_data = load_data(args.real)
    
    print(f"Loading synthetic data from {args.synthetic}...")
    synthetic_data = load_data(args.synthetic)
    
    # Parse sensitive attributes
    sensitive_attrs = None
    if args.sensitive_attrs:
        sensitive_attrs = [s.strip() for s in args.sensitive_attrs.split(",")]
    
    # Create report generator
    generator = ReportGenerator(
        real_data=real_data,
        synthetic_data=synthetic_data,
        sensitive_attrs=sensitive_attrs,
        target_col=args.target_col,
        title=args.title,
        description=args.description,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Compute metrics
    results = generator.compute_all_metrics()
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        output_dir = project_root / "artifacts" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = {"html": "html", "json": "json", "markdown": "md", "all": "html"}[args.format]
        output_path = str(output_dir / f"report_{timestamp}.{ext}")
    
    # Save report
    if args.format == "all":
        # Save in all formats
        base_path = Path(output_path).stem
        output_dir = Path(output_path).parent
        
        for fmt in ["html", "json", "markdown"]:
            ext = {"html": "html", "json": "json", "markdown": "md"}[fmt]
            path = str(output_dir / f"{base_path}.{ext}")
            generator.save_report(results, path, fmt, args.include_viz)
    else:
        generator.save_report(results, output_path, args.format, args.include_viz)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Report Summary")
    print("=" * 60)
    print(f"Fidelity Score: {results['overall'].get('fidelity_score', 0):.2%}")
    print(f"Fairness Improvement: {results['overall'].get('fairness_improvement', 0):.2%}")
    print(f"Overall Score: {results['overall'].get('overall_score', 0):.2%}")
    print("=" * 60)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
