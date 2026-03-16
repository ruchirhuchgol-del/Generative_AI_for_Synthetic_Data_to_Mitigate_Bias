"""
Report Generator
================

Generates comprehensive fairness evaluation reports including:
- Fidelity assessment
- Fairness metrics
- Privacy evaluation
- Multimodal analysis
- Executive summaries
- HTML/JSON/Markdown export
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import json
import os
import numpy as np

from src.evaluation.fidelity.statistical_similarity import (
    JensenShannonDivergence,
    WassersteinDistance,
    CorrelationPreservation,
    StatisticalSimilarityEvaluator,
)
from src.evaluation.fairness.group_metrics import (
    DemographicParityMetric,
    EqualizedOddsMetric,
    DisparateImpactMetric,
    EqualOpportunityMetric,
    GroupFairnessEvaluator,
)
from src.evaluation.fairness.individual_metrics import (
    ConsistencyScore,
    LipschitzEstimator,
    IndividualFairnessEvaluator,
)


class FairnessReport:
    """
    Comprehensive fairness evaluation report generator.
    
    Generates reports including:
    - Fidelity metrics (statistical similarity)
    - Group fairness metrics (demographic parity, equalized odds)
    - Individual fairness metrics (consistency, Lipschitz)
    - Privacy metrics (if applicable)
    - Multimodal metrics (if applicable)
    """
    
    def __init__(
        self,
        sensitive_attributes: List[str],
        group_fairness_threshold: float = 0.05,
        consistency_threshold: float = 0.9,
        fidelity_threshold: float = 0.8
    ):
        """
        Initialize report generator.
        
        Args:
            sensitive_attributes: List of sensitive attribute names
            group_fairness_threshold: Threshold for group fairness metrics
            consistency_threshold: Threshold for consistency score
            fidelity_threshold: Threshold for fidelity score
        """
        self.sensitive_attributes = sensitive_attributes
        self.group_fairness_threshold = group_fairness_threshold
        self.consistency_threshold = consistency_threshold
        self.fidelity_threshold = fidelity_threshold
        
        # Initialize fidelity metrics
        self.fidelity_evaluator = StatisticalSimilarityEvaluator()
        
        # Initialize group fairness metrics
        self.group_metrics = {
            "demographic_parity": DemographicParityMetric(group_fairness_threshold),
            "equalized_odds": EqualizedOddsMetric(group_fairness_threshold),
            "equal_opportunity": EqualOpportunityMetric(group_fairness_threshold),
            "disparate_impact": DisparateImpactMetric(),
        }
        
        # Initialize individual fairness metrics
        self.individual_metrics = {
            "consistency": ConsistencyScore(),
            "lipschitz": LipschitzEstimator(),
        }
    
    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            predictions: Model predictions (for fairness metrics)
            groups: Group membership (for fairness metrics)
            labels: Ground truth labels (for equalized odds)
            features: Feature matrix (for individual fairness)
            feature_names: Names of features
            
        Returns:
            Complete evaluation report
        """
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "sensitive_attributes": self.sensitive_attributes,
                "data_shapes": {
                    "real": list(real_data.shape),
                    "synthetic": list(synthetic_data.shape),
                },
            },
            "fidelity": {},
            "fairness": {},
            "overall": {},
        }
        
        # Evaluate fidelity
        report["fidelity"] = self._evaluate_fidelity(
            real_data, synthetic_data, feature_names
        )
        
        # Evaluate fairness if predictions and groups provided
        if predictions is not None and groups is not None:
            report["fairness"] = self._evaluate_fairness(
                predictions, groups, labels, features, feature_names
            )
        
        # Compute overall scores
        report["overall"] = self._compute_overall_scores(report)
        
        return report
    
    def _evaluate_fidelity(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate fidelity metrics.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            feature_names: Optional feature names
            
        Returns:
            Fidelity evaluation results
        """
        return self.fidelity_evaluator.evaluate(
            real_data, synthetic_data, feature_names
        )
    
    def _evaluate_fairness(
        self,
        predictions: np.ndarray,
        groups: np.ndarray,
        labels: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate fairness metrics.
        
        Args:
            predictions: Model predictions
            groups: Group membership
            labels: Ground truth labels
            features: Feature matrix
            feature_names: Feature names
            
        Returns:
            Fairness evaluation results
        """
        fairness_report = {
            "group_fairness": {},
            "individual_fairness": {},
        }
        
        # Group names
        unique_groups = np.unique(groups)
        group_names = [f"Group_{g}" for g in unique_groups]
        
        # Demographic parity
        dp = self.group_metrics["demographic_parity"]
        dp_diff = dp.compute(predictions, groups)
        fairness_report["group_fairness"]["demographic_parity"] = {
            "difference": float(dp_diff),
            "threshold": dp.threshold,
            "satisfied": dp.is_fair(dp_diff),
            "interpretation": self._interpret_dp(dp_diff),
        }
        
        # Equalized odds (if labels provided)
        if labels is not None:
            eo = self.group_metrics["equalized_odds"]
            eo_diff = eo.compute(predictions, groups, labels)
            fairness_report["group_fairness"]["equalized_odds"] = {
                "difference": float(eo_diff),
                "threshold": eo.threshold,
                "satisfied": eo.is_fair(eo_diff),
                "interpretation": self._interpret_eo(eo_diff),
            }
            
            # Equal opportunity
            eopp = self.group_metrics["equal_opportunity"]
            eopp_diff = eopp.compute(predictions, groups, labels)
            fairness_report["group_fairness"]["equal_opportunity"] = {
                "difference": float(eopp_diff),
                "threshold": eopp.threshold,
                "satisfied": eopp.is_fair(eopp_diff),
            }
        
        # Disparate impact
        di = self.group_metrics["disparate_impact"]
        di_ratio = di.compute(predictions, groups)
        fairness_report["group_fairness"]["disparate_impact"] = {
            "ratio": float(di_ratio),
            "min_threshold": di.min_ratio,
            "max_threshold": di.max_ratio,
            "satisfied": di.is_fair(di_ratio),
            "interpretation": self._interpret_di(di_ratio),
        }
        
        # Individual fairness (if features provided)
        if features is not None:
            # Consistency
            consistency = self.individual_metrics["consistency"].compute(
                predictions, features
            )
            fairness_report["individual_fairness"]["consistency"] = {
                "score": float(consistency),
                "threshold": self.consistency_threshold,
                "satisfied": consistency >= self.consistency_threshold,
            }
            
            # Lipschitz
            lipschitz = self.individual_metrics["lipschitz"].compute(
                predictions, features
            )
            fairness_report["individual_fairness"]["lipschitz"] = {
                "estimate": float(lipschitz),
                "interpretation": "Lower values indicate better individual fairness",
            }
        
        return fairness_report
    
    def _interpret_dp(self, value: float) -> str:
        """Interpret demographic parity value."""
        if value <= 0.05:
            return "Excellent - predictions are well-balanced across groups"
        elif value <= 0.1:
            return "Good - minor disparity in prediction rates"
        elif value <= 0.2:
            return "Moderate - noticeable disparity in prediction rates"
        else:
            return "Poor - significant disparity in prediction rates"
    
    def _interpret_eo(self, value: float) -> str:
        """Interpret equalized odds value."""
        if value <= 0.05:
            return "Excellent - error rates are well-balanced across groups"
        elif value <= 0.1:
            return "Good - minor disparity in error rates"
        elif value <= 0.2:
            return "Moderate - noticeable disparity in error rates"
        else:
            return "Poor - significant disparity in error rates"
    
    def _interpret_di(self, ratio: float) -> str:
        """Interpret disparate impact ratio."""
        if 0.8 <= ratio <= 1.25:
            return "Passes the 80% rule for disparate impact"
        elif ratio < 0.8:
            return "Fails the 80% rule - underprivileged group disadvantaged"
        else:
            return "Fails the 80% rule - privileged group disadvantaged"
    
    def _compute_overall_scores(self, report: Dict) -> Dict[str, Any]:
        """
        Compute overall fairness and fidelity scores.
        
        Args:
            report: Full evaluation report
            
        Returns:
            Overall scores dictionary
        """
        scores = {}
        
        # Fidelity score
        if "fidelity" in report and "overall" in report["fidelity"]:
            scores["fidelity"] = report["fidelity"]["overall"].get("fidelity_score", 1.0)
        else:
            scores["fidelity"] = 1.0
        
        # Fairness score
        if "fairness" in report and "group_fairness" in report["fairness"]:
            fairness_values = []
            for metric, data in report["fairness"]["group_fairness"].items():
                if "satisfied" in data:
                    fairness_values.append(1.0 if data["satisfied"] else 0.5)
            
            # Include individual fairness if available
            if "individual_fairness" in report["fairness"]:
                for metric, data in report["fairness"]["individual_fairness"].items():
                    if "satisfied" in data:
                        fairness_values.append(1.0 if data["satisfied"] else 0.5)
            
            scores["fairness"] = np.mean(fairness_values) if fairness_values else 1.0
        else:
            scores["fairness"] = 1.0
        
        # Overall score (harmonic mean for balance)
        if scores["fidelity"] > 0 and scores["fairness"] > 0:
            scores["overall"] = 2 * (scores["fidelity"] * scores["fairness"]) / (
                scores["fidelity"] + scores["fairness"]
            )
        else:
            scores["overall"] = 0.0
        
        # Grade
        scores["grade"] = self._compute_grade(scores["overall"])
        
        # Recommendations
        scores["recommendations"] = self._generate_recommendations(report)
        
        return scores
    
    def _compute_grade(self, score: float) -> str:
        """Compute letter grade from score."""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Fidelity recommendations
        if "fidelity" in report:
            fidelity_score = report["overall"].get("fidelity", 1.0)
            if fidelity_score < 0.8:
                recommendations.append(
                    "Fidelity is low. Consider improving the synthetic data "
                    "generation model to better capture real data distribution."
                )
            
            if "metrics" in report["fidelity"]:
                if report["fidelity"]["metrics"].get("correlation", 0) > 0.2:
                    recommendations.append(
                        "Correlation structure is not well preserved. "
                        "Consider using models that explicitly capture feature dependencies."
                    )
        
        # Fairness recommendations
        if "fairness" in report:
            gf = report["fairness"].get("group_fairness", {})
            
            if "demographic_parity" in gf and not gf["demographic_parity"].get("satisfied", True):
                recommendations.append(
                    "Demographic parity is violated. Consider using fairness-aware "
                    "training methods like adversarial debiasing or reweighing."
                )
            
            if "equalized_odds" in gf and not gf["equalized_odds"].get("satisfied", True):
                recommendations.append(
                    "Equalized odds is violated. Consider post-processing methods "
                    "like equalized odds post-processing or calibrated equalized odds."
                )
            
            if "disparate_impact" in gf and not gf["disparate_impact"].get("satisfied", True):
                recommendations.append(
                    "Disparate impact threshold is violated. Consider adjusting "
                    "decision thresholds or using fair classification methods."
                )
        
        if not recommendations:
            recommendations.append(
                "The synthetic data meets quality and fairness standards. "
                "Continue monitoring for potential drift."
            )
        
        return recommendations
    
    def save_report(
        self,
        report: Dict[str, Any],
        output_path: str,
        format: Optional[str] = None
    ) -> None:
        """
        Save report to file.
        
        Args:
            report: Report dictionary
            output_path: Path to save
            format: Output format ('json', 'html', 'md'). Auto-detected if None.
        """
        _, ext = os.path.splitext(output_path)
        
        if format is None:
            format = ext.lstrip('.') if ext else 'json'
        
        if format == "json":
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
        
        elif format == "html":
            html = self._generate_html_report(report)
            with open(output_path, "w") as f:
                f.write(html)
        
        elif format == "md":
            markdown = self._generate_markdown_report(report)
            with open(output_path, "w") as f:
                f.write(markdown)
        
        else:
            # Default to JSON
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
    
    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML report from dictionary."""
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fair Synthetic Data Evaluation Report</title>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0;
            padding: 0;
            background: #f5f7fa;
            color: #333;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .header p {{
            margin: 10px 0 0;
            opacity: 0.9;
        }}
        .section {{ 
            background: white;
            border-radius: 10px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}
        .section h2 {{ 
            color: #333;
            margin-top: 0;
            padding-bottom: 10px;
            border-bottom: 2px solid #667eea;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .metric-card {{ 
            padding: 20px;
            border-radius: 8px;
            background: #f8f9fa;
            border-left: 4px solid #667eea;
        }}
        .metric-card h3 {{
            margin: 0 0 10px;
            color: #555;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .satisfied {{ 
            color: #27ae60;
            border-left-color: #27ae60;
        }}
        .violated {{ 
            color: #e74c3c;
            border-left-color: #e74c3c;
        }}
        .warning {{
            color: #f39c12;
            border-left-color: #f39c12;
        }}
        .status-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}
        .status-pass {{
            background: #d4edda;
            color: #155724;
        }}
        .status-fail {{
            background: #f8d7da;
            color: #721c24;
        }}
        .score-circle {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 0 auto 20px;
            font-weight: bold;
        }}
        .score-value {{
            font-size: 48px;
        }}
        .grade-A {{ background: #27ae60; color: white; }}
        .grade-B {{ background: #2ecc71; color: white; }}
        .grade-C {{ background: #f39c12; color: white; }}
        .grade-D {{ background: #e67e22; color: white; }}
        .grade-F {{ background: #e74c3c; color: white; }}
        .recommendations {{
            background: #fff3cd;
            border-left: 4px solid #f39c12;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin: 10px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .metadata {{
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Fair Synthetic Data Report</h1>
            <p>Generated: {report['metadata']['timestamp']}</p>
        </div>
        
        <div class="section">
            <h2>📊 Overall Assessment</h2>
            <div style="text-align: center;">
                <div class="score-circle grade-{report['overall']['grade']}">
                    <span class="score-value">{report['overall']['overall']:.2f}</span>
                    <span>Grade: {report['overall']['grade']}</span>
                </div>
            </div>
            <div class="metric-grid">
                <div class="metric-card {'satisfied' if report['overall']['fidelity'] >= 0.8 else 'violated'}">
                    <h3>Fidelity Score</h3>
                    <div class="metric-value">{report['overall']['fidelity']:.4f}</div>
                </div>
                <div class="metric-card {'satisfied' if report['overall']['fairness'] >= 0.8 else 'violated'}">
                    <h3>Fairness Score</h3>
                    <div class="metric-value">{report['overall']['fairness']:.4f}</div>
                </div>
            </div>
        </div>
        
        {self._render_fidelity_html(report)}
        
        {self._render_fairness_html(report)}
        
        {self._render_recommendations_html(report)}
        
        <div class="section metadata">
            <h2>📋 Metadata</h2>
            <table>
                <tr><th>Attribute</th><th>Value</th></tr>
                <tr><td>Sensitive Attributes</td><td>{', '.join(report['metadata']['sensitive_attributes'])}</td></tr>
                <tr><td>Real Data Shape</td><td>{report['metadata']['data_shapes']['real']}</td></tr>
                <tr><td>Synthetic Data Shape</td><td>{report['metadata']['data_shapes']['synthetic']}</td></tr>
            </table>
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _render_fidelity_html(self, report: Dict) -> str:
        """Render fidelity section as HTML."""
        if "fidelity" not in report:
            return ""
        
        fidelity = report["fidelity"]
        metrics = fidelity.get("metrics", {})
        
        html = """
        <div class="section">
            <h2>🎯 Fidelity Metrics</h2>
            <div class="metric-grid">
        """
        
        metric_names = {
            "js_divergence": ("Jensen-Shannon Divergence", "Lower is better"),
            "wasserstein": ("Wasserstein Distance", "Lower is better"),
            "correlation": ("Correlation Difference", "Lower is better"),
            "mutual_information": ("Mutual Information", "Higher is better"),
            "ks_statistic": ("KS Statistic", "Lower is better"),
            "moment_matching": ("Moment Difference", "Lower is better"),
        }
        
        for metric_key, value in metrics.items():
            name, interpretation = metric_names.get(metric_key, (metric_key, ""))
            html += f"""
                <div class="metric-card">
                    <h3>{name}</h3>
                    <div class="metric-value">{value:.4f}</div>
                    <small>{interpretation}</small>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        
        return html
    
    def _render_fairness_html(self, report: Dict) -> str:
        """Render fairness section as HTML."""
        if "fairness" not in report or not report["fairness"]:
            return ""
        
        fairness = report["fairness"]
        
        html = """
        <div class="section">
            <h2>⚖️ Fairness Metrics</h2>
        """
        
        # Group fairness
        if "group_fairness" in fairness:
            html += "<h3>Group Fairness</h3><table>"
            html += "<tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>"
            
            for metric, data in fairness["group_fairness"].items():
                status = "✓ Passed" if data.get("satisfied", True) else "✗ Failed"
                status_class = "status-pass" if data.get("satisfied", True) else "status-fail"
                
                if "difference" in data:
                    value = f"{data['difference']:.4f}"
                    threshold = str(data.get("threshold", "N/A"))
                elif "ratio" in data:
                    value = f"{data['ratio']:.4f}"
                    threshold = f"[{data['min_threshold']}, {data['max_threshold']}]"
                else:
                    value = "N/A"
                    threshold = "N/A"
                
                html += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{value}</td>
                    <td>{threshold}</td>
                    <td><span class="status-badge {status_class}">{status}</span></td>
                </tr>
                """
            
            html += "</table>"
        
        # Individual fairness
        if "individual_fairness" in fairness and fairness["individual_fairness"]:
            html += "<h3>Individual Fairness</h3><table>"
            html += "<tr><th>Metric</th><th>Value</th><th>Status</th></tr>"
            
            for metric, data in fairness["individual_fairness"].items():
                status = "✓ Passed" if data.get("satisfied", True) else "✗ Failed"
                status_class = "status-pass" if data.get("satisfied", True) else "status-fail"
                
                if "score" in data:
                    value = f"{data['score']:.4f}"
                elif "estimate" in data:
                    value = f"{data['estimate']:.4f}"
                else:
                    value = "N/A"
                
                html += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{value}</td>
                    <td><span class="status-badge {status_class}">{status}</span></td>
                </tr>
                """
            
            html += "</table>"
        
        html += "</div>"
        
        return html
    
    def _render_recommendations_html(self, report: Dict) -> str:
        """Render recommendations section as HTML."""
        if "recommendations" not in report.get("overall", {}):
            return ""
        
        recommendations = report["overall"]["recommendations"]
        
        html = """
        <div class="section recommendations">
            <h2>💡 Recommendations</h2>
            <ul>
        """
        
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        
        html += """
            </ul>
        </div>
        """
        
        return html
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate Markdown report from dictionary."""
        md = f"""# Fair Synthetic Data Evaluation Report

**Generated:** {report['metadata']['timestamp']}

## Overall Assessment

| Metric | Score | Grade |
|--------|-------|-------|
| Overall Score | {report['overall']['overall']:.4f} | {report['overall']['grade']} |
| Fidelity Score | {report['overall']['fidelity']:.4f} | - |
| Fairness Score | {report['overall']['fairness']:.4f} | - |

## Fidelity Metrics

| Metric | Value |
|--------|-------|
"""
        
        # Add fidelity metrics
        if "fidelity" in report and "metrics" in report["fidelity"]:
            for metric, value in report["fidelity"]["metrics"].items():
                md += f"| {metric} | {value:.4f} |\n"
        
        md += "\n## Fairness Metrics\n\n"
        
        # Add group fairness metrics
        if "fairness" in report and "group_fairness" in report["fairness"]:
            md += "### Group Fairness\n\n"
            md += "| Metric | Value | Status |\n|--------|-------|--------|\n"
            
            for metric, data in report["fairness"]["group_fairness"].items():
                status = "✓" if data.get("satisfied", True) else "✗"
                if "difference" in data:
                    value = f"{data['difference']:.4f}"
                elif "ratio" in data:
                    value = f"{data['ratio']:.4f}"
                else:
                    value = "N/A"
                md += f"| {metric} | {value} | {status} |\n"
        
        # Add recommendations
        if "recommendations" in report.get("overall", {}):
            md += "\n## Recommendations\n\n"
            for rec in report["overall"]["recommendations"]:
                md += f"- {rec}\n"
        
        return md


class ComprehensiveReportGenerator:
    """
    Comprehensive report generator that combines all evaluation components.
    """
    
    def __init__(
        self,
        sensitive_attributes: List[str],
        include_privacy: bool = True,
        include_multimodal: bool = True
    ):
        """
        Initialize comprehensive report generator.
        
        Args:
            sensitive_attributes: List of sensitive attribute names
            include_privacy: Whether to include privacy evaluation
            include_multimodal: Whether to include multimodal evaluation
        """
        self.sensitive_attributes = sensitive_attributes
        self.include_privacy = include_privacy
        self.include_multimodal = include_multimodal
        
        self.fairness_report = FairnessReport(sensitive_attributes)
    
    def generate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        privacy_metrics: Optional[Dict] = None,
        multimodal_metrics: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            predictions: Model predictions
            groups: Group membership
            labels: Ground truth labels
            features: Feature matrix
            feature_names: Feature names
            privacy_metrics: Pre-computed privacy metrics
            multimodal_metrics: Pre-computed multimodal metrics
            
        Returns:
            Comprehensive evaluation report
        """
        # Generate base fairness report
        report = self.fairness_report.evaluate(
            real_data, synthetic_data,
            predictions, groups, labels,
            features, feature_names
        )
        
        # Add privacy metrics if provided
        if self.include_privacy and privacy_metrics:
            report["privacy"] = privacy_metrics
            report["overall"]["privacy_score"] = self._compute_privacy_score(privacy_metrics)
        
        # Add multimodal metrics if provided
        if self.include_multimodal and multimodal_metrics:
            report["multimodal"] = multimodal_metrics
            report["overall"]["multimodal_score"] = multimodal_metrics.get(
                "overall", {}
            ).get("multimodal_score", 1.0)
        
        return report
    
    def _compute_privacy_score(self, privacy_metrics: Dict) -> float:
        """Compute overall privacy score from metrics."""
        scores = []
        
        if "membership_inference" in privacy_metrics:
            mia_auc = privacy_metrics["membership_inference"].get("attack_auc", 0.5)
            scores.append(1 - max(0, mia_auc - 0.5) * 2)
        
        if "attribute_inference" in privacy_metrics:
            aia_risk = privacy_metrics["attribute_inference"].get("privacy_risk", 0)
            scores.append(1 - aia_risk)
        
        return float(np.mean(scores)) if scores else 1.0
