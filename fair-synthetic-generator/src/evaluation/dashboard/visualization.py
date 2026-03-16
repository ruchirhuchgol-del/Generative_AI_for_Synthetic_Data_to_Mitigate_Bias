"""
Visualization Module for Fair Synthetic Data Evaluation
========================================================

Comprehensive visualization tools for fairness, fidelity,
privacy, and multimodal evaluation metrics.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import json
import os
from datetime import datetime


class FairnessVisualizer:
    """
    Visualizer for fairness metrics.
    
    Creates visualizations for group fairness, individual fairness,
    counterfactual fairness, and intersectional fairness metrics.
    """
    
    def __init__(
        self,
        style: str = "default",
        color_scheme: Optional[Dict[str, str]] = None
    ):
        """
        Initialize fairness visualizer.
        
        Args:
            style: Visualization style ('default', 'dark', 'minimal')
            color_scheme: Custom color scheme
        """
        self.style = style
        self.color_scheme = color_scheme or self._get_default_colors()
    
    def _get_default_colors(self) -> Dict[str, str]:
        """Get default color scheme."""
        return {
            "fair": "#2ecc71",
            "unfair": "#e74c3c",
            "warning": "#f39c12",
            "neutral": "#3498db",
            "background": "#ffffff",
            "text": "#2c3e50",
        }
    
    def plot_group_fairness(
        self,
        metrics: Dict[str, Any],
        group_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create group fairness visualization data.
        
        Args:
            metrics: Dictionary of fairness metrics
            group_names: Names of demographic groups
            
        Returns:
            Dictionary with visualization data
        """
        # Prepare bar chart data
        bar_data = []
        
        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                if "difference" in metric_data:
                    bar_data.append({
                        "metric": metric_name,
                        "value": metric_data["difference"],
                        "threshold": metric_data.get("threshold", 0.1),
                        "satisfied": metric_data.get("satisfied", False),
                    })
                elif "group_rates" in metric_data:
                    # Per-group rates
                    for group, rate in metric_data["group_rates"].items():
                        bar_data.append({
                            "metric": f"{metric_name}_{group}",
                            "value": rate,
                            "type": "rate",
                        })
        
        return {
            "type": "group_fairness",
            "data": bar_data,
            "layout": {
                "title": "Group Fairness Metrics",
                "x_axis": "Metric",
                "y_axis": "Value",
                "colors": self.color_scheme,
            },
            "summary": self._compute_fairness_summary(bar_data),
        }
    
    def plot_intersectional_heatmap(
        self,
        intersectional_data: Dict[str, Dict[str, float]],
        attribute_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create intersectional fairness heatmap data.
        
        Args:
            intersectional_data: Dictionary with group combinations and rates
            attribute_names: Names of intersecting attributes
            
        Returns:
            Dictionary with heatmap visualization data
        """
        # Extract matrix data
        groups = list(intersectional_data.keys())
        values = [list(v.values()) if isinstance(v, dict) else v for v in intersectional_data.values()]
        
        # Flatten for heatmap
        heatmap_cells = []
        
        for i, (group_key, group_data) in enumerate(intersectional_data.items()):
            if isinstance(group_data, dict):
                for sub_key, value in group_data.items():
                    heatmap_cells.append({
                        "row": str(group_key),
                        "col": str(sub_key),
                        "value": value,
                    })
            else:
                heatmap_cells.append({
                    "row": "All",
                    "col": str(group_key),
                    "value": group_data,
                })
        
        return {
            "type": "intersectional_heatmap",
            "data": heatmap_cells,
            "layout": {
                "title": "Intersectional Fairness Analysis",
                "x_axis": "Attribute Combination",
                "y_axis": "Metric",
                "colorscale": "RdYlGn",
            },
        }
    
    def plot_fairness_radar(
        self,
        metrics: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Create radar chart for fairness metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            thresholds: Optional thresholds for each metric
            
        Returns:
            Dictionary with radar chart data
        """
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Normalize values (lower is better for fairness metrics)
        normalized_values = []
        for i, (cat, val) in enumerate(zip(categories, values)):
            if thresholds and cat in thresholds:
                # Score: 1 if below threshold, decreasing as exceeds
                norm_val = max(0, 1 - val / thresholds[cat])
            else:
                # Simple normalization (assuming lower is better)
                norm_val = max(0, 1 - val)
            normalized_values.append(norm_val)
        
        return {
            "type": "radar",
            "data": {
                "categories": categories,
                "values": normalized_values,
                "original_values": values,
            },
            "layout": {
                "title": "Fairness Profile",
                "range": [0, 1],
            },
        }
    
    def _compute_fairness_summary(self, bar_data: List[Dict]) -> Dict[str, Any]:
        """Compute summary statistics for fairness metrics."""
        if not bar_data:
            return {"overall_fair": True, "summary": "No metrics to evaluate"}
        
        satisfied = [d for d in bar_data if d.get("satisfied", True)]
        violated = [d for d in bar_data if not d.get("satisfied", True)]
        
        return {
            "overall_fair": len(violated) == 0,
            "n_metrics": len(bar_data),
            "n_satisfied": len(satisfied),
            "n_violated": len(violated),
            "fairness_ratio": len(satisfied) / len(bar_data) if bar_data else 1.0,
        }


class FidelityVisualizer:
    """
    Visualizer for fidelity metrics.
    
    Creates visualizations for distribution similarity,
    statistical metrics, and downstream utility.
    """
    
    def __init__(
        self,
        style: str = "default"
    ):
        """
        Initialize fidelity visualizer.
        
        Args:
            style: Visualization style
        """
        self.style = style
    
    def plot_distribution_comparison(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create distribution comparison visualization data.
        
        Args:
            real_data: Real data samples
            synthetic_data: Synthetic data samples
            feature_names: Names of features
            
        Returns:
            Dictionary with distribution plot data
        """
        n_features = real_data.shape[1] if real_data.ndim > 1 else 1
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        distributions = []
        
        for i in range(min(n_features, 20)):  # Limit to 20 features
            if real_data.ndim > 1:
                real_col = real_data[:, i]
                synth_col = synthetic_data[:, i]
            else:
                real_col = real_data
                synth_col = synthetic_data
            
            # Create histogram bins
            all_data = np.concatenate([real_col, synth_col])
            bins = np.linspace(all_data.min(), all_data.max(), 30)
            
            real_hist, _ = np.histogram(real_col, bins=bins, density=True)
            synth_hist, _ = np.histogram(synth_col, bins=bins, density=True)
            
            distributions.append({
                "feature": feature_names[i],
                "bins": bins.tolist(),
                "real_hist": real_hist.tolist(),
                "synthetic_hist": synth_hist.tolist(),
            })
        
        return {
            "type": "distribution_comparison",
            "data": distributions,
            "layout": {
                "title": "Real vs Synthetic Distribution Comparison",
                "x_axis": "Value",
                "y_axis": "Density",
            },
        }
    
    def plot_correlation_matrices(
        self,
        real_corr: np.ndarray,
        synthetic_corr: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create correlation matrix comparison visualization.
        
        Args:
            real_corr: Correlation matrix for real data
            synthetic_corr: Correlation matrix for synthetic data
            feature_names: Names of features
            
        Returns:
            Dictionary with correlation heatmap data
        """
        n_features = real_corr.shape[0]
        
        if feature_names is None:
            feature_names = [f"F{i}" for i in range(n_features)]
        
        # Compute difference
        diff = real_corr - synthetic_corr
        
        return {
            "type": "correlation_matrices",
            "data": {
                "real_correlation": real_corr.tolist(),
                "synthetic_correlation": synthetic_corr.tolist(),
                "difference": diff.tolist(),
                "feature_names": feature_names,
            },
            "layout": {
                "title": "Correlation Matrix Comparison",
                "colorscale": "RdBu",
            },
        }
    
    def plot_fidelity_summary(
        self,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Create fidelity summary visualization.
        
        Args:
            metrics: Dictionary of fidelity metrics
            
        Returns:
            Dictionary with summary visualization data
        """
        # Gauge chart data
        gauge_data = []
        
        for metric_name, value in metrics.items():
            if "divergence" in metric_name.lower() or "distance" in metric_name.lower():
                # Lower is better
                score = max(0, 1 - value)
            else:
                score = value
            
            gauge_data.append({
                "metric": metric_name,
                "value": value,
                "score": score,
            })
        
        return {
            "type": "fidelity_summary",
            "data": gauge_data,
            "layout": {
                "title": "Fidelity Metrics Summary",
            },
        }


class PrivacyVisualizer:
    """
    Visualizer for privacy metrics.
    
    Creates visualizations for membership inference attacks,
    attribute inference, and differential privacy metrics.
    """
    
    def __init__(self, style: str = "default"):
        """
        Initialize privacy visualizer.
        
        Args:
            style: Visualization style
        """
        self.style = style
    
    def plot_attack_results(
        self,
        mia_results: Dict[str, float],
        aia_results: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Create attack results visualization.
        
        Args:
            mia_results: Membership inference attack results
            aia_results: Attribute inference attack results
            
        Returns:
            Dictionary with attack visualization data
        """
        attack_data = []
        
        # MIA results
        for key, value in mia_results.items():
            attack_data.append({
                "attack_type": "Membership Inference",
                "metric": key,
                "value": value,
                "baseline": 0.5 if "auc" in key.lower() else None,
            })
        
        # AIA results
        if aia_results:
            for key, value in aia_results.items():
                attack_data.append({
                    "attack_type": "Attribute Inference",
                    "metric": key,
                    "value": value,
                })
        
        return {
            "type": "attack_results",
            "data": attack_data,
            "layout": {
                "title": "Privacy Attack Results",
                "x_axis": "Metric",
                "y_axis": "Value",
            },
        }
    
    def plot_privacy_budget(
        self,
        spent_epsilon: float,
        target_epsilon: float,
        history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Create privacy budget visualization.
        
        Args:
            spent_epsilon: Epsilon spent so far
            target_epsilon: Target epsilon budget
            history: Optional history of budget expenditure
            
        Returns:
            Dictionary with budget visualization data
        """
        gauge_data = {
            "spent": spent_epsilon,
            "target": target_epsilon,
            "remaining": max(0, target_epsilon - spent_epsilon),
            "usage_ratio": spent_epsilon / target_epsilon,
        }
        
        timeline_data = None
        if history:
            timeline_data = {
                "steps": [h["step"] for h in history],
                "cumulative_epsilon": [h["cumulative_epsilon"] for h in history],
            }
        
        return {
            "type": "privacy_budget",
            "data": {
                "gauge": gauge_data,
                "timeline": timeline_data,
            },
            "layout": {
                "title": "Privacy Budget Expenditure",
            },
        }
    
    def plot_privacy_risk_matrix(
        self,
        risk_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Create privacy risk matrix visualization.
        
        Args:
            risk_scores: Dictionary of risk scores
            
        Returns:
            Dictionary with risk matrix data
        """
        # Risk categories
        categories = ["Low", "Medium", "High", "Critical"]
        
        risk_data = []
        for risk_name, score in risk_scores.items():
            if score < 0.25:
                category = "Low"
            elif score < 0.5:
                category = "Medium"
            elif score < 0.75:
                category = "High"
            else:
                category = "Critical"
            
            risk_data.append({
                "risk_type": risk_name,
                "score": score,
                "category": category,
            })
        
        return {
            "type": "risk_matrix",
            "data": risk_data,
            "layout": {
                "title": "Privacy Risk Assessment",
                "categories": categories,
            },
        }


class MultimodalVisualizer:
    """
    Visualizer for multimodal metrics.
    
    Creates visualizations for cross-modal consistency,
    alignment, and coherence metrics.
    """
    
    def __init__(self, style: str = "default"):
        """
        Initialize multimodal visualizer.
        
        Args:
            style: Visualization style
        """
        self.style = style
    
    def plot_cross_modal_alignment(
        self,
        alignment_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Create cross-modal alignment visualization.
        
        Args:
            alignment_scores: Dictionary of alignment scores
            
        Returns:
            Dictionary with alignment visualization data
        """
        return {
            "type": "cross_modal_alignment",
            "data": {
                "pairs": list(alignment_scores.keys()),
                "scores": list(alignment_scores.values()),
            },
            "layout": {
                "title": "Cross-Modal Alignment Scores",
                "x_axis": "Modality Pair",
                "y_axis": "Alignment Score",
            },
        }
    
    def plot_modality_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Create modality embedding visualization (2D projection).
        
        Args:
            embeddings: Dictionary of modality embeddings
            labels: Optional labels for coloring
            
        Returns:
            Dictionary with embedding visualization data
        """
        # Use PCA or t-SNE for 2D projection
        projected = {}
        
        for mod_name, emb in embeddings.items():
            if emb.shape[1] > 2:
                try:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    projected[mod_name] = pca.fit_transform(emb).tolist()
                except:
                    projected[mod_name] = emb[:, :2].tolist()
            else:
                projected[mod_name] = emb[:, :2].tolist()
        
        return {
            "type": "embedding_scatter",
            "data": {
                "embeddings": projected,
                "labels": labels.tolist() if labels is not None else None,
            },
            "layout": {
                "title": "Modality Embedding Space",
                "x_axis": "Dimension 1",
                "y_axis": "Dimension 2",
            },
        }


class DashboardGenerator:
    """
    Comprehensive dashboard generator.
    
    Combines all visualizers into a unified dashboard.
    """
    
    def __init__(
        self,
        style: str = "default",
        output_format: str = "html"
    ):
        """
        Initialize dashboard generator.
        
        Args:
            style: Visualization style
            output_format: Output format ('html', 'json')
        """
        self.style = style
        self.output_format = output_format
        
        self.fairness_viz = FairnessVisualizer(style)
        self.fidelity_viz = FidelityVisualizer(style)
        self.privacy_viz = PrivacyVisualizer(style)
        self.multimodal_viz = MultimodalVisualizer(style)
    
    def generate(
        self,
        evaluation_results: Dict[str, Any],
        title: str = "Fair Synthetic Data Evaluation Dashboard"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive dashboard.
        
        Args:
            evaluation_results: Complete evaluation results
            title: Dashboard title
            
        Returns:
            Dictionary with complete dashboard data
        """
        dashboard = {
            "metadata": {
                "title": title,
                "generated_at": datetime.now().isoformat(),
                "style": self.style,
            },
            "sections": {},
        }
        
        # Fairness section
        if "fairness" in evaluation_results:
            dashboard["sections"]["fairness"] = self._create_fairness_section(
                evaluation_results["fairness"]
            )
        
        # Fidelity section
        if "fidelity" in evaluation_results:
            dashboard["sections"]["fidelity"] = self._create_fidelity_section(
                evaluation_results["fidelity"]
            )
        
        # Privacy section
        if "privacy" in evaluation_results:
            dashboard["sections"]["privacy"] = self._create_privacy_section(
                evaluation_results["privacy"]
            )
        
        # Multimodal section
        if "multimodal" in evaluation_results:
            dashboard["sections"]["multimodal"] = self._create_multimodal_section(
                evaluation_results["multimodal"]
            )
        
        # Overall summary
        dashboard["summary"] = self._create_overall_summary(evaluation_results)
        
        return dashboard
    
    def _create_fairness_section(self, fairness_data: Dict) -> Dict[str, Any]:
        """Create fairness visualization section."""
        section = {"title": "Fairness Evaluation", "visualizations": []}
        
        # Group fairness
        if "group_fairness" in fairness_data:
            viz = self.fairness_viz.plot_group_fairness(
                fairness_data["group_fairness"]
            )
            section["visualizations"].append(viz)
        
        # Intersectional fairness
        if "intersectional" in fairness_data:
            viz = self.fairness_viz.plot_intersectional_heatmap(
                fairness_data["intersectional"]
            )
            section["visualizations"].append(viz)
        
        return section
    
    def _create_fidelity_section(self, fidelity_data: Dict) -> Dict[str, Any]:
        """Create fidelity visualization section."""
        section = {"title": "Fidelity Evaluation", "visualizations": []}
        
        # Fidelity summary
        if "metrics" in fidelity_data:
            viz = self.fidelity_viz.plot_fidelity_summary(fidelity_data["metrics"])
            section["visualizations"].append(viz)
        
        return section
    
    def _create_privacy_section(self, privacy_data: Dict) -> Dict[str, Any]:
        """Create privacy visualization section."""
        section = {"title": "Privacy Evaluation", "visualizations": []}
        
        # Attack results
        if "membership_inference" in privacy_data:
            viz = self.privacy_viz.plot_attack_results(
                privacy_data["membership_inference"]
            )
            section["visualizations"].append(viz)
        
        # Privacy budget
        if "dp_budget" in privacy_data:
            viz = self.privacy_viz.plot_privacy_budget(
                privacy_data["dp_budget"]["spent"],
                privacy_data["dp_budget"]["target"]
            )
            section["visualizations"].append(viz)
        
        return section
    
    def _create_multimodal_section(self, multimodal_data: Dict) -> Dict[str, Any]:
        """Create multimodal visualization section."""
        section = {"title": "Multimodal Evaluation", "visualizations": []}
        
        # Cross-modal alignment
        if "alignment" in multimodal_data:
            viz = self.multimodal_viz.plot_cross_modal_alignment(
                multimodal_data["alignment"]
            )
            section["visualizations"].append(viz)
        
        return section
    
    def _create_overall_summary(self, results: Dict) -> Dict[str, Any]:
        """Create overall summary section."""
        scores = {}
        
        if "fairness" in results:
            if "overall" in results["fairness"]:
                scores["fairness"] = results["fairness"]["overall"].get("fairness_score", 0.5)
        
        if "fidelity" in results:
            if "overall" in results["fidelity"]:
                scores["fidelity"] = results["fidelity"]["overall"].get("fidelity_score", 0.5)
        
        if "privacy" in results:
            if "summary" in results["privacy"]:
                scores["privacy"] = results["privacy"]["summary"].get("privacy_score", 0.5)
        
        if "multimodal" in results:
            if "overall" in results["multimodal"]:
                scores["multimodal"] = results["multimodal"]["overall"].get("multimodal_score", 0.5)
        
        overall_score = np.mean(list(scores.values())) if scores else 0.5
        
        return {
            "overall_score": float(overall_score),
            "component_scores": scores,
            "grade": self._compute_grade(overall_score),
        }
    
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
    
    def save(
        self,
        dashboard: Dict[str, Any],
        output_path: str
    ) -> None:
        """
        Save dashboard to file.
        
        Args:
            dashboard: Dashboard data
            output_path: Output file path
        """
        _, ext = os.path.splitext(output_path)
        
        if ext == ".json" or self.output_format == "json":
            with open(output_path, "w") as f:
                json.dump(dashboard, f, indent=2)
        else:
            # Generate HTML
            html = self._generate_html(dashboard)
            with open(output_path, "w") as f:
                f.write(html)
    
    def _generate_html(self, dashboard: Dict) -> str:
        """Generate HTML representation of dashboard."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{dashboard['metadata']['title']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; padding: 20px; background: #fafafa; border-radius: 5px; }}
        .section h2 {{ color: #333; margin-top: 0; }}
        .visualization {{ margin: 15px 0; padding: 15px; background: white; border: 1px solid #ddd; border-radius: 5px; }}
        .summary {{ text-align: center; padding: 30px; }}
        .score-circle {{ width: 150px; height: 150px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-size: 48px; font-weight: bold; }}
        .grade-A {{ background: #2ecc71; color: white; }}
        .grade-B {{ background: #27ae60; color: white; }}
        .grade-C {{ background: #f39c12; color: white; }}
        .grade-D {{ background: #e67e22; color: white; }}
        .grade-F {{ background: #e74c3c; color: white; }}
        .metadata {{ color: #666; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>{dashboard['metadata']['title']}</h1>
            <p class="metadata">Generated: {dashboard['metadata']['generated_at']}</p>
        </div>
        
        <div class="summary">
            <div class="score-circle grade-{dashboard['summary']['grade']}">
                {dashboard['summary']['overall_score']:.2f}
            </div>
            <p>Overall Score (Grade: {dashboard['summary']['grade']})</p>
        </div>
        
        {self._render_sections_html(dashboard.get('sections', {}))}
    </div>
</body>
</html>
"""
        return html
    
    def _render_sections_html(self, sections: Dict) -> str:
        """Render sections as HTML."""
        html = ""
        
        for section_name, section_data in sections.items():
            html += f"""
        <div class="section">
            <h2>{section_data.get('title', section_name.title())}</h2>
            {self._render_visualizations_html(section_data.get('visualizations', []))}
        </div>
"""
        
        return html
    
    def _render_visualizations_html(self, visualizations: List) -> str:
        """Render visualizations as HTML."""
        html = ""
        
        for viz in visualizations:
            viz_type = viz.get('type', 'unknown')
            html += f"""
            <div class="visualization">
                <h4>{viz_type.replace('_', ' ').title()}</h4>
                <pre>{json.dumps(viz.get('data', {}), indent=2)}</pre>
            </div>
"""
        
        return html
