"""
Dashboard Module
================

Tools for generating evaluation reports and visualizations.
"""

from src.evaluation.dashboard.report_generator import FairnessReport
from src.evaluation.dashboard.visualization import (
    FairnessVisualizer,
    FidelityVisualizer,
    PrivacyVisualizer,
    MultimodalVisualizer,
    DashboardGenerator,
)

__all__ = [
    "FairnessReport",
    "FairnessVisualizer",
    "FidelityVisualizer",
    "PrivacyVisualizer",
    "MultimodalVisualizer",
    "DashboardGenerator",
]
