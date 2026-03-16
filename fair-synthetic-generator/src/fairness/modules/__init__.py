"""
Fairness Modules
================
"""

from src.fairness.modules.gradient_reversal import (
    GradientReversalLayer,
    ScheduledGradientReversalLayer,
)
from src.fairness.modules.adversary_network import (
    FairnessAdversary,
    MultiTaskAdversary,
)

__all__ = [
    "GradientReversalLayer",
    "ScheduledGradientReversalLayer",
    "FairnessAdversary",
    "MultiTaskAdversary",
]
