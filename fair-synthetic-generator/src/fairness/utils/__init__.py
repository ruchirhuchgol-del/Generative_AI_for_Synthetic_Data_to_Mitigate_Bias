"""
Fairness Utilities
==================
"""

from src.fairness.utils.sensitive_attribute_handler import (
    SensitiveAttributeHandler,
    MultiSensitiveAttributeHandler,
    SensitiveAttributeEncoder,
    GroupIndexMapper,
    EncodingType,
)
from src.fairness.utils.fairness_bounds import (
    FairnessBounds,
    AdaptiveFairnessBounds,
    FairnessThresholdScheduler,
    BoundType,
    compute_fairness_bounds,
    compute_statistical_bounds,
)

__all__ = [
    "SensitiveAttributeHandler",
    "MultiSensitiveAttributeHandler",
    "SensitiveAttributeEncoder",
    "GroupIndexMapper",
    "EncodingType",
    "FairnessBounds",
    "AdaptiveFairnessBounds",
    "FairnessThresholdScheduler",
    "BoundType",
    "compute_fairness_bounds",
    "compute_statistical_bounds",
]
