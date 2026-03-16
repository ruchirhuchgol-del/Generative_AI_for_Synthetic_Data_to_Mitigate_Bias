"""
Individual Fairness Constraints
===============================
"""

from src.fairness.individual_fairness.lipschitz_constraint import (
    LipschitzConstraint,
    AdaptiveLipschitzConstraint,
    FeatureWeightedLipschitzConstraint,
    DistanceMetric,
)
from src.fairness.individual_fairness.consistency_constraint import (
    ConsistencyConstraint,
    LocalFairnessConstraint,
    SmoothedConsistencyConstraint,
)

__all__ = [
    "LipschitzConstraint",
    "AdaptiveLipschitzConstraint",
    "FeatureWeightedLipschitzConstraint",
    "DistanceMetric",
    "ConsistencyConstraint",
    "LocalFairnessConstraint",
    "SmoothedConsistencyConstraint"
]
