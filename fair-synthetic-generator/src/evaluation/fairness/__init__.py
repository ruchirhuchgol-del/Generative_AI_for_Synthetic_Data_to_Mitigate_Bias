"""
Fairness Metrics Module
=======================

Comprehensive metrics for fairness evaluation including:
- Group fairness metrics
- Individual fairness metrics
- Counterfactual fairness metrics
- Intersectional fairness metrics
"""

from src.evaluation.fairness.group_metrics import (
    GroupFairnessMetric,
    DemographicParityMetric,
    EqualizedOddsMetric,
    DisparateImpactMetric,
)
from src.evaluation.fairness.individual_metrics import (
    IndividualFairnessMetric,
    ConsistencyScore,
    LipschitzEstimator,
)
from src.evaluation.fairness.counterfactual_metrics import (
    CounterfactualFairnessMetric,
    CounterfactualInvariance,
    CounterfactualEffectSize,
    CausalEffectRatio,
    CounterfactualDemographicParity,
    CounterfactualEqualizedOdds,
    IndividualTreatmentEffect,
    CounterfactualConsistency,
    CounterfactualFairnessEvaluator,
)
from src.evaluation.fairness.intersectional_metrics import (
    IntersectionalFairnessMetric,
    IntersectionalDemographicParity,
    IntersectionalEqualizedOdds,
    IntersectionalDisparateImpact,
    SubgroupAnalysis,
    AttributeInteractionEffect,
    IntersectionalFairnessEvaluator,
)

__all__ = [
    # Base classes
    "GroupFairnessMetric",
    "IndividualFairnessMetric",
    "CounterfactualFairnessMetric",
    "IntersectionalFairnessMetric",
    # Group fairness
    "DemographicParityMetric",
    "EqualizedOddsMetric",
    "DisparateImpactMetric",
    # Individual fairness
    "ConsistencyScore",
    "LipschitzEstimator",
    # Counterfactual fairness
    "CounterfactualInvariance",
    "CounterfactualEffectSize",
    "CausalEffectRatio",
    "CounterfactualDemographicParity",
    "CounterfactualEqualizedOdds",
    "IndividualTreatmentEffect",
    "CounterfactualConsistency",
    "CounterfactualFairnessEvaluator",
    # Intersectional fairness
    "IntersectionalDemographicParity",
    "IntersectionalEqualizedOdds",
    "IntersectionalDisparateImpact",
    "SubgroupAnalysis",
    "AttributeInteractionEffect",
    "IntersectionalFairnessEvaluator",
]
