"""
Fidelity Metrics Module
=======================

Metrics for measuring synthetic data fidelity including:
- Distribution metrics
- Statistical similarity
- Downstream utility
"""

from src.evaluation.fidelity.statistical_similarity import (
    FidelityMetric,
    JensenShannonDivergence,
    WassersteinDistance,
    CorrelationPreservation,
)
from src.evaluation.fidelity.distribution_metrics import (
    BaseDistributionMetric,
    KolmogorovSmirnovTest,
    AndersonDarlingTest,
    CramervonMisesTest,
    MaximumMeanDiscrepancy,
    EnergyDistance,
    TotalVariationDistance,
    HistogramIntersection,
    ChiSquaredTest,
    DistributionComparator,
)
from src.evaluation.fidelity.downstream_utility import (
    DownstreamUtilityMetric,
    TrainOnSyntheticTestOnReal,
    TrainOnRealTestOnSynthetic,
    CrossValidationUtility,
    FeatureImportancePreservation,
    QueryWorkloadUtility,
    DownstreamUtilityEvaluator,
)

__all__ = [
    # Base
    "FidelityMetric",
    "BaseDistributionMetric",
    "DownstreamUtilityMetric",
    # Statistical similarity
    "JensenShannonDivergence",
    "WassersteinDistance",
    "CorrelationPreservation",
    # Distribution metrics
    "KolmogorovSmirnovTest",
    "AndersonDarlingTest",
    "CramervonMisesTest",
    "MaximumMeanDiscrepancy",
    "EnergyDistance",
    "TotalVariationDistance",
    "HistogramIntersection",
    "ChiSquaredTest",
    "DistributionComparator",
    # Downstream utility
    "TrainOnSyntheticTestOnReal",
    "TrainOnRealTestOnSynthetic",
    "CrossValidationUtility",
    "FeatureImportancePreservation",
    "QueryWorkloadUtility",
    "DownstreamUtilityEvaluator",
]
