"""
Postprocessing Module
=====================

Comprehensive postprocessing pipeline for synthetic data quality control.
"""

from src.synthesis.postprocessing.consistency_checker import (
    BaseConstraint,
    RangeConstraint,
    DependencyConstraint,
    UniquenessConstraint,
    CategoricalConstraint,
    CrossModalConstraint,
    ConsistencyChecker,
)
from src.synthesis.postprocessing.fairness_auditor import (
    FairnessAuditResult,
    RepresentationAuditor,
    DistributionAuditor,
    CorrelationAuditor,
    ProxyAuditor,
    FairnessAuditor,
)
from src.synthesis.postprocessing.quality_filter import (
    QualityMetric,
    DistanceQualityMetric,
    DensityQualityMetric,
    ReconstructionQualityMetric,
    OutlierDetector,
    QualityFilter,
    AdaptiveQualityFilter,
    StreamingQualityFilter,
)

__all__ = [
    # Consistency checking
    "BaseConstraint",
    "RangeConstraint",
    "DependencyConstraint",
    "UniquenessConstraint",
    "CategoricalConstraint",
    "CrossModalConstraint",
    "ConsistencyChecker",
    # Fairness auditing
    "FairnessAuditResult",
    "RepresentationAuditor",
    "DistributionAuditor",
    "CorrelationAuditor",
    "ProxyAuditor",
    "FairnessAuditor",
    # Quality filtering
    "QualityMetric",
    "DistanceQualityMetric",
    "DensityQualityMetric",
    "ReconstructionQualityMetric",
    "OutlierDetector",
    "QualityFilter",
    "AdaptiveQualityFilter",
    "StreamingQualityFilter",
]
