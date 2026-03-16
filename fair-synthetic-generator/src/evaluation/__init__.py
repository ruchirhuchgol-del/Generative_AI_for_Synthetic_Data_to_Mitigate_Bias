"""
Evaluation Module
=================

Comprehensive evaluation metrics for fair synthetic data generation.

This module provides:
- Fidelity metrics for data quality assessment
- Fairness metrics for bias evaluation
- Privacy metrics for privacy risk assessment
- Multimodal metrics for cross-modal evaluation
- Dashboard tools for reporting and visualization
"""

# Fidelity metrics
from src.evaluation.fidelity import (
    # Base classes
    FidelityMetric,
    BaseDistributionMetric,
    DownstreamUtilityMetric,
    # Statistical similarity
    JensenShannonDivergence,
    WassersteinDistance,
    CorrelationPreservation,
    # Distribution metrics
    KolmogorovSmirnovTest,
    AndersonDarlingTest,
    CramervonMisesTest,
    MaximumMeanDiscrepancy,
    EnergyDistance,
    TotalVariationDistance,
    HistogramIntersection,
    ChiSquaredTest,
    DistributionComparator,
    # Downstream utility
    TrainOnSyntheticTestOnReal,
    TrainOnRealTestOnSynthetic,
    CrossValidationUtility,
    FeatureImportancePreservation,
    QueryWorkloadUtility,
    DownstreamUtilityEvaluator,
)

# Fairness metrics
from src.evaluation.fairness import (
    # Base classes
    GroupFairnessMetric,
    IndividualFairnessMetric,
    CounterfactualFairnessMetric,
    IntersectionalFairnessMetric,
    # Group fairness
    DemographicParityMetric,
    EqualizedOddsMetric,
    DisparateImpactMetric,
    # Individual fairness
    ConsistencyScore,
    LipschitzEstimator,
    # Counterfactual fairness
    CounterfactualInvariance,
    CounterfactualEffectSize,
    CausalEffectRatio,
    CounterfactualDemographicParity,
    CounterfactualEqualizedOdds,
    IndividualTreatmentEffect,
    CounterfactualConsistency,
    CounterfactualFairnessEvaluator,
    # Intersectional fairness
    IntersectionalDemographicParity,
    IntersectionalEqualizedOdds,
    IntersectionalDisparateImpact,
    SubgroupAnalysis,
    AttributeInteractionEffect,
    IntersectionalFairnessEvaluator,
)

# Privacy metrics
from src.evaluation.privacy import (
    # Base
    PrivacyMetric,
    # Membership inference
    MembershipInferenceAttack,
    ShadowModelMIA,
    LossBasedMIA,
    MIADefenseEvaluator,
    MembershipInferenceEvaluator,
    # Attribute inference
    AttributeInferenceAttack,
    CorrelationBasedAIA,
    ModelBasedAIA,
    AttributeInferenceEvaluator,
    # Differential privacy
    DifferentialPrivacyAccountant,
    EpsilonDeltaCalculator,
    PrivacyBudgetScheduler,
    DifferentialPrivacyVerifier,
    DifferentialPrivacyEvaluator,
    # Privacy checks
    KAnonymityChecker,
    LDiversityChecker,
    PrivacyReport,
)

# Multimodal metrics
from src.evaluation.multimodal import (
    CrossModalConsistencyMetric,
    MultimodalAlignmentScore,
    ModalityCompletenessMetric,
    MultimodalEvaluator,
    CrossModalRetrievalMetric,
    CrossModalSemanticConsistency,
    CrossModalCoherenceScore,
    CrossModalConsistencyEvaluator,
    FairMultimodalAlignment,
    ModalityGapMetric,
    ContrastiveAlignmentMetric,
    MultimodalAlignmentEvaluator,
)

# Dashboard tools
from src.evaluation.dashboard import (
    FairnessReport,
    FairnessVisualizer,
    FidelityVisualizer,
    PrivacyVisualizer,
    MultimodalVisualizer,
    DashboardGenerator,
)


__all__ = [
    # Fidelity metrics
    "FidelityMetric",
    "BaseDistributionMetric",
    "DownstreamUtilityMetric",
    "JensenShannonDivergence",
    "WassersteinDistance",
    "CorrelationPreservation",
    "KolmogorovSmirnovTest",
    "AndersonDarlingTest",
    "CramervonMisesTest",
    "MaximumMeanDiscrepancy",
    "EnergyDistance",
    "TotalVariationDistance",
    "HistogramIntersection",
    "ChiSquaredTest",
    "DistributionComparator",
    "TrainOnSyntheticTestOnReal",
    "TrainOnRealTestOnSynthetic",
    "CrossValidationUtility",
    "FeatureImportancePreservation",
    "QueryWorkloadUtility",
    "DownstreamUtilityEvaluator",
    # Fairness metrics
    "GroupFairnessMetric",
    "IndividualFairnessMetric",
    "CounterfactualFairnessMetric",
    "IntersectionalFairnessMetric",
    "DemographicParityMetric",
    "EqualizedOddsMetric",
    "DisparateImpactMetric",
    "ConsistencyScore",
    "LipschitzEstimator",
    "CounterfactualInvariance",
    "CounterfactualEffectSize",
    "CausalEffectRatio",
    "CounterfactualDemographicParity",
    "CounterfactualEqualizedOdds",
    "IndividualTreatmentEffect",
    "CounterfactualConsistency",
    "CounterfactualFairnessEvaluator",
    "IntersectionalDemographicParity",
    "IntersectionalEqualizedOdds",
    "IntersectionalDisparateImpact",
    "SubgroupAnalysis",
    "AttributeInteractionEffect",
    "IntersectionalFairnessEvaluator",
    # Privacy metrics
    "PrivacyMetric",
    "MembershipInferenceAttack",
    "ShadowModelMIA",
    "LossBasedMIA",
    "MIADefenseEvaluator",
    "MembershipInferenceEvaluator",
    "AttributeInferenceAttack",
    "CorrelationBasedAIA",
    "ModelBasedAIA",
    "AttributeInferenceEvaluator",
    "DifferentialPrivacyAccountant",
    "EpsilonDeltaCalculator",
    "PrivacyBudgetScheduler",
    "DifferentialPrivacyVerifier",
    "DifferentialPrivacyEvaluator",
    "KAnonymityChecker",
    "LDiversityChecker",
    "PrivacyReport",
    # Multimodal metrics
    "CrossModalConsistencyMetric",
    "MultimodalAlignmentScore",
    "ModalityCompletenessMetric",
    "MultimodalEvaluator",
    "CrossModalRetrievalMetric",
    "CrossModalSemanticConsistency",
    "CrossModalCoherenceScore",
    "CrossModalConsistencyEvaluator",
    "FairMultimodalAlignment",
    "ModalityGapMetric",
    "ContrastiveAlignmentMetric",
    "MultimodalAlignmentEvaluator",
    # Dashboard
    "FairnessReport",
    "FairnessVisualizer",
    "FidelityVisualizer",
    "PrivacyVisualizer",
    "MultimodalVisualizer",
    "DashboardGenerator",
]
