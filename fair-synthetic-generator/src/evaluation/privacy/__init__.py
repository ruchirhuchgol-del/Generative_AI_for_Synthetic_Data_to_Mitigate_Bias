"""
Privacy Metrics Module
======================

Comprehensive metrics for privacy evaluation including:
- Membership inference attacks
- Attribute inference attacks
- Differential privacy accounting
"""

from src.evaluation.privacy.membership_inference import (
    MembershipInferenceAttack,
    ShadowModelMIA,
    LossBasedMIA,
    MIADefenseEvaluator,
    MembershipInferenceEvaluator,
)
from src.evaluation.privacy.attribute_inference import (
    AttributeInferenceAttack,
    CorrelationBasedAIA,
    ModelBasedAIA,
    AttributeInferenceEvaluator,
)
from src.evaluation.privacy.differential_privacy import (
    DifferentialPrivacyAccountant,
    EpsilonDeltaCalculator,
    PrivacyBudgetScheduler,
    DifferentialPrivacyVerifier,
    DifferentialPrivacyEvaluator,
)
from src.evaluation.privacy.privacy_metrics import (
    PrivacyMetric,
    KAnonymityChecker,
    LDiversityChecker,
    PrivacyReport,
)

__all__ = [
    # Base
    "PrivacyMetric",
    # Membership inference
    "MembershipInferenceAttack",
    "ShadowModelMIA",
    "LossBasedMIA",
    "MIADefenseEvaluator",
    "MembershipInferenceEvaluator",
    # Attribute inference
    "AttributeInferenceAttack",
    "CorrelationBasedAIA",
    "ModelBasedAIA",
    "AttributeInferenceEvaluator",
    # Differential privacy
    "DifferentialPrivacyAccountant",
    "EpsilonDeltaCalculator",
    "PrivacyBudgetScheduler",
    "DifferentialPrivacyVerifier",
    "DifferentialPrivacyEvaluator",
    # Privacy checks
    "KAnonymityChecker",
    "LDiversityChecker",
    "PrivacyReport",
]
