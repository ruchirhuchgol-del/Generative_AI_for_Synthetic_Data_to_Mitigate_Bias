"""
Multimodal Metrics Module
=========================

Metrics for evaluating multimodal synthetic data including:
- Cross-modal consistency
- Alignment metrics
"""

from src.evaluation.multimodal.multimodal_metrics import (
    CrossModalConsistencyMetric,
    MultimodalAlignmentScore,
    ModalityCompletenessMetric,
    MultimodalEvaluator,
)
from src.evaluation.multimodal.cross_modal_consistency import (
    CrossModalRetrievalMetric,
    CrossModalSemanticConsistency,
    CrossModalCoherenceScore,
    CrossModalConsistencyEvaluator,
)
from src.evaluation.multimodal.alignment_metrics import (
    MultimodalAlignmentScore as AlignmentScore,
    FairMultimodalAlignment,
    ModalityGapMetric,
    ContrastiveAlignmentMetric,
    MultimodalAlignmentEvaluator,
)

__all__ = [
    # From multimodal_metrics
    "CrossModalConsistencyMetric",
    "MultimodalAlignmentScore",
    "ModalityCompletenessMetric",
    "MultimodalEvaluator",
    # From cross_modal_consistency
    "CrossModalRetrievalMetric",
    "CrossModalSemanticConsistency",
    "CrossModalCoherenceScore",
    "CrossModalConsistencyEvaluator",
    # From alignment_metrics
    "FairMultimodalAlignment",
    "ModalityGapMetric",
    "ContrastiveAlignmentMetric",
    "MultimodalAlignmentEvaluator",
]
