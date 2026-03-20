"""
Synthesis Module
================

End-to-end pipeline for generating fair synthetic data.

This module provides:
- GeneratorPipeline: Main pipeline for synthetic data generation
- Postprocessing: Consistency checking, fairness auditing, quality filtering
- Output: Data export and format conversion

Example Usage:
    >>> from src.synthesis import GeneratorPipeline, GenerationConfig
    >>> from src.synthesis.postprocessing import ConsistencyChecker, FairnessAuditor
    >>> 
    >>> # Create pipeline
    >>> config = GenerationConfig(
    ...     n_samples=10000,
    ...     ensure_fairness=True,
    ...     sensitive_attributes=["gender", "race"]
    ... )
    >>> pipeline = GeneratorPipeline(model=trained_model, config=config)
    >>> 
    >>> # Generate data
    >>> result = pipeline.generate_with_fairness(n_samples=10000)
    >>> synthetic_data = result["data"]
"""

from src.synthesis.generator_pipeline import (
    GenerationConfig,
    GeneratorPipeline,
    MultiModelPipeline,
    StreamingGenerator,
    BaseGenerator,
)

from src.synthesis.postprocessing import (
    # Consistency checking
    ConsistencyChecker,
    RangeConstraint,
    DependencyConstraint,
    UniquenessConstraint,
    CategoricalConstraint,
    CrossModalConstraint,
    # Fairness auditing
    FairnessAuditor,
    RepresentationAuditor,
    DistributionAuditor,
    CorrelationAuditor,
    ProxyAuditor,
    # Quality filtering
    QualityFilter,
    AdaptiveQualityFilter,
    StreamingQualityFilter,
    OutlierDetector,
)

from src.synthesis.output import (
    DataExporter,
    MetadataWriter,
    SyntheticDataPackage,
    FormatConverter,
    MultiModalConverter,
)

__all__ = [
    # Generator pipeline
    "GenerationConfig",
    "GeneratorPipeline",
    "MultiModelPipeline",
    "StreamingGenerator",
    "BaseGenerator",
    # Consistency checking
    "ConsistencyChecker",
    "RangeConstraint",
    "DependencyConstraint",
    "UniquenessConstraint",
    "CategoricalConstraint",
    "CrossModalConstraint",
    # Fairness auditing
    "FairnessAuditor",
    "RepresentationAuditor",
    "DistributionAuditor",
    "CorrelationAuditor",
    "ProxyAuditor",
    # Quality filtering
    "QualityFilter",
    "AdaptiveQualityFilter",
    "StreamingQualityFilter",
    "OutlierDetector",
    # Output
    "DataExporter",
    "MetadataWriter",
    "SyntheticDataPackage",
    "FormatConverter",
    "MultiModalConverter",
]
