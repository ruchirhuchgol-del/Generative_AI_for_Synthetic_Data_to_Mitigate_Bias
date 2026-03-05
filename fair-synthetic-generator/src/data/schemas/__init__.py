"""
Data Schemas Package
====================

Schema definitions for multimodal data with fairness considerations.
"""

from .data_schema import (
    FeatureType,
    FeatureSchema,
    TabularSchema,
    TextSchema,
    ImageSchema,
    DataSchema,
)
from .sensitive_attribute import (
    AttributeType,
    SensitiveAttribute,
    IntersectionalGroup,
    SensitiveAttributeManager,
    COMMON_SENSITIVE_ATTRIBUTES,
)

__all__ = [
    # Feature types
    "FeatureType",
    "FeatureSchema",
    # Schema classes
    "TabularSchema",
    "TextSchema",
    "ImageSchema",
    "DataSchema",
    # Sensitive attributes
    "AttributeType",
    "SensitiveAttribute",
    "IntersectionalGroup",
    "SensitiveAttributeManager",
    "COMMON_SENSITIVE_ATTRIBUTES",
]
