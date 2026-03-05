"""
Data Schema Definitions
=======================

Schema classes for defining data structures across modalities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


class FeatureType(Enum):
    """Feature data types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"
    DATETIME = "datetime"
    TEXT = "text"


@dataclass
class FeatureSchema:
    """
    Schema definition for a single feature.
    
    Attributes:
        name: Feature name
        dtype: Feature data type
        nullable: Whether feature can be null
        min_value: Minimum value for numerical features
        max_value: Maximum value for numerical features
        categories: List of categories for categorical features
        embedding_dim: Embedding dimension for categorical features
        transform: Transformation to apply
        sensitive: Whether this is a sensitive/protected attribute
    """
    name: str
    dtype: FeatureType
    nullable: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    categories: Optional[List[Any]] = None
    embedding_dim: Optional[int] = None
    transform: Optional[str] = None
    sensitive: bool = False
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate feature schema."""
        if self.dtype == FeatureType.CATEGORICAL and self.categories is None:
            raise ValueError(f"Categorical feature '{self.name}' requires categories")
        
        if self.dtype == FeatureType.NUMERICAL:
            if self.min_value is not None and self.max_value is not None:
                if self.min_value >= self.max_value:
                    raise ValueError(f"min_value must be less than max_value for '{self.name}'")
        
        # Auto-set embedding dimension for categorical features
        if self.dtype == FeatureType.CATEGORICAL and self.embedding_dim is None:
            cardinality = len(self.categories) if self.categories else 0
            self.embedding_dim = min(32, (cardinality + 1) // 2)
    
    @property
    def cardinality(self) -> Optional[int]:
        """Get cardinality for categorical features."""
        if self.dtype in [FeatureType.CATEGORICAL, FeatureType.ORDINAL]:
            return len(self.categories) if self.categories else None
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "nullable": self.nullable,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "categories": self.categories,
            "embedding_dim": self.embedding_dim,
            "transform": self.transform,
            "sensitive": self.sensitive,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureSchema":
        """Create schema from dictionary."""
        data["dtype"] = FeatureType(data["dtype"])
        return cls(**data)


@dataclass
class TabularSchema:
    """
    Schema definition for tabular data.
    
    Attributes:
        features: List of feature schemas
        target: Target variable schema (if supervised)
        name: Dataset name
    """
    features: List[FeatureSchema]
    target: Optional[FeatureSchema] = None
    name: str = "tabular_data"
    
    @property
    def numerical_features(self) -> List[FeatureSchema]:
        """Get all numerical features."""
        return [f for f in self.features if f.dtype == FeatureType.NUMERICAL]
    
    @property
    def categorical_features(self) -> List[FeatureSchema]:
        """Get all categorical features."""
        return [f for f in self.features 
                if f.dtype in [FeatureType.CATEGORICAL, FeatureType.ORDINAL]]
    
    @property
    def binary_features(self) -> List[FeatureSchema]:
        """Get all binary features."""
        return [f for f in self.features if f.dtype == FeatureType.BINARY]
    
    @property
    def sensitive_features(self) -> List[FeatureSchema]:
        """Get all sensitive/protected features."""
        return [f for f in self.features if f.sensitive]
    
    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names."""
        return [f.name for f in self.features]
    
    @property
    def input_dim(self) -> int:
        """Get total input dimension (excluding target)."""
        dim = len(self.numerical_features)
        for f in self.categorical_features:
            dim += f.embedding_dim or 0
        return dim
    
    @property
    def output_dim(self) -> int:
        """Get output dimension for generation."""
        return len(self.features)
    
    def get_feature(self, name: str) -> Optional[FeatureSchema]:
        """Get feature by name."""
        for f in self.features:
            if f.name == name:
                return f
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "features": [f.to_dict() for f in self.features],
            "target": self.target.to_dict() if self.target else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabularSchema":
        """Create schema from dictionary."""
        features = [FeatureSchema.from_dict(f) for f in data["features"]]
        target = FeatureSchema.from_dict(data["target"]) if data.get("target") else None
        return cls(features=features, target=target, name=data.get("name", "tabular_data"))


@dataclass
class TextSchema:
    """
    Schema definition for text data.
    
    Attributes:
        name: Field name
        max_length: Maximum sequence length
        vocab_size: Vocabulary size (None for character-level)
        tokenizer: Tokenizer type
        lowercase: Whether to lowercase text
        padding_strategy: Padding strategy ('max_length', 'longest')
        truncation: Whether to truncate long sequences
    """
    name: str
    max_length: int = 256
    vocab_size: Optional[int] = None
    tokenizer: str = "whitespace"  # whitespace, bert, gpt2, sentencepiece
    lowercase: bool = True
    padding_strategy: str = "max_length"
    truncation: bool = True
    
    @property
    def input_dim(self) -> int:
        """Get input dimension."""
        return self.max_length
    
    @property
    def output_dim(self) -> int:
        """Get output dimension."""
        return self.max_length
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "max_length": self.max_length,
            "vocab_size": self.vocab_size,
            "tokenizer": self.tokenizer,
            "lowercase": self.lowercase,
            "padding_strategy": self.padding_strategy,
            "truncation": self.truncation,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextSchema":
        """Create schema from dictionary."""
        return cls(**data)


@dataclass
class ImageSchema:
    """
    Schema definition for image data.
    
    Attributes:
        name: Field name
        height: Image height in pixels
        width: Image width in pixels
        channels: Number of color channels
        color_mode: Color mode ('RGB', 'L', 'RGBA')
        normalization: Normalization method
        augmentations: List of augmentation names
    """
    name: str
    height: int = 256
    width: int = 256
    channels: int = 3
    color_mode: str = "RGB"
    normalization: str = "imagenet"  # imagenet, minmax, none
    augmentations: List[str] = field(default_factory=list)
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get image shape as (channels, height, width)."""
        return (self.channels, self.height, self.width)
    
    @property
    def input_dim(self) -> int:
        """Get total input dimension."""
        return self.channels * self.height * self.width
    
    @property
    def output_dim(self) -> int:
        """Get total output dimension."""
        return self.channels * self.height * self.width
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "color_mode": self.color_mode,
            "normalization": self.normalization,
            "augmentations": self.augmentations,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageSchema":
        """Create schema from dictionary."""
        return cls(**data)


@dataclass
class DataSchema:
    """
    Combined schema for multimodal data.
    
    Attributes:
        name: Dataset name
        tabular: Tabular data schema
        text: Text data schema
        image: Image data schema
        metadata: Additional metadata
    """
    name: str
    tabular: Optional[TabularSchema] = None
    text: Optional[TextSchema] = None
    image: Optional[ImageSchema] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def modalities(self) -> List[str]:
        """Get list of available modalities."""
        mods = []
        if self.tabular is not None:
            mods.append("tabular")
        if self.text is not None:
            mods.append("text")
        if self.image is not None:
            mods.append("image")
        return mods
    
    @property
    def sensitive_attributes(self) -> List[str]:
        """Get list of sensitive attribute names."""
        if self.tabular is None:
            return []
        return [f.name for f in self.tabular.sensitive_features]
    
    @property
    def total_input_dim(self) -> Dict[str, int]:
        """Get total input dimensions per modality."""
        dims = {}
        if self.tabular is not None:
            dims["tabular"] = self.tabular.input_dim
        if self.text is not None:
            dims["text"] = self.text.input_dim
        if self.image is not None:
            dims["image"] = self.image.input_dim
        return dims
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "name": self.name,
            "tabular": self.tabular.to_dict() if self.tabular else None,
            "text": self.text.to_dict() if self.text else None,
            "image": self.image.to_dict() if self.image else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataSchema":
        """Create schema from dictionary."""
        tabular = TabularSchema.from_dict(data["tabular"]) if data.get("tabular") else None
        text = TextSchema.from_dict(data["text"]) if data.get("text") else None
        image = ImageSchema.from_dict(data["image"]) if data.get("image") else None
        return cls(
            name=data["name"],
            tabular=tabular,
            text=text,
            image=image,
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def load(cls, path: str) -> "DataSchema":
        """Load schema from JSON file."""
        import json
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save(self, path: str) -> None:
        """Save schema to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
