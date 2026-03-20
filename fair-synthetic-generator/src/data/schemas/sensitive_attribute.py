"""
Sensitive Attribute Definition
==============================

Definition and handling of sensitive/protected attributes for fairness.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class AttributeType(Enum):
    """Types of sensitive attributes."""
    BINARY = "binary"
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"


@dataclass
class SensitiveAttribute:
    """
    Definition of a sensitive/protected attribute.
    
    Sensitive attributes are characteristics that should not influence
    the models predictions for fairness reasons (e.g., gender, race, age).
    
    Attributes:
        name: Name of the sensitive attribute
        attr_type: Type of attribute (binary, categorical, continuous)
        values: Possible values for categorical/binary attributes
        privileged: The privileged/unprivileged value(s)
        description: Human-readable description
        groups: Groupings for analysis (e.g., intersectionality)
    """
    name: str
    attr_type: AttributeType
    values: Optional[List[Any]] = None
    privileged: Optional[Union[Any, List[Any]]] = None
    description: Optional[str] = None
    groups: Optional[Dict[str, List[Any]]] = None
    
    def __post_init__(self):
        """Validate sensitive attribute definition."""
        if self.attr_type == AttributeType.BINARY:
            if self.values is None or len(self.values) != 2:
                raise ValueError(f"Binary attribute '{self.name}' must have exactly 2 values")
        
        if self.attr_type in [AttributeType.BINARY, AttributeType.CATEGORICAL]:
            if self.values is None:
                raise ValueError(f"Attribute '{self.name}' of type {self.attr_type} requires values")
        
        # Validate privileged value
        if self.privileged is not None and self.values is not None:
            if isinstance(self.privileged, list):
                for p in self.privileged:
                    if p not in self.values:
                        raise ValueError(f"Privileged value '{p}' not in values for '{self.name}'")
            else:
                if self.privileged not in self.values:
                    raise ValueError(f"Privileged value '{self.privileged}' not in values for '{self.name}'")
    
    @property
    def num_groups(self) -> int:
        """Get number of distinct groups."""
        if self.attr_type == AttributeType.CONTINUOUS:
            return 1  # Continuous attributes don't have discrete groups
        return len(self.values) if self.values else 1
    
    @property
    def unprivileged(self) -> Optional[List[Any]]:
        """Get unprivileged value(s)."""
        if self.privileged is None or self.values is None:
            return None
        
        privileged_list = self.privileged if isinstance(self.privileged, list) else [self.privileged]
        return [v for v in self.values if v not in privileged_list]
    
    def get_privilege_mask(self, values: List[Any]) -> List[bool]:
        """
        Get boolean mask indicating privileged status.
        
        Args:
            values: List of attribute values
            
        Returns:
            List of booleans (True = privileged)
        """
        if self.privileged is None:
            return [True] * len(values)
        
        privileged_set = set(self.privileged if isinstance(self.privileged, list) else [self.privileged])
        return [v in privileged_set for v in values]
    
    def get_group_index(self, value: Any) -> int:
        """
        Get integer group index for a value.
        
        Args:
            value: Attribute value
            
        Returns:
            Integer index
        """
        if self.values is None:
            return 0
        try:
            return self.values.index(value)
        except ValueError:
            return -1  # Unknown value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "attr_type": self.attr_type.value,
            "values": self.values,
            "privileged": self.privileged,
            "description": self.description,
            "groups": self.groups,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensitiveAttribute":
        """Create from dictionary."""
        data["attr_type"] = AttributeType(data["attr_type"])
        return cls(**data)
    
    def __repr__(self) -> str:
        return f"SensitiveAttribute(name='{self.name}', type={self.attr_type.value}, groups={self.num_groups})"


@dataclass
class IntersectionalGroup:
    """
    Definition of an intersectional group.
    
    Represents a group defined by the intersection of multiple sensitive attributes.
    
    Attributes:
        name: Name for the intersectional group
        attributes: Dictionary mapping attribute names to values
        description: Human-readable description
    """
    name: str
    attributes: Dict[str, Any]
    description: Optional[str] = None
    
    def matches(self, sample: Dict[str, Any]) -> bool:
        """
        Check if a sample matches this intersectional group.
        
        Args:
            sample: Dictionary of attribute values
            
        Returns:
            True if sample matches all attribute conditions
        """
        for attr_name, attr_value in self.attributes.items():
            if sample.get(attr_name) != attr_value:
                return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "attributes": self.attributes,
            "description": self.description,
        }


class SensitiveAttributeManager:
    """
    Manager for multiple sensitive attributes.
    
    Handles:
    - Multiple sensitive attributes
    - Intersectional groups
    - Privilege calculations
    """
    
    def __init__(self, attributes: List[SensitiveAttribute]):
        """
        Initialize the manager.
        
        Args:
            attributes: List of sensitive attributes
        """
        self.attributes = {attr.name: attr for attr in attributes}
        self._intersectional_groups: List[IntersectionalGroup] = []
    
    def add_intersectional_group(self, group: IntersectionalGroup) -> None:
        """Add an intersectional group definition."""
        self._intersectional_groups.append(group)
    
    def get_attribute(self, name: str) -> Optional[SensitiveAttribute]:
        """Get attribute by name."""
        return self.attributes.get(name)
    
    @property
    def attribute_names(self) -> List[str]:
        """Get list of attribute names."""
        return list(self.attributes.keys())
    
    @property
    def total_groups(self) -> int:
        """Get total number of groups across all attributes."""
        return sum(attr.num_groups for attr in self.attributes.values())
    
    def compute_intersectional_groups(
        self,
        attribute_names: List[str]
    ) -> List[IntersectionalGroup]:
        """
        Compute all intersectional groups for given attributes.
        
        Args:
            attribute_names: List of attribute names to intersect
            
        Returns:
            List of all intersectional group combinations
        """
        import itertools
        
        groups = []
        
        # Get all value combinations
        value_lists = []
        for name in attribute_names:
            attr = self.attributes.get(name)
            if attr is None:
                raise ValueError(f"Unknown attribute: {name}")
            value_lists.append(attr.values or [])
        
        # Generate all combinations
        for combo in itertools.product(*value_lists):
            attr_dict = dict(zip(attribute_names, combo))
            name = "_".join(f"{k}={v}" for k, v in attr_dict.items())
            groups.append(IntersectionalGroup(name=name, attributes=attr_dict))
        
        return groups
    
    def get_privilege_vector(
        self,
        sample: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Get privilege status for each attribute.
        
        Args:
            sample: Dictionary of attribute values
            
        Returns:
            Dictionary mapping attribute names to privilege status
        """
        privilege = {}
        for name, attr in self.attributes.items():
            value = sample.get(name)
            privilege[name] = attr.get_privilege_mask([value])[0] if value is not None else None
        return privilege
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "attributes": {name: attr.to_dict() for name, attr in self.attributes.items()},
            "intersectional_groups": [g.to_dict() for g in self._intersectional_groups],
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SensitiveAttributeManager":
        """Create from dictionary."""
        attributes = [SensitiveAttribute.from_dict(a) for a in data["attributes"].values()]
        manager = cls(attributes)
        for g in data.get("intersectional_groups", []):
            manager.add_intersectional_group(IntersectionalGroup(**g))
        return manager


# Common sensitive attribute definitions
COMMON_SENSITIVE_ATTRIBUTES = {
    "gender": SensitiveAttribute(
        name="gender",
        attr_type=AttributeType.BINARY,
        values=["male", "female"],
        privileged="male",
        description="Gender identity"
    ),
    "race": SensitiveAttribute(
        name="race",
        attr_type=AttributeType.CATEGORICAL,
        values=["white", "black", "asian", "hispanic", "other"],
        privileged="white",
        description="Racial/ethnic group"
    ),
    "age_group": SensitiveAttribute(
        name="age_group",
        attr_type=AttributeType.CATEGORICAL,
        values=["young", "middle", "senior"],
        privileged="middle",
        description="Age group classification"
    ),
}
