"""
Sensitive Attribute Handler
===========================

Utilities for handling sensitive attributes in fairness-aware ML.

This module provides:
- SensitiveAttributeHandler: Main handler for sensitive attributes
- MultiSensitiveAttributeHandler: Handler for multiple sensitive attributes
- SensitiveAttributeEncoder: Encoders for sensitive attributes
- GroupIndexMapper: Mapping utilities for group indices
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum
from collections import Counter

import torch
import torch.nn as nn
import numpy as np


class EncodingType(Enum):
    """Encoding types for sensitive attributes."""
    ONE_HOT = "one_hot"
    LABEL = "label"
    EMBEDDING = "embedding"
    BINARY = "binary"


class SensitiveAttributeHandler:
    """
    Handler for sensitive attribute management.
    
    Provides utilities for:
        - Encoding and decoding sensitive attributes
        - Computing group statistics
        - Creating masks for group subsets
        - Validating sensitive attribute data
    
    Example:
        >>> handler = SensitiveAttributeHandler(
        ...     name="gender",
        ...     values=["male", "female", "other"]
        ... )
        >>> 
        >>> # Encode
        >>> encoded = handler.encode(raw_values)
        >>> 
        >>> # Get group mask
        >>> mask = handler.get_group_mask(encoded, group_id=1)
    """
    
    def __init__(
        self,
        name: str,
        values: List[Any],
        privileged: Optional[Any] = None,
        encoding: str = "label",
        embedding_dim: int = 16,
    ):
        """
        Initialize sensitive attribute handler.
        
        Args:
            name: Name of the sensitive attribute
            values: List of possible values
            privileged: Which value is privileged (optional)
            encoding: Encoding type ("label", "one_hot", "embedding", "binary")
            embedding_dim: Dimension for embedding encoding
        """
        self.name = name
        self.values = list(values)
        self.privileged = privileged
        self.encoding = EncodingType(encoding)
        self.embedding_dim = embedding_dim
        
        # Build value to index mapping
        self.value_to_idx = {v: i for i, v in enumerate(self.values)}
        self.idx_to_value = {i: v for i, v in enumerate(self.values)}
        
        self.num_groups = len(self.values)
        self._privileged_idx = self.value_to_idx.get(privileged) if privileged else None
        
        # Embedding layer (if using embedding encoding)
        if self.encoding == EncodingType.EMBEDDING:
            self.embedding = nn.Embedding(self.num_groups, embedding_dim)
        else:
            self.embedding = None
    
    def encode(
        self,
        values: Union[List[Any], torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Encode raw values to tensor.
        
        Args:
            values: Raw attribute values
            
        Returns:
            Encoded tensor
        """
        if isinstance(values, torch.Tensor):
            return values
        
        if isinstance(values, np.ndarray):
            values = values.tolist()
        
        # Convert to indices
        indices = []
        for v in values:
            if v in self.value_to_idx:
                indices.append(self.value_to_idx[v])
            else:
                raise ValueError(f"Unknown value: {v}")
        
        indices = torch.tensor(indices, dtype=torch.long)
        
        if self.encoding == EncodingType.LABEL:
            return indices
        
        elif self.encoding == EncodingType.ONE_HOT:
            return F.one_hot(indices, num_classes=self.num_groups).float()
        
        elif self.encoding == EncodingType.BINARY:
            if self._privileged_idx is None:
                raise ValueError("Binary encoding requires privileged value")
            return (indices == self._privileged_idx).long()
        
        elif self.encoding == EncodingType.EMBEDDING:
            if self.embedding is None:
                raise ValueError("Embedding not initialized")
            return self.embedding(indices)
        
        else:
            return indices
    
    def decode(
        self,
        encoded: torch.Tensor
    ) -> List[Any]:
        """
        Decode tensor back to raw values.
        
        Args:
            encoded: Encoded tensor
            
        Returns:
            List of raw values
        """
        if self.encoding == EncodingType.ONE_HOT:
            indices = encoded.argmax(dim=-1)
        elif self.encoding == EncodingType.EMBEDDING:
            # Find nearest embedding
            with torch.no_grad():
                distances = torch.cdist(encoded, self.embedding.weight)
                indices = distances.argmin(dim=-1)
        else:
            indices = encoded
        
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        
        return [self.idx_to_value[i] for i in indices]
    
    def get_group_mask(
        self,
        groups: torch.Tensor,
        group_id: Union[int, str]
    ) -> torch.Tensor:
        """
        Get boolean mask for a specific group.
        
        Args:
            groups: Group indices tensor
            group_id: Group to select (index or value)
            
        Returns:
            Boolean mask tensor
        """
        if isinstance(group_id, str):
            group_idx = self.value_to_idx[group_id]
        else:
            group_idx = group_id
        
        return (groups == group_idx)
    
    def get_privileged_mask(
        self,
        groups: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get mask for privileged group."""
        if self._privileged_idx is None:
            return None
        return self.get_group_mask(groups, self._privileged_idx)
    
    def get_unprivileged_mask(
        self,
        groups: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Get mask for unprivileged groups."""
        if self._privileged_idx is None:
            return None
        return (groups != self._privileged_idx)
    
    def get_group_statistics(
        self,
        groups: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute group statistics.
        
        Args:
            groups: Group indices tensor
            
        Returns:
            Dictionary with counts and proportions
        """
        stats = {}
        
        if isinstance(groups, torch.Tensor):
            groups_np = groups.detach().cpu().numpy()
        else:
            groups_np = np.array(groups)
        
        total = len(groups_np)
        
        for idx, value in self.idx_to_value.items():
            count = int((groups_np == idx).sum())
            stats[value] = {
                "count": count,
                "proportion": count / total if total > 0 else 0
            }
        
        stats["total"] = total
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "values": self.values,
            "privileged": self.privileged,
            "encoding": self.encoding.value,
            "embedding_dim": self.embedding_dim,
            "num_groups": self.num_groups
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SensitiveAttributeHandler":
        """Create from dictionary configuration."""
        return cls(
            name=config["name"],
            values=config["values"],
            privileged=config.get("privileged"),
            encoding=config.get("encoding", "label"),
            embedding_dim=config.get("embedding_dim", 16)
        )
    
    def __repr__(self) -> str:
        return (
            f"SensitiveAttributeHandler("
            f"name={self.name}, "
            f"num_groups={self.num_groups}, "
            f"encoding={self.encoding.value})"
        )


class MultiSensitiveAttributeHandler(nn.Module):
    """
    Handler for multiple sensitive attributes.
    
    Manages multiple sensitive attributes and provides:
        - Joint encoding/decoding
        - Intersectional group computation
        - Multi-attribute statistics
    
    Example:
        >>> handler = MultiSensitiveAttributeHandler(
        ...     attributes={
        ...         "gender": SensitiveAttributeHandler("gender", ["M", "F"]),
        ...         "race": SensitiveAttributeHandler("race", ["A", "B", "C"])
        ...     }
        ... )
        >>> 
        >>> # Get intersectional groups
        >>> intersectional = handler.get_intersectional_groups(data)
    """
    
    def __init__(
        self,
        attributes: Dict[str, SensitiveAttributeHandler],
        compute_intersections: bool = True
    ):
        """
        Initialize multi-attribute handler.
        
        Args:
            attributes: Dictionary of attribute handlers
            compute_intersections: Whether to compute intersectional groups
        """
        super().__init__()
        
        self.attributes = attributes
        self.compute_intersections = compute_intersections
        self.attribute_names = list(attributes.keys())
        
        # Compute intersectional groups
        if compute_intersections:
            self._compute_intersectional_mapping()
        else:
            self._intersectional_mapping = None
            self.num_intersectional_groups = 0
    
    def _compute_intersectional_mapping(self) -> None:
        """Compute mapping for intersectional groups."""
        # Create all combinations
        from itertools import product
        
        group_sizes = [attr.num_groups for attr in self.attributes.values()]
        
        self._intersectional_mapping = {}
        idx = 0
        
        for combo in product(*[range(s) for s in group_sizes]):
            self._intersectional_mapping[combo] = idx
            idx += 1
        
        self.num_intersectional_groups = idx
    
    def encode(
        self,
        data: Dict[str, Union[List, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Encode multiple attributes.
        
        Args:
            data: Dictionary of attribute values
            
        Returns:
            Dictionary of encoded tensors
        """
        encoded = {}
        
        for name, values in data.items():
            if name in self.attributes:
                encoded[name] = self.attributes[name].encode(values)
        
        return encoded
    
    def get_intersectional_groups(
        self,
        data: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get intersectional group indices.
        
        Args:
            data: Dictionary of encoded attributes
            
        Returns:
            Intersectional group indices tensor
        """
        if not self.compute_intersections:
            raise ValueError("Intersectional computation not enabled")
        
        batch_size = next(iter(data.values())).size(0)
        device = next(iter(data.values())).device
        
        # Convert to indices if one-hot
        indices = {}
        for name, tensor in data.items():
            if tensor.dim() > 1 and tensor.size(-1) > 1:
                # One-hot or similar
                indices[name] = tensor.argmax(dim=-1)
            else:
                indices[name] = tensor
        
        # Compute intersectional group for each sample
        intersectional = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            combo = tuple(indices[name][i].item() for name in self.attribute_names)
            intersectional[i] = self._intersectional_mapping[combo]
        
        return intersectional
    
    def get_group_mask(
        self,
        data: Dict[str, torch.Tensor],
        attribute: str,
        group_id: Union[int, str]
    ) -> torch.Tensor:
        """
        Get mask for a specific group on one attribute.
        
        Args:
            data: Dictionary of encoded attributes
            attribute: Attribute name
            group_id: Group to select
            
        Returns:
            Boolean mask tensor
        """
        if attribute not in self.attributes:
            raise ValueError(f"Unknown attribute: {attribute}")
        
        return self.attributes[attribute].get_group_mask(data[attribute], group_id)
    
    def get_intersectional_mask(
        self,
        data: Dict[str, torch.Tensor],
        groups: Dict[str, Union[int, str]]
    ) -> torch.Tensor:
        """
        Get mask for intersectional group.
        
        Args:
            data: Dictionary of encoded attributes
            groups: Dictionary of {attribute: group_id}
            
        Returns:
            Boolean mask tensor
        """
        masks = []
        
        for attr_name, group_id in groups.items():
            mask = self.get_group_mask(data, attr_name, group_id)
            masks.append(mask)
        
        # Intersection of all masks
        result = masks[0]
        for mask in masks[1:]:
            result = result & mask
        
        return result
    
    def get_statistics(
        self,
        data: Dict[str, torch.Tensor]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all attributes.
        
        Args:
            data: Dictionary of encoded attributes
            
        Returns:
            Nested dictionary of statistics
        """
        stats = {}
        
        for name, tensor in data.items():
            if name in self.attributes:
                stats[name] = self.attributes[name].get_group_statistics(tensor)
        
        if self.compute_intersections:
            intersectional = self.get_intersectional_groups(data)
            stats["intersectional"] = {
                "num_groups": self.num_intersectional_groups,
                "counts": dict(Counter(intersectional.tolist()))
            }
        
        return stats
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "attributes": {k: v.to_dict() for k, v in self.attributes.items()},
            "compute_intersections": self.compute_intersections,
            "num_intersectional_groups": self.num_intersectional_groups
        }


class SensitiveAttributeEncoder(nn.Module):
    """
    Neural network encoder for sensitive attributes.
    
    Transforms sensitive attribute information into learned embeddings
    that can be used for conditioning or fairness constraints.
    
    Example:
        >>> encoder = SensitiveAttributeEncoder(
        ...     num_attributes=2,
        ...     num_groups=[2, 3],
        ...     output_dim=64
        ... )
        >>> 
        >>> encoded = encoder([gender_tensor, race_tensor])
    """
    
    def __init__(
        self,
        num_attributes: int,
        num_groups: List[int],
        output_dim: int,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        use_attention: bool = False,
        dropout: float = 0.1
    ):
        """
        Initialize sensitive attribute encoder.
        
        Args:
            num_attributes: Number of sensitive attributes
            num_groups: Number of groups per attribute
            output_dim: Output embedding dimension
            embedding_dim: Per-attribute embedding dimension
            hidden_dim: Hidden layer dimension
            use_attention: Whether to use attention for combining attributes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_attributes = num_attributes
        self.num_groups = num_groups
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Per-attribute embeddings
        self.embeddings = nn.ModuleList([
            nn.Embedding(n, embedding_dim) for n in num_groups
        ])
        
        # Combination layer
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=2,
                dropout=dropout,
                batch_first=True
            )
            self.combiner = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            self.combiner = nn.Sequential(
                nn.Linear(embedding_dim * num_attributes, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim)
            )
    
    def forward(
        self,
        attribute_tensors: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Encode sensitive attributes.
        
        Args:
            attribute_tensors: List of attribute index tensors
            
        Returns:
            Combined encoded tensor
        """
        if len(attribute_tensors) != self.num_attributes:
            raise ValueError(
                f"Expected {self.num_attributes} attributes, "
                f"got {len(attribute_tensors)}"
            )
        
        # Get embeddings for each attribute
        embeddings = []
        for i, tensor in enumerate(attribute_tensors):
            emb = self.embeddings[i](tensor)
            embeddings.append(emb)
        
        # Combine embeddings
        if self.use_attention:
            stacked = torch.stack(embeddings, dim=1)  # (batch, num_attr, emb_dim)
            attended, _ = self.attention(stacked, stacked, stacked)
            combined = attended.mean(dim=1)  # (batch, emb_dim)
        else:
            combined = torch.cat(embeddings, dim=-1)  # (batch, num_attr * emb_dim)
        
        return self.combiner(combined)


class GroupIndexMapper:
    """
    Utility for mapping between different group representations.
    
    Handles conversion between:
        - Raw values (e.g., "male", "female")
        - Integer indices (e.g., 0, 1)
        - One-hot encoding
        - Binary privileged/unprivileged
    
    Example:
        >>> mapper = GroupIndexMapper(
        ...     group_names=["privileged", "unprivileged"]
        ... )
        >>> 
        >>> indices = mapper.names_to_indices(["unprivileged", "privileged"])
        >>> one_hot = mapper.indices_to_onehot(indices)
    """
    
    def __init__(
        self,
        group_names: List[str],
        privileged_idx: Optional[int] = None
    ):
        """
        Initialize group index mapper.
        
        Args:
            group_names: List of group names
            privileged_idx: Index of privileged group (if applicable)
        """
        self.group_names = group_names
        self.num_groups = len(group_names)
        self.privileged_idx = privileged_idx
        
        self._name_to_idx = {name: i for i, name in enumerate(group_names)}
        self._idx_to_name = {i: name for i, name in enumerate(group_names)}
    
    def names_to_indices(
        self,
        names: List[str]
    ) -> torch.Tensor:
        """Convert group names to indices."""
        return torch.tensor([self._name_to_idx[n] for n in names])
    
    def indices_to_names(
        self,
        indices: torch.Tensor
    ) -> List[str]:
        """Convert indices to group names."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return [self._idx_to_name[i] for i in indices]
    
    def indices_to_onehot(
        self,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """Convert indices to one-hot encoding."""
        return F.one_hot(indices.long(), num_classes=self.num_groups).float()
    
    def onehot_to_indices(
        self,
        onehot: torch.Tensor
    ) -> torch.Tensor:
        """Convert one-hot encoding to indices."""
        return onehot.argmax(dim=-1)
    
    def indices_to_binary(
        self,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert indices to binary privileged/unprivileged.
        
        Returns 1 for privileged, 0 for unprivileged.
        """
        if self.privileged_idx is None:
            raise ValueError("No privileged group specified")
        return (indices == self.privileged_idx).long()
    
    def binary_to_indices(
        self,
        binary: torch.Tensor,
        unprivileged_idx: int = 0
    ) -> torch.Tensor:
        """
        Convert binary to indices.
        
        Args:
            binary: Binary tensor (1 = privileged, 0 = unprivileged)
            unprivileged_idx: Index for unprivileged group
        """
        if self.privileged_idx is None:
            raise ValueError("No privileged group specified")
        
        result = torch.full_like(binary, unprivileged_idx)
        result[binary == 1] = self.privileged_idx
        return result
    
    def get_group_sizes(
        self,
        indices: torch.Tensor
    ) -> Dict[str, int]:
        """Get sizes of each group."""
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        
        counts = Counter(indices)
        return {self._idx_to_name[i]: count for i, count in counts.items()}


# Import F for one_hot function
import torch.nn.functional as F
