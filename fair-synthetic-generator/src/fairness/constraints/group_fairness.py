"""
Group Fairness Constraints
==========================

Base classes and common logic for group-based fairness constraints.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from src.fairness.constraints.base_constraint import BaseFairnessConstraint, ConstraintType


class GroupFairnessConstraint(BaseFairnessConstraint):
    """
    Base class for group fairness constraints.
    
    Group fairness (or statistical fairness) requires that some statistic of 
    the model's behavior is equal across different groups defined by 
    sensitive attributes.
    """
    
    def __init__(
        self,
        threshold: float = 0.05,
        relaxation: str = "soft",
        weight: float = 1.0,
        name: Optional[str] = None
    ):
        super().__init__(
            threshold=threshold,
            relaxation=relaxation,
            weight=weight,
            name=name
        )
        self._constraint_type = ConstraintType.GROUP

    def get_group_masks(self, groups: torch.Tensor) -> Dict[Any, torch.Tensor]:
        """
        Create boolean masks for each unique group.
        
        Args:
            groups: Group membership tensor
            
        Returns:
            Dictionary mapping group IDs to boolean masks
        """
        unique_groups = torch.unique(groups)
        return {g.item(): (groups == g) for g in unique_groups}

    def compute_group_means(
        self, 
        values: torch.Tensor, 
        groups: torch.Tensor
    ) -> Dict[Any, torch.Tensor]:
        """
        Compute mean of values for each group.
        
        Args:
            values: Values to average
            groups: Group membership tensor
            
        Returns:
            Dictionary mapping group IDs to mean values
        """
        masks = self.get_group_masks(groups)
        return {g: values[mask].mean() for g, mask in masks.items() if mask.any()}
