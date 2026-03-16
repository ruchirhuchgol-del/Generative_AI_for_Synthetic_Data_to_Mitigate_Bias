"""
Individual Fairness Constraints
===============================

Base classes and common logic for individual-based fairness constraints.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from src.fairness.constraints.base_constraint import BaseFairnessConstraint, ConstraintType


class IndividualFairnessConstraint(BaseFairnessConstraint):
    """
    Base class for individual fairness constraints.
    
    Individual fairness requires that "similar individuals should be treated 
    similarly". This usually involves a distance metric in the feature space 
    and a distance metric in the output space.
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
        self._constraint_type = ConstraintType.INDIVIDUAL

    def compute_pairwise_distances(
        self, 
        x: torch.Tensor, 
        p: int = 2
    ) -> torch.Tensor:
        """
        Compute pairwise distances between individuals.
        
        Args:
            x: Input tensor (batch_size, features)
            p: Norm degree
            
        Returns:
            Distance matrix (batch_size, batch_size)
        """
        return torch.cdist(x, x, p=p)
