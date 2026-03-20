"""
Fairness Module
===============

Comprehensive fairness constraints, losses, and utilities for fair synthetic data generation.

This module provides:

## Fairness Paradigms

### Group Fairness
- DemographicParity: Statistical parity across groups
- EqualizedOdds: Equal TPR and FPR across groups
- EqualOpportunity: Equal TPR across groups
- PredictiveEquality: Equal FPR across groups
- DisparateImpact: 80% rule compliance
- StatisticalParityDifference: Difference in positive rates

### Individual Fairness
- LipschitzConstraint: Lipschitz continuity for similar predictions
- ConsistencyConstraint: k-NN based consistency
- LocalFairnessConstraint: Group fairness within neighborhoods

### Counterfactual Fairness
- CounterfactualFairness: What-if sensitive attribute were different

## Loss Functions
- AdversarialDebiasingLoss: Adversarial training for debiasing
- MultiObjectiveLoss: Multi-objective optimization
- ContrastiveFairnessLoss: Contrastive learning for fairness
- MutualInformationLoss: MI minimization for independence

## Regularizers
- FairnessRegularizer: Main regularization module
- AdversarialRegularizer: Adversarial debiasing regularizer
- ReweighingRegularizer: Sample reweighing for fairness
- CombinedFairnessRegularizer: Combined methods

## Utilities
- SensitiveAttributeHandler: Manage sensitive attributes
- FairnessBounds: Fairness threshold management
- FairnessThresholdScheduler: Scheduled threshold adjustment

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │        Fairness Module              │
                    └─────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │   Group      │          │  Individual  │          │Counterfactual│
    │   Fairness   │          │   Fairness   │          │   Fairness   │
    │ - Dem. Parity│          │ - Lipschitz  │          │ - What-if    │
    │ - Eq. Odds   │          │ - Consistency│          │ - Causal     │
    │ - Disp. Imp. │          │              │          │              │
    └──────────────┘          └──────────────┘          └──────────────┘
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │   Losses & Regularization           │
                    │   - Adversarial Debiasing           │
                    │   - Multi-Objective Optimization    │
                    │   - Gradient Reversal Layer         │
                    └─────────────────────────────────────┘
```

## Example Usage

### Group Fairness Constraint
```python
from src.fairness import DemographicParity

constraint = DemographicParity(threshold=0.05)
violation = constraint.compute(predictions, groups)
loss = constraint.loss(predictions, groups)
```

### Fairness Regularization
```python
from src.fairness import FairnessRegularizer, DemographicParity, EqualizedOdds

regularizer = FairnessRegularizer(
    constraints=[
        DemographicParity(threshold=0.05),
        EqualizedOdds(threshold=0.05)
    ],
    initial_weight=0.1
)

result = regularizer(predictions, groups, labels)
total_loss = task_loss + result["regularization_loss"]
```

### Adversarial Debiasing
```python
from src.fairness import AdversarialDebiasingLoss

debiasing_loss = AdversarialDebiasingLoss(
    adversary=adversary_network,
    fairness_weight=1.0,
    use_grl=True
)

# During main model training
loss_dict = debiasing_loss(latent, sensitive_attrs)
loss_dict["total"].backward()
```

### Sensitive Attribute Handling
```python
from src.fairness import SensitiveAttributeHandler

handler = SensitiveAttributeHandler(
    name="gender",
    values=["male", "female", "other"],
    privileged="male"
)

encoded = handler.encode(raw_values)
mask = handler.get_unprivileged_mask(encoded)
```
"""

# Base classes
from src.fairness.constraints.base_constraint import (
    BaseFairnessConstraint,
    ConstraintType,
    RelaxationType,
    ConstraintCombiner,
)

# Group fairness constraints
from src.fairness.constraints.group_fairness import GroupFairnessConstraint
from src.fairness.constraints.demographic_parity import (
    DemographicParity,
    ConditionalDemographicParity,
    AggregationType,
)
from src.fairness.constraints.equalized_odds import (
    EqualizedOdds,
    EqualOpportunity,
    PredictiveEquality,
    AccuracyParity,
    EqualizedOddsType,
)
from src.fairness.constraints.disparate_impact import (
    DisparateImpact,
    FourFifthsRule,
    StatisticalParityDifference,
    CalibrationDifference,
)

# Individual fairness constraints
from src.fairness.constraints.individual_fairness import IndividualFairnessConstraint
from src.fairness.individual_fairness.lipschitz_constraint import (
    LipschitzConstraint,
    AdaptiveLipschitzConstraint,
    FeatureWeightedLipschitzConstraint,
    DistanceMetric,
)
from src.fairness.individual_fairness.consistency_constraint import (
    ConsistencyConstraint,
    LocalFairnessConstraint,
    SmoothedConsistencyConstraint,
)

# Counterfactual fairness
from src.fairness.constraints.counterfactual_fairness import (
    CounterfactualFairnessConstraint,
)

from src.fairness.losses.fairness_loss import (
    FairnessLoss,
    MultiObjectiveFairnessLoss,
)
from src.fairness.losses.adversarial_loss import (
    AdversarialDebiasingLoss,
    MultiAdversaryLoss,
    ContrastiveFairnessLoss,
    MutualInformationLoss,
)
from src.fairness.losses.multi_objective_loss import (
    MultiObjectiveLoss,
    DynamicFairnessWeightScheduler,
    EpsilonConstraintHandler,
    ScalarizationMethod,
    WeightUpdateStrategy,
)

# Modules
from src.fairness.modules.adversary_network import (
    FairnessAdversary,
    MultiTaskAdversary,
)
from src.fairness.modules.gradient_reversal import (
    GradientReversalFunction,
    GradientReversalLayer,
    ScheduledGradientReversalLayer,
)

# Regularizers
from src.fairness.fairness_regularizer import (
    FairnessRegularizer,
    AdversarialRegularizer,
    ReweighingRegularizer,
    CombinedFairnessRegularizer,
    RegularizationType,
    create_fairness_regularizer,
)

# Utils
from src.fairness.utils.sensitive_attribute_handler import (
    SensitiveAttributeHandler,
    MultiSensitiveAttributeHandler,
    SensitiveAttributeEncoder,
    GroupIndexMapper,
    EncodingType,
)
from src.fairness.utils.fairness_bounds import (
    FairnessBounds,
    AdaptiveFairnessBounds,
    FairnessThresholdScheduler,
    BoundType,
    compute_fairness_bounds,
    compute_statistical_bounds,
)


__all__ = [
    # Base classes
    "BaseFairnessConstraint",
    "ConstraintType",
    "RelaxationType",
    "ConstraintCombiner",
    
    # Group fairness constraints
    "DemographicParity",
    "ConditionalDemographicParity",
    "AggregationType",
    "EqualizedOdds",
    "EqualOpportunity",
    "PredictiveEquality",
    "AccuracyParity",
    "EqualizedOddsType",
    "DisparateImpact",
    "FourFifthsRule",
    "StatisticalParityDifference",
    "CalibrationDifference",
    
    # Individual fairness constraints
    "IndividualFairnessConstraint",
    "LipschitzConstraint",
    "AdaptiveLipschitzConstraint",
    "FeatureWeightedLipschitzConstraint",
    "DistanceMetric",
    "ConsistencyConstraint",
    "LocalFairnessConstraint",
    "SmoothedConsistencyConstraint",
    
    # Counterfactual fairness
    "CounterfactualFairnessConstraint",
    
    # Losses
    "FairnessLoss",
    "MultiObjectiveFairnessLoss",
    "AdversarialDebiasingLoss",
    "MultiAdversaryLoss",
    "ContrastiveFairnessLoss",
    "MutualInformationLoss",
    "MultiObjectiveLoss",
    "DynamicFairnessWeightScheduler",
    "EpsilonConstraintHandler",
    "ScalarizationMethod",
    "WeightUpdateStrategy",
    
    # Modules
    "FairnessAdversary",
    "MultiTaskAdversary",
    "GradientReversalFunction",
    "GradientReversalLayer",
    "ScheduledGradientReversalLayer",
    
    # Regularizers
    "FairnessRegularizer",
    "AdversarialRegularizer",
    "ReweighingRegularizer",
    "CombinedFairnessRegularizer",
    "RegularizationType",
    "create_fairness_regularizer",
    
    # Utils
    "SensitiveAttributeHandler",
    "MultiSensitiveAttributeHandler",
    "SensitiveAttributeEncoder",
    "GroupIndexMapper",
    "EncodingType",
    "FairnessBounds",
    "AdaptiveFairnessBounds",
    "FairnessThresholdScheduler",
    "BoundType",
    "compute_fairness_bounds",
    "compute_statistical_bounds",
]


# Constraint Registry for easy lookup
CONSTRAINT_REGISTRY = {
    # Group fairness
    "demographic_parity": DemographicParity,
    "conditional_demographic_parity": ConditionalDemographicParity,
    "equalized_odds": EqualizedOdds,
    "equal_opportunity": EqualOpportunity,
    "predictive_equality": PredictiveEquality,
    "accuracy_parity": AccuracyParity,
    "disparate_impact": DisparateImpact,
    "four_fifths_rule": FourFifthsRule,
    "statistical_parity_difference": StatisticalParityDifference,
    "calibration_difference": CalibrationDifference,
    
    # Individual fairness
    "lipschitz": LipschitzConstraint,
    "adaptive_lipschitz": AdaptiveLipschitzConstraint,
    "feature_weighted_lipschitz": FeatureWeightedLipschitzConstraint,
    "consistency": ConsistencyConstraint,
    "local_fairness": LocalFairnessConstraint,
    "smoothed_consistency": SmoothedConsistencyConstraint,
    
    # Counterfactual
    "counterfactual": CounterfactualFairnessConstraint,
}


def get_constraint(name: str, **kwargs) -> BaseFairnessConstraint:
    """
    Get a constraint by name from the registry.
    
    Args:
        name: Constraint name
        **kwargs: Arguments to pass to the constraint constructor
        
    Returns:
        Constraint instance
        
    Raises:
        ValueError: If constraint name is not found
        
    Example:
        >>> constraint = get_constraint("demographic_parity", threshold=0.05)
    """
    if name not in CONSTRAINT_REGISTRY:
        available = list(CONSTRAINT_REGISTRY.keys())
        raise ValueError(
            f"Unknown constraint: {name}. Available constraints: {available}"
        )
    
    return CONSTRAINT_REGISTRY[name](**kwargs)

from typing import List

def list_constraints() -> List[str]:
    """List all available constraint names."""
    return list(CONSTRAINT_REGISTRY.keys())
