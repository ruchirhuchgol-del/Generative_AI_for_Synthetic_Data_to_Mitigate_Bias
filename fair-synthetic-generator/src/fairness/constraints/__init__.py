"""
Fairness Constraints Module
===========================

Comprehensive implementations of fairness constraints for different paradigms.

This module provides:

## Group Fairness Constraints
- DemographicParity: Statistical parity across groups
- ConditionalDemographicParity: Conditional on legitimate features
- EqualizedOdds: Equal TPR and FPR across groups
- EqualOpportunity: Equal TPR across groups
- PredictiveEquality: Equal FPR across groups
- DisparateImpact: 80% rule compliance
- StatisticalParityDifference: Difference in positive rates

## Individual Fairness Constraints
- LipschitzConstraint: Similar inputs → similar outputs
- AdaptiveLipschitzConstraint: Learned distance metric
- ConsistencyConstraint: k-NN based consistency
- LocalFairnessConstraint: Group fairness in neighborhoods

## Counterfactual Fairness
- CounterfactualFairness: What if sensitive attribute were different

## Base Classes
- BaseFairnessConstraint: Abstract base class
- ConstraintCombiner: Combine multiple constraints

## Example Usage

### Demographic Parity
```python
from src.fairness.constraints import DemographicParity

constraint = DemographicParity(threshold=0.05)

# Check if satisfied
is_fair = constraint.is_satisfied(predictions, groups)

# Compute loss for training
loss = constraint.loss(predictions, groups)
```

### Constraint Combination
```python
from src.fairness.constraints import (
    ConstraintCombiner,
    DemographicParity,
    EqualizedOdds
)

combiner = ConstraintCombiner([
    DemographicParity(threshold=0.05),
    EqualizedOdds(threshold=0.05)
])

total_loss, metrics = combiner(predictions, groups, labels)
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

# Counterfactual fairness
from src.fairness.constraints.counterfactual_fairness import (
    CounterfactualFairnessConstraint,
)

# Individual fairness constraints
from src.fairness.constraints.individual_fairness import IndividualFairnessConstraint
# Individual fairness (from subdirectory)
from src.fairness.individual_fairness import (
    LipschitzConstraint,
    AdaptiveLipschitzConstraint,
    FeatureWeightedLipschitzConstraint,
    ConsistencyConstraint,
    LocalFairnessConstraint,
    SmoothedConsistencyConstraint,
)


__all__ = [
    # Base classes
    "BaseFairnessConstraint",
    "ConstraintType",
    "RelaxationType",
    "ConstraintCombiner",
    
    # Group fairness - base
    "GroupFairnessConstraint",
    
    # Group fairness - demographic parity
    "DemographicParity",
    "ConditionalDemographicParity",
    "AggregationType",
    
    # Group fairness - equalized odds
    "EqualizedOdds",
    "EqualOpportunity",
    "PredictiveEquality",
    "AccuracyParity",
    "EqualizedOddsType",
    
    # Group fairness - disparate impact
    "DisparateImpact",
    "FourFifthsRule",
    "StatisticalParityDifference",
    "CalibrationDifference",
    
    # Individual fairness
    "IndividualFairnessConstraint",
    "LipschitzConstraint",
    "AdaptiveLipschitzConstraint",
    "FeatureWeightedLipschitzConstraint",
    "ConsistencyConstraint",
    "LocalFairnessConstraint",
    "SmoothedConsistencyConstraint",
    
    # Counterfactual fairness
    "CounterfactualFairnessConstraint",
]


# Group fairness specifically
GROUP_FAIRNESS_CONSTRAINTS = [
    "DemographicParity",
    "ConditionalDemographicParity",
    "EqualizedOdds",
    "EqualOpportunity",
    "PredictiveEquality",
    "AccuracyParity",
    "DisparateImpact",
    "FourFifthsRule",
    "StatisticalParityDifference",
    "CalibrationDifference",
]

# Individual fairness specifically
INDIVIDUAL_FAIRNESS_CONSTRAINTS = [
    "LipschitzConstraint",
    "AdaptiveLipschitzConstraint",
    "FeatureWeightedLipschitzConstraint",
    "ConsistencyConstraint",
    "LocalFairnessConstraint",
    "SmoothedConsistencyConstraint",
]

# Counterfactual fairness
COUNTERFACTUAL_CONSTRAINTS = [
    "CounterfactualFairnessConstraint",
]
