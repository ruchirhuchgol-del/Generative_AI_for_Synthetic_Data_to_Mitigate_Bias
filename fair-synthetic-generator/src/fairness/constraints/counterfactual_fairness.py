import torch
from src.fairness.constraints.base_constraint import BaseFairnessConstraint

class CounterfactualFairnessConstraint(BaseFairnessConstraint):
    """
    Counterfactual fairness constraint.
    Requires f(x|S=s) = f(x|S=s') for an individual.
    Often implemented using a counterfactual generator.
    """
    def __init__(self, name: str = "counterfactual_fairness", config: dict = None):
        super().__init__(name, config)

    def forward(self, outputs: torch.Tensor, cf_outputs: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        outputs: Model outputs for original data
        cf_outputs: Model outputs for counterfactual data (same individuals, different S)
        """
        # Distance between original and counterfactual outputs
        return torch.nn.functional.mse_loss(outputs, cf_outputs)
