import torch
import torch.nn as nn
from src.core.base_module import BaseModule

class DemographicParityConstraint(BaseModule):
    """Demographic Parity fairness constraint."""
    def __init__(self, config=None):
        super().__init__(config)
        self.sensitive_attr_idx = config.get('sensitive_attr_idx', 0)

    def forward(self, outputs, sensitive_attrs):
        """
        Calculates the demographic parity loss.
        DP requires P(Y_hat=1 | S=0) = P(Y_hat=1 | S=1)
        """
        group0_mask = (sensitive_attrs == 0)
        group1_mask = (sensitive_attrs == 1)
        
        if group0_mask.sum() == 0 or group1_mask.sum() == 0:
            return torch.tensor(0.0, requires_grad=True)

        prob0 = outputs[group0_mask].mean()
        prob1 = outputs[group1_mask].mean()
        
        return torch.abs(prob0 - prob1)
