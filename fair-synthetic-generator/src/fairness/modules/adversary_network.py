import torch
import torch.nn as nn
from src.core.base_module import BaseModule

class FairnessAdversary(BaseModule):
    """
    Standard adversary network for predicting sensitive attributes
    from latent representations.
    """
    def __init__(self, name: str = "fairness_adversary", config: dict = None):
        super().__init__(name, config)
        input_dim = self.config.get('input_dim', 128)
        output_dim = self.config.get('n_sensitive_classes', 2)
        hidden_dims = self.config.get('hidden_dims', [64, 32])
        
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=self.config.get('dropout', 0.1))
            ])
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiTaskAdversary(BaseModule):
    """
    Multi-task adversary for predicting multiple sensitive attributes
    simultaneously from shared latent representations.
    """
    def __init__(self, name: str = "multi_task_adversary", config: dict = None):
        super().__init__(name, config)
        input_dim = self.config.get('input_dim', 128)
        sensitive_configs = self.config.get('sensitive_configs', {'attr1': 2})
        hidden_dims = self.config.get('hidden_dims', [64, 32])
        
        # Shared encoder
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(p=self.config.get('dropout', 0.1))
            ])
            curr_dim = h_dim
        self.shared_encoder = nn.Sequential(*layers)
        
        # Individual output heads
        self.heads = nn.ModuleDict({
            attr: nn.Linear(curr_dim, n_classes) 
            for attr, n_classes in sensitive_configs.items()
        })

    def forward(self, x):
        latent = self.shared_encoder(x)
        return {attr: head(latent) for attr, head in self.heads.items()}
