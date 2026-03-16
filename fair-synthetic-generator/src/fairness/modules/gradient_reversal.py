import torch
import torch.nn as nn
import numpy as np

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    """
    GRL reverses the gradient during backpropagation.
    Used for adversarial debiasing.
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


class ScheduledGradientReversalLayer(GradientReversalLayer):
    """
    GRL with scheduled alpha that increases over training steps.
    Helps stabilize adversarial training.
    """
    def __init__(
        self, 
        lambda_start=0.0, 
        lambda_end=1.0, 
        warmup_epochs=10, 
        schedule_type="linear"
    ):
        super().__init__(alpha=lambda_start)
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.current_epoch = 0
        self.lambda_ = lambda_start

    def step(self):
        """Advance the schedule."""
        self.current_epoch += 1
        progress = min(self.current_epoch / self.warmup_epochs, 1.0)
        
        if self.schedule_type == "linear":
            self.lambda_ = self.lambda_start + progress * (self.lambda_end - self.lambda_start)
        elif self.schedule_type == "cosine":
            self.lambda_ = self.lambda_start + 0.5 * (self.lambda_end - self.lambda_start) * (
                1 - np.cos(np.pi * progress)
            )
        
        self.alpha = self.lambda_
