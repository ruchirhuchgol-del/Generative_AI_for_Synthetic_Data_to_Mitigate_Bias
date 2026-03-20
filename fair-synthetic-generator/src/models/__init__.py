"""
Models Module
=============

This module provides encoders, decoders, generators, discriminators, and complete 
architectures for fair synthetic data generation.

Available Architectures:
- FairGAN: GAN with adversarial debiasing
- FairDiffusion: Diffusion with adversarial debiasing
- DebiasedVAE: VAE with adversarial debiasing
- CounterfactualGenerator: Causal-based counterfactual generation
"""

from src.models.architectures import (
    FairGAN,
    FairDiffusion,
    DebiasedVAE,
    CounterfactualGenerator,
    get_architecture,
    list_architectures,
)
from src.models.encoders import *
from src.models.decoders import *
from src.models.generators import *
from src.models.discriminators import *

__all__ = [
    "FairGAN",
    "FairDiffusion",
    "DebiasedVAE",
    "CounterfactualGenerator",
    "get_architecture",
    "list_architectures",
]
