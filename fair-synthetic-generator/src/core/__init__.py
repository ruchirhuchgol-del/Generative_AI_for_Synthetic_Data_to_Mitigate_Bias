"""
Core Module
===========

Base classes, constants, and utility functions for the Fair Synthetic Data Generator.

This module provides the foundational components for the entire framework:
- Abstract base classes for all components (encoders, decoders, generators, etc.)
- Centralized constants and enumerations
- Utility functions for common operations

Usage:
    from src.core import BaseEncoder, set_seed, DEFAULT_SEED
    
    # Set random seed for reproducibility
    set_seed(DEFAULT_SEED)
    
    # Create a custom encoder
    class MyEncoder(BaseEncoder):
        def __init__(self, input_dim, latent_dim):
            super().__init__("my_encoder", {"input_dim": input_dim, "latent_dim": latent_dim})
            self._input_dim = input_dim
            self._latent_dim = latent_dim
            self.encoder = nn.Sequential(...)
            
        def encode(self, x, **kwargs):
            return self.encoder(x)
            
        @property
        def latent_dim(self):
            return self._latent_dim
            
        @property
        def input_dim(self):
            return self._input_dim
"""

from src.core.base_module import (
    # Base classes
    BaseModule,
    BaseEncoder,
    BaseDecoder,
    BaseGenerator,
    BaseDiscriminator,
    BaseLoss,
    BaseMetric,
    BaseTrainer,
    BaseCallback,
    # Mixin classes
    SerializableMixin,
    ConfigurableMixin,
    FairnessMixin,
)
from src.core.constants import (
    # Version
    __version__,
    # Seeds
    DEFAULT_SEED,
    # Enums
    ModalityType,
    FairnessParadigm,
    GroupFairnessMetric,
    IndividualFairnessMetric,
    CounterfactualMetric,
    LossType,
    MetricType,
    EncoderType,
    DecoderType,
    GeneratorType,
    DataType,
    SensitiveAttributeType,
    # Lists
    MODALITY_TYPES,
    FAIRNESS_PARADIGMS,
    GROUP_FAIRNESS_METRICS,
    INDIVIDUAL_FAIRNESS_METRICS,
    COUNTERFACTUAL_METRICS,
    LOSS_TYPES,
    METRIC_TYPES,
    # Dictionaries
    ACTIVATION_FUNCTIONS,
    INITIALIZATION_METHODS,
    OPTIMIZER_TYPES,
    LR_SCHEDULER_TYPES,
    PRIVACY_DEFAULTS,
    FAIRNESS_THRESHOLDS,
    DIFFUSION_DEFAULTS,
    # File extensions
    CHECKPOINT_EXTENSIONS,
    DATA_EXTENSIONS,
    # Numerical
    EPS,
    INFINITY,
    NEG_INFINITY,
)
from src.core.utils import (
    # Random seed
    set_seed,
    # Device management
    get_device,
    move_to_device,
    # Parameter utilities
    count_parameters,
    get_parameter_info,
    print_model_summary,
    get_gradient_norm,
    clip_gradients,
    # Logging
    get_logger,
    # File operations
    ensure_dir,
    load_config,
    save_config,
    merge_configs,
    # Model utilities
    get_activation_function,
    initialize_weights,
    freeze_module,
    unfreeze_module,
    # Distance computations
    compute_pairwise_distances,
    # Formatting
    format_time,
    format_number,
    # Data utilities
    tensor_to_numpy,
    numpy_to_tensor,
    one_hot_encode,
    normalize_tensor,
    # Registry
    Registry,
    register_component,
    get_component,
    list_components,
)

__all__ = [
    # ==========================================
    # Base Classes
    # ==========================================
    "BaseModule",
    "BaseEncoder",
    "BaseDecoder",
    "BaseGenerator",
    "BaseDiscriminator",
    "BaseLoss",
    "BaseMetric",
    "BaseTrainer",
    "BaseCallback",
    # Mixins
    "SerializableMixin",
    "ConfigurableMixin",
    "FairnessMixin",
    
    # ==========================================
    # Constants & Enums
    # ==========================================
    "__version__",
    "DEFAULT_SEED",
    "EPS",
    "INFINITY",
    "NEG_INFINITY",
    # Enums
    "ModalityType",
    "FairnessParadigm",
    "GroupFairnessMetric",
    "IndividualFairnessMetric",
    "CounterfactualMetric",
    "LossType",
    "MetricType",
    "EncoderType",
    "DecoderType",
    "GeneratorType",
    "DataType",
    "SensitiveAttributeType",
    # Lists
    "MODALITY_TYPES",
    "FAIRNESS_PARADIGMS",
    "GROUP_FAIRNESS_METRICS",
    "INDIVIDUAL_FAIRNESS_METRICS",
    "COUNTERFACTUAL_METRICS",
    "LOSS_TYPES",
    "METRIC_TYPES",
    # Dictionaries
    "ACTIVATION_FUNCTIONS",
    "INITIALIZATION_METHODS",
    "OPTIMIZER_TYPES",
    "LR_SCHEDULER_TYPES",
    "PRIVACY_DEFAULTS",
    "FAIRNESS_THRESHOLDS",
    "DIFFUSION_DEFAULTS",
    "CHECKPOINT_EXTENSIONS",
    "DATA_EXTENSIONS",
    
    # ==========================================
    # Utility Functions
    # ==========================================
    # Random seed
    "set_seed",
    # Device management
    "get_device",
    "move_to_device",
    # Parameter utilities
    "count_parameters",
    "get_parameter_info",
    "print_model_summary",
    "get_gradient_norm",
    "clip_gradients",
    # Logging
    "get_logger",
    # File operations
    "ensure_dir",
    "load_config",
    "save_config",
    "merge_configs",
    # Model utilities
    "get_activation_function",
    "initialize_weights",
    "freeze_module",
    "unfreeze_module",
    # Distance computations
    "compute_pairwise_distances",
    # Formatting
    "format_time",
    "format_number",
    # Data utilities
    "tensor_to_numpy",
    "numpy_to_tensor",
    "one_hot_encode",
    "normalize_tensor",
    # Registry
    "Registry",
    "register_component",
    "get_component",
    "list_components",
]
