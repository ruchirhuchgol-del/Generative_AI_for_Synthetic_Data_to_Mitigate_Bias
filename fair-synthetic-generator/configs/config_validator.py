""
Validation utilities for ensuring configuration correctness.
"""

from typing import Any, Dict, List, Optional, Set
from pathlib import Path
import re


class ConfigValidationError(Exception):
    """Exception raised for configuration validation errors."""
    
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Configuration validation failed with {len(errors)} errors")


class ConfigValidator:
    """
    Validates configuration files for correctness and completeness.
    """
    
    # Required keys for different config types
    REQUIRED_KEYS = {
        "model_config": {
            "model.name",
            "latent_dim",
            "encoders",
            "decoders"
        },
        "training_config": {
            "training.n_epochs",
            "training.batch_size",
            "optimizer",
            "loss_weights"
        },
        "fairness_config": {
            "protected_attributes",
            "fairness_paradigms"
        },
        "data_config": {
            "schema"
        },
        "evaluation_config": {
            "evaluation_modes"
        }
    }
    
    # Valid values for specific keys
    VALID_VALUES = {
        "optimizer.type": {"adam", "adamw", "sgd", "rmsprop", "lion"},
        "activation": {"relu", "leaky_relu", "gelu", "selu", "silu", "tanh", "sigmoid"},
        "precision": {"float16", "bfloat16", "float32", "float64"},
        "device": {"cuda", "cpu", "mps", "auto"},
        "fusion.method": {"cross_attention", "concatenation", "gated", "hierarchical"},
        "generator.type": {"vae", "gan", "vae_gan", "diffusion"},
        "lr_scheduler.type": {"constant", "linear", "cosine", "cosine_warmup", "polynomial", "exponential"}
    }
    
    # Value range constraints
    VALUE_RANGES = {
        "latent_dim": (16, 4096),
        "training.n_epochs": (1, 100000),
        "training.batch_size": (1, 4096),
        "optimizer.lr": (1e-8, 1.0),
        "training.max_grad_norm": (0.0, 100.0),
        "loss_weights.*": (0.0, 100.0)
    }
    
    def __init__(self, strict: bool = True):
        """
        Initialize the validator.
        
        Args:
            strict: If True, raise errors on validation failure.
                   If False, collect warnings.
        """
        self.strict = strict
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def validate(
        self,
        config: Dict[str, Any],
        config_type: Optional[str] = None
    ) -> bool:
        """
        Validate a configuration dictionary.
        
        Args:
            config: Configuration to validate
            config_type: Type of configuration for specific validation
            
        Returns:
            True if valid, False otherwise
        """
        self.errors = []
        self.warnings = []
        
        # Check required keys if config type specified
        if config_type and config_type in self.REQUIRED_KEYS:
            self._check_required_keys(config, self.REQUIRED_KEYS[config_type])
            
        # Validate specific sections
        self._validate_model_config(config)
        self._validate_training_config(config)
        self._validate_fairness_config(config)
        self._validate_loss_weights(config)
        self._validate_consistency(config)
        
        if self.strict and self.errors:
            raise ConfigValidationError(self.errors)
            
        return len(self.errors) == 0
    
    def _check_required_keys(
        self,
        config: Dict[str, Any],
        required_keys: Set[str]
    ) -> None:
        """Check that all required keys are present."""
        for key_path in required_keys:
            value = self._get_nested(config, key_path)
            if value is None:
                self.errors.append(f"Missing required key: {key_path}")
                
    def _validate_model_config(self, config: Dict[str, Any]) -> None:
        """Validate model configuration section."""
        model = config.get("model", {})
        
        # Check latent dimension
        latent_dim = config.get("latent_dim", model.get("latent_dim"))
        if latent_dim is not None:
            if not isinstance(latent_dim, int) or latent_dim < 1:
                self.errors.append(f"latent_dim must be positive integer, got {latent_dim}")
                
        # Check encoders
        encoders = config.get("encoders", {})
        for enc_name, enc_config in encoders.items():
            if enc_config.get("enabled", True):
                self._validate_encoder(enc_name, enc_config)
                
        # Check decoders
        decoders = config.get("decoders", {})
        for dec_name, dec_config in decoders.items():
            if dec_config.get("enabled", True):
                self._validate_decoder(dec_name, dec_config)
                
    def _validate_encoder(self, name: str, config: Dict[str, Any]) -> None:
        """Validate a single encoder configuration."""
        if "type" not in config:
            self.warnings.append(f"Encoder '{name}' has no type specified, using default")
            
        # Check hidden dimensions are valid
        hidden_dims = config.get("hidden_dims", [])
        if hidden_dims:
            if not all(isinstance(d, int) and d > 0 for d in hidden_dims):
                self.errors.append(f"Encoder '{name}' has invalid hidden_dims")
                
    def _validate_decoder(self, name: str, config: Dict[str, Any]) -> None:
        """Validate a single decoder configuration."""
        if "type" not in config:
            self.warnings.append(f"Decoder '{name}' has no type specified, using default")
            
    def _validate_training_config(self, config: Dict[str, Any]) -> None:
        """Validate training configuration section."""
        training = config.get("training", {})
        
        # Check epochs
        n_epochs = training.get("n_epochs")
        if n_epochs is not None:
            if not isinstance(n_epochs, int) or n_epochs < 1:
                self.errors.append(f"n_epochs must be positive integer, got {n_epochs}")
                
        # Check batch size
        batch_size = training.get("batch_size")
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size < 1:
                self.errors.append(f"batch_size must be positive integer, got {batch_size}")
                
        # Check optimizer
        optimizer = config.get("optimizer", {})
        opt_type = optimizer.get("type")
        if opt_type and opt_type not in self.VALID_VALUES.get("optimizer.type", set()):
            self.warnings.append(f"Unknown optimizer type: {opt_type}")
            
        # Check learning rate
        lr = optimizer.get("lr")
        if lr is not None:
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1:
                self.errors.append(f"Learning rate should be between 0 and 1, got {lr}")
                
    def _validate_fairness_config(self, config: Dict[str, Any]) -> None:
        """Validate fairness configuration section."""
        protected_attrs = config.get("protected_attributes", [])
        
        if protected_attrs:
            for i, attr in enumerate(protected_attrs):
                if "name" not in attr:
                    self.errors.append(f"Protected attribute {i} missing 'name'")
                if "type" not in attr:
                    self.warnings.append(f"Protected attribute {i} missing 'type'")
                    
        # Check fairness paradigms
        fairness_paradigms = config.get("fairness_paradigms", {})
        
        for paradigm, paradigm_config in fairness_paradigms.items():
            if paradigm_config.get("enabled", False):
                weight = paradigm_config.get("weight")
                if weight is not None and weight < 0:
                    self.errors.append(f"Fairness weight for '{paradigm}' should be non-negative")
                    
    def _validate_loss_weights(self, config: Dict[str, Any]) -> None:
        """Validate loss weights configuration."""
        loss_weights = config.get("loss_weights", config.get("training", {}).get("loss_weights", {}))
        
        for key, value in loss_weights.items():
            if not isinstance(value, (int, float)):
                self.errors.append(f"Loss weight '{key}' must be numeric, got {type(value).__name__}")
            elif value < 0:
                self.warnings.append(f"Loss weight '{key}' is negative, this may cause unexpected behavior")
                
    def _validate_consistency(self, config: Dict[str, Any]) -> None:
        """Validate consistency across configuration sections."""
        # Check encoder/decoder latent dim consistency
        encoders = config.get("encoders", {})
        decoders = config.get("decoders", {})
        latent_dim = config.get("latent_dim", config.get("model", {}).get("latent_dim"))
        
        if latent_dim:
            for enc_name, enc_config in encoders.items():
                enc_latent = enc_config.get("latent_dim")
                if enc_latent and enc_latent != latent_dim:
                    self.warnings.append(
                        f"Encoder '{enc_name}' latent_dim ({enc_latent}) differs from "
                        f"global latent_dim ({latent_dim})"
                    )
                    
        # Check fairness loss weights vs enabled paradigms
        fairness_paradigms = config.get("fairness_paradigms", {})
        loss_weights = config.get("loss_weights", config.get("training", {}).get("loss_weights", {}))
        
        for paradigm, paradigm_config in fairness_paradigms.items():
            if paradigm_config.get("enabled", False):
                weight_key = f"{paradigm}"
                if weight_key in loss_weights and loss_weights[weight_key] == 0:
                    self.warnings.append(
                        f"Fairness paradigm '{paradigm}' is enabled but loss weight is 0"
                    )
                    
    def _get_nested(self, config: Dict[str, Any], path: str) -> Any:
        """Get a nested value from config using dot notation."""
        keys = path.split(".")
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
                
        return value
    
    def get_errors(self) -> List[str]:
        """Get list of validation errors."""
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Get list of validation warnings."""
        return self.warnings.copy()
    
    def has_errors(self) -> bool:
        """Check if there are any validation errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any validation warnings."""
        return len(self.warnings) > 0


def validate_config(
    config: Dict[str, Any],
    config_type: Optional[str] = None,
    strict: bool = False
) -> tuple[bool, List[str], List[str]]:
    """
    Convenience function to validate a configuration.
    
    Args:
        config: Configuration dictionary to validate
        config_type: Type of configuration
        strict: Whether to raise on errors
        
    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    validator = ConfigValidator(strict=strict)
    is_valid = validator.validate(config, config_type)
    return is_valid, validator.get_errors(), validator.get_warnings()
