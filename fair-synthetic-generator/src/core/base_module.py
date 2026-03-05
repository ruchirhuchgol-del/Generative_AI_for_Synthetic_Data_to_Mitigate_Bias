"""
Base Module Classes
===================

Abstract base classes for all components in the Fair Synthetic Data Generator.
These classes define the interface that all implementations must follow.

The module provides:
- BaseModule: Foundation class for all neural network modules
- BaseEncoder/BaseDecoder: Encoder/decoder interfaces
- BaseGenerator/BaseDiscriminator: Generative model interfaces
- BaseLoss/BaseMetric: Loss and metric interfaces
- BaseTrainer/BaseCallback: Training infrastructure
- Mixin classes for common functionality (Serializable, Configurable, Fairness)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Generic
import copy
import json
import pickle
from pathlib import Path

import torch
import torch.nn as nn


T = TypeVar('T')


class SerializableMixin:
    """
    Mixin class providing serialization capabilities.
    
    Provides methods for saving and loading object state to/from disk.
    Supports multiple serialization formats: pickle, json, torch.
    """
    
    def save_state(self, path: Union[str, Path], format: str = "torch") -> None:
        """
        Save object state to file.
        
        Args:
            path: Output file path
            format: Serialization format ('torch', 'pickle', 'json')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = self._get_serializable_state()
        
        if format == "torch":
            torch.save(state, path)
        elif format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(state, f)
        elif format == "json":
            with open(path, "w") as f:
                json.dump(state, f, indent=2, default=str)
        else:
            raise ValueError(f"Unknown format: {format}")
            
    def load_state(self, path: Union[str, Path], format: str = "torch") -> None:
        """
        Load object state from file.
        
        Args:
            path: Input file path
            format: Serialization format ('torch', 'pickle', 'json')
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")
            
        if format == "torch":
            state = torch.load(path)
        elif format == "pickle":
            with open(path, "rb") as f:
                state = pickle.load(f)
        elif format == "json":
            with open(path, "r") as f:
                state = json.load(f)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        self._set_serializable_state(state)
        
    def _get_serializable_state(self) -> Dict[str, Any]:
        """Get state dictionary for serialization."""
        state = {}
        
        # Get config if available
        if hasattr(self, "config"):
            state["config"] = self.config
            
        # Get name if available
        if hasattr(self, "name"):
            state["name"] = self.name
            
        # Get module state dict if it's an nn.Module
        if isinstance(self, nn.Module):
            state["state_dict"] = self.state_dict()
            
        return state
    
    def _set_serializable_state(self, state: Dict[str, Any]) -> None:
        """Set state from dictionary."""
        if "config" in state and hasattr(self, "config"):
            self.config = state["config"]
            
        if "name" in state and hasattr(self, "name"):
            self.name = state["name"]
            
        if "state_dict" in state and isinstance(self, nn.Module):
            self.load_state_dict(state["state_dict"])


class ConfigurableMixin:
    """
    Mixin class providing configuration management.
    
    Allows objects to be configured from dictionaries, files, or keyword arguments.
    Supports configuration validation and default values.
    """
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConfigurableMixin":
        """
        Create instance from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured instance
        """
        return cls(**config)
    
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ConfigurableMixin":
        """
        Create instance from configuration file.
        
        Args:
            path: Path to config file (YAML or JSON)
            
        Returns:
            Configured instance
        """
        path = Path(path)
        
        if path.suffix in [".yaml", ".yml"]:
            import yaml
            with open(path) as f:
                config = yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path) as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
            
        return cls.from_config(config)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Configuration dictionary
        """
        if hasattr(self, "config"):
            return copy.deepcopy(self.config)
        return {}
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
        """
        if hasattr(self, "config"):
            self.config = self._deep_merge(self.config, updates)
            
    def _deep_merge(
        self, 
        base: Dict[str, Any], 
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if (
                key in result 
                and isinstance(result[key], dict) 
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
                
        return result


class FairnessMixin:
    """
    Mixin class providing fairness-related functionality.
    
    Adds support for:
    - Sensitive attribute tracking
    - Fairness constraint configuration
    - Protected attribute handling
    """
    
    def set_sensitive_attributes(self, attributes: List[str]) -> None:
        """
        Set the list of sensitive attributes.
        
        Args:
            attributes: List of sensitive attribute names
        """
        if hasattr(self, "config"):
            self.config["sensitive_attributes"] = attributes
        self._sensitive_attributes = attributes
        
    def get_sensitive_attributes(self) -> List[str]:
        """
        Get the list of sensitive attributes.
        
        Returns:
            List of sensitive attribute names
        """
        if hasattr(self, "_sensitive_attributes"):
            return self._sensitive_attributes
        if hasattr(self, "config") and "sensitive_attributes" in self.config:
            return self.config["sensitive_attributes"]
        return []
    
    def set_fairness_weights(self, weights: Dict[str, float]) -> None:
        """
        Set fairness constraint weights.
        
        Args:
            weights: Dictionary mapping fairness metric names to weights
        """
        if hasattr(self, "config"):
            self.config["fairness_weights"] = weights
        self._fairness_weights = weights
        
    def get_fairness_weights(self) -> Dict[str, float]:
        """
        Get fairness constraint weights.
        
        Returns:
            Dictionary of fairness weights
        """
        if hasattr(self, "_fairness_weights"):
            return self._fairness_weights
        if hasattr(self, "config") and "fairness_weights" in self.config:
            return self.config["fairness_weights"]
        return {}
    
    def is_privileged(self, attribute: str, value: Any) -> bool:
        """
        Check if a value is the privileged value for an attribute.
        
        Args:
            attribute: Attribute name
            value: Value to check
            
        Returns:
            True if value is privileged
        """
        if hasattr(self, "config") and "privileged_values" in self.config:
            privileged = self.config["privileged_values"].get(attribute)
            return value == privileged
        return False


class BaseModule(nn.Module, ABC, SerializableMixin, ConfigurableMixin, FairnessMixin):
    """
    Base class for all neural network modules in the framework.
    
    Provides common functionality:
    - Parameter initialization
    - Device management
    - Checkpoint save/load
    - Parameter counting
    
    Attributes:
        name (str): Name of the module
        config (Dict[str, Any]): Configuration dictionary
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base module.
        
        Args:
            name: Name identifier for the module
            config: Optional configuration dictionary
        """
        super().__init__()
        self.name = name
        self.config = config or {}
        self._device = None
        
    @property
    def device(self) -> torch.device:
        """Get the device of the module's parameters."""
        if self._device is None:
            try:
                param = next(self.parameters())
                self._device = param.device
            except StopIteration:
                self._device = torch.device("cpu")
        return self._device
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count the number of parameters in the module.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_parameter_groups(
        self, lr_mult: float = 1.0, weight_decay_mult: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Get parameter groups for optimizer with custom learning rates.
        
        Args:
            lr_mult: Learning rate multiplier for this module
            weight_decay_mult: Weight decay multiplier for this module
            
        Returns:
            List of parameter group dictionaries
        """
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
        params_with_decay = []
        params_without_decay = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in no_decay):
                params_without_decay.append(param)
            else:
                params_with_decay.append(param)
        
        return [
            {
                "params": params_with_decay,
                "lr_mult": lr_mult,
                "weight_decay_mult": weight_decay_mult,
            },
            {
                "params": params_without_decay,
                "lr_mult": lr_mult,
                "weight_decay_mult": 0.0,
            },
        ]
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """
        Save module checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional items to save (optimizer, scheduler, etc.)
        """
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.config,
            "name": self.name,
        }
        checkpoint.update(kwargs)
        torch.save(checkpoint, path)
    
    def load_checkpoint(
        self, 
        path: str, 
        strict: bool = True,
        map_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load module checkpoint.
        
        Args:
            path: Path to checkpoint file
            strict: Whether to strictly enforce state_dict matching
            map_location: Device to map tensors to
            
        Returns:
            Loaded checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint["state_dict"], strict=strict)
        return checkpoint
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass - must be implemented by subclasses."""
        pass
    
    def __repr__(self) -> str:
        param_count = self.count_parameters()
        return f"{self.__class__.__name__}(name={self.name}, params={param_count:,})"


class BaseEncoder(BaseModule):
    """
    Base class for encoder modules.
    
    Encoders transform input data into latent representations.
    Each modality (tabular, text, image) has its own encoder implementation.
    
    Methods:
        encode: Transform input to latent representation
        get_latent_dim: Return dimension of latent space
    """
    
    @abstractmethod
    def encode(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Encode input tensor to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, *)
            
        Returns:
            Latent tensor of shape (batch_size, latent_dim)
        """
        pass
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass calls encode."""
        return self.encode(x, **kwargs)
    
    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        pass
    
    @property
    @abstractmethod
    def input_dim(self) -> Union[int, Tuple[int, ...]]:
        """Return the dimension of the input."""
        pass


class BaseDecoder(BaseModule):
    """
    Base class for decoder modules.
    
    Decoders transform latent representations back to the original data space.
    Each modality has its own decoder implementation.
    
    Methods:
        decode: Transform latent representation to output
        get_output_dim: Return dimension of output space
    """
    
    @abstractmethod
    def decode(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decode latent tensor to output space.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Output tensor of shape (batch_size, *)
        """
        pass
    
    def forward(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass calls decode."""
        return self.decode(z, **kwargs)
    
    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> Union[int, Tuple[int, ...]]:
        """Return the dimension of the output."""
        pass


class BaseGenerator(BaseModule):
    """
    Base class for generative models.
    
    Generators create synthetic data samples. They combine encoders and decoders
    with various generative mechanisms (VAE, GAN, Diffusion).
    
    Methods:
        generate: Generate synthetic samples
        sample_latent: Sample from latent prior
        reconstruct: Reconstruct input through encode-decode
    """
    
    @abstractmethod
    def generate(
        self, 
        n_samples: int, 
        conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate synthetic samples.
        
        Args:
            n_samples: Number of samples to generate
            conditions: Optional conditioning dictionary
            **kwargs: Additional generation parameters
            
        Returns:
            Generated samples tensor
        """
        pass
    
    @abstractmethod
    def sample_latent(self, n_samples: int, **kwargs) -> torch.Tensor:
        """
        Sample from the latent prior distribution.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Latent samples tensor of shape (n_samples, latent_dim)
        """
        pass
    
    def reconstruct(
        self, 
        x: torch.Tensor, 
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct input through encode-decode cycle.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed tensor, latent representation)
        """
        z = self.encode(x, **kwargs)
        x_recon = self.decode(z, **kwargs)
        return x_recon, z
    
    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Return the dimension of the latent space."""
        pass


class BaseDiscriminator(BaseModule):
    """
    Base class for discriminator modules.
    
    Discriminators are used in adversarial training (GANs) and
    fairness adversarial debiasing.
    
    Methods:
        discriminate: Compute discriminator output
        get_probability: Get probability output (for binary classification)
    """
    
    @abstractmethod
    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute discriminator output.
        
        Args:
            x: Input tensor
            
        Returns:
            Discriminator output (logits or features)
        """
        pass
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Forward pass calls discriminate."""
        return self.discriminate(x, **kwargs)
    
    def get_probability(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Get probability output using sigmoid or softmax.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor
        """
        logits = self.discriminate(x, **kwargs)
        if logits.shape[-1] == 1:
            return torch.sigmoid(logits)
        else:
            return torch.softmax(logits, dim=-1)


class BaseLoss(nn.Module, ABC):
    """
    Base class for loss functions.
    
    All custom loss functions should inherit from this class.
    Provides common functionality for loss tracking and weighting.
    
    Attributes:
        weight (float): Weight for this loss in total loss computation
        reduction (str): Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, weight: float = 1.0, reduction: str = "mean"):
        """
        Initialize the loss.
        
        Args:
            weight: Weight for this loss
            reduction: Reduction method
        """
        super().__init__()
        self.weight = weight
        self.reduction = reduction
    
    @abstractmethod
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        """
        Compute the raw loss value.
        
        Returns:
            Loss tensor
        """
        pass
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass computes weighted loss.
        
        Returns:
            Weighted loss tensor
        """
        loss = self.compute_loss(*args, **kwargs)
        return self.weight * loss
    
    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class BaseMetric(ABC):
    """
    Base class for evaluation metrics.
    
    All metrics should inherit from this class and implement
    the compute method.
    
    Attributes:
        name (str): Name of the metric
        higher_is_better (bool): Whether higher values are better
    """
    
    def __init__(self, name: str, higher_is_better: bool = True):
        """
        Initialize the metric.
        
        Args:
            name: Name of the metric
            higher_is_better: Whether higher values indicate better performance
        """
        self.name = name
        self.higher_is_better = higher_is_better
    
    @abstractmethod
    def compute(
        self, 
        predictions: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        groups: Optional[torch.Tensor] = None,
        **kwargs
    ) -> float:
        """
        Compute the metric value.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets (optional for some metrics)
            groups: Group membership tensor (for fairness metrics)
            **kwargs: Additional arguments
            
        Returns:
            Metric value as float
        """
        pass
    
    def __call__(self, *args, **kwargs) -> float:
        """Call compute method."""
        return self.compute(*args, **kwargs)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, higher_is_better={self.higher_is_better})"


class BaseTrainer(ABC):
    """
    Base class for training loops.
    
    Provides common functionality for:
    - Training loop management
    - Validation
    - Checkpointing
    - Logging
    - Early stopping
    
    Attributes:
        model: The model to train
        optimizer: The optimizer
        scheduler: Learning rate scheduler
        device: Training device
    """
    
    def __init__(
        self,
        model: BaseModule,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            optimizer: Optimizer for training
            scheduler: Optional learning rate scheduler
            device: Device to train on
            config: Training configuration dictionary
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf") if config.get("mode", "min") == "min" else float("-inf")
        self.callbacks: List[BaseCallback] = []
    
    def add_callback(self, callback: "BaseCallback") -> None:
        """Add a callback to the training loop."""
        self.callbacks.append(callback)
    
    def _trigger_callbacks(self, event: str, **kwargs) -> None:
        """Trigger callbacks for a specific event."""
        for callback in self.callbacks:
            method = getattr(callback, f"on_{event}", None)
            if method is not None:
                method(trainer=self, **kwargs)
    
    @abstractmethod
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Run validation.
        
        Args:
            dataloader: Validation dataloader
            
        Returns:
            Dictionary of validation metrics
        """
        pass
    
    def fit(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        n_epochs: int = 100,
    ) -> Dict[str, List[float]]:
        """
        Full training loop.
        
        Args:
            train_dataloader: Training dataloader
            val_dataloader: Optional validation dataloader
            n_epochs: Number of epochs to train
            
        Returns:
            Training history dictionary
        """
        history = {"train_loss": [], "val_loss": [], "metrics": []}
        
        self._trigger_callbacks("train_begin")
        
        for epoch in range(n_epochs):
            self.current_epoch = epoch
            self._trigger_callbacks("epoch_begin", epoch=epoch)
            
            # Train
            train_metrics = self.train_epoch(train_dataloader)
            history["train_loss"].append(train_metrics.get("loss", 0))
            
            # Validate
            if val_dataloader is not None:
                val_metrics = self.validate(val_dataloader)
                history["val_loss"].append(val_metrics.get("loss", 0))
                
                # Check for improvement
                monitor_metric = self.config.get("monitor", "val_loss")
                current = val_metrics.get(monitor_metric, float("inf"))
                mode = self.config.get("mode", "min")
                
                if self._is_improvement(current, mode):
                    self.best_metric = current
                    self._trigger_callbacks("best_model", metrics=val_metrics)
            
            self._trigger_callbacks("epoch_end", epoch=epoch, metrics=train_metrics)
        
        self._trigger_callbacks("train_end")
        return history
    
    def _is_improvement(self, current: float, mode: str) -> bool:
        """Check if current metric is an improvement."""
        if mode == "min":
            return current < self.best_metric
        return current > self.best_metric


class BaseCallback(ABC):
    """
    Base class for training callbacks.
    
    Callbacks can hook into various points in the training loop
    to execute custom logic (logging, checkpointing, etc.)
    """
    
    def on_train_begin(self, trainer: BaseTrainer) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer: BaseTrainer) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: BaseTrainer, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: BaseTrainer, epoch: int, metrics: Dict) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_begin(self, trainer: BaseTrainer, batch: Any) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, trainer: BaseTrainer, batch: Any, metrics: Dict) -> None:
        """Called at the end of each batch."""
        pass
    
    def on_best_model(self, trainer: BaseTrainer, metrics: Dict) -> None:
        """Called when a new best model is found."""
        pass
