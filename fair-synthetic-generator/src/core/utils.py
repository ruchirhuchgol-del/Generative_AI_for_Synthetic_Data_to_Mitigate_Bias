"""
Utility Functions
=================

Common utility functions used throughout the framework.

This module provides:
- Random seed management
- Device handling utilities
- Parameter counting and model inspection
- Configuration loading/saving
- Model weight initialization
- Gradient manipulation
- Data conversion utilities
- Component registry system
"""

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, Generic

import numpy as np
import torch
import yaml


# ==========================================
# RANDOM SEED MANAGEMENT
# ==========================================

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: If True, sets PyTorch to deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(device: Optional[str] = None) -> torch.device:
    """
    Get the appropriate torch device.
    
    Args:
        device: Device string ('cuda', 'cpu', 'mps', or None for auto)
        
    Returns:
        torch.device object
    """
    if device is not None:
        return torch.device(device)
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_logger(
    name: str,
    level: int = logging.INFO,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name: Logger name
        level: Logging level
        format_str: Optional custom format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        
        if format_str is None:
            format_str = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
        
        formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save a configuration dictionary to YAML file.
    
    Args:
        config: Configuration dictionary
        path: Path to save config
    """
    path = Path(path)
    ensure_dir(path.parent)
    
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Override configuration (takes precedence)
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def get_activation_function(name: str) -> torch.nn.Module:
    """
    Get an activation function by name.
    
    Args:
        name: Activation function name
        
    Returns:
        Activation module
    """
    activations = {
        "relu": torch.nn.ReLU,
        "leaky_relu": torch.nn.LeakyReLU,
        "gelu": torch.nn.GELU,
        "silu": torch.nn.SiLU,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
        "softmax": torch.nn.Softmax,
        "linear": torch.nn.Identity,
        "elu": torch.nn.ELU,
        "selu": torch.nn.SELU,
        "prelu": torch.nn.PReLU,
        "mish": torch.nn.Mish,
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation function: {name}")
    
    return activations[name]()


def initialize_weights(
    module: torch.nn.Module,
    method: str = "xavier_uniform",
    gain: float = 1.0
) -> None:
    """
    Initialize module weights.
    
    Args:
        module: Module to initialize
        method: Initialization method
        gain: Gain value for certain methods
    """
    for name, param in module.named_parameters():
        if "weight" in name:
            if method == "xavier_uniform":
                torch.nn.init.xavier_uniform_(param, gain=gain)
            elif method == "xavier_normal":
                torch.nn.init.xavier_normal_(param, gain=gain)
            elif method == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
            elif method == "kaiming_normal":
                torch.nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
            elif method == "orthogonal":
                torch.nn.init.orthogonal_(param, gain=gain)
            elif method == "normal":
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
        elif "bias" in name:
            torch.nn.init.zeros_(param)


def move_to_device(
    data: Any,
    device: torch.device,
    non_blocking: bool = True
) -> Any:
    """
    Recursively move data to device.
    
    Args:
        data: Data to move (tensor, dict, list, tuple)
        device: Target device
        non_blocking: Use non-blocking transfer
        
    Returns:
        Data on target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device, non_blocking) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in data)
    return data


def freeze_module(module: torch.nn.Module) -> None:
    """Freeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: torch.nn.Module) -> None:
    """Unfreeze all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def get_gradient_norm(module: torch.nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute total gradient norm of a module.
    
    Args:
        module: Module to compute gradient norm for
        norm_type: Type of norm (L2 by default)
        
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for param in module.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    
    return total_norm ** (1.0 / norm_type)


def clip_gradients(
    module: torch.nn.Module,
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by norm.
    
    Args:
        module: Module to clip gradients for
        max_norm: Maximum gradient norm
        norm_type: Type of norm
        
    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(
        module.parameters(),
        max_norm=max_norm,
        norm_type=norm_type
    )


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(num: Union[int, float], precision: int = 2) -> str:
    """
    Format number with appropriate suffixes (K, M, B).
    
    Args:
        num: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted number string
    """
    if abs(num) >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif abs(num) >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    return f"{num:.{precision}f}"


def compute_pairwise_distances(
    x: torch.Tensor,
    metric: str = "euclidean"
) -> torch.Tensor:
    """
    Compute pairwise distances between samples.
    
    Args:
        x: Input tensor of shape (batch_size, features)
        metric: Distance metric ('euclidean', 'cosine', 'manhattan')
        
    Returns:
        Distance matrix of shape (batch_size, batch_size)
    """
    if metric == "euclidean":
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T y
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        distances = x_norm + x_norm.t() - 2 * torch.mm(x, x.t())
        distances = torch.clamp(distances, min=0.0)
        distances = torch.sqrt(distances)
    elif metric == "cosine":
        x_normalized = x / (x.norm(dim=1, keepdim=True) + 1e-8)
        similarities = torch.mm(x_normalized, x_normalized.t())
        distances = 1.0 - similarities
    elif metric == "manhattan":
        # |x - y| = sum of absolute differences
        distances = torch.cdist(x, x, p=1)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")
    
    return distances


def get_parameter_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get detailed parameter information for a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    param_sizes = {}
    for name, param in model.named_parameters():
        param_sizes[name] = {
            "shape": list(param.shape),
            "numel": param.numel(),
            "trainable": param.requires_grad,
            "dtype": str(param.dtype),
        }
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
        "num_layers": len(param_sizes),
        "param_details": param_sizes,
    }


def print_model_summary(model: torch.nn.Module, input_size: Optional[tuple] = None) -> None:
    """
    Print a summary of the model.
    
    Args:
        model: PyTorch model
        input_size: Optional input size for forward pass
    """
    info = get_parameter_info(model)
    
    print("=" * 60)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 60)
    print(f"Total parameters: {info['total_params']:,}")
    print(f"Trainable parameters: {info['trainable_params']:,}")
    print(f"Frozen parameters: {info['frozen_params']:,}")
    print("-" * 60)
    print("Layers:")
    
    for name, details in info['param_details'].items():
        shape_str = "x".join(str(s) for s in details['shape'])
        status = "trainable" if details['trainable'] else "frozen"
        print(f"  {name}: [{shape_str}] ({details['numel']:,}) [{status}]")
    
    print("=" * 60)


# ==========================================
# DATA CONVERSION UTILITIES
# ==========================================

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Numpy array
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    return tensor.cpu().numpy()


def numpy_to_tensor(
    array: np.ndarray,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.
    
    Args:
        array: Numpy array
        dtype: Optional target dtype
        device: Optional target device
        
    Returns:
        PyTorch tensor
    """
    tensor = torch.from_numpy(array)
    if dtype is not None:
        tensor = tensor.to(dtype)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def one_hot_encode(
    labels: torch.Tensor,
    num_classes: int,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    One-hot encode labels.
    
    Args:
        labels: Label tensor of shape (batch_size,)
        num_classes: Number of classes
        dtype: Output dtype
        
    Returns:
        One-hot encoded tensor of shape (batch_size, num_classes)
    """
    return torch.nn.functional.one_hot(labels, num_classes).to(dtype)


def normalize_tensor(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    L2 normalize a tensor along a dimension.
    
    Args:
        tensor: Input tensor
        dim: Dimension to normalize along
        eps: Small constant for numerical stability
        
    Returns:
        Normalized tensor
    """
    norm = tensor.norm(dim=dim, keepdim=True)
    return tensor / (norm + eps)


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Union[List[float], torch.Tensor],
    std: Union[List[float], torch.Tensor]
) -> torch.Tensor:
    """
    Denormalize a tensor using mean and std.
    
    Args:
        tensor: Normalized tensor
        mean: Mean values
        std: Standard deviation values
        
    Returns:
        Denormalized tensor
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean, device=tensor.device)
    if isinstance(std, list):
        std = torch.tensor(std, device=tensor.device)
        
    # Reshape for broadcasting
    while mean.dim() < tensor.dim():
        mean = mean.unsqueeze(-1)
    while std.dim() < tensor.dim():
        std = std.unsqueeze(-1)
        
    return tensor * std + mean


# ==========================================
# COMPONENT REGISTRY
# ==========================================

T = TypeVar('T')


class Registry(Generic[T]):
    """
    A registry for mapping names to classes or functions.
    
    Provides a way to register and retrieve components by name,
    useful for creating factories and plugin systems.
    
    Example:
        # Create a registry
        encoders = Registry["BaseEncoder"]("encoders")
        
        # Register a component
        @encoders.register("mlp")
        class MLPEncoder(BaseEncoder):
            ...
            
        # Retrieve a component
        encoder_class = encoders.get("mlp")
        encoder = encoder_class(input_dim=100, latent_dim=32)
    """
    
    def __init__(self, name: str):
        """
        Initialize the registry.
        
        Args:
            name: Name of the registry
        """
        self.name = name
        self._registry: Dict[str, Type[T]] = {}
        self._aliases: Dict[str, str] = {}
        
    def register(
        self, 
        name: str, 
        aliases: Optional[List[str]] = None
    ) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a class.
        
        Args:
            name: Name to register under
            aliases: Optional list of aliases
            
        Returns:
            Decorator function
        """
        def decorator(cls: Type[T]) -> Type[T]:
            self._registry[name] = cls
            
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name
                    
            return cls
        return decorator
    
    def register_class(self, name: str, cls: Type[T]) -> None:
        """
        Register a class directly.
        
        Args:
            name: Name to register under
            cls: Class to register
        """
        self._registry[name] = cls
        
    def get(self, name: str) -> Type[T]:
        """
        Get a registered class by name.
        
        Args:
            name: Registered name or alias
            
        Returns:
            Registered class
            
        Raises:
            KeyError: If name not found
        """
        # Check aliases first
        if name in self._aliases:
            name = self._aliases[name]
            
        if name not in self._registry:
            available = list(self._registry.keys())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
            
        return self._registry[name]
    
    def list_available(self) -> List[str]:
        """
        List all registered names.
        
        Returns:
            List of registered names
        """
        return list(self._registry.keys())
    
    def contains(self, name: str) -> bool:
        """
        Check if a name is registered.
        
        Args:
            name: Name to check
            
        Returns:
            True if registered
        """
        return name in self._registry or name in self._aliases
    
    def __contains__(self, name: str) -> bool:
        return self.contains(name)
    
    def __repr__(self) -> str:
        return f"Registry({self.name}, components={self.list_available()})"


# Global registries for common components
_ENCODERS = Registry("encoders")
_DECODERS = Registry("decoders")
_GENERATORS = Registry("generators")
_DISCRIMINATORS = Registry("discriminators")
_LOSSES = Registry("losses")
_METRICS = Registry("metrics")


def register_component(
    registry_name: str, 
    name: str,
    aliases: Optional[List[str]] = None
) -> Callable:
    """
    Decorator to register a component in a global registry.
    
    Args:
        registry_name: Name of registry ('encoders', 'decoders', etc.)
        name: Name to register under
        aliases: Optional aliases
        
    Returns:
        Decorator function
    """
    registries = {
        "encoders": _ENCODERS,
        "decoders": _DECODERS,
        "generators": _GENERATORS,
        "discriminators": _DISCRIMINATORS,
        "losses": _LOSSES,
        "metrics": _METRICS,
    }
    
    if registry_name not in registries:
        raise ValueError(f"Unknown registry: {registry_name}")
        
    return registries[registry_name].register(name, aliases)


def get_component(registry_name: str, name: str) -> Type:
    """
    Get a component from a global registry.
    
    Args:
        registry_name: Name of registry
        name: Component name
        
    Returns:
        Registered class
    """
    registries = {
        "encoders": _ENCODERS,
        "decoders": _DECODERS,
        "generators": _GENERATORS,
        "discriminators": _DISCRIMINATORS,
        "losses": _LOSSES,
        "metrics": _METRICS,
    }
    
    if registry_name not in registries:
        raise ValueError(f"Unknown registry: {registry_name}")
        
    return registries[registry_name].get(name)


def list_components(registry_name: str) -> List[str]:
    """
    List available components in a registry.
    
    Args:
        registry_name: Name of registry
        
    Returns:
        List of component names
    """
    registries = {
        "encoders": _ENCODERS,
        "decoders": _DECODERS,
        "generators": _GENERATORS,
        "discriminators": _DISCRIMINATORS,
        "losses": _LOSSES,
        "metrics": _METRICS,
    }
    
    if registry_name not in registries:
        raise ValueError(f"Unknown registry: {registry_name}")
        
    return registries[registry_name].list_available()


# ==========================================
# ADDITIONAL UTILITIES
# ==========================================

def check_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Check GPU memory usage.
    
    Args:
        device: Device to check (None for current)
        
    Returns:
        Dictionary with memory statistics (in GB)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
        
        return {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
        }
    else:
        return {
            "allocated_gb": 0.0,
            "reserved_gb": 0.0,
            "max_allocated_gb": 0.0,
        }


def clear_cuda_cache() -> None:
    """Clear CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_torch_dtype(name: str) -> torch.dtype:
    """
    Get torch dtype from string name.
    
    Args:
        name: Dtype name ('float32', 'float16', 'bfloat16', etc.)
        
    Returns:
        torch.dtype
    """
    dtypes = {
        "float32": torch.float32,
        "float": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int": torch.int32,
        "int64": torch.int64,
        "long": torch.int64,
        "int16": torch.int16,
        "short": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    
    if name not in dtypes:
        raise ValueError(f"Unknown dtype: {name}. Available: {list(dtypes.keys())}")
        
    return dtypes[name]


def estimate_model_size(model: torch.nn.Module) -> Dict[str, float]:
    """
    Estimate model size in memory.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with size estimates
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
        
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
        
    total_size = param_size + buffer_size
    
    return {
        "param_size_mb": param_size / 1e6,
        "buffer_size_mb": buffer_size / 1e6,
        "total_size_mb": total_size / 1e6,
    }
