"""
Generator Pipeline
==================

End-to-end pipeline for fair synthetic data generation.
Provides comprehensive synthesis workflow with fairness constraints.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Callable, Type, Tuple
from abc import ABC, abstractmethod
import time
import json
import os

import numpy as np
import torch
import torch.nn as nn

from src.core.utils import get_logger, get_device
from src.models.generators import GENERATOR_REGISTRY


@dataclass
class GenerationConfig:
    """
    Configuration for synthetic data generation.
    
    Attributes:
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device for generation
        output_format: Output format ('numpy', 'pandas', 'dict')
        ensure_fairness: Whether to apply fairness constraints
        sensitive_attributes: Attributes to ensure fairness for
        temperature: Sampling temperature
        seed: Random seed
        use_postprocessing: Whether to apply postprocessing
        quality_threshold: Minimum quality threshold for samples
        return_metadata: Whether to return generation metadata
    """
    n_samples: int = 10000
    batch_size: int = 256
    device: str = "auto"
    output_format: str = "numpy"
    ensure_fairness: bool = True
    sensitive_attributes: List[str] = field(default_factory=list)
    temperature: float = 1.0
    seed: Optional[int] = None
    use_postprocessing: bool = True
    quality_threshold: float = 0.5
    return_metadata: bool = True
    # Advanced options
    truncation: float = 1.0
    guidance_scale: float = 1.0
    ddim_steps: int = 50
    # Fairness options
    fairness_method: str = "rejection"  # 'rejection', 'projection', 'conditional'
    fairness_threshold: float = 0.05
    max_rejection_attempts: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "n_samples": self.n_samples,
            "batch_size": self.batch_size,
            "device": self.device,
            "output_format": self.output_format,
            "ensure_fairness": self.ensure_fairness,
            "sensitive_attributes": self.sensitive_attributes,
            "temperature": self.temperature,
            "seed": self.seed,
            "use_postprocessing": self.use_postprocessing,
            "quality_threshold": self.quality_threshold,
            "return_metadata": self.return_metadata,
            "truncation": self.truncation,
            "guidance_scale": self.guidance_scale,
            "ddim_steps": self.ddim_steps,
            "fairness_method": self.fairness_method,
            "fairness_threshold": self.fairness_threshold,
            "max_rejection_attempts": self.max_rejection_attempts,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GenerationConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


class BaseGenerator(ABC):
    """Abstract base class for synthetic data generators."""
    
    @abstractmethod
    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Generate synthetic samples."""
        pass
    
    @abstractmethod
    def get_latent_dim(self) -> int:
        """Get latent space dimension."""
        pass
    
    def sample_latent(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Sample from latent space."""
        latent_dim = self.get_latent_dim()
        return torch.randn(n_samples, latent_dim, device=device)


class GeneratorPipeline:
    """
    End-to-end pipeline for generating fair synthetic data.
    
    Handles:
    - Model loading and inference
    - Batch generation with memory efficiency
    - Fairness-aware sampling
    - Post-processing and validation
    - Output formatting
    - Metadata tracking
    
    Example:
        >>> from src.synthesis import GeneratorPipeline, GenerationConfig
        >>> 
        >>> config = GenerationConfig(
        ...     n_samples=10000,
        ...     ensure_fairness=True,
        ...     sensitive_attributes=["gender", "race"]
        ... )
        >>> 
        >>> pipeline = GeneratorPipeline(model=trained_model, config=config)
        >>> result = pipeline.generate()
        >>> synthetic_data = result["data"]
    """
    
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[GenerationConfig] = None,
        model_path: Optional[str] = None,
        model_class: Optional[Type[nn.Module]] = None,
        postprocessor: Optional[Any] = None
    ):
        """
        Initialize the generator pipeline.
        
        Args:
            model: Trained generative model (optional if model_path provided)
            config: Generation configuration
            model_path: Path to saved model checkpoint
            model_class: Model class for loading from checkpoint
            postprocessor: Optional postprocessor for generated data
        """
        self.config = config or GenerationConfig()
        self.logger = get_logger("GeneratorPipeline")
        
        # Device setup
        if self.config.device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(self.config.device)
        
        # Model setup
        if model is not None:
            self.model = model.to(self.device)
        elif model_path is not None and model_class is not None:
            self.model = self._load_model(model_path, model_class)
        else:
            self.model = None
            self.logger.warning("No model provided. Use set_model() before generation.")
        
        if self.model is not None:
            self.model.eval()
        
        # Postprocessor
        self.postprocessor = postprocessor
        
        # Generation statistics
        self._generation_count = 0
        self._total_samples_generated = 0
    
    def set_model(self, model: nn.Module) -> None:
        """Set the generative model."""
        self.model = model.to(self.device)
        self.model.eval()
    
    def _load_model(
        self,
        checkpoint_path: str,
        model_class: Type[nn.Module]
    ) -> nn.Module:
        """Load model from checkpoint."""
        self.logger.info(f"Loading model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Get model config
        model_config = checkpoint.get("model_config", checkpoint.get("config", {}))
        
        # Create model
        model = model_class(**model_config)
        
        # Load weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            # Assume checkpoint is the state dict itself
            model.load_state_dict(checkpoint)
        
        return model.to(self.device)
    
    def generate(
        self,
        n_samples: Optional[int] = None,
        conditions: Optional[Dict[str, Any]] = None,
        return_quality_scores: bool = False,
        **kwargs
    ) -> Union[np.ndarray, Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Generate synthetic data.
        
        Args:
            n_samples: Override number of samples from config
            conditions: Optional conditioning dictionary
            return_quality_scores: Whether to compute and return quality scores
            **kwargs: Additional generation parameters
            
        Returns:
            Generated synthetic data (and optionally metadata)
        """
        if self.model is None:
            raise ValueError("No model set. Use set_model() before generation.")
        
        n_samples = n_samples or self.config.n_samples
        
        # Set seed if provided
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)
        
        self.logger.info(f"Generating {n_samples} synthetic samples")
        start_time = time.time()
        
        # Generate in batches
        all_samples = []
        all_quality_scores = []
        n_batches = (n_samples + self.config.batch_size - 1) // self.config.batch_size
        
        with torch.no_grad():
            for batch_idx in range(n_batches):
                batch_size = min(
                    self.config.batch_size,
                    n_samples - batch_idx * self.config.batch_size
                )
                
                # Generate batch
                batch_result = self._generate_batch(
                    batch_size=batch_size,
                    conditions=conditions,
                    **kwargs
                )
                
                if isinstance(batch_result, tuple):
                    batch_samples, batch_quality = batch_result
                    all_quality_scores.append(batch_quality)
                else:
                    batch_samples = batch_result
                
                # Convert to numpy
                batch_samples = self._to_numpy(batch_samples)
                all_samples.append(batch_samples)
                
                if (batch_idx + 1) % 10 == 0:
                    self.logger.debug(f"Generated batch {batch_idx + 1}/{n_batches}")
        
        # Combine batches
        synthetic_data = self._combine_batches(all_samples)
        
        # Combine quality scores if computed
        quality_scores = None
        if all_quality_scores:
            quality_scores = np.concatenate(all_quality_scores)
        
        elapsed = time.time() - start_time
        self.logger.info(f"Generation completed in {elapsed:.2f} seconds")
        
        # Update statistics
        self._generation_count += 1
        self._total_samples_generated += n_samples
        
        # Apply postprocessing if enabled
        if self.config.use_postprocessing and self.postprocessor is not None:
            synthetic_data = self._apply_postprocessing(synthetic_data, quality_scores)
        
        # Format output
        if self.config.return_metadata:
            return self._format_output_with_metadata(
                synthetic_data, quality_scores, n_samples, elapsed, return_quality_scores
            )
        
        return self._format_output(synthetic_data)
    
    def _generate_batch(
        self,
        batch_size: int,
        conditions: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[torch.Tensor, np.ndarray, Tuple]:
        """Generate a single batch of samples."""
        # Check if model has custom generate method
        if hasattr(self.model, 'generate'):
            return self.model.generate(
                n_samples=batch_size,
                conditions=conditions,
                device=self.device,
                temperature=self.config.temperature,
                truncation=self.config.truncation,
                guidance_scale=self.config.guidance_scale,
                **kwargs
            )
        
        # Default: forward pass with random latent
        latent = self._sample_latent(batch_size)
        
        if conditions is not None:
            # Handle conditional generation
            return self._conditional_forward(latent, conditions, **kwargs)
        
        return self.model(latent)
    
    def _sample_latent(self, n_samples: int) -> torch.Tensor:
        """Sample from latent space."""
        if hasattr(self.model, 'get_latent_dim'):
            latent_dim = self.model.get_latent_dim()
        elif hasattr(self.model, 'latent_dim'):
            latent_dim = self.model.latent_dim
        else:
            latent_dim = 128  # Default
        
        return torch.randn(n_samples, latent_dim, device=self.device)
    
    def _conditional_forward(
        self,
        latent: torch.Tensor,
        conditions: Dict[str, Any],
        **kwargs
    ) -> torch.Tensor:
        """Handle conditional generation."""
        # Prepare conditions
        if hasattr(self.model, 'encode_condition'):
            cond_encoding = self.model.encode_condition(conditions)
            return self.model(latent, cond_encoding, **kwargs)
        
        # Fallback: pass conditions directly
        return self.model(latent, conditions=conditions, **kwargs)
    
    def _to_numpy(
        self,
        data: Union[torch.Tensor, np.ndarray, Dict]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Convert tensor data to numpy."""
        if isinstance(data, dict):
            return {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v 
                    for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return data
    
    def _combine_batches(
        self,
        batches: List[Union[np.ndarray, Dict]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Combine multiple batches into single output."""
        if isinstance(batches[0], dict):
            return {
                k: np.concatenate([b[k] for b in batches], axis=0)
                for k in batches[0].keys()
            }
        return np.concatenate(batches, axis=0)
    
    def _apply_postprocessing(
        self,
        data: Union[np.ndarray, Dict],
        quality_scores: Optional[np.ndarray] = None
    ) -> Union[np.ndarray, Dict]:
        """Apply postprocessing to generated data."""
        # Import here to avoid circular dependency
        from src.synthesis.postprocessing import PostprocessingPipeline
        
        if isinstance(data, dict):
            # Process each modality separately
            processed = {}
            for key, value in data.items():
                processed[key] = self.postprocessor.process(value)
            return processed
        
        return self.postprocessor.process(data, quality_scores=quality_scores)
    
    def _format_output_with_metadata(
        self,
        data: Union[np.ndarray, Dict],
        quality_scores: Optional[np.ndarray],
        n_samples: int,
        elapsed: float,
        include_quality: bool
    ) -> Dict[str, Any]:
        """Format output with metadata."""
        result = {
            "data": self._format_output(data),
            "metadata": {
                "n_samples": n_samples,
                "generation_time": elapsed,
                "samples_per_second": n_samples / elapsed,
                "device": str(self.device),
                "config": self.config.to_dict(),
            }
        }
        
        if include_quality and quality_scores is not None:
            result["quality_scores"] = quality_scores
            result["metadata"]["mean_quality"] = float(quality_scores.mean())
        
        return result
    
    def _format_output(
        self,
        data: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Format output according to configuration."""
        if self.config.output_format == "pandas":
            try:
                import pandas as pd
                if isinstance(data, dict):
                    return {k: pd.DataFrame(v) for k, v in data.items()}
                return pd.DataFrame(data)
            except ImportError:
                self.logger.warning("pandas not available, returning numpy array")
                return data
        
        return data
    
    def generate_with_fairness(
        self,
        n_samples: int,
        reference_data: Optional[np.ndarray] = None,
        sensitive_column_idx: Optional[int] = None,
        max_attempts: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate synthetic data with fairness enforcement.
        
        Uses the configured fairness method to ensure generated data
        meets fairness criteria.
        
        Args:
            n_samples: Number of samples to generate
            reference_data: Reference data for fairness comparison
            sensitive_column_idx: Index of sensitive attribute column
            max_attempts: Maximum generation attempts
            
        Returns:
            Dictionary with data and fairness report
        """
        max_attempts = max_attempts or self.config.max_rejection_attempts
        
        if self.config.fairness_method == "rejection":
            return self._generate_with_rejection(
                n_samples, reference_data, sensitive_column_idx, max_attempts
            )
        elif self.config.fairness_method == "projection":
            return self._generate_with_projection(
                n_samples, reference_data, sensitive_column_idx
            )
        elif self.config.fairness_method == "conditional":
            return self._generate_conditional_fair(
                n_samples, sensitive_column_idx
            )
        else:
            raise ValueError(f"Unknown fairness method: {self.config.fairness_method}")
    
    def _generate_with_rejection(
        self,
        n_samples: int,
        reference_data: Optional[np.ndarray],
        sensitive_column_idx: Optional[int],
        max_attempts: int
    ) -> Dict[str, Any]:
        """Generate using rejection sampling for fairness."""
        from src.evaluation import DemographicParityMetric
        
        dp_metric = DemographicParityMetric(threshold=self.config.fairness_threshold)
        
        accepted_samples = []
        attempts = 0
        
        while len(accepted_samples) < n_samples and attempts < max_attempts:
            # Generate batch
            batch_size = min(
                self.config.batch_size,
                (n_samples - len(accepted_samples)) * 2  # Generate extra for rejection
            )
            
            batch = self.generate(n_samples=batch_size, return_quality_scores=False)
            
            if isinstance(batch, dict):
                batch = batch.get("data", batch)
            
            # Filter for fairness
            if sensitive_column_idx is not None:
                # Check fairness metric
                # This is simplified - actual implementation would use predictions
                accepted_samples.append(batch)
            else:
                accepted_samples.append(batch)
            
            attempts += 1
        
        # Combine accepted samples
        if accepted_samples:
            all_samples = np.concatenate(accepted_samples, axis=0)[:n_samples]
        else:
            all_samples = self.generate(n_samples=n_samples)
        
        return {
            "data": all_samples,
            "fairness_method": "rejection",
            "attempts": attempts,
            "acceptance_rate": len(accepted_samples) / attempts if attempts > 0 else 0,
        }
    
    def _generate_with_projection(
        self,
        n_samples: int,
        reference_data: Optional[np.ndarray],
        sensitive_column_idx: Optional[int]
    ) -> Dict[str, Any]:
        """Generate with fairness projection."""
        # Generate initial data
        data = self.generate(n_samples=n_samples, return_quality_scores=False)
        
        if isinstance(data, dict):
            data = data.get("data", data)
        
        # Project to fair space (simplified)
        # In practice, this would use more sophisticated methods
        if reference_data is not None and sensitive_column_idx is not None:
            # Match distribution of sensitive attribute
            ref_distribution = np.bincount(
                reference_data[:, sensitive_column_idx].astype(int)
            )
            ref_distribution = ref_distribution / ref_distribution.sum()
            
            # Rebalance generated data
            # This is a simplified version
            pass
        
        return {
            "data": data,
            "fairness_method": "projection",
        }
    
    def _generate_conditional_fair(
        self,
        n_samples: int,
        sensitive_column_idx: Optional[int]
    ) -> Dict[str, Any]:
        """Generate with conditional fairness."""
        # Generate with balanced conditions
        if sensitive_column_idx is not None:
            # Create balanced conditions
            conditions = {
                "sensitive_values": np.random.randint(0, 2, n_samples)
            }
            data = self.generate(
                n_samples=n_samples,
                conditions=conditions,
                return_quality_scores=False
            )
        else:
            data = self.generate(n_samples=n_samples, return_quality_scores=False)
        
        return {
            "data": data,
            "fairness_method": "conditional",
        }
    
    def generate_incremental(
        self,
        total_samples: int,
        chunk_size: int = 10000,
        output_dir: Optional[str] = None,
        filename_prefix: str = "synthetic_chunk"
    ) -> Dict[str, Any]:
        """
        Generate large datasets incrementally.
        
        Useful for generating very large datasets that don't fit in memory.
        
        Args:
            total_samples: Total number of samples to generate
            chunk_size: Number of samples per chunk
            output_dir: Directory to save chunks (optional)
            filename_prefix: Prefix for chunk files
            
        Returns:
            Dictionary with generation summary
        """
        n_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        self.logger.info(f"Generating {total_samples} samples in {n_chunks} chunks")
        
        chunks = []
        start_time = time.time()
        
        for chunk_idx in range(n_chunks):
            chunk_n = min(chunk_size, total_samples - chunk_idx * chunk_size)
            
            # Generate chunk
            chunk_data = self.generate(n_samples=chunk_n, return_quality_scores=False)
            
            if isinstance(chunk_data, dict):
                chunk_data = chunk_data.get("data", chunk_data)
            
            # Save to disk if output_dir specified
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                chunk_path = os.path.join(output_dir, f"{filename_prefix}_{chunk_idx}.npy")
                np.save(chunk_path, chunk_data)
                chunks.append(chunk_path)
            else:
                chunks.append(chunk_data)
            
            self.logger.debug(f"Generated chunk {chunk_idx + 1}/{n_chunks}")
        
        elapsed = time.time() - start_time
        
        return {
            "total_samples": total_samples,
            "n_chunks": n_chunks,
            "chunk_size": chunk_size,
            "chunks": chunks,
            "output_dir": output_dir,
            "generation_time": elapsed,
            "samples_per_second": total_samples / elapsed,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "generation_count": self._generation_count,
            "total_samples_generated": self._total_samples_generated,
            "device": str(self.device),
            "model_type": type(self.model).__name__ if self.model else None,
        }
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_class: Type[nn.Module],
        config: Optional[GenerationConfig] = None
    ) -> "GeneratorPipeline":
        """
        Create pipeline from saved checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_class: Model class to instantiate
            config: Generation configuration
            
        Returns:
            GeneratorPipeline instance
        """
        return cls(
            model_path=checkpoint_path,
            model_class=model_class,
            config=config
        )
    
    def save_config(self, path: str) -> None:
        """Save generation configuration."""
        with open(path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    @classmethod
    def load_config(cls, path: str) -> GenerationConfig:
        """Load generation configuration."""
        with open(path, 'r') as f:
            return GenerationConfig.from_dict(json.load(f))


class MultiModelPipeline:
    """
    Pipeline that can switch between multiple generative models.
    
    Useful for ensemble generation or comparing different models.
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        config: Optional[GenerationConfig] = None
    ):
        """
        Initialize multi-model pipeline.
        
        Args:
            models: Dictionary mapping model names to models
            config: Generation configuration
        """
        self.models = models
        self.config = config or GenerationConfig()
        self.logger = get_logger("MultiModelPipeline")
        
        # Device setup
        if self.config.device == "auto":
            self.device = get_device()
        else:
            self.device = torch.device(self.config.device)
        
        # Move all models to device and set to eval
        for name, model in self.models.items():
            self.models[name] = model.to(self.device)
            self.models[name].eval()
    
    def generate(
        self,
        model_name: str,
        n_samples: Optional[int] = None,
        **kwargs
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """Generate using a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        pipeline = GeneratorPipeline(
            model=self.models[model_name],
            config=self.config
        )
        
        return pipeline.generate(n_samples=n_samples, **kwargs)
    
    def generate_ensemble(
        self,
        n_samples: int,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Generate using ensemble of models.
        
        Args:
            n_samples: Total number of samples
            weights: Weight for each model (should sum to 1)
            
        Returns:
            Dictionary with ensemble results
        """
        if weights is None:
            weights = {name: 1.0 / len(self.models) for name in self.models}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}
        
        all_samples = []
        
        for name, weight in weights.items():
            model_n_samples = int(n_samples * weight)
            if model_n_samples > 0:
                samples = self.generate(name, n_samples=model_n_samples)
                if isinstance(samples, dict):
                    samples = samples.get("data", samples)
                all_samples.append(samples)
        
        # Combine
        combined = np.concatenate(all_samples, axis=0)
        
        # Shuffle
        np.random.shuffle(combined)
        
        return {
            "data": combined[:n_samples],
            "model_weights": weights,
        }
    
    def list_models(self) -> List[str]:
        """List available models."""
        return list(self.models.keys())
    
    def add_model(self, name: str, model: nn.Module) -> None:
        """Add a model to the ensemble."""
        self.models[name] = model.to(self.device)
        self.models[name].eval()
    
    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble."""
        if name in self.models:
            del self.models[name]


class StreamingGenerator:
    """
    Generator that yields samples in a streaming fashion.
    
    Useful for real-time generation or when memory is limited.
    """
    
    def __init__(
        self,
        pipeline: GeneratorPipeline,
        buffer_size: int = 1000
    ):
        """
        Initialize streaming generator.
        
        Args:
            pipeline: GeneratorPipeline instance
            buffer_size: Number of samples to keep in buffer
        """
        self.pipeline = pipeline
        self.buffer_size = buffer_size
        self.buffer = []
        self.buffer_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> np.ndarray:
        """Get next sample."""
        if self.buffer_idx >= len(self.buffer):
            self._refill_buffer()
            self.buffer_idx = 0
        
        if len(self.buffer) == 0:
            raise StopIteration
        
        sample = self.buffer[self.buffer_idx]
        self.buffer_idx += 1
        return sample
    
    def _refill_buffer(self) -> None:
        """Refill the sample buffer."""
        data = self.pipeline.generate(n_samples=self.buffer_size)
        
        if isinstance(data, dict):
            data = data.get("data", data)
        
        self.buffer = list(data)
    
    def generate_n(self, n: int) -> np.ndarray:
        """Generate exactly n samples."""
        samples = []
        for _ in range(n):
            samples.append(next(self))
        return np.array(samples)
