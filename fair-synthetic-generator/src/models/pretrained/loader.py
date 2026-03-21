"""
Pretrained Model Loader - Integrated Version

This module provides a unified interface for loading and using pretrained models.
Works with NumPy-only inference.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Project paths - Updated for integration into src/models/pretrained/
# Original was Path(__file__).parent.parent.parent
# Now being in src/models/pretrained/, parent.parent.parent is fair-synthetic-generator/
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PRETRAINED_DIR = PROJECT_ROOT / "models" / "pretrained"


@dataclass
class ModelInfo:
    """Information about a pretrained model."""
    model_id: str
    architecture: str
    dataset: str
    input_dim: int
    latent_dim: int
    metrics: Dict[str, float]
    created_at: str
    version: str = "1.0.0"
    
    def __repr__(self):
        return (f"ModelInfo(id='{self.model_id}', arch='{self.architecture}', "
                f"fidelity={self.metrics.get('fidelity', 0):.2f}, "
                f"fairness={self.metrics.get('fairness', 0):.2f})")


class NumPyVAE:
    """NumPy-only VAE decoder for inference."""
    
    def __init__(self, weights: Dict, config: Dict):
        self.weights = weights
        self.latent_dim = config.get('latent_dim', 32)
        self.input_dim = config.get('input_dim', 14)
        self.hidden_dims = config.get('hidden_dims', [256, 128, 64])
    
    def _leaky_relu(self, x, alpha=0.2):
        return np.where(x > 0, x, alpha * x)
    
    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vectors to data space."""
        x = z
        
        # Decoder: latent_dim -> hidden_dims reversed -> input_dim
        # First layer: latent -> first hidden
        w0 = self.weights.get('decoder.layers.0.weight')
        b0 = self.weights.get('decoder.layers.0.bias')
        if w0 is not None:
            x = x @ w0 + b0
            x = self._leaky_relu(x)
        
        # Second layer
        w4 = self.weights.get('decoder.layers.4.weight')
        b4 = self.weights.get('decoder.layers.4.bias')
        if w4 is not None:
            x = x @ w4 + b4
            x = self._leaky_relu(x)
        
        # Third layer
        w8 = self.weights.get('decoder.layers.8.weight')
        b8 = self.weights.get('decoder.layers.8.bias')
        if w8 is not None:
            x = x @ w8 + b8
            x = self._leaky_relu(x)
        
        # Output layer
        out_w = self.weights.get('decoder.output.weight')
        out_b = self.weights.get('decoder.output.bias')
        if out_w is not None:
            x = x @ out_w + out_b
        
        return x
    
    def generate(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate synthetic samples."""
        if seed is not None:
            np.random.seed(seed)
        z = np.random.randn(n_samples, self.latent_dim).astype(np.float32)
        return self.decode(z)


class NumPyGAN:
    """NumPy-only GAN Generator for inference."""
    
    def __init__(self, weights: Dict, config: Dict):
        self.weights = weights
        self.latent_dim = config.get('latent_dim', 64)
        self.input_dim = config.get('input_dim', 14)
        self.gen_dims = config.get('gen_dims', [256, 512, 256])
    
    def _leaky_relu(self, x, alpha=0.2):
        return np.where(x > 0, x, alpha * x)
    
    def generate(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate synthetic samples."""
        if seed is not None:
            np.random.seed(seed)
        
        z = np.random.randn(n_samples, self.latent_dim).astype(np.float32)
        x = z
        
        # Helper to get weight with fallback
        def get_weight(primary_key, fallback_key):
            w = self.weights.get(primary_key)
            if w is None:
                w = self.weights.get(fallback_key)
            return w
        
        # Layer 0
        w0 = get_weight('generator.model.0.weight', 'generator.layers.0.weight')
        b0 = get_weight('generator.model.0.bias', 'generator.layers.0.bias')
        if w0 is not None:
            x = x @ w0 + b0
            x = self._leaky_relu(x)
        
        # Layer 3/4
        w3 = get_weight('generator.model.3.weight', 'generator.layers.4.weight')
        b3 = get_weight('generator.model.3.bias', 'generator.layers.4.bias')
        if w3 is not None:
            x = x @ w3 + b3
            x = self._leaky_relu(x)
        
        # Layer 6/8
        w6 = get_weight('generator.model.6.weight', 'generator.layers.8.weight')
        b6 = get_weight('generator.model.6.bias', 'generator.layers.8.bias')
        if w6 is not None:
            x = x @ w6 + b6
            x = self._leaky_relu(x)
        
        # Output layer
        out_w = get_weight('generator.model.final.weight', 'generator.output.weight')
        out_b = get_weight('generator.model.final.bias', 'generator.output.bias')
        if out_w is not None:
            x = x @ out_w + out_b
        
        return x


class NumPyDiffusion:
    """NumPy-only Diffusion model for simplified inference."""
    
    def __init__(self, weights: Dict, config: Dict):
        self.weights = weights
        self.input_dim = config.get('input_dim', 14)
        self.timesteps = config.get('timesteps', 1000)
    
    def generate(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate samples using simplified diffusion."""
        if seed is not None:
            np.random.seed(seed)
        
        x = np.random.randn(n_samples, self.input_dim).astype(np.float32)
        
        # Apply denoising transformation if weights available
        out_w = self.weights.get('denoiser.output.weight')
        out_b = self.weights.get('denoiser.output.bias')
        
        if out_w is not None:
            x = x @ out_w[:self.input_dim, :self.input_dim] + out_b[:self.input_dim]
        
        return x


class PretrainedModel:
    """Unified interface for pretrained models."""
    
    def __init__(self, model_id: str, model, info: ModelInfo):
        self.model_id = model_id
        self._model = model
        self.info = info
    
    def generate(self, n_samples: int = 1000, seed: Optional[int] = None,
                 **kwargs) -> pd.DataFrame:
        """Generate synthetic data."""
        samples = self._model.generate(n_samples, seed)
        
        # Convert to DataFrame
        columns = self._get_column_names()
        df = pd.DataFrame(samples, columns=columns[:samples.shape[1]])
        
        return df
    
    def _get_column_names(self) -> List[str]:
        """Get column names based on dataset."""
        dataset = self.info.dataset
        if dataset == 'adult':
            return ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                   'marital_status', 'occupation', 'relationship', 'race', 'sex',
                   'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
        elif dataset == 'credit':
            return ['limit_bal', 'sex', 'education', 'marriage', 'age',
                   'pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6',
                   'bill_amt1', 'bill_amt2', 'bill_amt3', 'bill_amt4', 'bill_amt5', 'bill_amt6',
                   'pay_amt1', 'pay_amt2', 'pay_amt3', 'pay_amt4', 'pay_amt5', 'pay_amt6', 'default']
        else:
            return [f'feature_{i}' for i in range(self.info.input_dim)]
    
    def get_metrics(self) -> Dict[str, float]:
        """Get the model's reported metrics."""
        return self.info.metrics
    
    def __repr__(self):
        return f"PretrainedModel(id='{self.model_id}')"


def list_available_models() -> List[str]:
    """List all available pretrained models."""
    models = []
    if not PRETRAINED_DIR.exists():
        return []
        
    for arch in ['vae', 'gan', 'diffusion']:
        arch_dir = PRETRAINED_DIR / arch
        if arch_dir.exists():
            for f in arch_dir.glob('*.npz'):
                models.append(f.stem)
    return sorted(models)


def get_model_info(model_id: str) -> ModelInfo:
    """Get information about a specific model."""
    for arch in ['vae', 'gan', 'diffusion']:
        json_path = PRETRAINED_DIR / arch / f"{model_id}.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            return ModelInfo(
                model_id=model_id,
                architecture=data.get('architecture', arch),
                dataset=data.get('training_data', 'unknown'),
                input_dim=data.get('input_dim', 14),
                latent_dim=data.get('config', {}).get('latent_dim', 32),
                metrics=data.get('metrics', {}),
                created_at=data.get('created_at', 'unknown'),
                version=data.get('version', '1.0.0')
            )
    raise ValueError(f"Model not found: {model_id}. Search path: {PRETRAINED_DIR}")


def load_model(model_id: str) -> PretrainedModel:
    """Load a pretrained model by ID."""
    for arch in ['vae', 'gan', 'diffusion']:
        npz_path = PRETRAINED_DIR / arch / f"{model_id}.npz"
        json_path = PRETRAINED_DIR / arch / f"{model_id}.json"
        
        if npz_path.exists():
            # Load weights
            weights = dict(np.load(npz_path))
            
            # Load config
            config = {}
            if json_path.exists():
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    config = data.get('config', data)
            
            # Get model info
            info = get_model_info(model_id)
            
            # Create model instance
            if arch == 'vae':
                model = NumPyVAE(weights, config)
            elif arch == 'gan':
                model = NumPyGAN(weights, config)
            else:
                model = NumPyDiffusion(weights, config)
            
            return PretrainedModel(model_id, model, info)
    
    raise ValueError(f"Model not found: {model_id}. Available: {list_available_models()}")


def generate_synthetic_data(model_id: str, n_samples: int = 1000,
                           seed: Optional[int] = None) -> pd.DataFrame:
    """Convenience function to generate synthetic data."""
    model = load_model(model_id)
    return model.generate(n_samples, seed)
