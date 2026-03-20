"""
Pytest Fixtures
===============

Shared fixtures for testing the Fair Synthetic Data Generator.

This module provides:
- Sample data fixtures for different modalities
- Model fixtures for encoders, decoders, generators
- Configuration fixtures
- Temporary directory fixtures
- Mock fixtures for external dependencies

Usage:
    def test_something(sample_tabular_data, vae_generator):
        # Use fixtures in tests
        pass
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ==========================================
# Configuration
# ==========================================

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==========================================
# Directory Fixtures
# ==========================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    dir_path = tempfile.mkdtemp()
    yield Path(dir_path)
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def temp_data_dir(temp_dir):
    """Create temporary data directory structure."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (data_dir / "raw").mkdir()
    (data_dir / "processed").mkdir()
    (data_dir / "synthetic").mkdir()
    
    return data_dir


@pytest.fixture
def temp_model_dir(temp_dir):
    """Create temporary model directory for checkpoints."""
    model_dir = temp_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def temp_output_dir(temp_dir):
    """Create temporary output directory."""
    output_dir = temp_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ==========================================
# Data Fixtures
# ==========================================

@pytest.fixture
def sample_tabular_data() -> pd.DataFrame:
    """
    Create sample tabular data for testing.
    
    Returns:
        DataFrame with synthetic tabular data including sensitive attributes.
    """
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'id': range(n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.5, 0.5]),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples),
        'income': np.random.exponential(50000, n_samples) + 20000,
        'education_years': np.random.randint(8, 20, n_samples),
        'credit_score': np.random.normal(700, 50, n_samples).clip(300, 850),
        'employment_years': np.random.exponential(5, n_samples).clip(0, 40),
        'loan_amount': np.random.exponential(15000, n_samples).clip(1000, 100000),
    })
    
    # Create biased outcome
    base_prob = 0.3
    data['outcome_prob'] = base_prob + (
        (data['age'] - 40) / 100 +
        (data['education_years'] - 12) / 50 +
        (data['credit_score'] - 700) / 1000
    )
    # Add gender bias
    data.loc[data['gender'] == 'Female', 'outcome_prob'] -= 0.1
    # Add race bias
    data.loc[data['race'] == 'Black', 'outcome_prob'] -= 0.08
    
    data['outcome_prob'] = data['outcome_prob'].clip(0.05, 0.95)
    data['outcome'] = (np.random.random(n_samples) < data['outcome_prob']).astype(int)
    data = data.drop(['id', 'outcome_prob'], axis=1)
    
    return data


@pytest.fixture
def sample_tabular_data_small() -> pd.DataFrame:
    """Create small sample data for quick tests."""
    np.random.seed(42)
    n_samples = 100
    
    return pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'sensitive': np.random.choice([0, 1], n_samples),
        'outcome': np.random.choice([0, 1], n_samples),
    })


@pytest.fixture
def sample_text_data() -> List[str]:
    """Create sample text data for testing."""
    np.random.seed(42)
    
    templates = [
        "A {age} year old {gender} with income ${income}",
        "Individual aged {age}, {gender}, earns ${income} annually",
        "{gender} person, {age} years old, annual income: ${income}",
    ]
    
    n_samples = 200
    texts = []
    
    for _ in range(n_samples):
        template = np.random.choice(templates)
        text = template.format(
            age=np.random.randint(18, 70),
            gender=np.random.choice(['male', 'female']),
            income=np.random.randint(20000, 100000)
        )
        texts.append(text)
    
    return texts


@pytest.fixture
def sample_image_data() -> np.ndarray:
    """
    Create sample image data for testing.
    
    Returns:
        Numpy array of shape (n_samples, channels, height, width)
    """
    np.random.seed(42)
    n_samples = 50
    channels = 3
    height = 32
    width = 32
    
    # Generate random images
    images = np.random.randint(0, 256, (n_samples, channels, height, width), dtype=np.uint8)
    
    return images


@pytest.fixture
def sample_multimodal_data(
    sample_tabular_data,
    sample_text_data,
    sample_image_data
) -> Dict[str, Any]:
    """Create sample multimodal data combining all modalities."""
    return {
        'tabular': sample_tabular_data[:len(sample_text_data)],
        'text': sample_text_data,
        'image': sample_image_data[:len(sample_text_data)]
    }


@pytest.fixture
def sensitive_attributes() -> List[str]:
    """Return list of sensitive attribute names."""
    return ['gender', 'race']


@pytest.fixture
def target_column() -> str:
    """Return target column name."""
    return 'outcome'


# ==========================================
# Tensor Fixtures
# ==========================================

@pytest.fixture
def sample_tensor_batch() -> torch.Tensor:
    """Create sample tensor batch for model testing."""
    return torch.randn(32, 20)  # batch_size=32, features=20


@pytest.fixture
def sample_latent_vector() -> torch.Tensor:
    """Create sample latent vector."""
    return torch.randn(32, 64)  # batch_size=32, latent_dim=64


@pytest.fixture
def sample_sensitive_labels() -> torch.Tensor:
    """Create sample sensitive attribute labels."""
    return torch.randint(0, 2, (32,))  # binary sensitive attribute


@pytest.fixture
def sample_target_labels() -> torch.Tensor:
    """Create sample target labels."""
    return torch.randint(0, 2, (32,))


# ==========================================
# Model Fixtures
# ==========================================

@pytest.fixture
def simple_encoder():
    """Create a simple encoder for testing."""
    class SimpleEncoder(nn.Module):
        def __init__(self, input_dim=20, latent_dim=64):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc_mu = nn.Linear(128, latent_dim)
            self.fc_var = nn.Linear(128, latent_dim)
        
        def forward(self, x):
            h = torch.relu(self.fc1(x))
            return self.fc_mu(h), self.fc_var(h)
    
    return SimpleEncoder()


@pytest.fixture
def simple_decoder():
    """Create a simple decoder for testing."""
    class SimpleDecoder(nn.Module):
        def __init__(self, latent_dim=64, output_dim=20):
            super().__init__()
            self.fc1 = nn.Linear(latent_dim, 128)
            self.fc2 = nn.Linear(128, output_dim)
        
        def forward(self, z):
            h = torch.relu(self.fc1(z))
            return self.fc2(h)
    
    return SimpleDecoder()


@pytest.fixture
def simple_vae(simple_encoder, simple_decoder):
    """Create a simple VAE for testing."""
    class SimpleVAE(nn.Module):
        def __init__(self, encoder, decoder):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
        
        def reparameterize(self, mu, log_var):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        
        def forward(self, x):
            mu, log_var = self.encoder(x)
            z = self.reparameterize(mu, log_var)
            x_recon = self.decoder(z)
            return x_recon, mu, log_var
        
        def sample(self, n_samples, device='cpu'):
            z = torch.randn(n_samples, 64).to(device)
            return self.decoder(z)
    
    return SimpleVAE(simple_encoder, simple_decoder)


@pytest.fixture
def simple_discriminator():
    """Create a simple discriminator for testing."""
    class SimpleDiscriminator(nn.Module):
        def __init__(self, input_dim=20):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.net(x)
    
    return SimpleDiscriminator()


@pytest.fixture
def gradient_reversal_layer():
    """Create gradient reversal layer for testing."""
    from torch.autograd import Function
    
    class GradReverse(Function):
        @staticmethod
        def forward(ctx, x, lambda_):
            ctx.lambda_ = lambda_
            return x.clone()
        
        @staticmethod
        def backward(ctx, grad_output):
            return -ctx.lambda_ * grad_output, None
    
    class GRL(nn.Module):
        def __init__(self, lambda_=1.0):
            super().__init__()
            self.lambda_ = lambda_
        
        def forward(self, x):
            return GradReverse.apply(x, self.lambda_)
    
    return GRL()


# ==========================================
# Configuration Fixtures
# ==========================================

@pytest.fixture
def default_config() -> Dict[str, Any]:
    """Create default configuration dictionary."""
    return {
        'model': {
            'type': 'vae',
            'latent_dim': 64,
            'hidden_dims': [128, 256, 128],
            'activation': 'relu',
        },
        'fairness': {
            'enabled': True,
            'type': 'adversarial',
            'lambda_fairness': 0.5,
            'sensitive_attributes': ['gender', 'race'],
            'constraint_threshold': 0.1,
        },
        'training': {
            'epochs': 10,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'weight_decay': 0.0,
            'early_stopping_patience': 5,
        },
        'data': {
            'modality': 'tabular',
            'normalize': True,
            'test_split': 0.2,
        },
        'evaluation': {
            'metrics': ['demographic_parity', 'equalized_odds', 'fidelity'],
            'generate_report': True,
        }
    }


@pytest.fixture
def fairness_config() -> Dict[str, Any]:
    """Create fairness-specific configuration."""
    return {
        'paradigm': 'group',
        'constraints': {
            'demographic_parity': {'threshold': 0.1},
            'equalized_odds': {'threshold': 0.15},
        },
        'adversary': {
            'hidden_dims': [64, 32],
            'learning_rate': 1e-3,
        },
        'lambda_fairness': 0.5,
    }


@pytest.fixture
def training_config() -> Dict[str, Any]:
    """Create training-specific configuration."""
    return {
        'epochs': 100,
        'batch_size': 256,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'gradient_clip': 1.0,
        'mixed_precision': True,
    }


# ==========================================
# Evaluation Fixtures
# ==========================================

@pytest.fixture
def fairness_metrics_results() -> Dict[str, float]:
    """Create sample fairness metrics results."""
    return {
        'demographic_parity_difference': 0.08,
        'demographic_parity_ratio': 0.85,
        'equalized_odds_difference': 0.12,
        'equal_opportunity_difference': 0.10,
        'disparate_impact_ratio': 0.82,
        'calibration_difference': 0.05,
        'overall_fairness_score': 0.85,
    }


@pytest.fixture
def fidelity_metrics_results() -> Dict[str, float]:
    """Create sample fidelity metrics results."""
    return {
        'js_divergence': 0.08,
        'wasserstein_distance': 0.15,
        'correlation_preservation': 0.92,
        'ks_statistic_max': 0.12,
        'mutual_information': 0.85,
        'moment_matching_score': 0.90,
        'overall_fidelity_score': 0.88,
    }


@pytest.fixture
def privacy_metrics_results() -> Dict[str, float]:
    """Create sample privacy metrics results."""
    return {
        'membership_inference_accuracy': 0.55,
        'attribute_inference_accuracy': 0.48,
        'epsilon': 1.0,
        'delta': 1e-5,
        'privacy_risk_score': 0.35,
    }


# ==========================================
# Mock Fixtures
# ==========================================

@pytest.fixture
def mock_model():
    """Create a mock model for testing without actual computation."""
    mock = MagicMock()
    mock.return_value = torch.randn(32, 20)
    mock.parameters.return_value = [torch.randn(10, 10)]
    mock.state_dict.return_value = {'weight': torch.randn(10, 10)}
    return mock


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader."""
    mock = MagicMock()
    mock.__len__ = lambda self: 10
    mock.__iter__ = lambda self: iter([
        (torch.randn(32, 20), torch.randint(0, 2, (32,)))
        for _ in range(10)
    ])
    return mock


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer."""
    mock = MagicMock()
    mock.zero_grad = MagicMock()
    mock.step = MagicMock()
    return mock


# ==========================================
# File Fixtures
# ==========================================

@pytest.fixture
def sample_csv_file(temp_data_dir, sample_tabular_data) -> Path:
    """Create a sample CSV file for testing file I/O."""
    file_path = temp_data_dir / "raw" / "sample_data.csv"
    sample_tabular_data.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_json_config(temp_dir, default_config) -> Path:
    """Create a sample JSON config file."""
    config_path = temp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    return config_path


@pytest.fixture
def sample_checkpoint(temp_model_dir, simple_vae) -> Path:
    """Create a sample model checkpoint file."""
    checkpoint_path = temp_model_dir / "model_checkpoint.pt"
    torch.save({
        'model_state_dict': simple_vae.state_dict(),
        'optimizer_state_dict': {},
        'epoch': 10,
        'loss': 0.5,
    }, checkpoint_path)
    return checkpoint_path


# ==========================================
# Utility Fixtures
# ==========================================

@pytest.fixture
def assert_tensors_close():
    """Helper to assert two tensors are close."""
    def _assert(a, b, rtol=1e-4, atol=1e-6):
        assert torch.allclose(a, b, rtol=rtol, atol=atol), \
            f"Tensors not close: max diff = {(a - b).abs().max().item()}"
    return _assert


@pytest.fixture
def assert_dataframe_schema():
    """Helper to assert DataFrame has expected schema."""
    def _assert(df, expected_columns, expected_types=None):
        assert list(df.columns) == expected_columns, \
            f"Columns mismatch: {list(df.columns)} vs {expected_columns}"
        if expected_types:
            for col, dtype in expected_types.items():
                assert df[col].dtype == dtype, \
                    f"Type mismatch for {col}: {df[col].dtype} vs {dtype}"
    return _assert


# ==========================================
# Skip Markers
# ==========================================

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks end-to-end tests"
    )


# ==========================================
# Device Selection
# ==========================================

@pytest.fixture(scope="session")
def device():
    """Get the device to use for testing."""
    return DEVICE


@pytest.fixture
def use_cuda():
    """Check if CUDA is available."""
    return torch.cuda.is_available()
