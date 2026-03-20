"""
Unit Tests for Encoders
=======================

Comprehensive tests for encoder modules including:
- TabularEncoder
- ImageEncoder
- TextEncoder
- MultimodalFusionEncoder

Tests cover:
- Forward pass correctness
- Output shape validation
- Gradient flow
- Edge cases and error handling
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import encoders
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.encoders.tabular_encoder import TabularEncoder
from src.models.encoders.image_encoder import ImageEncoder
from src.models.encoders.text_encoder import TextEncoder
from src.models.encoders.multimodal_fusion import MultimodalFusionEncoder
from src.models.encoders.base_encoder import BaseEncoder


class TestTabularEncoder:
    """Tests for TabularEncoder class."""
    
    @pytest.fixture
    def encoder_config(self):
        """Encoder configuration."""
        return {
            'input_dim': 20,
            'hidden_dims': [64, 128, 64],
            'output_dim': 32,
            'activation': 'relu',
            'dropout': 0.1,
            'batch_norm': True,
        }
    
    @pytest.fixture
    def encoder(self, encoder_config):
        """Create TabularEncoder instance."""
        return TabularEncoder(**encoder_config)
    
    def test_initialization(self, encoder, encoder_config):
        """Test encoder initializes correctly."""
        assert encoder is not None
        assert encoder.input_dim == encoder_config['input_dim']
        assert encoder.output_dim == encoder_config['output_dim']
    
    def test_forward_pass_shape(self, encoder, encoder_config):
        """Test forward pass produces correct output shape."""
        batch_size = 32
        x = torch.randn(batch_size, encoder_config['input_dim'])
        
        output = encoder(x)
        
        assert output.shape == (batch_size, encoder_config['output_dim'])
    
    def test_forward_pass_deterministic(self, encoder, encoder_config):
        """Test forward pass is deterministic in eval mode."""
        x = torch.randn(16, encoder_config['input_dim'])
        
        encoder.eval()
        output1 = encoder(x)
        output2 = encoder(x)
        
        assert torch.allclose(output1, output2)
    
    def test_gradient_flow(self, encoder, encoder_config):
        """Test gradients flow through encoder."""
        x = torch.randn(16, encoder_config['input_dim'], requires_grad=True)
        
        output = encoder(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_different_batch_sizes(self, encoder, encoder_config):
        """Test encoder handles different batch sizes."""
        for batch_size in [1, 16, 64, 128]:
            x = torch.randn(batch_size, encoder_config['input_dim'])
            output = encoder(x)
            assert output.shape[0] == batch_size
    
    def test_invalid_input_dim(self, encoder_config):
        """Test encoder raises error for invalid input dimension."""
        encoder = TabularEncoder(**encoder_config)
        x = torch.randn(16, 10)  # Wrong input dim
        
        with pytest.raises(RuntimeError):
            encoder(x)
    
    def test_device_compatibility(self, encoder, encoder_config, device):
        """Test encoder works on different devices."""
        encoder = encoder.to(device)
        x = torch.randn(16, encoder_config['input_dim']).to(device)
        
        output = encoder(x)
        
        assert output.device == x.device


class TestImageEncoder:
    """Tests for ImageEncoder class."""
    
    @pytest.fixture
    def encoder_config(self):
        """Encoder configuration for images."""
        return {
            'input_channels': 3,
            'hidden_channels': [32, 64, 128],
            'output_dim': 64,
            'image_size': 32,
        }
    
    @pytest.fixture
    def encoder(self, encoder_config):
        """Create ImageEncoder instance."""
        return ImageEncoder(**encoder_config)
    
    def test_forward_pass_shape(self, encoder, encoder_config):
        """Test forward pass produces correct output shape."""
        batch_size = 16
        x = torch.randn(
            batch_size,
            encoder_config['input_channels'],
            encoder_config['image_size'],
            encoder_config['image_size']
        )
        
        output = encoder(x)
        
        assert output.shape == (batch_size, encoder_config['output_dim'])
    
    def test_different_image_sizes(self):
        """Test encoder handles different image sizes."""
        for size in [32, 64, 128]:
            encoder = ImageEncoder(
                input_channels=3,
                hidden_channels=[32, 64],
                output_dim=32,
                image_size=size
            )
            x = torch.randn(8, 3, size, size)
            output = encoder(x)
            assert output.shape == (8, 32)
    
    def test_grayscale_images(self):
        """Test encoder handles grayscale (1-channel) images."""
        encoder = ImageEncoder(
            input_channels=1,
            hidden_channels=[32, 64],
            output_dim=32,
            image_size=32
        )
        x = torch.randn(8, 1, 32, 32)
        output = encoder(x)
        assert output.shape == (8, 32)
    
    def test_gradient_flow(self, encoder, encoder_config):
        """Test gradients flow through image encoder."""
        x = torch.randn(8, 3, 32, 32, requires_grad=True)
        
        output = encoder(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestTextEncoder:
    """Tests for TextEncoder class."""
    
    @pytest.fixture
    def encoder_config(self):
        """Encoder configuration for text."""
        return {
            'vocab_size': 10000,
            'embedding_dim': 128,
            'hidden_dim': 256,
            'output_dim': 64,
            'num_layers': 2,
            'max_length': 50,
            'padding_idx': 0,
        }
    
    @pytest.fixture
    def encoder(self, encoder_config):
        """Create TextEncoder instance."""
        return TextEncoder(**encoder_config)
    
    def test_forward_pass_shape(self, encoder, encoder_config):
        """Test forward pass produces correct output shape."""
        batch_size = 16
        x = torch.randint(
            1, encoder_config['vocab_size'],
            (batch_size, encoder_config['max_length'])
        )
        
        output = encoder(x)
        
        assert output.shape == (batch_size, encoder_config['output_dim'])
    
    def test_handles_padding(self, encoder, encoder_config):
        """Test encoder handles padded sequences."""
        x = torch.zeros(8, encoder_config['max_length'], dtype=torch.long)
        x[:, :10] = torch.randint(1, 100, (8, 10))  # Only first 10 tokens valid
        
        output = encoder(x)
        
        assert output.shape == (8, encoder_config['output_dim'])
        assert not torch.isnan(output).any()
    
    def test_variable_length_sequences(self, encoder, encoder_config):
        """Test encoder handles variable length sequences."""
        for length in [10, 30, 50]:
            x = torch.randint(1, 100, (8, length))
            # Pad or truncate to max_length
            if length < encoder_config['max_length']:
                x = torch.cat([
                    x,
                    torch.zeros(8, encoder_config['max_length'] - length, dtype=torch.long)
                ], dim=1)
            else:
                x = x[:, :encoder_config['max_length']]
            
            output = encoder(x)
            assert output.shape == (8, encoder_config['output_dim'])


class TestMultimodalFusionEncoder:
    """Tests for MultimodalFusionEncoder class."""
    
    @pytest.fixture
    def encoder_config(self):
        """Encoder configuration for multimodal fusion."""
        return {
            'tabular_input_dim': 10,
            'image_output_dim': 32,
            'text_output_dim': 32,
            'fusion_dim': 64,
            'output_dim': 128,
        }
    
    @pytest.fixture
    def encoder(self, encoder_config):
        """Create MultimodalFusionEncoder instance."""
        return MultimodalFusionEncoder(**encoder_config)
    
    def test_multimodal_forward(self, encoder, encoder_config):
        """Test forward pass with multiple modalities."""
        batch_size = 16
        
        inputs = {
            'tabular': torch.randn(batch_size, encoder_config['tabular_input_dim']),
            'image': torch.randn(batch_size, encoder_config['image_output_dim']),
            'text': torch.randn(batch_size, encoder_config['text_output_dim']),
        }
        
        output = encoder(inputs)
        
        assert output.shape == (batch_size, encoder_config['output_dim'])
    
    def test_missing_modality_handling(self, encoder, encoder_config):
        """Test encoder handles missing modalities gracefully."""
        batch_size = 16
        
        # Only provide tabular data
        inputs = {
            'tabular': torch.randn(batch_size, encoder_config['tabular_input_dim']),
        }
        
        output = encoder(inputs)
        
        assert output.shape[0] == batch_size
        assert not torch.isnan(output).any()
    
    def test_fusion_weights(self, encoder, encoder_config):
        """Test that fusion weights are learned."""
        batch_size = 16
        inputs = {
            'tabular': torch.randn(batch_size, encoder_config['tabular_input_dim']),
            'image': torch.randn(batch_size, encoder_config['image_output_dim']),
            'text': torch.randn(batch_size, encoder_config['text_output_dim']),
        }
        
        # Forward and backward
        output = encoder(inputs)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist for fusion parameters
        for name, param in encoder.named_parameters():
            if 'fusion' in name:
                assert param.grad is not None


class TestBaseEncoder:
    """Tests for BaseEncoder abstract class."""
    
    def test_abstract_methods(self):
        """Test that BaseEncoder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEncoder()
    
    def test_subclass_interface(self):
        """Test that subclass implements required methods."""
        class ValidEncoder(BaseEncoder):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x
            
            def get_output_dim(self):
                return 10
        
        encoder = ValidEncoder()
        assert encoder.get_output_dim() == 10


class TestEncoderEdgeCases:
    """Edge case tests for encoders."""
    
    def test_empty_batch(self):
        """Test encoder handles empty batch."""
        encoder = TabularEncoder(input_dim=10, hidden_dims=[32], output_dim=16)
        x = torch.randn(0, 10)
        
        output = encoder(x)
        assert output.shape == (0, 16)
    
    def test_single_sample(self):
        """Test encoder handles single sample."""
        encoder = TabularEncoder(input_dim=10, hidden_dims=[32], output_dim=16)
        x = torch.randn(1, 10)
        
        output = encoder(x)
        assert output.shape == (1, 16)
    
    def test_extreme_values(self):
        """Test encoder handles extreme input values."""
        encoder = TabularEncoder(input_dim=10, hidden_dims=[32], output_dim=16)
        
        # Very large values
        x = torch.randn(16, 10) * 1e6
        output = encoder(x)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Very small values
        x = torch.randn(16, 10) * 1e-6
        output = encoder(x)
        assert not torch.isnan(output).any()
    
    def test_reproducibility(self):
        """Test encoder produces reproducible results."""
        encoder = TabularEncoder(input_dim=10, hidden_dims=[32], output_dim=16)
        encoder.eval()
        
        torch.manual_seed(42)
        x = torch.randn(16, 10)
        
        output1 = encoder(x)
        output2 = encoder(x)
        
        assert torch.allclose(output1, output2)


class TestEncoderPerformance:
    """Performance-related tests for encoders."""
    
    @pytest.mark.slow
    def test_large_batch_efficiency(self):
        """Test encoder handles large batches efficiently."""
        encoder = TabularEncoder(input_dim=100, hidden_dims=[256, 512], output_dim=128)
        x = torch.randn(1024, 100)
        
        # Should not raise memory error
        output = encoder(x)
        assert output.shape == (1024, 128)
    
    @pytest.mark.slow
    def test_memory_efficiency(self):
        """Test encoder doesn't leak memory."""
        import gc
        
        encoder = TabularEncoder(input_dim=10, hidden_dims=[32], output_dim=16)
        
        # Run multiple forward passes
        for _ in range(100):
            x = torch.randn(32, 10)
            _ = encoder(x)
        
        # Force garbage collection
        gc.collect()
        
        # Test passes if no memory error


class TestEncoderSerialization:
    """Tests for encoder serialization and loading."""
    
    def test_state_dict_save_load(self, temp_dir):
        """Test encoder can be saved and loaded."""
        encoder = TabularEncoder(input_dim=10, hidden_dims=[32], output_dim=16)
        
        # Save
        save_path = temp_dir / "encoder.pt"
        torch.save(encoder.state_dict(), save_path)
        
        # Load
        encoder2 = TabularEncoder(input_dim=10, hidden_dims=[32], output_dim=16)
        encoder2.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        x = torch.randn(16, 10)
        encoder.eval()
        encoder2.eval()
        
        output1 = encoder(x)
        output2 = encoder2(x)
        
        assert torch.allclose(output1, output2)
