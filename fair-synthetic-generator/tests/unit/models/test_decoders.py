"""
Unit Tests for Decoders
=======================

Comprehensive tests for decoder modules including:
- TabularDecoder
- ImageDecoder
- TextDecoder
- MultimodalDecoder
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.decoders.tabular_decoder import TabularDecoder
from src.models.decoders.image_decoder import ImageDecoder
from src.models.decoders.text_decoder import TextDecoder
from src.models.decoders.multimodal_decoder import MultimodalDecoder
from src.models.decoders.base_decoder import BaseDecoder


class TestTabularDecoder:
    """Tests for TabularDecoder class."""
    
    @pytest.fixture
    def decoder_config(self):
        """Decoder configuration."""
        return {
            'latent_dim': 32,
            'hidden_dims': [64, 128, 64],
            'output_dim': 20,
            'activation': 'relu',
            'output_activation': None,
        }
    
    @pytest.fixture
    def decoder(self, decoder_config):
        """Create TabularDecoder instance."""
        return TabularDecoder(**decoder_config)
    
    def test_initialization(self, decoder, decoder_config):
        """Test decoder initializes correctly."""
        assert decoder is not None
        assert decoder.latent_dim == decoder_config['latent_dim']
        assert decoder.output_dim == decoder_config['output_dim']
    
    def test_forward_pass_shape(self, decoder, decoder_config):
        """Test forward pass produces correct output shape."""
        batch_size = 32
        z = torch.randn(batch_size, decoder_config['latent_dim'])
        
        output = decoder(z)
        
        assert output.shape == (batch_size, decoder_config['output_dim'])
    
    def test_latent_space_sampling(self, decoder, decoder_config):
        """Test decoder can generate from random latent vectors."""
        n_samples = 100
        z = torch.randn(n_samples, decoder_config['latent_dim'])
        
        output = decoder(z)
        
        assert output.shape == (n_samples, decoder_config['output_dim'])
    
    def test_gradient_flow(self, decoder, decoder_config):
        """Test gradients flow through decoder."""
        z = torch.randn(16, decoder_config['latent_dim'], requires_grad=True)
        
        output = decoder(z)
        loss = output.sum()
        loss.backward()
        
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()
    
    def test_output_activation(self):
        """Test different output activations."""
        # No activation
        decoder1 = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10, 
                                   output_activation=None)
        
        # Sigmoid activation
        decoder2 = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10,
                                   output_activation='sigmoid')
        
        # Tanh activation
        decoder3 = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10,
                                   output_activation='tanh')
        
        z = torch.randn(16, 16)
        
        out1 = decoder1(z)
        out2 = decoder2(z)
        out3 = decoder3(z)
        
        # Check output ranges
        assert out1.min() < 0 or out1.max() > 1  # No bounds
        assert (out2 >= 0).all() and (out2 <= 1).all()  # Sigmoid: [0, 1]
        assert (out3 >= -1).all() and (out3 <= 1).all()  # Tanh: [-1, 1]
    
    def test_conditional_generation(self):
        """Test conditional generation with labels."""
        decoder = TabularDecoder(
            latent_dim=16,
            hidden_dims=[32],
            output_dim=10,
            num_classes=2
        )
        
        z = torch.randn(16, 16)
        labels = torch.randint(0, 2, (16,))
        
        output = decoder(z, labels=labels)
        
        assert output.shape == (16, 10)


class TestImageDecoder:
    """Tests for ImageDecoder class."""
    
    @pytest.fixture
    def decoder_config(self):
        """Decoder configuration for images."""
        return {
            'latent_dim': 64,
            'hidden_channels': [128, 64, 32],
            'output_channels': 3,
            'image_size': 32,
        }
    
    @pytest.fixture
    def decoder(self, decoder_config):
        """Create ImageDecoder instance."""
        return ImageDecoder(**decoder_config)
    
    def test_forward_pass_shape(self, decoder, decoder_config):
        """Test forward pass produces correct image shape."""
        batch_size = 16
        z = torch.randn(batch_size, decoder_config['latent_dim'])
        
        output = decoder(z)
        
        assert output.shape == (
            batch_size,
            decoder_config['output_channels'],
            decoder_config['image_size'],
            decoder_config['image_size']
        )
    
    def test_output_range(self, decoder, decoder_config):
        """Test image output is in valid range."""
        z = torch.randn(16, decoder_config['latent_dim'])
        
        output = decoder(z)
        
        # Check output is bounded (depends on final activation)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_upsampling_methods(self):
        """Test different upsampling methods."""
        for method in ['transposed_conv', 'upsample']:
            decoder = ImageDecoder(
                latent_dim=32,
                hidden_channels=[64, 32],
                output_channels=3,
                image_size=32,
                upsampling=method
            )
            
            z = torch.randn(8, 32)
            output = decoder(z)
            
            assert output.shape == (8, 3, 32, 32)
    
    def test_different_output_sizes(self):
        """Test decoder handles different output image sizes."""
        for size in [32, 64, 128]:
            decoder = ImageDecoder(
                latent_dim=32,
                hidden_channels=[64, 32],
                output_channels=3,
                image_size=size
            )
            
            z = torch.randn(8, 32)
            output = decoder(z)
            
            assert output.shape == (8, 3, size, size)


class TestTextDecoder:
    """Tests for TextDecoder class."""
    
    @pytest.fixture
    def decoder_config(self):
        """Decoder configuration for text."""
        return {
            'latent_dim': 64,
            'hidden_dim': 256,
            'vocab_size': 10000,
            'max_length': 50,
            'num_layers': 2,
        }
    
    @pytest.fixture
    def decoder(self, decoder_config):
        """Create TextDecoder instance."""
        return TextDecoder(**decoder_config)
    
    def test_forward_pass_shape(self, decoder, decoder_config):
        """Test forward pass produces correct output shape."""
        batch_size = 16
        z = torch.randn(batch_size, decoder_config['latent_dim'])
        
        output = decoder(z)
        
        # Output should be sequence of token probabilities
        assert output.shape == (
            batch_size,
            decoder_config['max_length'],
            decoder_config['vocab_size']
        )
    
    def test_sampling_mode(self, decoder, decoder_config):
        """Test text generation via sampling."""
        decoder.eval()
        z = torch.randn(1, decoder_config['latent_dim'])
        
        # Should produce token indices
        output = decoder.sample(z, temperature=1.0)
        
        assert output.shape == (1, decoder_config['max_length'])
        assert (output >= 0).all() and (output < decoder_config['vocab_size']).all()
    
    def test_beam_search(self, decoder, decoder_config):
        """Test beam search decoding."""
        decoder.eval()
        z = torch.randn(1, decoder_config['latent_dim'])
        
        output = decoder.beam_search(z, beam_width=5)
        
        assert output.shape[0] == 1
        assert output.shape[1] <= decoder_config['max_length']


class TestMultimodalDecoder:
    """Tests for MultimodalDecoder class."""
    
    @pytest.fixture
    def decoder_config(self):
        """Decoder configuration for multimodal."""
        return {
            'latent_dim': 64,
            'tabular_output_dim': 10,
            'image_output_shape': (3, 32, 32),
            'text_vocab_size': 1000,
        }
    
    @pytest.fixture
    def decoder(self, decoder_config):
        """Create MultimodalDecoder instance."""
        return MultimodalDecoder(**decoder_config)
    
    def test_multimodal_forward(self, decoder, decoder_config):
        """Test forward pass produces all modalities."""
        batch_size = 16
        z = torch.randn(batch_size, decoder_config['latent_dim'])
        
        outputs = decoder(z)
        
        assert 'tabular' in outputs
        assert 'image' in outputs
        assert 'text' in outputs
        
        assert outputs['tabular'].shape == (batch_size, decoder_config['tabular_output_dim'])
        assert outputs['image'].shape == (batch_size, 3, 32, 32)
    
    def test_selective_generation(self, decoder, decoder_config):
        """Test generating specific modalities only."""
        batch_size = 16
        z = torch.randn(batch_size, decoder_config['latent_dim'])
        
        # Generate only tabular
        outputs = decoder(z, modalities=['tabular'])
        
        assert 'tabular' in outputs
        assert 'image' not in outputs or outputs.get('image') is None


class TestDecoderEdgeCases:
    """Edge case tests for decoders."""
    
    def test_zero_latent(self):
        """Test decoder handles zero latent vector."""
        decoder = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10)
        z = torch.zeros(16, 16)
        
        output = decoder(z)
        
        assert output.shape == (16, 10)
        assert not torch.isnan(output).any()
    
    def test_extreme_latent_values(self):
        """Test decoder handles extreme latent values."""
        decoder = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10)
        
        # Very large values
        z = torch.randn(16, 16) * 10
        output = decoder(z)
        assert not torch.isnan(output).any()
        
        # Very small values
        z = torch.randn(16, 16) * 0.01
        output = decoder(z)
        assert not torch.isnan(output).any()
    
    def test_batch_size_one(self):
        """Test decoder handles batch size of 1."""
        decoder = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10)
        z = torch.randn(1, 16)
        
        output = decoder(z)
        
        assert output.shape == (1, 10)


class TestDecoderDeterminism:
    """Tests for decoder determinism and reproducibility."""
    
    def test_eval_mode_deterministic(self):
        """Test decoder is deterministic in eval mode."""
        decoder = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10)
        decoder.eval()
        
        z = torch.randn(16, 16)
        
        output1 = decoder(z)
        output2 = decoder(z)
        
        assert torch.allclose(output1, output2)
    
    def test_reproducibility_with_seed(self):
        """Test reproducible results with same seed."""
        torch.manual_seed(42)
        decoder1 = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10)
        decoder1.eval()
        
        torch.manual_seed(42)
        decoder2 = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10)
        decoder2.eval()
        
        z = torch.randn(16, 16)
        
        output1 = decoder1(z)
        output2 = decoder2(z)
        
        assert torch.allclose(output1, output2)


class TestDecoderSerialization:
    """Tests for decoder serialization."""
    
    def test_save_load_state_dict(self, temp_dir):
        """Test decoder state dict save/load."""
        decoder = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10)
        
        # Save
        save_path = temp_dir / "decoder.pt"
        torch.save(decoder.state_dict(), save_path)
        
        # Load
        decoder2 = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10)
        decoder2.load_state_dict(torch.load(save_path))
        
        # Compare
        z = torch.randn(16, 16)
        decoder.eval()
        decoder2.eval()
        
        assert torch.allclose(decoder(z), decoder2(z))
    
    def test_script_compatibility(self):
        """Test decoder is TorchScript compatible."""
        decoder = TabularDecoder(latent_dim=16, hidden_dims=[32], output_dim=10)
        decoder.eval()
        
        # Should not raise
        scripted = torch.jit.script(decoder)
        
        z = torch.randn(16, 16)
        output = scripted(z)
        
        assert output.shape == (16, 10)
