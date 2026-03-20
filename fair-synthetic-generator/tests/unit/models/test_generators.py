"""
Unit Tests for Generators
=========================

Comprehensive tests for generator modules including:
- VAEGenerator
- GANGenerator
- DiffusionGenerator
- MultimodalGenerator
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.models.generators.vae_generator import VAEGenerator
from src.models.generators.gan_generator import GANGenerator
from src.models.generators.diffusion_generator import DiffusionGenerator
from src.models.generators.multimodal_generator import MultimodalGenerator
from src.models.generators.base_generator import BaseGenerator


class TestVAEGenerator:
    """Tests for VAEGenerator class."""
    
    @pytest.fixture
    def generator_config(self):
        """Generator configuration."""
        return {
            'input_dim': 20,
            'latent_dim': 32,
            'hidden_dims': [64, 128, 64],
            'beta': 1.0,
        }
    
    @pytest.fixture
    def generator(self, generator_config):
        """Create VAEGenerator instance."""
        return VAEGenerator(**generator_config)
    
    def test_initialization(self, generator, generator_config):
        """Test generator initializes correctly."""
        assert generator is not None
        assert generator.latent_dim == generator_config['latent_dim']
    
    def test_encode_decode_shape(self, generator, generator_config):
        """Test encode-decode produces correct shapes."""
        batch_size = 32
        x = torch.randn(batch_size, generator_config['input_dim'])
        
        # Encode
        mu, log_var = generator.encode(x)
        assert mu.shape == (batch_size, generator_config['latent_dim'])
        assert log_var.shape == (batch_size, generator_config['latent_dim'])
        
        # Decode
        z = generator.reparameterize(mu, log_var)
        x_recon = generator.decode(z)
        assert x_recon.shape == (batch_size, generator_config['input_dim'])
    
    def test_forward_pass(self, generator, generator_config):
        """Test full forward pass."""
        x = torch.randn(32, generator_config['input_dim'])
        
        x_recon, mu, log_var = generator(x)
        
        assert x_recon.shape == x.shape
        assert mu.shape == (32, generator_config['latent_dim'])
        assert log_var.shape == (32, generator_config['latent_dim'])
    
    def test_sample_generation(self, generator, generator_config):
        """Test sample generation from latent."""
        n_samples = 100
        
        samples = generator.sample(n_samples)
        
        assert samples.shape == (n_samples, generator_config['input_dim'])
    
    def test_reconstruction_loss(self, generator, generator_config):
        """Test reconstruction loss computation."""
        x = torch.randn(32, generator_config['input_dim'])
        x_recon, mu, log_var = generator(x)
        
        loss_dict = generator.compute_loss(x, x_recon, mu, log_var)
        
        assert 'reconstruction_loss' in loss_dict
        assert 'kl_loss' in loss_dict
        assert 'total_loss' in loss_dict
        assert loss_dict['total_loss'] > 0
    
    def test_beta_vae(self):
        """Test beta-VAE with different beta values."""
        for beta in [0.5, 1.0, 2.0, 5.0]:
            generator = VAEGenerator(
                input_dim=20,
                latent_dim=16,
                hidden_dims=[32],
                beta=beta
            )
            
            x = torch.randn(32, 20)
            x_recon, mu, log_var = generator(x)
            loss_dict = generator.compute_loss(x, x_recon, mu, log_var)
            
            # Higher beta should weight KL more
            assert loss_dict['kl_loss'] >= 0
    
    def test_latent_interpolation(self, generator, generator_config):
        """Test latent space interpolation."""
        n_steps = 10
        
        z1 = torch.randn(1, generator_config['latent_dim'])
        z2 = torch.randn(1, generator_config['latent_dim'])
        
        interpolated = generator.interpolate(z1, z2, n_steps)
        
        assert interpolated.shape == (n_steps, generator_config['input_dim'])
    
    def test_reparameterization_trick(self, generator, generator_config):
        """Test reparameterization trick enables gradient flow."""
        mu = torch.randn(16, generator_config['latent_dim'], requires_grad=True)
        log_var = torch.randn(16, generator_config['latent_dim'], requires_grad=True)
        
        z = generator.reparameterize(mu, log_var)
        loss = z.sum()
        loss.backward()
        
        assert mu.grad is not None
        assert log_var.grad is not None


class TestGANGenerator:
    """Tests for GANGenerator class."""
    
    @pytest.fixture
    def generator_config(self):
        """GAN generator configuration."""
        return {
            'latent_dim': 64,
            'output_dim': 20,
            'hidden_dims': [128, 256, 128],
            'generator_type': 'standard',
        }
    
    @pytest.fixture
    def generator(self, generator_config):
        """Create GANGenerator instance."""
        return GANGenerator(**generator_config)
    
    def test_generation_shape(self, generator, generator_config):
        """Test generator produces correct output shape."""
        n_samples = 32
        z = torch.randn(n_samples, generator_config['latent_dim'])
        
        output = generator(z)
        
        assert output.shape == (n_samples, generator_config['output_dim'])
    
    def test_generator_deterministic_in_eval(self, generator, generator_config):
        """Test generator is deterministic in eval mode."""
        generator.eval()
        z = torch.randn(16, generator_config['latent_dim'])
        
        output1 = generator(z)
        output2 = generator(z)
        
        assert torch.allclose(output1, output2)
    
    def test_gradient_flow(self, generator, generator_config):
        """Test gradients flow through generator."""
        z = torch.randn(16, generator_config['latent_dim'], requires_grad=True)
        
        output = generator(z)
        loss = output.sum()
        loss.backward()
        
        assert z.grad is not None
    
    def test_conditional_generation(self):
        """Test conditional GAN generation."""
        generator = GANGenerator(
            latent_dim=64,
            output_dim=20,
            hidden_dims=[128],
            num_classes=2
        )
        
        z = torch.randn(16, 64)
        labels = torch.randint(0, 2, (16,))
        
        output = generator(z, labels=labels)
        
        assert output.shape == (16, 20)
    
    def test_wasserstein_mode(self):
        """Test Wasserstein GAN mode."""
        generator = GANGenerator(
            latent_dim=64,
            output_dim=20,
            hidden_dims=[128],
            generator_type='wasserstein'
        )
        
        z = torch.randn(16, 64)
        output = generator(z)
        
        assert output.shape == (16, 20)


class TestDiffusionGenerator:
    """Tests for DiffusionGenerator class."""
    
    @pytest.fixture
    def generator_config(self):
        """Diffusion generator configuration."""
        return {
            'data_dim': 20,
            'hidden_dim': 128,
            'num_timesteps': 100,
            'beta_start': 0.0001,
            'beta_end': 0.02,
        }
    
    @pytest.fixture
    def generator(self, generator_config):
        """Create DiffusionGenerator instance."""
        return DiffusionGenerator(**generator_config)
    
    def test_forward_diffusion(self, generator, generator_config):
        """Test forward diffusion process."""
        x = torch.randn(16, generator_config['data_dim'])
        t = torch.randint(0, generator_config['num_timesteps'], (16,))
        
        noisy_x, noise = generator.forward_diffusion(x, t)
        
        assert noisy_x.shape == x.shape
        assert noise.shape == x.shape
    
    def test_reverse_diffusion(self, generator, generator_config):
        """Test reverse diffusion (denoising) process."""
        n_samples = 16
        
        samples = generator.sample(n_samples)
        
        assert samples.shape == (n_samples, generator_config['data_dim'])
    
    def test_noise_prediction(self, generator, generator_config):
        """Test noise prediction network."""
        x = torch.randn(16, generator_config['data_dim'])
        t = torch.randint(0, generator_config['num_timesteps'], (16,))
        
        predicted_noise = generator.predict_noise(x, t)
        
        assert predicted_noise.shape == x.shape
    
    def test_ddim_sampling(self):
        """Test DDIM fast sampling."""
        generator = DiffusionGenerator(
            data_dim=20,
            hidden_dim=64,
            num_timesteps=100,
            sampling_method='ddim'
        )
        
        samples = generator.sample(16, ddim_steps=20)
        
        assert samples.shape == (16, 20)
    
    def test_time_embedding(self, generator, generator_config):
        """Test time step embedding."""
        t = torch.randint(0, generator_config['num_timesteps'], (16,))
        
        embedding = generator.time_embedding(t)
        
        assert embedding.shape == (16, generator_config['hidden_dim'])


class TestMultimodalGenerator:
    """Tests for MultimodalGenerator class."""
    
    @pytest.fixture
    def generator_config(self):
        """Multimodal generator configuration."""
        return {
            'latent_dim': 64,
            'tabular_dim': 10,
            'image_shape': (3, 32, 32),
            'text_vocab_size': 1000,
            'fusion_type': 'attention',
        }
    
    @pytest.fixture
    def generator(self, generator_config):
        """Create MultimodalGenerator instance."""
        return MultimodalGenerator(**generator_config)
    
    def test_multimodal_generation(self, generator, generator_config):
        """Test generation of all modalities."""
        n_samples = 16
        
        outputs = generator.sample(n_samples)
        
        assert 'tabular' in outputs
        assert 'image' in outputs
        assert 'text' in outputs
        
        assert outputs['tabular'].shape == (n_samples, generator_config['tabular_dim'])
        assert outputs['image'].shape == (n_samples, 3, 32, 32)
    
    def test_conditional_generation(self, generator, generator_config):
        """Test conditional generation across modalities."""
        n_samples = 16
        
        # Condition on tabular, generate others
        tabular_condition = torch.randn(n_samples, generator_config['tabular_dim'])
        
        outputs = generator.sample(
            n_samples,
            condition={'tabular': tabular_condition}
        )
        
        assert outputs['tabular'].shape == (n_samples, generator_config['tabular_dim'])
    
    def test_cross_modal_consistency(self, generator, generator_config):
        """Test cross-modal consistency in generated samples."""
        n_samples = 32
        
        outputs = generator.sample(n_samples)
        
        # Check that generated modalities are consistent
        # (e.g., income in tabular matches text description)
        consistency_score = generator.compute_cross_modal_consistency(outputs)
        
        assert 0 <= consistency_score <= 1


class TestBaseGenerator:
    """Tests for BaseGenerator abstract class."""
    
    def test_abstract_methods(self):
        """Test BaseGenerator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseGenerator()
    
    def test_required_interface(self):
        """Test subclass must implement required methods."""
        class ValidGenerator(BaseGenerator):
            def sample(self, n_samples):
                return torch.randn(n_samples, 10)
            
            def forward(self, x):
                return x
        
        generator = ValidGenerator()
        samples = generator.sample(16)
        
        assert samples.shape == (16, 10)


class TestGeneratorEdgeCases:
    """Edge case tests for generators."""
    
    def test_vae_empty_batch(self):
        """Test VAE handles empty batch."""
        generator = VAEGenerator(input_dim=10, latent_dim=16, hidden_dims=[32])
        x = torch.randn(0, 10)
        
        x_recon, mu, log_var = generator(x)
        
        assert x_recon.shape == (0, 10)
    
    def test_gan_single_sample(self):
        """Test GAN generates single sample."""
        generator = GANGenerator(latent_dim=16, output_dim=10, hidden_dims=[32])
        z = torch.randn(1, 16)
        
        output = generator(z)
        
        assert output.shape == (1, 10)
    
    def test_extreme_latent_values(self):
        """Test generators handle extreme latent values."""
        generator = GANGenerator(latent_dim=16, output_dim=10, hidden_dims=[32])
        
        # Large values
        z = torch.randn(16, 16) * 10
        output = generator(z)
        assert not torch.isnan(output).any()
        
        # Small values
        z = torch.randn(16, 16) * 0.01
        output = generator(z)
        assert not torch.isnan(output).any()


class TestGeneratorTraining:
    """Tests for generator training utilities."""
    
    def test_vae_loss_components(self):
        """Test VAE loss has correct components."""
        generator = VAEGenerator(input_dim=10, latent_dim=16, hidden_dims=[32])
        
        x = torch.randn(16, 10)
        x_recon, mu, log_var = generator(x)
        
        loss_dict = generator.compute_loss(x, x_recon, mu, log_var)
        
        # Loss should be positive
        assert loss_dict['reconstruction_loss'] >= 0
        assert loss_dict['kl_loss'] >= 0
        assert loss_dict['total_loss'] >= 0
    
    def test_generator_training_step(self):
        """Test single training step updates parameters."""
        generator = VAEGenerator(input_dim=10, latent_dim=16, hidden_dims=[32])
        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
        
        # Store initial parameters
        initial_params = {name: p.clone() for name, p in generator.named_parameters()}
        
        # Training step
        x = torch.randn(16, 10)
        x_recon, mu, log_var = generator(x)
        loss_dict = generator.compute_loss(x, x_recon, mu, log_var)
        
        optimizer.zero_grad()
        loss_dict['total_loss'].backward()
        optimizer.step()
        
        # Check parameters changed
        params_changed = False
        for name, p in generator.named_parameters():
            if not torch.allclose(p, initial_params[name]):
                params_changed = True
                break
        
        assert params_changed


class TestGeneratorSerialization:
    """Tests for generator serialization."""
    
    def test_save_load_vae(self, temp_dir):
        """Test VAE save and load."""
        generator = VAEGenerator(input_dim=10, latent_dim=16, hidden_dims=[32])
        
        # Save
        save_path = temp_dir / "vae.pt"
        torch.save(generator.state_dict(), save_path)
        
        # Load
        generator2 = VAEGenerator(input_dim=10, latent_dim=16, hidden_dims=[32])
        generator2.load_state_dict(torch.load(save_path))
        
        # Compare outputs
        generator.eval()
        generator2.eval()
        
        x = torch.randn(16, 10)
        with torch.no_grad():
            out1 = generator(x)
            out2 = generator2(x)
        
        assert torch.allclose(out1[0], out2[0], atol=1e-5)
    
    def test_export_for_inference(self, temp_dir):
        """Test generator export for inference."""
        generator = VAEGenerator(input_dim=10, latent_dim=16, hidden_dims=[32])
        generator.eval()
        
        # Should be TorchScript compatible
        scripted = torch.jit.script(generator)
        
        # Test inference
        x = torch.randn(16, 10)
        output = scripted(x)
        
        assert output[0].shape == (16, 10)
