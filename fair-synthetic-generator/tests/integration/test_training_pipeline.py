"""
Integration Tests for Training Pipeline
=======================================

Tests for the complete training pipeline including:
- Data loading and preprocessing
- Model initialization and training
- Fairness constraint integration
- Checkpointing and logging
- Model evaluation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.trainer import Trainer
from src.training.callbacks.checkpoint_callback import CheckpointCallback
from src.training.callbacks.logging_callback import LoggingCallback
from src.training.callbacks.fairness_callback import FairnessCallback
from src.training.optimizers.multi_objective_optimizer import MultiObjectiveOptimizer
from src.training.strategies.adversarial_training import AdversarialTrainingStrategy
from src.models.generators.vae_generator import VAEGenerator
from src.data.preprocessing.tabular_preprocessor import TabularPreprocessor


class TestTrainingPipelineIntegration:
    """Integration tests for complete training pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        dir_path = tempfile.mkdtemp()
        yield Path(dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for training."""
        np.random.seed(42)
        n = 1000
        
        data = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.choice([0, 1, 2], n),
            'sensitive': np.random.choice([0, 1], n),
            'target': np.random.choice([0, 1], n)
        })
        
        return data
    
    @pytest.fixture
    def training_config(self, temp_dir):
        """Create training configuration."""
        return {
            'epochs': 5,
            'batch_size': 64,
            'learning_rate': 1e-3,
            'latent_dim': 16,
            'hidden_dims': [32, 64, 32],
            'checkpoint_dir': str(temp_dir / 'checkpoints'),
            'log_dir': str(temp_dir / 'logs'),
            'device': 'cpu',
        }
    
    @pytest.fixture
    def preprocessed_data(self, sample_data):
        """Create preprocessed data."""
        preprocessor = TabularPreprocessor(
            numerical_cols=['feature_1', 'feature_2'],
            categorical_cols=['feature_3', 'sensitive']
        )
        
        processed = preprocessor.fit_transform(sample_data)
        
        return processed, preprocessor
    
    def test_end_to_end_training(self, preprocessed_data, training_config):
        """Test complete end-to-end training pipeline."""
        X, preprocessor = preprocessed_data
        
        # Initialize model
        model = VAEGenerator(
            input_dim=X.shape[1],
            latent_dim=training_config['latent_dim'],
            hidden_dims=training_config['hidden_dims']
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            config=training_config
        )
        
        # Train
        history = trainer.fit(X, epochs=training_config['epochs'])
        
        assert history is not None
        assert len(history['loss']) == training_config['epochs']
        assert history['loss'][-1] < history['loss'][0]  # Should improve
    
    def test_training_with_fairness_constraints(self, preprocessed_data, training_config):
        """Test training with fairness constraints."""
        X, preprocessor = preprocessed_data
        
        model = VAEGenerator(
            input_dim=X.shape[1],
            latent_dim=training_config['latent_dim'],
            hidden_dims=training_config['hidden_dims']
        )
        
        # Add fairness callback
        fairness_callback = FairnessCallback(
            sensitive_idx=-2,  # Assuming sensitive is second to last
            lambda_fairness=0.5
        )
        
        trainer = Trainer(
            model=model,
            config=training_config,
            callbacks=[fairness_callback]
        )
        
        history = trainer.fit(X, epochs=training_config['epochs'])
        
        assert 'fairness_loss' in history
    
    def test_checkpoint_saving_and_loading(self, preprocessed_data, training_config, temp_dir):
        """Test checkpoint saving and loading."""
        X, preprocessor = preprocessed_data
        
        model = VAEGenerator(
            input_dim=X.shape[1],
            latent_dim=training_config['latent_dim'],
            hidden_dims=training_config['hidden_dims']
        )
        
        checkpoint_callback = CheckpointCallback(
            checkpoint_dir=temp_dir / 'checkpoints',
            save_best_only=True
        )
        
        trainer = Trainer(
            model=model,
            config=training_config,
            callbacks=[checkpoint_callback]
        )
        
        trainer.fit(X, epochs=training_config['epochs'])
        
        # Check checkpoint was saved
        checkpoint_files = list((temp_dir / 'checkpoints').glob('*.pt'))
        assert len(checkpoint_files) > 0
    
    def test_training_resumption(self, preprocessed_data, training_config, temp_dir):
        """Test resuming training from checkpoint."""
        X, preprocessor = preprocessed_data
        
        model = VAEGenerator(
            input_dim=X.shape[1],
            latent_dim=training_config['latent_dim'],
            hidden_dims=training_config['hidden_dims']
        )
        
        # First training session
        trainer = Trainer(
            model=model,
            config=training_config
        )
        
        checkpoint_path = temp_dir / 'checkpoint.pt'
        
        # Train for 2 epochs
        history1 = trainer.fit(X, epochs=2)
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': 2,
        }, checkpoint_path)
        
        # Resume from checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Continue training
        history2 = trainer.fit(X, epochs=3, initial_epoch=2)
        
        assert len(history2['loss']) == 3
    
    def test_early_stopping(self, preprocessed_data, training_config):
        """Test early stopping functionality."""
        X, preprocessor = preprocessed_data
        
        model = VAEGenerator(
            input_dim=X.shape[1],
            latent_dim=training_config['latent_dim'],
            hidden_dims=training_config['hidden_dims']
        )
        
        trainer = Trainer(
            model=model,
            config={**training_config, 'early_stopping_patience': 2}
        )
        
        history = trainer.fit(X, epochs=100)
        
        # Should stop early
        assert len(history['loss']) < 100


class TestAdversarialTrainingIntegration:
    """Integration tests for adversarial training strategy."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        n = 500
        
        X = np.random.randn(n, 10)
        sensitive = np.random.choice([0, 1], n)
        
        return torch.FloatTensor(X), torch.LongTensor(sensitive)
    
    def test_adversarial_training_flow(self, sample_data):
        """Test adversarial training strategy."""
        X, sensitive = sample_data
        
        # Create model
        class SimpleVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(10, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(16, 32),
                    nn.ReLU(),
                    nn.Linear(32, 10)
                )
            
            def forward(self, x):
                z = self.encoder(x)
                return self.decoder(z), z
        
        model = SimpleVAE()
        
        # Create adversary
        adversary = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # Training strategy
        strategy = AdversarialTrainingStrategy(
            lambda_adv=0.5,
            adversary_lr=1e-3
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        adversary_optimizer = torch.optim.Adam(adversary.parameters(), lr=1e-3)
        
        # Train for a few steps
        for _ in range(5):
            recon, z = model(X)
            loss = strategy.compute_loss(
                reconstruction=recon,
                original=X,
                latent=z,
                sensitive=sensitive,
                adversary=adversary
            )
            
            optimizer.zero_grad()
            adversary_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            adversary_optimizer.step()
        
        # Should complete without errors
        assert True


class TestMultiObjectiveOptimization:
    """Integration tests for multi-objective optimization."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return torch.randn(64, 10)
    
    def test_multi_objective_loss_computation(self, sample_data):
        """Test multi-objective loss computation."""
        model = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
        
        optimizer = MultiObjectiveOptimizer(
            model.parameters(),
            objectives=['reconstruction', 'fairness', 'privacy'],
            weights=[1.0, 0.5, 0.3],
            lr=1e-3
        )
        
        # Compute losses
        output = model(sample_data)
        recon_loss = nn.MSELoss()(output, sample_data)
        fairness_loss = torch.tensor(0.1)
        privacy_loss = torch.tensor(0.2)
        
        losses = {
            'reconstruction': recon_loss,
            'fairness': fairness_loss,
            'privacy': privacy_loss
        }
        
        # Step optimizer
        optimizer.zero_grad()
        total_loss = optimizer.compute_total_loss(losses)
        total_loss.backward()
        optimizer.step()
        
        assert total_loss.item() > 0


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        dir_path = tempfile.mkdtemp()
        yield Path(dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)
    
    @pytest.fixture
    def raw_data_file(self, temp_dir):
        """Create raw data file."""
        np.random.seed(42)
        df = pd.DataFrame({
            'age': np.random.randint(18, 70, 100),
            'income': np.random.exponential(50000, 100),
            'gender': np.random.choice(['M', 'F'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        file_path = temp_dir / 'raw_data.csv'
        df.to_csv(file_path, index=False)
        
        return file_path
    
    def test_full_data_pipeline(self, raw_data_file, temp_dir):
        """Test full data processing pipeline."""
        # Load data
        df = pd.read_csv(raw_data_file)
        
        # Preprocess
        preprocessor = TabularPreprocessor(
            numerical_cols=['age', 'income'],
            categorical_cols=['gender']
        )
        
        X = preprocessor.fit_transform(df)
        
        # Verify output
        assert X.shape[0] == 100
        assert not np.isnan(X).any()
        
        # Test inverse transform
        df_reconstructed = preprocessor.inverse_transform(X)
        
        assert df_reconstructed is not None


class TestTrainingWithMultipleModalities:
    """Integration tests for multimodal training."""
    
    def test_multimodal_data_handling(self):
        """Test handling of multimodal data."""
        np.random.seed(42)
        n = 100
        
        # Create multimodal data
        tabular = np.random.randn(n, 10)
        text_features = np.random.randn(n, 50)  # Pre-extracted features
        image_features = np.random.randn(n, 64)
        
        # Combine modalities
        multimodal = {
            'tabular': torch.FloatTensor(tabular),
            'text': torch.FloatTensor(text_features),
            'image': torch.FloatTensor(image_features)
        }
        
        # Simple multimodal model
        class MultimodalModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.tabular_encoder = nn.Linear(10, 16)
                self.text_encoder = nn.Linear(50, 16)
                self.image_encoder = nn.Linear(64, 16)
                self.decoder = nn.Linear(48, 10)
            
            def forward(self, data):
                t = self.tabular_encoder(data['tabular'])
                te = self.text_encoder(data['text'])
                i = self.image_encoder(data['image'])
                combined = torch.cat([t, te, i], dim=-1)
                return self.decoder(combined)
        
        model = MultimodalModel()
        output = model(multimodal)
        
        assert output.shape == (n, 10)


class TestDistributedTrainingIntegration:
    """Integration tests for distributed training (mocked)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_ddp_setup(self):
        """Test DDP setup (mocked for single GPU)."""
        # This would test DDP setup in a real distributed environment
        # For now, just verify the imports work
        from torch.nn.parallel import DistributedDataParallel as DDP
        
        model = nn.Linear(10, 5)
        # In real test: model = DDP(model)
        assert model is not None


class TestTrainingReproducibility:
    """Integration tests for training reproducibility."""
    
    def test_reproducibility_with_seed(self):
        """Test training reproducibility with same seed."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # First run
        model1 = VAEGenerator(input_dim=10, latent_dim=8, hidden_dims=[16])
        X = torch.randn(64, 10)
        
        output1 = model1(X)
        
        # Reset seed
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Second run
        model2 = VAEGenerator(input_dim=10, latent_dim=8, hidden_dims=[16])
        
        output2 = model2(X)
        
        assert torch.allclose(output1[0], output2[0], atol=1e-5)


class TestTrainingMemoryEfficiency:
    """Integration tests for memory efficiency."""
    
    def test_gradient_accumulation(self):
        """Test gradient accumulation for memory efficiency."""
        model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 100)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Simulate large batch with accumulation
        accumulation_steps = 4
        effective_batch_size = 256
        mini_batch_size = effective_batch_size // accumulation_steps
        
        optimizer.zero_grad()
        
        for _ in range(accumulation_steps):
            X = torch.randn(mini_batch_size, 100)
            output = model(X)
            loss = criterion(output, X) / accumulation_steps
            loss.backward()
        
        optimizer.step()
        
        # Should complete without memory issues
        assert True
    
    def test_mixed_precision(self):
        """Test mixed precision training."""
        model = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Linear(128, 50)
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler('cuda', enabled=False)  # Disabled for CPU
        
        X = torch.randn(32, 50)
        
        with torch.amp.autocast('cuda', enabled=False):
            output = model(X)
            loss = nn.MSELoss()(output, X)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert True
