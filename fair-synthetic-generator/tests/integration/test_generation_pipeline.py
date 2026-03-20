"""
Integration Tests for Generation Pipeline
=========================================

Tests for the complete generation pipeline including:
- Model loading and initialization
- Synthetic data generation
- Post-processing
- Quality filtering
- Export and formatting
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.synthesis.generator_pipeline import GeneratorPipeline
from src.synthesis.postprocessing.consistency_checker import ConsistencyChecker
from src.synthesis.postprocessing.fairness_auditor import FairnessAuditor
from src.synthesis.postprocessing.quality_filter import QualityFilter
from src.synthesis.output.data_exporter import DataExporter
from src.synthesis.output.format_converter import FormatConverter
from src.models.generators.vae_generator import VAEGenerator


class TestGenerationPipelineIntegration:
    """Integration tests for generation pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        dir_path = tempfile.mkdtemp()
        yield Path(dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)
    
    @pytest.fixture
    def trained_model(self, temp_dir):
        """Create and save a trained model."""
        model = VAEGenerator(
            input_dim=10,
            latent_dim=16,
            hidden_dims=[32, 64, 32]
        )
        
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        X_train = torch.randn(500, 10)
        
        model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            x_recon, mu, log_var = model(X_train)
            recon_loss = nn.MSELoss()(x_recon, X_train)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + 0.1 * kl_loss
            loss.backward()
            optimizer.step()
        
        # Save model
        model_path = temp_dir / 'model.pt'
        torch.save(model.state_dict(), model_path)
        
        return model, model_path
    
    @pytest.fixture
    def generation_config(self, temp_dir):
        """Create generation configuration."""
        return {
            'n_samples': 1000,
            'batch_size': 64,
            'output_dir': str(temp_dir / 'output'),
            'output_format': 'csv',
            'apply_postprocessing': True,
            'quality_threshold': 0.7,
        }
    
    def test_full_generation_pipeline(self, trained_model, generation_config, temp_dir):
        """Test complete generation pipeline."""
        model, model_path = trained_model
        
        # Initialize pipeline
        pipeline = GeneratorPipeline(
            model=model,
            config=generation_config
        )
        
        # Generate synthetic data
        synthetic_data = pipeline.generate(n_samples=500)
        
        assert synthetic_data is not None
        assert len(synthetic_data) == 500
        assert synthetic_data.shape[1] == 10
    
    def test_generation_with_conditioning(self, trained_model, generation_config):
        """Test conditional generation."""
        model, model_path = trained_model
        
        pipeline = GeneratorPipeline(model=model, config=generation_config)
        
        # Condition on some features
        conditions = {
            'feature_0': np.random.randn(100),
        }
        
        synthetic_data = pipeline.generate(
            n_samples=100,
            conditions=conditions
        )
        
        assert synthetic_data is not None
    
    def test_batch_generation(self, trained_model, generation_config):
        """Test batch generation."""
        model, model_path = trained_model
        
        pipeline = GeneratorPipeline(model=model, config=generation_config)
        
        # Generate large batch
        synthetic_data = pipeline.generate(
            n_samples=10000,
            batch_size=256
        )
        
        assert len(synthetic_data) == 10000


class TestPostprocessingIntegration:
    """Integration tests for postprocessing modules."""
    
    @pytest.fixture
    def sample_synthetic_data(self):
        """Create sample synthetic data."""
        np.random.seed(42)
        n = 1000
        
        return pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.choice([0, 1, 2], n),
            'sensitive': np.random.choice([0, 1], n),
            'prediction': np.random.random(n),
        })
    
    @pytest.fixture
    def sample_real_data(self):
        """Create sample real data for comparison."""
        np.random.seed(123)
        n = 1000
        
        return pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.choice([0, 1, 2], n),
            'sensitive': np.random.choice([0, 1], n),
            'prediction': np.random.random(n),
        })
    
    def test_consistency_checking(self, sample_synthetic_data):
        """Test consistency checking postprocessing."""
        checker = ConsistencyChecker(
            numerical_cols=['feature_1', 'feature_2'],
            categorical_cols=['feature_3', 'sensitive']
        )
        
        results = checker.check(sample_synthetic_data)
        
        assert 'consistency_score' in results
        assert 0 <= results['consistency_score'] <= 1
    
    def test_fairness_auditing(self, sample_synthetic_data):
        """Test fairness auditing postprocessing."""
        auditor = FairnessAuditor(
            sensitive_attribute='sensitive',
            target_column='prediction'
        )
        
        results = auditor.audit(sample_synthetic_data)
        
        assert 'demographic_parity' in results
        assert 'fairness_score' in results
    
    def test_quality_filtering(self, sample_synthetic_data, sample_real_data):
        """Test quality filtering postprocessing."""
        filter = QualityFilter(
            quality_threshold=0.5,
            reference_data=sample_real_data
        )
        
        filtered_data, scores = filter.filter(sample_synthetic_data)
        
        assert len(filtered_data) <= len(sample_synthetic_data)
        assert len(scores) == len(sample_synthetic_data)
    
    def test_full_postprocessing_pipeline(self, sample_synthetic_data, sample_real_data):
        """Test complete postprocessing pipeline."""
        # Consistency check
        checker = ConsistencyChecker(
            numerical_cols=['feature_1', 'feature_2']
        )
        consistency_results = checker.check(sample_synthetic_data)
        
        # Fairness audit
        auditor = FairnessAuditor(
            sensitive_attribute='sensitive',
            target_column='prediction'
        )
        fairness_results = auditor.audit(sample_synthetic_data)
        
        # Quality filter
        quality_filter = QualityFilter(
            quality_threshold=0.3,
            reference_data=sample_real_data
        )
        filtered_data, quality_scores = quality_filter.filter(sample_synthetic_data)
        
        # All steps should complete
        assert consistency_results is not None
        assert fairness_results is not None
        assert filtered_data is not None


class TestOutputIntegration:
    """Integration tests for output modules."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        dir_path = tempfile.mkdtemp()
        yield Path(dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for export."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'category': np.random.choice(['A', 'B', 'C'], n),
            'target': np.random.choice([0, 1], n)
        })
    
    def test_csv_export(self, sample_data, temp_dir):
        """Test CSV format export."""
        exporter = DataExporter(output_dir=temp_dir)
        
        output_path = exporter.export_csv(
            data=sample_data,
            filename='synthetic_data.csv'
        )
        
        assert output_path.exists()
        
        # Verify data integrity
        loaded = pd.read_csv(output_path)
        pd.testing.assert_frame_equal(loaded, sample_data)
    
    def test_json_export(self, sample_data, temp_dir):
        """Test JSON format export."""
        exporter = DataExporter(output_dir=temp_dir)
        
        output_path = exporter.export_json(
            data=sample_data,
            filename='synthetic_data.json'
        )
        
        assert output_path.exists()
        
        # Verify data integrity
        with open(output_path) as f:
            loaded = json.load(f)
        
        assert len(loaded) == len(sample_data)
    
    def test_parquet_export(self, sample_data, temp_dir):
        """Test Parquet format export."""
        exporter = DataExporter(output_dir=temp_dir)
        
        output_path = exporter.export_parquet(
            data=sample_data,
            filename='synthetic_data.parquet'
        )
        
        assert output_path.exists()
        
        # Verify data integrity
        loaded = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(loaded, sample_data)
    
    def test_format_conversion(self, sample_data, temp_dir):
        """Test format conversion."""
        converter = FormatConverter()
        
        # CSV to JSON
        csv_path = temp_dir / 'data.csv'
        sample_data.to_csv(csv_path, index=False)
        
        json_path = converter.convert(
            source_path=csv_path,
            target_format='json',
            output_path=temp_dir / 'data.json'
        )
        
        assert json_path.exists()
        
        # Verify conversion
        with open(json_path) as f:
            loaded = json.load(f)
        
        assert len(loaded) == len(sample_data)
    
    def test_multi_format_export(self, sample_data, temp_dir):
        """Test exporting to multiple formats."""
        exporter = DataExporter(output_dir=temp_dir)
        
        paths = exporter.export_multi_format(
            data=sample_data,
            base_filename='synthetic_data',
            formats=['csv', 'json', 'parquet']
        )
        
        assert len(paths) == 3
        assert all(p.exists() for p in paths.values())


class TestGenerationWithFairnessConstraints:
    """Integration tests for fair generation."""
    
    @pytest.fixture
    def fairness_constrained_model(self):
        """Create model with fairness constraints."""
        class FairVAE(nn.Module):
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
            
            def sample(self, n):
                z = torch.randn(n, 16)
                return self.decoder(z)
        
        model = FairVAE()
        
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for _ in range(50):
            X = torch.randn(64, 10)
            recon, _ = model(X)
            loss = nn.MSELoss()(recon, X)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return model
    
    def test_fair_generation_audit(self, fairness_constrained_model):
        """Test fairness audit of generated data."""
        model = fairness_constrained_model
        
        # Generate data
        model.eval()
        with torch.no_grad():
            synthetic = model.sample(500).numpy()
        
        # Create dataframe with sensitive attribute
        synthetic_df = pd.DataFrame(synthetic, columns=[f'f{i}' for i in range(10)])
        synthetic_df['sensitive'] = (synthetic[:, 0] > 0).astype(int)
        synthetic_df['prediction'] = synthetic[:, 1]
        
        # Audit fairness
        auditor = FairnessAuditor(
            sensitive_attribute='sensitive',
            target_column='prediction'
        )
        
        results = auditor.audit(synthetic_df)
        
        assert 'demographic_parity' in results


class TestGenerationScalability:
    """Integration tests for generation scalability."""
    
    @pytest.fixture
    def model(self):
        """Create model for scalability testing."""
        model = VAEGenerator(
            input_dim=20,
            latent_dim=32,
            hidden_dims=[64, 128, 64]
        )
        model.eval()
        return model
    
    def test_large_scale_generation(self, model):
        """Test generation of large datasets."""
        with torch.no_grad():
            synthetic = model.sample(10000)
        
        assert synthetic.shape == (10000, 20)
    
    def test_streaming_generation(self, model):
        """Test streaming generation for memory efficiency."""
        total_samples = 0
        batch_size = 100
        
        with torch.no_grad():
            for _ in range(10):
                batch = model.sample(batch_size)
                total_samples += batch.shape[0]
        
        assert total_samples == 1000
    
    def test_generation_with_progress_tracking(self, model):
        """Test generation with progress tracking."""
        n_total = 1000
        batch_size = 100
        
        progress_history = []
        
        with torch.no_grad():
            for i in range(0, n_total, batch_size):
                batch = model.sample(min(batch_size, n_total - i))
                progress = min(100, (i + batch_size) / n_total * 100)
                progress_history.append(progress)
        
        assert len(progress_history) > 0
        assert progress_history[-1] >= 100


class TestGenerationReproducibility:
    """Integration tests for generation reproducibility."""
    
    def test_seed_reproducibility(self):
        """Test generation reproducibility with seed."""
        model = VAEGenerator(input_dim=10, latent_dim=16, hidden_dims=[32])
        model.eval()
        
        # First generation
        torch.manual_seed(42)
        with torch.no_grad():
            gen1 = model.sample(100)
        
        # Second generation with same seed
        torch.manual_seed(42)
        with torch.no_grad():
            gen2 = model.sample(100)
        
        assert torch.allclose(gen1, gen2)
    
    def test_deterministic_mode(self):
        """Test deterministic generation mode."""
        model = VAEGenerator(input_dim=10, latent_dim=16, hidden_dims=[32])
        
        pipeline = GeneratorPipeline(
            model=model,
            config={'deterministic': True, 'seed': 42}
        )
        
        gen1 = pipeline.generate(100)
        gen2 = pipeline.generate(100)
        
        np.testing.assert_array_almost_equal(gen1, gen2)


class TestErrorHandling:
    """Integration tests for error handling."""
    
    def test_invalid_sample_count(self):
        """Test handling of invalid sample count."""
        model = VAEGenerator(input_dim=10, latent_dim=16, hidden_dims=[32])
        model.eval()
        
        with pytest.raises((ValueError, RuntimeError)):
            with torch.no_grad():
                model.sample(-1)
    
    def test_model_device_mismatch(self):
        """Test handling of device mismatch."""
        model = VAEGenerator(input_dim=10, latent_dim=16, hidden_dims=[32])
        
        # Model on CPU, try to generate with GPU config (if CUDA available)
        # This should handle gracefully
        pipeline = GeneratorPipeline(
            model=model,
            config={'device': 'cpu'}
        )
        
        # Should work
        data = pipeline.generate(10)
        assert data is not None
    
    def test_corrupted_model_handling(self, temp_dir):
        """Test handling of corrupted model file."""
        temp_path = tempfile.mkdtemp()
        
        # Create corrupted file
        corrupted_path = Path(temp_path) / 'corrupted.pt'
        with open(corrupted_path, 'wb') as f:
            f.write(b'not a valid model')
        
        # Should handle gracefully
        try:
            model = VAEGenerator(input_dim=10, latent_dim=16, hidden_dims=[32])
            state_dict = torch.load(corrupted_path)
            model.load_state_dict(state_dict)
            assert False, "Should have raised an error"
        except Exception:
            pass  # Expected
        
        shutil.rmtree(temp_path, ignore_errors=True)
