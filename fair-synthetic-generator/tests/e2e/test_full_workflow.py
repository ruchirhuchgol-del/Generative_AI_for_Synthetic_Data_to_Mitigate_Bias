"""
End-to-End Tests for Full Workflow
===================================

Comprehensive end-to-end tests covering the complete workflow:
- Data loading and preprocessing
- Model training with fairness constraints
- Synthetic data generation
- Evaluation and reporting
- Export and deployment

These tests validate the entire pipeline works correctly together.
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
import time
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.generators.vae_generator import VAEGenerator
from src.models.architectures.debiased_vae import DebiasedVAE
from src.data.preprocessing.tabular_preprocessor import TabularPreprocessor
from src.training.trainer import Trainer
from src.training.callbacks.checkpoint_callback import CheckpointCallback
from src.synthesis.generator_pipeline import GeneratorPipeline
from src.evaluation.fairness.group_metrics import GroupFairnessEvaluator
from src.evaluation.fidelity.statistical_similarity import StatisticalSimilarityEvaluator
from src.evaluation.dashboard.report_generator import ComprehensiveReportGenerator
from src.fairness.constraints.demographic_parity import DemographicParity


@pytest.mark.e2e
class TestFullWorkflow:
    """End-to-end tests for complete fair synthetic data workflow."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for full workflow test."""
        workspace = tempfile.mkdtemp()
        workspace_path = Path(workspace)
        
        # Create directory structure
        (workspace_path / 'data' / 'raw').mkdir(parents=True)
        (workspace_path / 'data' / 'processed').mkdir(parents=True)
        (workspace_path / 'data' / 'synthetic').mkdir(parents=True)
        (workspace_path / 'models').mkdir(parents=True)
        (workspace_path / 'reports').mkdir(parents=True)
        (workspace_path / 'checkpoints').mkdir(parents=True)
        
        yield workspace_path
        
        shutil.rmtree(workspace, ignore_errors=True)
    
    @pytest.fixture
    def raw_dataset(self, temp_workspace):
        """Create and save raw dataset."""
        np.random.seed(42)
        n = 2000
        
        # Create realistic dataset with bias
        df = pd.DataFrame({
            'age': np.random.randint(18, 70, n),
            'gender': np.random.choice(['Male', 'Female'], n, p=[0.55, 0.45]),
            'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n),
            'income': np.random.exponential(50000, n) + 20000,
            'education_years': np.random.randint(8, 20, n),
            'credit_score': np.random.normal(700, 50, n).clip(300, 850),
            'employment_years': np.random.exponential(5, n).clip(0, 40),
        })
        
        # Create biased outcome
        base_prob = 0.3
        df['approval_prob'] = base_prob + (
            (df['age'] - 40) / 100 +
            (df['education_years'] - 12) / 50 +
            (df['credit_score'] - 700) / 1000
        )
        # Introduce bias
        df.loc[df['gender'] == 'Female', 'approval_prob'] -= 0.1
        df.loc[df['race'] == 'Black', 'approval_prob'] -= 0.08
        df['approval_prob'] = df['approval_prob'].clip(0.1, 0.9)
        df['approved'] = (np.random.random(n) < df['approval_prob']).astype(int)
        df = df.drop('approval_prob', axis=1)
        
        # Save
        raw_path = temp_workspace / 'data' / 'raw' / 'raw_data.csv'
        df.to_csv(raw_path, index=False)
        
        return df, raw_path
    
    def test_complete_workflow(self, temp_workspace, raw_dataset):
        """Test complete fair synthetic data generation workflow."""
        df, raw_path = raw_dataset
        
        # ==========================================
        # STEP 1: Data Preprocessing
        # ==========================================
        print("\n[Step 1] Data Preprocessing")
        
        preprocessor = TabularPreprocessor(
            numerical_cols=['age', 'income', 'education_years', 'credit_score', 'employment_years'],
            categorical_cols=['gender', 'race'],
            target_col='approved'
        )
        
        X = preprocessor.fit_transform(df)
        
        # Save preprocessed data
        processed_path = temp_workspace / 'data' / 'processed' / 'processed.npz'
        np.savez(processed_path, X=X)
        
        assert X.shape[0] == len(df)
        print(f"  - Preprocessed shape: {X.shape}")
        
        # ==========================================
        # STEP 2: Model Training
        # ==========================================
        print("\n[Step 2] Model Training")
        
        model = DebiasedVAE(
            input_dim=X.shape[1],
            latent_dim=32,
            hidden_dims=[64, 128, 64],
            sensitive_dim=2,  # gender + race encoded
            lambda_fairness=0.3
        )
        
        trainer = Trainer(
            model=model,
            config={
                'epochs': 20,
                'batch_size': 128,
                'learning_rate': 1e-3,
                'device': 'cpu'
            }
        )
        
        history = trainer.fit(X, epochs=20)
        
        # Save model
        model_path = temp_workspace / 'models' / 'fair_vae.pt'
        torch.save(model.state_dict(), model_path)
        
        assert 'loss' in history
        print(f"  - Training epochs: {len(history['loss'])}")
        print(f"  - Final loss: {history['loss'][-1]:.4f}")
        
        # ==========================================
        # STEP 3: Synthetic Data Generation
        # ==========================================
        print("\n[Step 3] Synthetic Data Generation")
        
        model.eval()
        pipeline = GeneratorPipeline(model=model, config={'device': 'cpu'})
        
        synthetic_array = pipeline.generate(n_samples=1000)
        
        # Inverse transform
        synthetic_df = preprocessor.inverse_transform(synthetic_array)
        
        # Save synthetic data
        synthetic_path = temp_workspace / 'data' / 'synthetic' / 'synthetic_data.csv'
        synthetic_df.to_csv(synthetic_path, index=False)
        
        assert len(synthetic_df) == 1000
        print(f"  - Generated samples: {len(synthetic_df)}")
        
        # ==========================================
        # STEP 4: Evaluation
        # ==========================================
        print("\n[Step 4] Evaluation")
        
        # Fairness evaluation
        fairness_evaluator = GroupFairnessEvaluator(
            sensitive_attributes=['gender', 'race'],
            target_column='approved'
        )
        fairness_results = fairness_evaluator.evaluate(
            real_data=df,
            synthetic_data=synthetic_df
        )
        
        # Fidelity evaluation
        fidelity_evaluator = StatisticalSimilarityEvaluator()
        fidelity_results = fidelity_evaluator.evaluate(
            real_data=df.select_dtypes(include=[np.number]),
            synthetic_data=synthetic_df.select_dtypes(include=[np.number])
        )
        
        print(f"  - Fairness score: {fairness_results.get('overall_fairness_score', 'N/A')}")
        print(f"  - Fidelity score: {fidelity_results.get('overall_fidelity_score', 'N/A')}")
        
        # ==========================================
        # STEP 5: Report Generation
        # ==========================================
        print("\n[Step 5] Report Generation")
        
        report_generator = ComprehensiveReportGenerator(
            output_dir=temp_workspace / 'reports'
        )
        
        all_results = {
            'fairness': fairness_results,
            'fidelity': fidelity_results,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'real_samples': len(df),
                'synthetic_samples': len(synthetic_df)
            }
        }
        
        report_path = report_generator.generate_html(
            results=all_results,
            filename='evaluation_report.html'
        )
        
        assert report_path.exists()
        print(f"  - Report saved: {report_path}")
        
        print("\n✅ Complete workflow test passed!")
    
    def test_workflow_with_checkpointing(self, temp_workspace, raw_dataset):
        """Test workflow with training checkpointing."""
        df, raw_path = raw_dataset
        
        # Preprocess
        preprocessor = TabularPreprocessor(
            numerical_cols=['age', 'income', 'education_years', 'credit_score', 'employment_years'],
            categorical_cols=['gender', 'race'],
            target_col='approved'
        )
        X = preprocessor.fit_transform(df)
        
        # Model with checkpointing
        model = VAEGenerator(
            input_dim=X.shape[1],
            latent_dim=32,
            hidden_dims=[64, 128, 64]
        )
        
        checkpoint_callback = CheckpointCallback(
            checkpoint_dir=temp_workspace / 'checkpoints',
            save_best_only=True,
            monitor='loss'
        )
        
        trainer = Trainer(
            model=model,
            config={'epochs': 10, 'batch_size': 128, 'device': 'cpu'},
            callbacks=[checkpoint_callback]
        )
        
        trainer.fit(X, epochs=10)
        
        # Check checkpoints exist
        checkpoints = list((temp_workspace / 'checkpoints').glob('*.pt'))
        assert len(checkpoints) > 0
    
    def test_workflow_with_fairness_constraints(self, temp_workspace, raw_dataset):
        """Test workflow with explicit fairness constraints."""
        df, raw_path = raw_dataset
        
        # Preprocess
        preprocessor = TabularPreprocessor(
            numerical_cols=['age', 'income', 'education_years', 'credit_score', 'employment_years'],
            categorical_cols=['gender', 'race'],
            target_col='approved'
        )
        X = preprocessor.fit_transform(df)
        
        # Create fairness constraint
        constraint = DemographicParity(
            sensitive_attribute='gender',
            threshold=0.1
        )
        
        # Model with fairness
        model = DebiasedVAE(
            input_dim=X.shape[1],
            latent_dim=32,
            hidden_dims=[64, 128, 64],
            sensitive_dim=2,
            lambda_fairness=0.5
        )
        
        trainer = Trainer(
            model=model,
            config={'epochs': 15, 'batch_size': 128, 'device': 'cpu'}
        )
        
        history = trainer.fit(X, epochs=15)
        
        assert history is not None


@pytest.mark.e2e
class TestWorkflowWithMultipleModalities:
    """End-to-end tests for multimodal workflow."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        workspace = tempfile.mkdtemp()
        yield Path(workspace)
        shutil.rmtree(workspace, ignore_errors=True)
    
    def test_multimodal_workflow(self, temp_workspace):
        """Test workflow with multimodal data."""
        np.random.seed(42)
        n = 500
        
        # Create multimodal data
        tabular = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'sensitive': np.random.choice([0, 1], n)
        })
        
        text_features = np.random.randn(n, 32)  # Pre-extracted features
        
        # Combine for model
        combined_features = np.concatenate([
            tabular[['feature_1', 'feature_2']].values,
            text_features
        ], axis=1)
        
        # Train simple model
        class SimpleMultimodalVAE(nn.Module):
            def __init__(self, input_dim, latent_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, latent_dim * 2)
                )
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, input_dim)
                )
                self.latent_dim = latent_dim
            
            def forward(self, x):
                h = self.encoder(x)
                mu, log_var = h[:, :self.latent_dim], h[:, self.latent_dim:]
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
                return self.decoder(z), mu, log_var
            
            def sample(self, n):
                z = torch.randn(n, self.latent_dim)
                return self.decoder(z)
        
        model = SimpleMultimodalVAE(input_dim=34, latent_dim=16)
        
        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        X_tensor = torch.FloatTensor(combined_features)
        
        for _ in range(20):
            optimizer.zero_grad()
            recon, mu, log_var = model(X_tensor)
            loss = nn.MSELoss()(recon, X_tensor)
            loss.backward()
            optimizer.step()
        
        # Generate
        model.eval()
        with torch.no_grad():
            synthetic = model.sample(100).numpy()
        
        assert synthetic.shape == (100, 34)


@pytest.mark.e2e
class TestWorkflowPerformance:
    """End-to-end tests for workflow performance."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        workspace = tempfile.mkdtemp()
        yield Path(workspace)
        shutil.rmtree(workspace, ignore_errors=True)
    
    def test_workflow_timing(self, temp_workspace):
        """Test workflow execution timing."""
        np.random.seed(42)
        
        # Create data
        n = 1000
        df = pd.DataFrame(np.random.randn(n, 10))
        df['sensitive'] = np.random.choice([0, 1], n)
        
        # Time preprocessing
        start = time.time()
        preprocessor = TabularPreprocessor(
            numerical_cols=list(range(10)),
            categorical_cols=['sensitive']
        )
        X = preprocessor.fit_transform(df)
        preprocess_time = time.time() - start
        
        # Time training
        start = time.time()
        model = VAEGenerator(input_dim=X.shape[1], latent_dim=16, hidden_dims=[32])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for _ in range(10):
            optimizer.zero_grad()
            recon, mu, log_var = model(torch.FloatTensor(X))
            loss = nn.MSELoss()(recon, torch.FloatTensor(X))
            loss.backward()
            optimizer.step()
        
        train_time = time.time() - start
        
        # Time generation
        start = time.time()
        model.eval()
        with torch.no_grad():
            synthetic = model.sample(500)
        gen_time = time.time() - start
        
        # Log timings
        print(f"\nPerformance Metrics:")
        print(f"  Preprocessing: {preprocess_time:.3f}s")
        print(f"  Training: {train_time:.3f}s")
        print(f"  Generation: {gen_time:.3f}s")
        
        # Basic timing assertions
        assert preprocess_time < 5.0  # Should be fast
        assert train_time < 30.0
        assert gen_time < 1.0


@pytest.mark.e2e
class TestWorkflowRobustness:
    """End-to-end tests for workflow robustness."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        workspace = tempfile.mkdtemp()
        yield Path(workspace)
        shutil.rmtree(workspace, ignore_errors=True)
    
    def test_workflow_with_missing_values(self, temp_workspace):
        """Test workflow handles missing values."""
        np.random.seed(42)
        n = 500
        
        # Create data with missing values
        df = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'sensitive': np.random.choice([0, 1, np.nan], n, p=[0.45, 0.45, 0.1])
        })
        
        # Should handle missing values
        preprocessor = TabularPreprocessor(
            numerical_cols=['feature_1', 'feature_2'],
            categorical_cols=['sensitive'],
            handle_missing='impute'
        )
        
        X = preprocessor.fit_transform(df)
        
        assert X is not None
        assert not np.isnan(X).any()
    
    def test_workflow_with_imbalanced_classes(self, temp_workspace):
        """Test workflow with imbalanced classes."""
        np.random.seed(42)
        n = 500
        
        # Create imbalanced data
        df = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'sensitive': np.random.choice([0, 1], n),
            'target': np.random.choice([0, 1], n, p=[0.9, 0.1])  # Imbalanced
        })
        
        preprocessor = TabularPreprocessor(
            numerical_cols=['feature_1', 'feature_2'],
            categorical_cols=['sensitive', 'target']
        )
        
        X = preprocessor.fit_transform(df)
        
        # Should complete successfully
        assert X.shape[0] == n
    
    def test_workflow_with_large_features(self, temp_workspace):
        """Test workflow with many features."""
        np.random.seed(42)
        n = 500
        n_features = 100
        
        df = pd.DataFrame(np.random.randn(n, n_features))
        df['sensitive'] = np.random.choice([0, 1], n)
        
        preprocessor = TabularPreprocessor(
            numerical_cols=list(range(n_features)),
            categorical_cols=['sensitive']
        )
        
        X = preprocessor.fit_transform(df)
        
        # Train model
        model = VAEGenerator(
            input_dim=X.shape[1],
            latent_dim=32,
            hidden_dims=[64, 128, 64]
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for _ in range(5):
            optimizer.zero_grad()
            recon, mu, log_var = model(torch.FloatTensor(X))
            loss = nn.MSELoss()(recon, torch.FloatTensor(X))
            loss.backward()
            optimizer.step()
        
        # Generate
        model.eval()
        with torch.no_grad():
            synthetic = model.sample(100)
        
        assert synthetic.shape == (100, X.shape[1])


@pytest.mark.e2e
class TestWorkflowIntegration:
    """End-to-end integration tests with all components."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        workspace = tempfile.mkdtemp()
        yield Path(workspace)
        shutil.rmtree(workspace, ignore_errors=True)
    
    def test_api_integration(self, temp_workspace):
        """Test workflow with API integration."""
        # This would test the full API workflow
        # For now, just verify imports work
        
        from src.api.app import app
        
        assert app is not None
    
    def test_config_driven_workflow(self, temp_workspace):
        """Test workflow driven by configuration file."""
        # Create config file
        config = {
            'data': {
                'numerical_cols': ['feature_1', 'feature_2'],
                'categorical_cols': ['sensitive'],
                'target_col': 'target'
            },
            'model': {
                'type': 'vae',
                'latent_dim': 16,
                'hidden_dims': [32, 64, 32]
            },
            'training': {
                'epochs': 10,
                'batch_size': 64,
                'learning_rate': 0.001
            },
            'generation': {
                'n_samples': 500
            }
        }
        
        config_path = temp_workspace / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Load config and run workflow
        with open(config_path) as f:
            loaded_config = json.load(f)
        
        # Create data based on config
        np.random.seed(42)
        n = 300
        
        df = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'sensitive': np.random.choice([0, 1], n),
            'target': np.random.choice([0, 1], n)
        })
        
        # Preprocess
        preprocessor = TabularPreprocessor(
            numerical_cols=loaded_config['data']['numerical_cols'],
            categorical_cols=loaded_config['data']['categorical_cols']
        )
        X = preprocessor.fit_transform(df)
        
        # Create model
        model = VAEGenerator(
            input_dim=X.shape[1],
            latent_dim=loaded_config['model']['latent_dim'],
            hidden_dims=loaded_config['model']['hidden_dims']
        )
        
        # Train
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=loaded_config['training']['learning_rate']
        )
        
        for _ in range(loaded_config['training']['epochs']):
            optimizer.zero_grad()
            recon, mu, log_var = model(torch.FloatTensor(X))
            loss = nn.MSELoss()(recon, torch.FloatTensor(X))
            loss.backward()
            optimizer.step()
        
        # Generate
        model.eval()
        with torch.no_grad():
            synthetic = model.sample(loaded_config['generation']['n_samples'])
        
        assert synthetic.shape[0] == loaded_config['generation']['n_samples']


@pytest.mark.e2e
@pytest.mark.slow
class TestWorkflowAtScale:
    """End-to-end tests for scaled workflows."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace."""
        workspace = tempfile.mkdtemp()
        yield Path(workspace)
        shutil.rmtree(workspace, ignore_errors=True)
    
    @pytest.mark.slow
    def test_large_scale_workflow(self, temp_workspace):
        """Test workflow with larger dataset."""
        np.random.seed(42)
        n = 5000
        
        # Create large dataset
        df = pd.DataFrame({
            **{f'feature_{i}': np.random.randn(n) for i in range(20)},
            'sensitive': np.random.choice([0, 1], n),
            'target': np.random.choice([0, 1], n)
        })
        
        # Preprocess
        preprocessor = TabularPreprocessor(
            numerical_cols=[f'feature_{i}' for i in range(20)],
            categorical_cols=['sensitive', 'target']
        )
        X = preprocessor.fit_transform(df)
        
        # Train
        model = VAEGenerator(
            input_dim=X.shape[1],
            latent_dim=32,
            hidden_dims=[64, 128, 64]
        )
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        for epoch in range(20):
            optimizer.zero_grad()
            recon, mu, log_var = model(torch.FloatTensor(X))
            loss = nn.MSELoss()(recon, torch.FloatTensor(X))
            loss.backward()
            optimizer.step()
        
        # Generate
        model.eval()
        with torch.no_grad():
            synthetic = model.sample(2000)
        
        assert synthetic.shape == (2000, X.shape[1])
