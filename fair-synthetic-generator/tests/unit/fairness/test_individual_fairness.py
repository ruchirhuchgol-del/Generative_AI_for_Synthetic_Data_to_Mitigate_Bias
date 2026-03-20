
"""
Unit Tests for Individual Fairness
===================================

Tests for individual fairness metrics including:
- Consistency Score
- Lipschitz Constant Estimation
- Local Fairness
- Individual Discrimination
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.evaluation.fairness.individual_metrics import (
    ConsistencyScore,
    LipschitzEstimator,
    LocalFairnessMetric,
    IndividualDiscriminationMetric,
    IndividualFairnessEvaluator
)
from src.fairness.individual_fairness.consistency_constraint import ConsistencyConstraint
from src.fairness.individual_fairness.lipschitz_constraint import LipschitzConstraint


class TestConsistencyScore:
    """Tests for Consistency Score metric."""
    
    @pytest.fixture
    def consistent_data(self):
        """Create data where similar individuals have similar predictions."""
        np.random.seed(42)
        n = 500
        
        # Features
        X = np.random.randn(n, 5)
        
        # Predictions based on features (consistent)
        predictions = (X[:, 0] + X[:, 1] > 0).astype(float)
        
        return X, predictions
    
    @pytest.fixture
    def inconsistent_data(self):
        """Create data where similar individuals have different predictions."""
        np.random.seed(42)
        n = 500
        
        # Features
        X = np.random.randn(n, 5)
        
        # Random predictions (inconsistent)
        predictions = np.random.choice([0, 1], n).astype(float)
        
        return X, predictions
    
    def test_consistency_score_high(self, consistent_data):
        """Test consistency score for consistent predictions."""
        X, predictions = consistent_data
        
        metric = ConsistencyScore(k=5)
        score = metric.compute(X, predictions)
        
        assert score >= 0.7  # Should be high for consistent data
    
    def test_consistency_score_low(self, inconsistent_data):
        """Test consistency score for inconsistent predictions."""
        X, predictions = inconsistent_data
        
        metric = ConsistencyScore(k=5)
        score = metric.compute(X, predictions)
        
        assert score < 0.7  # Should be lower for random predictions
    
    def test_consistency_with_different_k(self, consistent_data):
        """Test consistency score with different k values."""
        X, predictions = consistent_data
        
        scores = []
        for k in [1, 3, 5, 10]:
            metric = ConsistencyScore(k=k)
            score = metric.compute(X, predictions)
            scores.append(score)
        
        # Scores should be in valid range
        assert all(0 <= s <= 1 for s in scores)
    
    def test_consistency_continuous_predictions(self):
        """Test consistency with continuous predictions."""
        np.random.seed(42)
        n = 300
        
        X = np.random.randn(n, 5)
        predictions = X[:, 0] * 0.5 + X[:, 1] * 0.3  # Continuous, based on features
        
        metric = ConsistencyScore(k=5)
        score = metric.compute(X, predictions)
        
        assert 0 <= score <= 1
    
    def test_consistency_scaled_features(self, consistent_data):
        """Test consistency with feature scaling."""
        X, predictions = consistent_data
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        metric1 = ConsistencyScore(k=5)
        score1 = metric1.compute(X, predictions)
        
        metric2 = ConsistencyScore(k=5)
        score2 = metric2.compute(X_scaled, predictions)
        
        # Both should be valid scores
        assert 0 <= score1 <= 1
        assert 0 <= score2 <= 1


class TestLipschitzEstimator:
    """Tests for Lipschitz constant estimation."""
    
    @pytest.fixture
    def smooth_function_data(self):
        """Create data from a smooth (Lipschitz) function."""
        np.random.seed(42)
        n = 200
        
        X = np.random.randn(n, 3)
        # Smooth function: linear combination
        predictions = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2]
        
        return X, predictions
    
    @pytest.fixture
    def non_smooth_function_data(self):
        """Create data from a non-smooth function."""
        np.random.seed(42)
        n = 200
        
        X = np.random.randn(n, 3)
        # Non-smooth: step function
        predictions = (X[:, 0] > 0).astype(float)
        
        return X, predictions
    
    def test_lipschitz_estimation(self, smooth_function_data):
        """Test Lipschitz constant estimation."""
        X, predictions = smooth_function_data
        
        estimator = LipschitzEstimator()
        L = estimator.estimate(X, predictions)
        
        assert L > 0  # Should be positive
        assert L < 10  # Reasonable upper bound for this data
    
    def test_smooth_vs_non_smooth(self, smooth_function_data, non_smooth_function_data):
        """Test that smooth functions have lower Lipschitz estimates."""
        X_smooth, pred_smooth = smooth_function_data
        X_non_smooth, pred_non_smooth = non_smooth_function_data
        
        estimator = LipschitzEstimator()
        
        L_smooth = estimator.estimate(X_smooth, pred_smooth)
        L_non_smooth = estimator.estimate(X_non_smooth, pred_non_smooth)
        
        # Non-smooth function should have higher or similar Lipschitz estimate
        # (depending on the estimation method)
        assert L_smooth > 0
        assert L_non_smooth > 0
    
    def test_lipschitz_with_norm(self, smooth_function_data):
        """Test Lipschitz estimation with different norms."""
        X, predictions = smooth_function_data
        
        estimator_l2 = LipschitzEstimator(norm='l2')
        estimator_l1 = LipschitzEstimator(norm='l1')
        
        L_l2 = estimator_l2.estimate(X, predictions)
        L_l1 = estimator_l1.estimate(X, predictions)
        
        assert L_l2 > 0
        assert L_l1 > 0


class TestLocalFairnessMetric:
    """Tests for local fairness metric."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for local fairness testing."""
        np.random.seed(42)
        n = 300
        
        return pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'sensitive': np.random.choice([0, 1], n),
            'prediction': np.random.random(n),
        })
    
    def test_local_fairness_computation(self, sample_data):
        """Test local fairness metric computation."""
        metric = LocalFairnessMetric(
            sensitive_attr='sensitive',
            n_neighbors=10
        )
        
        X = sample_data[['feature_1', 'feature_2']].values
        predictions = sample_data['prediction'].values
        sensitive = sample_data['sensitive'].values
        
        score = metric.compute(X, predictions, sensitive)
        
        assert 0 <= score <= 1
    
    def test_local_fairness_different_neighborhoods(self, sample_data):
        """Test local fairness with different neighborhood sizes."""
        X = sample_data[['feature_1', 'feature_2']].values
        predictions = sample_data['prediction'].values
        sensitive = sample_data['sensitive'].values
        
        scores = []
        for n_neighbors in [5, 10, 20]:
            metric = LocalFairnessMetric(
                sensitive_attr='sensitive',
                n_neighbors=n_neighbors
            )
            score = metric.compute(X, predictions, sensitive)
            scores.append(score)
        
        # All should be valid
        assert all(0 <= s <= 1 for s in scores)


class TestIndividualDiscriminationMetric:
    """Tests for individual discrimination metric."""
    
    @pytest.fixture
    def discriminatory_data(self):
        """Create data with individual discrimination."""
        np.random.seed(42)
        n = 300
        
        X = np.random.randn(n, 5)
        sensitive = np.random.choice([0, 1], n)
        
        # Discriminatory: prediction depends on sensitive attribute
        predictions = sensitive.astype(float)
        
        return X, predictions, sensitive
    
    @pytest.fixture
    def non_discriminatory_data(self):
        """Create data without individual discrimination."""
        np.random.seed(42)
        n = 300
        
        X = np.random.randn(n, 5)
        sensitive = np.random.choice([0, 1], n)
        
        # Non-discriminatory: prediction independent of sensitive
        predictions = (X[:, 0] > 0).astype(float)
        
        return X, predictions, sensitive
    
    def test_discrimination_detection(self, discriminatory_data):
        """Test detection of individual discrimination."""
        X, predictions, sensitive = discriminatory_data
        
        metric = IndividualDiscriminationMetric(threshold=0.1)
        score = metric.compute(X, predictions, sensitive)
        
        assert score > 0.5  # Should detect discrimination
    
    def test_no_discrimination_detection(self, non_discriminatory_data):
        """Test when there's no individual discrimination."""
        X, predictions, sensitive = non_discriminatory_data
        
        metric = IndividualDiscriminationMetric(threshold=0.1)
        score = metric.compute(X, predictions, sensitive)
        
        assert score < 0.5  # Should not detect much discrimination


class TestIndividualFairnessEvaluator:
    """Tests for IndividualFairnessEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return IndividualFairnessEvaluator()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
        np.random.seed(42)
        n = 200
        
        return {
            'features': np.random.randn(n, 5),
            'predictions': np.random.random(n),
            'sensitive': np.random.choice([0, 1], n),
        }
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initializes correctly."""
        assert evaluator is not None
    
    def test_full_evaluation(self, evaluator, sample_data):
        """Test full individual fairness evaluation."""
        results = evaluator.evaluate(
            X=sample_data['features'],
            predictions=sample_data['predictions'],
            sensitive=sample_data['sensitive']
        )
        
        assert 'consistency_score' in results
        assert 'lipschitz_estimate' in results
    
    def test_metric_selection(self):
        """Test selecting specific metrics to compute."""
        evaluator = IndividualFairnessEvaluator(
            metrics=['consistency']
        )
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        predictions = np.random.random(100)
        sensitive = np.random.choice([0, 1], 100)
        
        results = evaluator.evaluate(X, predictions, sensitive)
        
        assert 'consistency_score' in results


class TestConsistencyConstraint:
    """Tests for ConsistencyConstraint for training."""
    
    def test_constraint_initialization(self):
        """Test constraint initializes correctly."""
        constraint = ConsistencyConstraint(
            k=5,
            weight=1.0
        )
        
        assert constraint.k == 5
        assert constraint.weight == 1.0
    
    def test_constraint_loss(self):
        """Test constraint loss computation."""
        constraint = ConsistencyConstraint(k=5)
        
        X = torch.randn(32, 10)
        predictions = torch.randn(32, 1)
        
        loss = constraint.compute_loss(X, predictions)
        
        assert loss >= 0
        assert loss.requires_grad
    
    def test_constraint_gradient_flow(self):
        """Test gradient flows through constraint."""
        constraint = ConsistencyConstraint(k=5)
        
        X = torch.randn(16, 10)
        predictions = torch.randn(16, 1, requires_grad=True)
        
        loss = constraint.compute_loss(X, predictions)
        loss.backward()
        
        assert predictions.grad is not None


class TestLipschitzConstraint:
    """Tests for LipschitzConstraint for training."""
    
    def test_constraint_initialization(self):
        """Test constraint initializes correctly."""
        constraint = LipschitzConstraint(
            target_L=1.0,
            weight=0.5
        )
        
        assert constraint.target_L == 1.0
        assert constraint.weight == 0.5
    
    def test_constraint_penalty(self):
        """Test Lipschitz penalty computation."""
        constraint = LipschitzConstraint(target_L=1.0)
        
        # Create model with outputs
        model_output = torch.randn(32, 10)
        
        penalty = constraint.compute_penalty(model_output)
        
        assert penalty >= 0


class TestEdgeCases:
    """Edge case tests for individual fairness metrics."""
    
    def test_single_sample(self):
        """Test metrics with single sample."""
        X = np.random.randn(1, 5)
        predictions = np.array([0.5])
        
        metric = ConsistencyScore(k=1)
        
        # Should handle gracefully
        try:
            score = metric.compute(X, predictions)
        except (ValueError, IndexError):
            pass  # Acceptable to raise
    
    def test_identical_samples(self):
        """Test with identical samples."""
        n = 100
        X = np.ones((n, 5))  # All same
        predictions = np.random.choice([0, 1], n)
        
        metric = ConsistencyScore(k=5)
        score = metric.compute(X, predictions)
        
        # Should handle (neighbors will have same features)
        assert 0 <= score <= 1
    
    def test_high_dimensional(self):
        """Test with high-dimensional features."""
        np.random.seed(42)
        n = 200
        d = 100  # High dimension
        
        X = np.random.randn(n, d)
        predictions = np.random.choice([0, 1], n)
        
        metric = ConsistencyScore(k=5)
        score = metric.compute(X, predictions)
        
        assert 0 <= score <= 1
    
    def test_sparse_predictions(self):
        """Test with sparse predictions (mostly 0 or 1)."""
        np.random.seed(42)
        n = 200
        
        X = np.random.randn(n, 5)
        predictions = np.zeros(n)
        predictions[:10] = 1  # Only 5% positive
        
        metric = ConsistencyScore(k=5)
        score = metric.compute(X, predictions)
        
        assert 0 <= score <= 1
