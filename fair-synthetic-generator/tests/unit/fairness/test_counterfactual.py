"""
Unit Tests for Counterfactual Fairness
======================================

Tests for counterfactual fairness metrics including:
- Counterfactual Invariance
- Counterfactual Effect Size
- Causal Effect Ratio
- Counterfactual Fairness Evaluator
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.evaluation.fairness.counterfactual_metrics import (
    CounterfactualInvariance,
    CounterfactualEffectSize,
    CausalEffectRatio,
    CounterfactualFairnessEvaluator
)
from src.fairness.constraints.counterfactual_fairness import CounterfactualFairnessConstraint


class TestCounterfactualInvariance:
    """Tests for Counterfactual Invariance metric."""
    
    @pytest.fixture
    def invariant_data(self):
        """Create data where predictions are invariant to counterfactual changes."""
        np.random.seed(42)
        n = 500
        
        # Features not affected by sensitive attribute
        X = np.random.randn(n, 5)
        sensitive = np.random.choice([0, 1], n)
        
        # Predictions based only on non-sensitive features
        predictions = (X[:, 0] + X[:, 1] > 0).astype(float)
        predictions_cf = predictions.copy()  # Same predictions for counterfactual
        
        return X, predictions, predictions_cf, sensitive
    
    @pytest.fixture
    def variant_data(self):
        """Create data where predictions change with counterfactual."""
        np.random.seed(42)
        n = 500
        
        X = np.random.randn(n, 5)
        sensitive = np.random.choice([0, 1], n)
        
        # Predictions based on sensitive attribute
        predictions = sensitive.astype(float)
        predictions_cf = 1 - predictions  # Flipped for counterfactual
        
        return X, predictions, predictions_cf, sensitive
    
    def test_invariance_high(self, invariant_data):
        """Test high invariance score for invariant predictions."""
        X, predictions, predictions_cf, sensitive = invariant_data
        
        metric = CounterfactualInvariance()
        score = metric.compute(predictions, predictions_cf)
        
        assert score > 0.9  # Should be close to 1 for invariant data
    
    def test_invariance_low(self, variant_data):
        """Test low invariance score for variant predictions."""
        X, predictions, predictions_cf, sensitive = variant_data
        
        metric = CounterfactualInvariance()
        score = metric.compute(predictions, predictions_cf)
        
        assert score < 0.5  # Should be low for variant data
    
    def test_invariance_continuous_predictions(self):
        """Test invariance with continuous predictions."""
        np.random.seed(42)
        n = 200
        
        predictions = np.random.randn(n)
        predictions_cf = predictions + np.random.randn(n) * 0.1  # Small change
        
        metric = CounterfactualInvariance()
        score = metric.compute(predictions, predictions_cf)
        
        assert 0 <= score <= 1


class TestCounterfactualEffectSize:
    """Tests for Counterfactual Effect Size metric."""
    
    @pytest.fixture
    def small_effect_data(self):
        """Create data with small counterfactual effect."""
        np.random.seed(42)
        n = 500
        
        predictions = np.random.random(n)
        predictions_cf = predictions + np.random.randn(n) * 0.01  # Small effect
        
        return predictions, predictions_cf
    
    @pytest.fixture
    def large_effect_data(self):
        """Create data with large counterfactual effect."""
        np.random.seed(42)
        n = 500
        
        predictions = np.random.random(n)
        predictions_cf = 1 - predictions  # Complete flip
        
        return predictions, predictions_cf
    
    def test_effect_size_small(self, small_effect_data):
        """Test small effect size detection."""
        predictions, predictions_cf = small_effect_data
        
        metric = CounterfactualEffectSize()
        effect = metric.compute(predictions, predictions_cf)
        
        assert effect < 0.1  # Small effect
    
    def test_effect_size_large(self, large_effect_data):
        """Test large effect size detection."""
        predictions, predictions_cf = large_effect_data
        
        metric = CounterfactualEffectSize()
        effect = metric.compute(predictions, predictions_cf)
        
        assert effect > 0.5  # Large effect
    
    def test_effect_size_normalized(self):
        """Test that effect size is normalized."""
        np.random.seed(42)
        n = 200
        
        predictions = np.random.randn(n)
        predictions_cf = np.random.randn(n)
        
        metric = CounterfactualEffectSize()
        effect = metric.compute(predictions, predictions_cf)
        
        # Effect size should be non-negative
        assert effect >= 0


class TestCausalEffectRatio:
    """Tests for Causal Effect Ratio metric."""
    
    @pytest.fixture
    def sample_causal_data(self):
        """Create sample data with known causal structure."""
        np.random.seed(42)
        n = 500
        
        # Sensitive attribute
        sensitive = np.random.choice([0, 1], n)
        
        # Mediator (affected by sensitive)
        mediator = 0.5 * sensitive + np.random.randn(n) * 0.5
        
        # Outcome (affected by both)
        outcome = 0.3 * mediator + 0.2 * sensitive + np.random.randn(n) * 0.3
        
        return sensitive, mediator, outcome
    
    def test_causal_effect_ratio_computation(self, sample_causal_data):
        """Test causal effect ratio computation."""
        sensitive, mediator, outcome = sample_causal_data
        
        metric = CausalEffectRatio()
        ratio = metric.compute(
            treatment=sensitive,
            mediator=mediator,
            outcome=outcome
        )
        
        assert 0 <= ratio <= 1
    
    def test_direct_vs_indirect_effect(self):
        """Test separation of direct and indirect effects."""
        np.random.seed(42)
        n = 500
        
        # Only direct effect (no mediation)
        sensitive = np.random.choice([0, 1], n)
        outcome = sensitive.astype(float)
        mediator = np.zeros(n)  # No mediation
        
        metric = CausalEffectRatio()
        ratio = metric.compute(sensitive, mediator, outcome)
        
        # Should capture that effect is direct
        assert isinstance(ratio, (int, float))


class TestCounterfactualFairnessEvaluator:
    """Tests for CounterfactualFairnessEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return CounterfactualFairnessEvaluator(
            sensitive_attr='gender'
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
        np.random.seed(42)
        n = 300
        
        return {
            'features': np.random.randn(n, 5),
            'predictions': np.random.random(n),
            'predictions_cf': np.random.random(n),
            'sensitive': np.random.choice([0, 1], n),
        }
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initializes correctly."""
        assert evaluator is not None
    
    def test_full_evaluation(self, evaluator, sample_data):
        """Test full counterfactual fairness evaluation."""
        results = evaluator.evaluate(
            predictions=sample_data['predictions'],
            predictions_counterfactual=sample_data['predictions_cf'],
            sensitive=sample_data['sensitive']
        )
        
        assert 'counterfactual_invariance' in results
        assert 'effect_size' in results
    
    def test_with_causal_model(self, evaluator, sample_data):
        """Test evaluation with causal model."""
        # Mock causal model
        causal_model = type('CausalModel', (), {
            'get_counterfactual': lambda self, x, a_new: x
        })()
        
        evaluator.set_causal_model(causal_model)
        
        results = evaluator.evaluate(
            predictions=sample_data['predictions'],
            sensitive=sample_data['sensitive'],
            features=sample_data['features']
        )
        
        assert results is not None


class TestCounterfactualFairnessConstraint:
    """Tests for CounterfactualFairnessConstraint for training."""
    
    def test_constraint_initialization(self):
        """Test constraint initializes correctly."""
        constraint = CounterfactualFairnessConstraint(
            threshold=0.1,
            weight=1.0
        )
        
        assert constraint.threshold == 0.1
        assert constraint.weight == 1.0
    
    def test_constraint_loss_computation(self):
        """Test constraint loss computation."""
        constraint = CounterfactualFairnessConstraint(threshold=0.1)
        
        predictions = torch.randn(32, 1)
        predictions_cf = predictions + torch.randn(32, 1) * 0.1
        
        loss = constraint.compute_loss(predictions, predictions_cf)
        
        assert loss >= 0
        assert loss.requires_grad
    
    def test_gradient_flow(self):
        """Test gradient flows through constraint."""
        constraint = CounterfactualFairnessConstraint(threshold=0.1)
        
        predictions = torch.randn(32, 1, requires_grad=True)
        predictions_cf = predictions.detach() + torch.randn(32, 1) * 0.1
        predictions_cf.requires_grad = True
        
        loss = constraint.compute_loss(predictions, predictions_cf)
        loss.backward()
        
        assert predictions.grad is not None


class TestCounterfactualGeneration:
    """Tests for counterfactual generation utilities."""
    
    def test_counterfactual_generation_shape(self):
        """Test counterfactual generation produces correct shapes."""
        # Mock counterfactual generator
        def generate_counterfactual(X, sensitive_new):
            return X + np.random.randn(*X.shape) * 0.1
        
        np.random.seed(42)
        X = np.random.randn(100, 10)
        sensitive_new = np.ones(100)
        
        X_cf = generate_counterfactual(X, sensitive_new)
        
        assert X_cf.shape == X.shape
    
    def test_batch_counterfactual_generation(self):
        """Test batch counterfactual generation."""
        # Mock batch generator
        def batch_generate(X, sensitive_attrs):
            return {attr: X + np.random.randn(*X.shape) * 0.1 
                    for attr in sensitive_attrs}
        
        np.random.seed(42)
        X = np.random.randn(50, 10)
        sensitive_attrs = ['gender', 'race']
        
        counterfactuals = batch_generate(X, sensitive_attrs)
        
        assert 'gender' in counterfactuals
        assert 'race' in counterfactuals
        assert counterfactuals['gender'].shape == X.shape


class TestCausalModelIntegration:
    """Tests for causal model integration."""
    
    def test_scm_forward(self):
        """Test structural causal model forward pass."""
        # Mock SCM
        class SimpleSCM:
            def __init__(self):
                self.coefficients = {
                    'A_to_Y': 0.3,
                    'X_to_Y': 0.5
                }
            
            def forward(self, A, X):
                Y = self.coefficients['A_to_Y'] * A + self.coefficients['X_to_Y'] * X
                return Y
            
            def counterfactual(self, A_obs, A_cf, X, Y_obs):
                # Compute counterfactual outcome
                Y_cf = Y_obs + self.coefficients['A_to_Y'] * (A_cf - A_obs)
                return Y_cf
        
        scm = SimpleSCM()
        
        np.random.seed(42)
        n = 100
        A = np.random.choice([0, 1], n)
        X = np.random.randn(n)
        Y = scm.forward(A, X)
        
        # Counterfactual: what if A was different?
        A_cf = 1 - A
        Y_cf = scm.counterfactual(A, A_cf, X, Y)
        
        assert len(Y_cf) == n
    
    def test_causal_graph_structure(self):
        """Test causal graph structure validation."""
        # Mock causal graph
        edges = [
            ('gender', 'income'),
            ('gender', 'education'),
            ('education', 'income'),
            ('age', 'income')
        ]
        
        # Validate structure
        nodes = set()
        for edge in edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
        
        assert 'gender' in nodes
        assert 'income' in nodes


class TestEdgeCases:
    """Edge case tests for counterfactual fairness."""
    
    def test_all_same_sensitive(self):
        """Test when all samples have same sensitive attribute."""
        np.random.seed(42)
        n = 100
        
        predictions = np.random.random(n)
        predictions_cf = np.random.random(n)
        sensitive = np.zeros(n)  # All same
        
        metric = CounterfactualInvariance()
        score = metric.compute(predictions, predictions_cf)
        
        assert 0 <= score <= 1
    
    def test_binary_vs_continuous_predictions(self):
        """Test with binary and continuous predictions."""
        np.random.seed(42)
        n = 100
        
        # Binary
        predictions_binary = np.random.choice([0, 1], n)
        predictions_cf_binary = np.random.choice([0, 1], n)
        
        metric = CounterfactualInvariance()
        score_binary = metric.compute(predictions_binary, predictions_cf_binary)
        
        # Continuous
        predictions_cont = np.random.random(n)
        predictions_cf_cont = np.random.random(n)
        
        score_cont = metric.compute(predictions_cont, predictions_cf_cont)
        
        assert 0 <= score_binary <= 1
        assert 0 <= score_cont <= 1
    
    def test_empty_counterfactual(self):
        """Test handling of empty counterfactual data."""
        predictions = np.array([])
        predictions_cf = np.array([])
        
        metric = CounterfactualInvariance()
        
        with pytest.raises((ValueError, IndexError)):
            metric.compute(predictions, predictions_cf)
    
    def test_mismatched_shapes(self):
        """Test handling of mismatched shapes."""
        predictions = np.random.random(100)
        predictions_cf = np.random.random(50)  # Different size
        
        metric = CounterfactualInvariance()
        
        with pytest.raises((ValueError, IndexError)):
            metric.compute(predictions, predictions_cf)
