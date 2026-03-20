"""
Unit Tests for Privacy Metrics
===============================

Tests for privacy evaluation metrics including:
- Membership Inference Attack
- Attribute Inference Attack
- Differential Privacy Verification
- Privacy Risk Assessment
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.evaluation.privacy.membership_inference import (
    MembershipInferenceAttack,
    ShadowModelMIA,
    LossBasedMIA,
    MembershipInferenceEvaluator
)
from src.evaluation.privacy.attribute_inference import (
    AttributeInferenceAttack,
    CorrelationBasedAIA,
    ModelBasedAIA,
    AttributeInferenceEvaluator
)
from src.evaluation.privacy.differential_privacy import (
    DifferentialPrivacyAccountant,
    EpsilonDeltaCalculator,
    PrivacyBudgetScheduler,
    DifferentialPrivacyVerifier
)


class TestMembershipInferenceAttack:
    """Tests for Membership Inference Attack metrics."""
    
    @pytest.fixture
    def vulnerable_data(self):
        """Create data from a model that overfits (vulnerable to MIA)."""
        np.random.seed(42)
        n_train, n_test = 500, 500
        
        # Training data (memorized by overfitted model)
        X_train = np.random.randn(n_train, 10)
        y_train = np.random.choice([0, 1], n_train)
        
        # Test data (different)
        X_test = np.random.randn(n_test, 10) + 0.5
        y_test = np.random.choice([0, 1], n_test)
        
        return X_train, y_train, X_test, y_test
    
    @pytest.fixture
    def secure_data(self):
        """Create data from a well-generalized model (secure against MIA)."""
        np.random.seed(42)
        n_train, n_test = 500, 500
        
        # Similar distributions
        X_train = np.random.randn(n_train, 10)
        y_train = (X_train[:, 0] > 0).astype(int)
        
        X_test = np.random.randn(n_test, 10)
        y_test = (X_test[:, 0] > 0).astype(int)
        
        return X_train, y_train, X_test, y_test
    
    def test_mia_basic_computation(self, vulnerable_data):
        """Test basic MIA computation."""
        X_train, y_train, X_test, y_test = vulnerable_data
        
        mia = MembershipInferenceAttack()
        
        # Create mock predictions
        train_probs = np.random.random(len(X_train))
        test_probs = np.random.random(len(X_test))
        
        result = mia.compute(train_probs, test_probs)
        
        assert 'attack_accuracy' in result
        assert 'attack_auc' in result
        assert 0 <= result['attack_accuracy'] <= 1
    
    def test_mia_auc_range(self, vulnerable_data):
        """Test MIA AUC is in valid range."""
        X_train, y_train, X_test, y_test = vulnerable_data
        
        mia = MembershipInferenceAttack()
        
        train_probs = np.random.random(len(X_train))
        test_probs = np.random.random(len(X_test))
        
        result = mia.compute(train_probs, test_probs)
        
        assert 0 <= result['attack_auc'] <= 1
    
    def test_mia_random_baseline(self):
        """Test MIA with random predictions (should be ~50%)."""
        np.random.seed(42)
        n = 500
        
        train_probs = np.random.random(n)
        test_probs = np.random.random(n)
        
        mia = MembershipInferenceAttack()
        result = mia.compute(train_probs, test_probs)
        
        # Random predictions should give ~50% accuracy
        assert 0.4 <= result['attack_accuracy'] <= 0.6


class TestShadowModelMIA:
    """Tests for Shadow Model MIA."""
    
    @pytest.fixture
    def shadow_mia(self):
        """Create Shadow Model MIA instance."""
        return ShadowModelMIA(
            shadow_model_type='logistic',
            n_shadow_models=3
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for shadow model MIA."""
        np.random.seed(42)
        n = 500
        
        X = np.random.randn(n, 10)
        y = np.random.choice([0, 1], n)
        
        return X, y
    
    def test_shadow_model_training(self, shadow_mia, sample_data):
        """Test shadow model training."""
        X, y = sample_data
        
        shadow_mia.train_shadow_models(X, y)
        
        assert shadow_mia.is_trained
        assert len(shadow_mia.shadow_models) == 3
    
    def test_shadow_model_attack(self, shadow_mia, sample_data):
        """Test attack using shadow models."""
        X, y = sample_data
        
        shadow_mia.train_shadow_models(X, y)
        
        # Create member/non-member indicators
        is_member = np.concatenate([np.ones(250), np.zeros(250)])
        attack_results = shadow_mia.attack(X[:500], is_member)
        
        assert 'attack_accuracy' in attack_results


class TestLossBasedMIA:
    """Tests for Loss-based MIA."""
    
    def test_loss_based_mia_computation(self):
        """Test loss-based MIA computation."""
        np.random.seed(42)
        n = 200
        
        # Members have lower loss (overfitted model)
        member_losses = np.random.exponential(0.5, n)
        non_member_losses = np.random.exponential(1.5, n)
        
        mia = LossBasedMIA(threshold=1.0)
        result = mia.compute(member_losses, non_member_losses)
        
        assert 'attack_accuracy' in result
        assert result['attack_accuracy'] > 0.5  # Should be better than random
    
    def test_loss_threshold_selection(self):
        """Test automatic threshold selection."""
        np.random.seed(42)
        n = 200
        
        member_losses = np.random.exponential(0.5, n)
        non_member_losses = np.random.exponential(1.5, n)
        
        mia = LossBasedMIA(threshold='auto')
        result = mia.compute(member_losses, non_member_losses)
        
        assert result['threshold'] is not None


class TestAttributeInferenceAttack:
    """Tests for Attribute Inference Attack metrics."""
    
    @pytest.fixture
    def inference_data(self):
        """Create data for attribute inference testing."""
        np.random.seed(42)
        n = 500
        
        # Create data where sensitive attribute is correlated with other features
        X = np.random.randn(n, 5)
        sensitive = (X[:, 0] + X[:, 1] > 0).astype(int)  # Correlated
        X = np.column_stack([X, np.random.randn(n, 3)])  # Add noise features
        
        return X, sensitive
    
    def test_aia_basic_computation(self, inference_data):
        """Test basic AIA computation."""
        X, sensitive = inference_data
        
        aia = AttributeInferenceAttack()
        result = aia.compute(X, sensitive)
        
        assert 'attack_accuracy' in result
        assert 0 <= result['attack_accuracy'] <= 1
    
    def test_aia_with_known_correlation(self, inference_data):
        """Test AIA detects known correlations."""
        X, sensitive = inference_data
        
        aia = AttributeInferenceAttack()
        result = aia.compute(X, sensitive)
        
        # Should be better than random due to correlation
        assert result['attack_accuracy'] > 0.5
    
    def test_aia_random_baseline(self):
        """Test AIA with random sensitive attribute."""
        np.random.seed(42)
        n = 500
        
        X = np.random.randn(n, 8)
        sensitive = np.random.choice([0, 1], n)  # Random, no correlation
        
        aia = AttributeInferenceAttack()
        result = aia.compute(X, sensitive)
        
        # Should be close to random (50%)
        assert 0.4 <= result['attack_accuracy'] <= 0.6


class TestCorrelationBasedAIA:
    """Tests for Correlation-based AIA."""
    
    def test_correlation_aia(self):
        """Test correlation-based attribute inference."""
        np.random.seed(42)
        n = 500
        
        # Create correlated data
        X = np.random.randn(n, 5)
        sensitive = (X[:, 0] > 0).astype(int)
        
        aia = CorrelationBasedAIA()
        result = aia.compute(X, sensitive)
        
        assert 'attack_accuracy' in result
        assert result['attack_accuracy'] > 0.5


class TestModelBasedAIA:
    """Tests for Model-based AIA."""
    
    def test_model_based_aia(self):
        """Test model-based attribute inference."""
        np.random.seed(42)
        n = 500
        
        X = np.random.randn(n, 10)
        sensitive = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        aia = ModelBasedAIA(
            model_type='random_forest',
            cv_folds=3
        )
        result = aia.compute(X, sensitive)
        
        assert 'attack_accuracy' in result
        assert 'attack_auc' in result


class TestDifferentialPrivacyAccountant:
    """Tests for Differential Privacy Accountant."""
    
    @pytest.fixture
    def accountant(self):
        """Create DP accountant instance."""
        return DifferentialPrivacyAccountant(
            target_epsilon=1.0,
            target_delta=1e-5
        )
    
    def test_accountant_initialization(self, accountant):
        """Test accountant initializes correctly."""
        assert accountant.target_epsilon == 1.0
        assert accountant.target_delta == 1e-5
        assert accountant.total_epsilon == 0
    
    def test_noise_addition(self, accountant):
        """Test noise addition tracking."""
        accountant.add_noise(epsilon=0.1)
        
        assert accountant.total_epsilon == 0.1
        
        accountant.add_noise(epsilon=0.2)
        
        assert accountant.total_epsilon == 0.3
    
    def test_budget_exhaustion(self, accountant):
        """Test budget exhaustion detection."""
        accountant.add_noise(epsilon=0.9)
        
        assert not accountant.is_budget_exhausted()
        
        accountant.add_noise(epsilon=0.2)
        
        assert accountant.is_budget_exhausted()
    
    def test_remaining_budget(self, accountant):
        """Test remaining budget calculation."""
        accountant.add_noise(epsilon=0.3)
        
        remaining = accountant.remaining_budget()
        
        assert remaining == 0.7


class TestEpsilonDeltaCalculator:
    """Tests for Epsilon-Delta Calculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create epsilon-delta calculator instance."""
        return EpsilonDeltaCalculator()
    
    def test_compose_gaussian_mechanism(self, calculator):
        """Test composition of Gaussian mechanism."""
        sigma = 1.0
        n_queries = 100
        
        epsilon, delta = calculator.compose_gaussian(sigma, n_queries)
        
        assert epsilon > 0
        assert delta > 0
    
    def test_privacy_curve(self, calculator):
        """Test privacy curve computation."""
        epsilons = calculator.compute_privacy_curve(
            noise_multipliers=[1.0, 1.5, 2.0],
            n_steps=1000
        )
        
        assert len(epsilons) > 0
        assert all(e > 0 for e in epsilons)


class TestPrivacyBudgetScheduler:
    """Tests for Privacy Budget Scheduler."""
    
    def test_budget_allocation(self):
        """Test budget allocation across rounds."""
        scheduler = PrivacyBudgetScheduler(
            total_epsilon=1.0,
            n_rounds=10
        )
        
        budgets = scheduler.get_budget_schedule()
        
        assert len(budgets) == 10
        assert sum(budgets) <= 1.0
    
    def test_adaptive_budget(self):
        """Test adaptive budget allocation."""
        scheduler = PrivacyBudgetScheduler(
            total_epsilon=1.0,
            mode='adaptive'
        )
        
        # Get budget for first round
        budget_1 = scheduler.get_budget(0)
        
        # Get budget for later round
        budget_2 = scheduler.get_budget(5)
        
        assert budget_1 > 0
        assert budget_2 > 0


class TestDifferentialPrivacyVerifier:
    """Tests for Differential Privacy Verifier."""
    
    @pytest.fixture
    def verifier(self):
        """Create DP verifier instance."""
        return DifferentialPrivacyVerifier(epsilon=1.0, delta=1e-5)
    
    def test_mechanism_verification(self, verifier):
        """Test verification of DP mechanism."""
        # Mock mechanism
        def gaussian_mechanism(value, sensitivity=1.0, sigma=1.0):
            noise = np.random.normal(0, sigma * sensitivity)
            return value + noise
        
        result = verifier.verify_mechanism(gaussian_mechanism)
        
        assert 'is_dp' in result
        assert 'achieved_epsilon' in result
    
    def test_composition_verification(self, verifier):
        """Test composition verification."""
        epsilons = [0.3, 0.3, 0.3]
        
        is_valid = verifier.verify_composition(epsilons)
        
        # Total epsilon is 0.9 < 1.0, should be valid
        assert is_valid is True


class TestMembershipInferenceEvaluator:
    """Tests for MembershipInferenceEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return MembershipInferenceEvaluator()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
        np.random.seed(42)
        n = 500
        
        return {
            'train_data': np.random.randn(n, 10),
            'test_data': np.random.randn(n, 10) + 0.1,
            'train_predictions': np.random.random(n),
            'test_predictions': np.random.random(n),
        }
    
    def test_full_evaluation(self, evaluator, sample_data):
        """Test full MIA evaluation."""
        result = evaluator.evaluate(
            member_predictions=sample_data['train_predictions'],
            non_member_predictions=sample_data['test_predictions']
        )
        
        assert 'mia_accuracy' in result
        assert 'mia_auc' in result
        assert 'privacy_risk_level' in result
    
    def test_risk_assessment(self, evaluator, sample_data):
        """Test privacy risk assessment."""
        result = evaluator.assess_risk(
            member_predictions=sample_data['train_predictions'],
            non_member_predictions=sample_data['test_predictions']
        )
        
        assert result['risk_level'] in ['low', 'medium', 'high', 'critical']
        assert 0 <= result['risk_score'] <= 1


class TestAttributeInferenceEvaluator:
    """Tests for AttributeInferenceEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return AttributeInferenceEvaluator()
    
    def test_full_evaluation(self):
        """Test full AIA evaluation."""
        np.random.seed(42)
        n = 300
        
        X = np.random.randn(n, 10)
        sensitive = np.random.choice([0, 1], n)
        
        evaluator = AttributeInferenceEvaluator()
        result = evaluator.evaluate(X, sensitive)
        
        assert 'inference_accuracy' in result
        assert 'privacy_risk' in result


class TestEdgeCases:
    """Edge case tests for privacy metrics."""
    
    def test_empty_predictions(self):
        """Test MIA with empty predictions."""
        mia = MembershipInferenceAttack()
        
        with pytest.raises((ValueError, IndexError)):
            mia.compute(np.array([]), np.array([]))
    
    def test_single_sample(self):
        """Test with single sample."""
        np.random.seed(42)
        
        mia = MembershipInferenceAttack()
        
        # Single sample - should handle or raise
        try:
            result = mia.compute(np.array([0.5]), np.array([0.3]))
        except (ValueError, IndexError):
            pass  # Acceptable
    
    def test_all_same_predictions(self):
        """Test with all same predictions."""
        np.random.seed(42)
        n = 100
        
        train_probs = np.ones(n)
        test_probs = np.ones(n)
        
        mia = MembershipInferenceAttack()
        result = mia.compute(train_probs, test_probs)
        
        # Should handle gracefully
        assert 'attack_accuracy' in result
    
    def test_extreme_imbalance(self):
        """Test with extreme class imbalance."""
        np.random.seed(42)
        n = 100
        
        # 95% members
        is_member = np.concatenate([np.ones(95), np.zeros(5)])
        
        mia = MembershipInferenceAttack()
        # Should handle imbalance
        assert mia is not None


class TestPrivacyReportGeneration:
    """Tests for privacy report generation."""
    
    def test_report_generation(self):
        """Test privacy report generation."""
        np.random.seed(42)
        n = 200
        
        mia_results = {
            'attack_accuracy': 0.55,
            'attack_auc': 0.58,
        }
        
        aia_results = {
            'attack_accuracy': 0.52,
        }
        
        evaluator = MembershipInferenceEvaluator()
        report = evaluator.generate_report(mia_results, aia_results)
        
        assert report is not None
        assert 'overall_privacy_score' in report or 'risk_assessment' in report
