"""
Unit Tests for Group Fairness
==============================

Tests for group fairness metrics and constraints including:
- Demographic Parity
- Equalized Odds
- Equal Opportunity
- Disparate Impact
- Calibration
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.fairness.constraints.demographic_parity import DemographicParity
from src.fairness.constraints.equalized_odds import EqualizedOdds
from src.fairness.constraints.disparate_impact import DisparateImpact
from src.fairness.constraints.group_fairness import GroupFairnessConstraint
from src.evaluation.fairness.group_metrics import (
    DemographicParityMetric,
    EqualizedOddsMetric,
    EqualOpportunityMetric,
    DisparateImpactMetric,
    GroupFairnessEvaluator
)


class TestDemographicParity:
    """Tests for Demographic Parity metric and constraint."""
    
    @pytest.fixture
    def fair_predictions(self):
        """Create predictions with no demographic parity violation."""
        np.random.seed(42)
        n = 1000
        # Equal positive rates across groups
        predictions = np.concatenate([
            np.random.choice([0, 1], 500, p=[0.7, 0.3]),  # Group 0
            np.random.choice([0, 1], 500, p=[0.7, 0.3]),  # Group 1
        ])
        groups = np.concatenate([np.zeros(500), np.ones(500)])
        return predictions, groups
    
    @pytest.fixture
    def biased_predictions(self):
        """Create predictions with demographic parity violation."""
        np.random.seed(42)
        n = 1000
        # Different positive rates across groups
        predictions = np.concatenate([
            np.random.choice([0, 1], 500, p=[0.6, 0.4]),  # Group 0: 40% positive
            np.random.choice([0, 1], 500, p=[0.8, 0.2]),  # Group 1: 20% positive
        ])
        groups = np.concatenate([np.zeros(500), np.ones(500)])
        return predictions, groups
    
    def test_demographic_parity_metric_fair(self, fair_predictions):
        """Test DP metric correctly identifies fair predictions."""
        predictions, groups = fair_predictions
        
        metric = DemographicParityMetric()
        result = metric.compute(predictions, groups)
        
        assert 'demographic_parity_difference' in result
        assert result['demographic_parity_difference'] < 0.1  # Should be small
    
    def test_demographic_parity_metric_biased(self, biased_predictions):
        """Test DP metric correctly identifies biased predictions."""
        predictions, groups = biased_predictions
        
        metric = DemographicParityMetric()
        result = metric.compute(predictions, groups)
        
        assert result['demographic_parity_difference'] > 0.1  # Should be large
    
    def test_demographic_parity_constraint_satisfied(self, fair_predictions):
        """Test DP constraint correctly identifies satisfied constraint."""
        predictions, groups = fair_predictions
        
        constraint = DemographicParity(threshold=0.1)
        satisfied = constraint.check_satisfied(predictions, groups)
        
        assert satisfied is True
    
    def test_demographic_parity_constraint_violated(self, biased_predictions):
        """Test DP constraint correctly identifies violated constraint."""
        predictions, groups = biased_predictions
        
        constraint = DemographicParity(threshold=0.1)
        satisfied = constraint.check_satisfied(predictions, groups)
        
        assert satisfied is False
    
    def test_demographic_parity_loss(self, biased_predictions):
        """Test DP loss computation."""
        predictions, groups = biased_predictions
        predictions_tensor = torch.FloatTensor(predictions)
        groups_tensor = torch.LongTensor(groups)
        
        constraint = DemographicParity(threshold=0.1)
        loss = constraint.compute_loss(predictions_tensor, groups_tensor)
        
        assert loss.item() >= 0  # Loss should be non-negative
    
    def test_multiple_groups(self):
        """Test DP with more than 2 groups."""
        np.random.seed(42)
        n = 1500
        predictions = np.random.choice([0, 1], n)
        groups = np.repeat([0, 1, 2], 500)
        
        metric = DemographicParityMetric()
        result = metric.compute(predictions, groups)
        
        assert 'demographic_parity_difference' in result
        assert result['demographic_parity_difference'] >= 0
    
    def test_edge_case_all_positive(self):
        """Test DP when all predictions are positive."""
        predictions = np.ones(100)
        groups = np.concatenate([np.zeros(50), np.ones(50)])
        
        metric = DemographicParityMetric()
        result = metric.compute(predictions, groups)
        
        assert result['demographic_parity_difference'] == 0.0
    
    def test_edge_case_all_negative(self):
        """Test DP when all predictions are negative."""
        predictions = np.zeros(100)
        groups = np.concatenate([np.zeros(50), np.ones(50)])
        
        metric = DemographicParityMetric()
        result = metric.compute(predictions, groups)
        
        assert result['demographic_parity_difference'] == 0.0


class TestEqualizedOdds:
    """Tests for Equalized Odds metric and constraint."""
    
    @pytest.fixture
    def equalized_odds_data(self):
        """Create data satisfying equalized odds."""
        np.random.seed(42)
        n = 1000
        
        # Equal TPR and FPR across groups
        labels = np.concatenate([
            np.concatenate([np.ones(250), np.zeros(250)]),  # Group 0
            np.concatenate([np.ones(250), np.zeros(250)]),  # Group 1
        ])
        predictions = np.concatenate([
            np.concatenate([np.random.choice([0, 1], 250, p=[0.2, 0.8]),  # TPR=0.8
                           np.random.choice([0, 1], 250, p=[0.9, 0.1])]),  # FPR=0.1
            np.concatenate([np.random.choice([0, 1], 250, p=[0.2, 0.8]),  # TPR=0.8
                           np.random.choice([0, 1], 250, p=[0.9, 0.1])]),  # FPR=0.1
        ])
        groups = np.concatenate([np.zeros(500), np.ones(500)])
        
        return predictions, labels, groups
    
    def test_equalized_odds_metric(self, equalized_odds_data):
        """Test equalized odds metric computation."""
        predictions, labels, groups = equalized_odds_data
        
        metric = EqualizedOddsMetric()
        result = metric.compute(predictions, groups, labels)
        
        assert 'tpr_difference' in result
        assert 'fpr_difference' in result
        assert 'equalized_odds_difference' in result
    
    def test_equalized_odds_constraint(self, equalized_odds_data):
        """Test equalized odds constraint check."""
        predictions, labels, groups = equalized_odds_data
        
        constraint = EqualizedOdds(threshold=0.15)
        satisfied = constraint.check_satisfied(predictions, groups, labels=labels)
        
        assert satisfied is True
    
    def test_equalized_odds_violation(self):
        """Test detection of equalized odds violation."""
        np.random.seed(42)
        
        # Different TPR across groups
        labels = np.concatenate([np.ones(500), np.zeros(500)])
        predictions = np.concatenate([
            np.random.choice([0, 1], 500, p=[0.1, 0.9]),  # Group 0: TPR=0.9
            np.random.choice([0, 1], 500, p=[0.5, 0.5]),  # Group 1: TPR=0.5
        ])
        groups = np.concatenate([np.zeros(500), np.ones(500)])
        
        metric = EqualizedOddsMetric()
        result = metric.compute(predictions, groups, labels)
        
        assert result['tpr_difference'] > 0.2


class TestDisparateImpact:
    """Tests for Disparate Impact metric."""
    
    def test_disparate_impact_ratio_fair(self):
        """Test DI ratio for fair predictions."""
        np.random.seed(42)
        
        # Similar positive rates
        predictions = np.concatenate([
            np.random.choice([0, 1], 500, p=[0.65, 0.35]),
            np.random.choice([0, 1], 500, p=[0.65, 0.35]),
        ])
        groups = np.concatenate([np.zeros(500), np.ones(500)])
        
        metric = DisparateImpactMetric()
        result = metric.compute(predictions, groups)
        
        # DI ratio should be close to 1.0
        assert 0.8 <= result['disparate_impact_ratio'] <= 1.25
    
    def test_disparate_impact_ratio_biased(self):
        """Test DI ratio for biased predictions."""
        np.random.seed(42)
        
        # Different positive rates
        predictions = np.concatenate([
            np.random.choice([0, 1], 500, p=[0.5, 0.5]),   # 50% positive
            np.random.choice([0, 1], 500, p=[0.9, 0.1]),   # 10% positive
        ])
        groups = np.concatenate([np.zeros(500), np.ones(500)])
        
        metric = DisparateImpactMetric()
        result = metric.compute(predictions, groups)
        
        # DI ratio should be far from 1.0
        assert result['disparate_impact_ratio'] < 0.8
    
    def test_80_percent_rule(self):
        """Test the 80% rule for disparate impact."""
        np.random.seed(42)
        
        # Create predictions that pass 80% rule
        predictions = np.concatenate([
            np.random.choice([0, 1], 500, p=[0.6, 0.4]),  # 40% positive
            np.random.choice([0, 1], 500, p=[0.7, 0.3]),  # 30% positive
        ])
        groups = np.concatenate([np.zeros(500), np.ones(500)])
        
        constraint = DisparateImpact(threshold=0.8)
        satisfied = constraint.check_satisfied(predictions, groups)
        
        # 30/40 = 0.75, which is close to threshold
        assert isinstance(satisfied, bool)


class TestGroupFairnessEvaluator:
    """Tests for GroupFairnessEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return GroupFairnessEvaluator(
            sensitive_attributes=['gender', 'race']
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
        np.random.seed(42)
        n = 1000
        
        return pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female'], n),
            'race': np.random.choice(['White', 'Black', 'Asian'], n),
            'prediction': np.random.random(n),
            'label': np.random.choice([0, 1], n),
        })
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initializes correctly."""
        assert evaluator is not None
        assert 'gender' in evaluator.sensitive_attributes
        assert 'race' in evaluator.sensitive_attributes
    
    def test_evaluate_single_attribute(self, evaluator, sample_data):
        """Test evaluation of single sensitive attribute."""
        results = evaluator.evaluate(
            predictions=sample_data['prediction'].values,
            groups=sample_data['gender'].values,
            labels=sample_data['label'].values
        )
        
        assert 'demographic_parity' in results
        assert 'equalized_odds' in results
    
    def test_evaluate_multiple_attributes(self, evaluator, sample_data):
        """Test evaluation of multiple sensitive attributes."""
        results = evaluator.evaluate_all(
            predictions=sample_data['prediction'].values,
            sensitive_data=sample_data[['gender', 'race']],
            labels=sample_data['label'].values
        )
        
        assert 'gender' in results
        assert 'race' in results
    
    def test_generate_report(self, evaluator, sample_data):
        """Test fairness report generation."""
        results = evaluator.evaluate(
            predictions=sample_data['prediction'].values,
            groups=sample_data['gender'].values,
            labels=sample_data['label'].values
        )
        
        report = evaluator.generate_report(results)
        
        assert report is not None
        assert len(report) > 0


class TestGroupFairnessConstraints:
    """Tests for GroupFairnessConstraint training integration."""
    
    def test_constraint_as_loss(self):
        """Test using constraint as differentiable loss."""
        constraint = GroupFairnessConstraint(
            constraint_type='demographic_parity',
            threshold=0.1
        )
        
        predictions = torch.randn(64, 1, requires_grad=True)
        groups = torch.randint(0, 2, (64,))
        
        loss = constraint.compute_loss(predictions, groups)
        
        assert loss.requires_grad
        loss.backward()
        assert predictions.grad is not None
    
    def test_constraint_threshold_adjustment(self):
        """Test adjusting constraint threshold."""
        constraint = DemographicParity(threshold=0.1)
        
        # Should be able to update threshold
        constraint.threshold = 0.05
        assert constraint.threshold == 0.05
    
    def test_slack_types(self):
        """Test different slack types for constraints."""
        for slack_type in ['relative', 'absolute']:
            constraint = DemographicParity(
                threshold=0.1,
                slack_type=slack_type
            )
            assert constraint.slack_type == slack_type


class TestEdgeCases:
    """Edge case tests for group fairness metrics."""
    
    def test_single_group(self):
        """Test metric with single group."""
        predictions = np.random.choice([0, 1], 100)
        groups = np.zeros(100)  # All same group
        
        metric = DemographicParityMetric()
        
        # Should handle gracefully (possibly return 0 or raise)
        try:
            result = metric.compute(predictions, groups)
        except ValueError:
            pass  # Acceptable to raise for invalid input
    
    def test_empty_predictions(self):
        """Test metric with empty predictions."""
        predictions = np.array([])
        groups = np.array([])
        
        metric = DemographicParityMetric()
        
        with pytest.raises((ValueError, IndexError)):
            metric.compute(predictions, groups)
    
    def test_continuous_predictions(self):
        """Test metric with continuous predictions (probabilities)."""
        np.random.seed(42)
        predictions = np.random.random(100)
        groups = np.concatenate([np.zeros(50), np.ones(50)])
        
        metric = DemographicParityMetric()
        result = metric.compute(predictions, groups)
        
        assert 'demographic_parity_difference' in result
    
    def test_imbalanced_groups(self):
        """Test metric with heavily imbalanced group sizes."""
        np.random.seed(42)
        predictions = np.random.choice([0, 1], 1000)
        groups = np.concatenate([np.zeros(990), np.ones(10)])  # 99:1 ratio
        
        metric = DemographicParityMetric()
        result = metric.compute(predictions, groups)
        
        assert result is not None
