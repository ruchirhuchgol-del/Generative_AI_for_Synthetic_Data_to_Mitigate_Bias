"""
Unit Tests for Fidelity Metrics
================================

Tests for fidelity evaluation metrics including:
- Statistical Similarity Metrics
- Distribution Comparison Metrics
- Downstream Utility Metrics
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.evaluation.fidelity.statistical_similarity import (
    JensenShannonDivergence,
    WassersteinDistance,
    CorrelationPreservation,
    MutualInformationMetric,
    KolmogorovSmirnovStatistic,
    MomentMatchingMetric,
    StatisticalSimilarityEvaluator
)
from src.evaluation.fidelity.distribution_metrics import (
    KolmogorovSmirnovTest,
    AndersonDarlingTest,
    MaximumMeanDiscrepancy,
    EnergyDistance,
    DistributionComparator
)
from src.evaluation.fidelity.downstream_utility import (
    TrainOnSyntheticTestOnReal,
    CrossValidationUtility,
    FeatureImportancePreservation,
    DownstreamUtilityEvaluator
)


class TestJensenShannonDivergence:
    """Tests for Jensen-Shannon Divergence metric."""
    
    @pytest.fixture
    def identical_distributions(self):
        """Create identical distributions."""
        np.random.seed(42)
        p = np.random.randn(1000)
        q = p.copy()
        return p, q
    
    @pytest.fixture
    def different_distributions(self):
        """Create different distributions."""
        np.random.seed(42)
        p = np.random.randn(1000)  # Standard normal
        q = np.random.randn(1000) + 2  # Shifted normal
        return p, q
    
    def test_identical_distributions(self, identical_distributions):
        """Test JSD for identical distributions."""
        p, q = identical_distributions
        
        metric = JensenShannonDivergence()
        jsd = metric.compute(p, q)
        
        assert jsd >= 0
        assert jsd < 0.01  # Should be very close to 0
    
    def test_different_distributions(self, different_distributions):
        """Test JSD for different distributions."""
        p, q = different_distributions
        
        metric = JensenShannonDivergence()
        jsd = metric.compute(p, q)
        
        assert jsd > 0.1  # Should be significant
    
    def test_jsd_bounds(self):
        """Test JSD is bounded between 0 and 1."""
        np.random.seed(42)
        
        for _ in range(10):
            p = np.random.randn(500)
            q = np.random.randn(500) + np.random.randn() * 3
            
            metric = JensenShannonDivergence()
            jsd = metric.compute(p, q)
            
            assert 0 <= jsd <= 1
    
    def test_jsd_symmetry(self):
        """Test JSD is symmetric."""
        np.random.seed(42)
        p = np.random.randn(500)
        q = np.random.randn(500) + 1
        
        metric = JensenShannonDivergence()
        jsd_pq = metric.compute(p, q)
        jsd_qp = metric.compute(q, p)
        
        assert np.isclose(jsd_pq, jsd_qp, rtol=1e-5)


class TestWassersteinDistance:
    """Tests for Wasserstein Distance metric."""
    
    def test_wasserstein_identical(self):
        """Test Wasserstein distance for identical distributions."""
        np.random.seed(42)
        p = np.random.randn(1000)
        q = p.copy()
        
        metric = WassersteinDistance()
        wd = metric.compute(p, q)
        
        assert wd < 0.01  # Should be near 0
    
    def test_wasserstein_different(self):
        """Test Wasserstein distance for different distributions."""
        np.random.seed(42)
        p = np.random.randn(1000)
        q = np.random.randn(1000) + 2  # Shifted by 2
        
        metric = WassersteinDistance()
        wd = metric.compute(p, q)
        
        assert wd > 1.5  # Should be close to 2 (shift amount)
    
    def test_wasserstein_non_negative(self):
        """Test Wasserstein distance is non-negative."""
        np.random.seed(42)
        
        for _ in range(10):
            p = np.random.randn(100)
            q = np.random.randn(100)
            
            metric = WassersteinDistance()
            wd = metric.compute(p, q)
            
            assert wd >= 0
    
    def test_multivariate_wasserstein(self):
        """Test Wasserstein distance for multivariate data."""
        np.random.seed(42)
        p = np.random.randn(100, 5)
        q = np.random.randn(100, 5) + 1
        
        metric = WassersteinDistance()
        wd = metric.compute(p, q)
        
        assert wd >= 0


class TestCorrelationPreservation:
    """Tests for Correlation Preservation metric."""
    
    @pytest.fixture
    def correlated_data(self):
        """Create data with known correlations."""
        np.random.seed(42)
        n = 500
        
        # Create correlated features
        X = np.random.randn(n, 3)
        X[:, 1] = X[:, 0] + np.random.randn(n) * 0.3  # Correlated
        X[:, 2] = -X[:, 0] + np.random.randn(n) * 0.3  # Negatively correlated
        
        # Synthetic data preserves correlations
        Y = X + np.random.randn(n, 3) * 0.1
        
        return X, Y
    
    def test_correlation_preservation_high(self, correlated_data):
        """Test high correlation preservation."""
        X, Y = correlated_data
        
        metric = CorrelationPreservation()
        score = metric.compute(X, Y)
        
        assert score > 0.8  # Should be high
    
    def test_correlation_preservation_low(self):
        """Test low correlation preservation."""
        np.random.seed(42)
        X = np.random.randn(500, 3)
        Y = np.random.randn(500, 3)  # Independent
        
        metric = CorrelationPreservation()
        score = metric.compute(X, Y)
        
        assert score < 0.5  # Should be low
    
    def test_correlation_matrix_shape(self):
        """Test correlation matrix computation."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        metric = CorrelationPreservation()
        corr_matrix = metric.compute_correlation_matrix(X)
        
        assert corr_matrix.shape == (5, 5)
        assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal is 1


class TestKolmogorovSmirnovStatistic:
    """Tests for KS Statistic metric."""
    
    def test_ks_identical_distributions(self):
        """Test KS statistic for identical distributions."""
        np.random.seed(42)
        p = np.random.randn(500)
        q = p.copy()
        
        metric = KolmogorovSmirnovStatistic()
        ks_stat, p_value = metric.compute(p, q)
        
        assert ks_stat < 0.05
        assert p_value > 0.5
    
    def test_ks_different_distributions(self):
        """Test KS statistic for different distributions."""
        np.random.seed(42)
        p = np.random.randn(500)
        q = np.random.randn(500) + 2
        
        metric = KolmogorovSmirnovStatistic()
        ks_stat, p_value = metric.compute(p, q)
        
        assert ks_stat > 0.5
        assert p_value < 0.01
    
    def test_ks_bounds(self):
        """Test KS statistic is in [0, 1]."""
        np.random.seed(42)
        
        for _ in range(10):
            p = np.random.randn(100)
            q = np.random.randn(100) + np.random.randn()
            
            metric = KolmogorovSmirnovStatistic()
            ks_stat, _ = metric.compute(p, q)
            
            assert 0 <= ks_stat <= 1


class TestMomentMatchingMetric:
    """Tests for Moment Matching metric."""
    
    def test_moment_matching_identical(self):
        """Test moment matching for identical distributions."""
        np.random.seed(42)
        p = np.random.randn(500)
        q = p.copy()
        
        metric = MomentMatchingMetric()
        score = metric.compute(p, q, moments=['mean', 'std', 'skew'])
        
        assert score > 0.99
    
    def test_moment_matching_different_mean(self):
        """Test moment matching with different means."""
        np.random.seed(42)
        p = np.random.randn(500)
        q = np.random.randn(500) + 1  # Different mean
        
        metric = MomentMatchingMetric()
        score = metric.compute(p, q, moments=['mean'])
        
        assert score < 0.9
    
    def test_higher_moments(self):
        """Test computation of higher moments."""
        np.random.seed(42)
        p = np.random.randn(500)
        q = np.random.randn(500)
        
        metric = MomentMatchingMetric()
        
        # Test different moments
        for moments in [['mean'], ['mean', 'std'], ['mean', 'std', 'skew', 'kurtosis']]:
            score = metric.compute(p, q, moments=moments)
            assert 0 <= score <= 1


class TestMaximumMeanDiscrepancy:
    """Tests for Maximum Mean Discrepancy metric."""
    
    def test_mmd_identical(self):
        """Test MMD for identical distributions."""
        np.random.seed(42)
        p = np.random.randn(200, 3)
        q = p.copy()
        
        metric = MaximumMeanDiscrepancy()
        mmd = metric.compute(p, q)
        
        assert mmd < 0.01
    
    def test_mmd_different(self):
        """Test MMD for different distributions."""
        np.random.seed(42)
        p = np.random.randn(200, 3)
        q = np.random.randn(200, 3) + 1
        
        metric = MaximumMeanDiscrepancy()
        mmd = metric.compute(p, q)
        
        assert mmd > 0.1
    
    def test_mmd_kernel_selection(self):
        """Test MMD with different kernels."""
        np.random.seed(42)
        p = np.random.randn(100, 3)
        q = np.random.randn(100, 3) + 0.5
        
        for kernel in ['rbf', 'linear', 'polynomial']:
            metric = MaximumMeanDiscrepancy(kernel=kernel)
            mmd = metric.compute(p, q)
            assert mmd >= 0


class TestTrainOnSyntheticTestOnReal:
    """Tests for TSTR utility metric."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for TSTR testing."""
        np.random.seed(42)
        n = 500
        
        # Real data
        X_real = np.random.randn(n, 5)
        y_real = (X_real[:, 0] + X_real[:, 1] > 0).astype(int)
        
        # Synthetic data (similar distribution)
        X_synth = np.random.randn(n, 5)
        y_synth = (X_synth[:, 0] + X_synth[:, 1] > 0).astype(int)
        
        return X_real, y_real, X_synth, y_synth
    
    def test_tstr_computation(self, sample_data):
        """Test TSTR metric computation."""
        X_real, y_real, X_synth, y_synth = sample_data
        
        metric = TrainOnSyntheticTestOnReal()
        result = metric.compute(X_synth, y_synth, X_real, y_real)
        
        assert 'accuracy' in result
        assert 0 <= result['accuracy'] <= 1
    
    def test_tstr_with_different_classifiers(self, sample_data):
        """Test TSTR with different classifiers."""
        X_real, y_real, X_synth, y_synth = sample_data
        
        results = {}
        for clf_type in ['logistic', 'random_forest', 'mlp']:
            metric = TrainOnSyntheticTestOnReal(classifier_type=clf_type)
            result = metric.compute(X_synth, y_synth, X_real, y_real)
            results[clf_type] = result['accuracy']
        
        # All should produce valid accuracies
        assert all(0 <= acc <= 1 for acc in results.values())


class TestStatisticalSimilarityEvaluator:
    """Tests for StatisticalSimilarityEvaluator class."""
    
    @pytest.fixture
    def evaluator(self):
        """Create evaluator instance."""
        return StatisticalSimilarityEvaluator()
    
    @pytest.fixture
    def sample_dataframes(self):
        """Create sample dataframes for comparison."""
        np.random.seed(42)
        n = 500
        
        df_real = pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'feature_3': np.random.choice(['A', 'B', 'C'], n),
            'target': np.random.choice([0, 1], n)
        })
        
        df_synth = pd.DataFrame({
            'feature_1': np.random.randn(n) + 0.1,
            'feature_2': np.random.randn(n),
            'feature_3': np.random.choice(['A', 'B', 'C'], n),
            'target': np.random.choice([0, 1], n)
        })
        
        return df_real, df_synth
    
    def test_full_evaluation(self, evaluator, sample_dataframes):
        """Test full statistical similarity evaluation."""
        df_real, df_synth = sample_dataframes
        
        results = evaluator.evaluate(df_real, df_synth)
        
        assert 'overall_score' in results
        assert 'feature_scores' in results
        assert 0 <= results['overall_score'] <= 1
    
    def test_feature_level_scores(self, evaluator, sample_dataframes):
        """Test feature-level scores."""
        df_real, df_synth = sample_dataframes
        
        results = evaluator.evaluate(df_real, df_synth)
        
        for col in df_real.columns:
            assert col in results['feature_scores']
    
    def test_custom_metrics(self):
        """Test evaluator with custom metrics."""
        evaluator = StatisticalSimilarityEvaluator(
            metrics=['js_divergence', 'wasserstein', 'ks']
        )
        
        np.random.seed(42)
        df_real = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        df_synth = pd.DataFrame(np.random.randn(100, 3), columns=['a', 'b', 'c'])
        
        results = evaluator.evaluate(df_real, df_synth)
        
        assert results is not None


class TestDistributionComparator:
    """Tests for DistributionComparator class."""
    
    @pytest.fixture
    def comparator(self):
        """Create comparator instance."""
        return DistributionComparator()
    
    def test_compare_distributions(self, comparator):
        """Test distribution comparison."""
        np.random.seed(42)
        p = np.random.randn(500)
        q = np.random.randn(500) + 0.5
        
        results = comparator.compare(p, q)
        
        assert 'ks_test' in results
        assert 'anderson_darling' in results
        assert 'energy_distance' in results
    
    def test_multivariate_comparison(self, comparator):
        """Test multivariate distribution comparison."""
        np.random.seed(42)
        p = np.random.randn(100, 5)
        q = np.random.randn(100, 5) + 0.5
        
        results = comparator.compare_multivariate(p, q)
        
        assert 'mmd' in results
        assert 'energy_distance' in results
    
    def test_significance_testing(self, comparator):
        """Test significance testing."""
        np.random.seed(42)
        
        # Identical distributions
        p = np.random.randn(500)
        q = p.copy()
        
        results = comparator.compare_with_significance(p, q)
        
        assert 'significant_difference' in results
        assert results['significant_difference'] is False


class TestEdgeCases:
    """Edge case tests for fidelity metrics."""
    
    def test_single_feature(self):
        """Test with single feature."""
        np.random.seed(42)
        p = np.random.randn(100)
        q = np.random.randn(100)
        
        metric = WassersteinDistance()
        wd = metric.compute(p, q)
        
        assert wd >= 0
    
    def test_constant_feature(self):
        """Test with constant feature."""
        p = np.ones(100)
        q = np.ones(100)
        
        metric = JensenShannonDivergence()
        jsd = metric.compute(p, q)
        
        # Should handle constant features
        assert not np.isnan(jsd)
    
    def test_different_sizes(self):
        """Test with different sample sizes."""
        np.random.seed(42)
        p = np.random.randn(100)
        q = np.random.randn(200)
        
        metric = WassersteinDistance()
        wd = metric.compute(p, q)
        
        assert wd >= 0
    
    def test_high_dimensional(self):
        """Test with high-dimensional data."""
        np.random.seed(42)
        p = np.random.randn(100, 50)
        q = np.random.randn(100, 50)
        
        metric = MaximumMeanDiscrepancy()
        mmd = metric.compute(p, q)
        
        assert mmd >= 0
    
    def test_missing_values(self):
        """Test handling of missing values."""
        np.random.seed(42)
        p = np.random.randn(100)
        q = np.random.randn(100)
        q[::10] = np.nan  # 10% missing
        
        metric = WassersteinDistance()
        
        # Should handle or raise
        try:
            wd = metric.compute(p, q)
            assert not np.isnan(wd)
        except ValueError:
            pass  # Acceptable to raise
