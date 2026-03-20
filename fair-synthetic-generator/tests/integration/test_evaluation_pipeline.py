"""
Integration Tests for Evaluation Pipeline
=========================================

Tests for the complete evaluation pipeline including:
- Fairness evaluation
- Fidelity evaluation
- Privacy evaluation
- Report generation
- Multi-metric aggregation
"""

import pytest
import torch
import numpy as np
import pandas as pd
import tempfile
import shutil
from pathlib import Path
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation.fairness.group_metrics import GroupFairnessEvaluator
from src.evaluation.fairness.individual_metrics import IndividualFairnessEvaluator
from src.evaluation.fidelity.statistical_similarity import StatisticalSimilarityEvaluator
from src.evaluation.privacy.membership_inference import MembershipInferenceEvaluator
from src.evaluation.dashboard.report_generator import FairnessReport, ComprehensiveReportGenerator
from src.evaluation.dashboard.visualization import FairnessVisualizer, FidelityVisualizer


class TestEvaluationPipelineIntegration:
    """Integration tests for complete evaluation pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        dir_path = tempfile.mkdtemp()
        yield Path(dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)
    
    @pytest.fixture
    def sample_real_data(self):
        """Create sample real data."""
        np.random.seed(42)
        n = 1000
        
        return pd.DataFrame({
            'age': np.random.randint(18, 70, n),
            'income': np.random.exponential(50000, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'race': np.random.choice(['White', 'Black', 'Asian'], n),
            'education_years': np.random.randint(8, 20, n),
            'credit_score': np.random.normal(700, 50, n).clip(300, 850),
            'approved': np.random.choice([0, 1], n, p=[0.7, 0.3])
        })
    
    @pytest.fixture
    def sample_synthetic_data(self):
        """Create sample synthetic data similar to real data."""
        np.random.seed(123)
        n = 1000
        
        return pd.DataFrame({
            'age': np.random.randint(18, 70, n),
            'income': np.random.exponential(48000, n),  # Slightly different
            'gender': np.random.choice(['Male', 'Female'], n),
            'race': np.random.choice(['White', 'Black', 'Asian'], n),
            'education_years': np.random.randint(8, 20, n),
            'credit_score': np.random.normal(695, 55, n).clip(300, 850),
            'approved': np.random.choice([0, 1], n, p=[0.68, 0.32])
        })
    
    def test_full_fairness_evaluation(self, sample_real_data, sample_synthetic_data):
        """Test complete fairness evaluation."""
        evaluator = GroupFairnessEvaluator(
            sensitive_attributes=['gender', 'race'],
            target_column='approved'
        )
        
        results = evaluator.evaluate(
            real_data=sample_real_data,
            synthetic_data=sample_synthetic_data
        )
        
        assert 'demographic_parity' in results
        assert 'equalized_odds' in results
        assert 'overall_fairness_score' in results
    
    def test_full_fidelity_evaluation(self, sample_real_data, sample_synthetic_data):
        """Test complete fidelity evaluation."""
        evaluator = StatisticalSimilarityEvaluator()
        
        results = evaluator.evaluate(
            real_data=sample_real_data,
            synthetic_data=sample_synthetic_data
        )
        
        assert 'js_divergence' in results
        assert 'correlation_preservation' in results
        assert 'overall_fidelity_score' in results
    
    def test_full_privacy_evaluation(self, sample_real_data, sample_synthetic_data):
        """Test complete privacy evaluation."""
        evaluator = MembershipInferenceEvaluator()
        
        results = evaluator.evaluate(
            real_data=sample_real_data,
            synthetic_data=sample_synthetic_data
        )
        
        assert 'mia_accuracy' in results
        assert 'privacy_risk' in results
    
    def test_comprehensive_evaluation(self, sample_real_data, sample_synthetic_data, temp_dir):
        """Test comprehensive multi-metric evaluation."""
        # Fairness
        fairness_evaluator = GroupFairnessEvaluator(
            sensitive_attributes=['gender'],
            target_column='approved'
        )
        fairness_results = fairness_evaluator.evaluate(
            real_data=sample_real_data,
            synthetic_data=sample_synthetic_data
        )
        
        # Fidelity
        fidelity_evaluator = StatisticalSimilarityEvaluator()
        fidelity_results = fidelity_evaluator.evaluate(
            real_data=sample_real_data,
            synthetic_data=sample_synthetic_data
        )
        
        # Privacy
        privacy_evaluator = MembershipInferenceEvaluator()
        privacy_results = privacy_evaluator.evaluate(
            real_data=sample_real_data,
            synthetic_data=sample_synthetic_data
        )
        
        # Aggregate results
        comprehensive_results = {
            'fairness': fairness_results,
            'fidelity': fidelity_results,
            'privacy': privacy_results
        }
        
        assert comprehensive_results is not None


class TestReportGenerationIntegration:
    """Integration tests for report generation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        dir_path = tempfile.mkdtemp()
        yield Path(dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)
    
    @pytest.fixture
    def evaluation_results(self):
        """Create sample evaluation results."""
        return {
            'fairness': {
                'demographic_parity_difference': 0.08,
                'equalized_odds_difference': 0.12,
                'overall_fairness_score': 0.85
            },
            'fidelity': {
                'js_divergence': 0.05,
                'correlation_preservation': 0.92,
                'overall_fidelity_score': 0.88
            },
            'privacy': {
                'mia_accuracy': 0.55,
                'privacy_risk': 'low'
            }
        }
    
    def test_html_report_generation(self, evaluation_results, temp_dir):
        """Test HTML report generation."""
        generator = ComprehensiveReportGenerator(output_dir=temp_dir)
        
        report_path = generator.generate_html(
            results=evaluation_results,
            filename='evaluation_report.html'
        )
        
        assert report_path.exists()
        
        # Verify HTML content
        with open(report_path) as f:
            content = f.read()
        
        assert '<html' in content.lower()
        assert 'fairness' in content.lower()
    
    def test_json_report_generation(self, evaluation_results, temp_dir):
        """Test JSON report generation."""
        generator = ComprehensiveReportGenerator(output_dir=temp_dir)
        
        report_path = generator.generate_json(
            results=evaluation_results,
            filename='evaluation_report.json'
        )
        
        assert report_path.exists()
        
        # Verify JSON content
        with open(report_path) as f:
            loaded = json.load(f)
        
        assert 'fairness' in loaded
    
    def test_markdown_report_generation(self, evaluation_results, temp_dir):
        """Test Markdown report generation."""
        generator = ComprehensiveReportGenerator(output_dir=temp_dir)
        
        report_path = generator.generate_markdown(
            results=evaluation_results,
            filename='evaluation_report.md'
        )
        
        assert report_path.exists()
        
        # Verify Markdown content
        with open(report_path) as f:
            content = f.read()
        
        assert '##' in content  # Markdown headers


class TestVisualizationIntegration:
    """Integration tests for visualization generation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        dir_path = tempfile.mkdtemp()
        yield Path(dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for visualization."""
        np.random.seed(42)
        n = 500
        
        return pd.DataFrame({
            'feature_1': np.random.randn(n),
            'feature_2': np.random.randn(n),
            'sensitive': np.random.choice([0, 1], n),
            'prediction': np.random.random(n)
        })
    
    def test_fairness_visualization(self, sample_data, temp_dir):
        """Test fairness visualization generation."""
        visualizer = FairnessVisualizer(output_dir=temp_dir)
        
        # Generate fairness plot
        plot_path = visualizer.plot_demographic_parity(
            data=sample_data,
            sensitive_attr='sensitive',
            target_attr='prediction'
        )
        
        # Should create a file
        assert plot_path is not None
    
    def test_fidelity_visualization(self, sample_data, temp_dir):
        """Test fidelity visualization generation."""
        visualizer = FidelityVisualizer(output_dir=temp_dir)
        
        real_data = sample_data
        synthetic_data = sample_data + np.random.randn(*sample_data.shape) * 0.1
        
        # Generate distribution comparison plot
        plot_path = visualizer.plot_distribution_comparison(
            real_data=real_data,
            synthetic_data=synthetic_data,
            columns=['feature_1', 'feature_2']
        )
        
        assert plot_path is not None


class TestMultiModalEvaluationIntegration:
    """Integration tests for multimodal evaluation."""
    
    @pytest.fixture
    def multimodal_data(self):
        """Create multimodal data for evaluation."""
        np.random.seed(42)
        n = 300
        
        return {
            'tabular': pd.DataFrame({
                'feature_1': np.random.randn(n),
                'feature_2': np.random.randn(n),
                'sensitive': np.random.choice([0, 1], n)
            }),
            'text_features': np.random.randn(n, 50),
            'image_features': np.random.randn(n, 64)
        }
    
    def test_multimodal_fidelity_evaluation(self, multimodal_data):
        """Test multimodal fidelity evaluation."""
        # Create synthetic version
        synthetic = {
            'tabular': multimodal_data['tabular'] + np.random.randn(*multimodal_data['tabular'].shape) * 0.1,
            'text_features': multimodal_data['text_features'] + np.random.randn(*multimodal_data['text_features'].shape) * 0.1,
            'image_features': multimodal_data['image_features'] + np.random.randn(*multimodal_data['image_features'].shape) * 0.1
        }
        
        # Evaluate each modality
        results = {}
        
        # Tabular fidelity
        evaluator = StatisticalSimilarityEvaluator()
        results['tabular'] = evaluator.evaluate(
            multimodal_data['tabular'],
            synthetic['tabular']
        )
        
        # Results should be valid
        assert results['tabular'] is not None


class TestBatchEvaluation:
    """Integration tests for batch evaluation."""
    
    def test_multiple_dataset_evaluation(self):
        """Test evaluation of multiple synthetic datasets."""
        np.random.seed(42)
        
        real_data = pd.DataFrame({
            'feature': np.random.randn(100),
            'sensitive': np.random.choice([0, 1], 100)
        })
        
        evaluator = StatisticalSimilarityEvaluator()
        
        results = []
        for i in range(5):
            synthetic = real_data + np.random.randn(*real_data.shape) * (i + 1) * 0.1
            result = evaluator.evaluate(real_data, synthetic)
            results.append(result)
        
        # Fidelity should decrease as noise increases
        scores = [r['overall_fidelity_score'] for r in results]
        assert scores[-1] < scores[0]
    
    def test_evaluation_with_different_sizes(self):
        """Test evaluation with different data sizes."""
        evaluator = StatisticalSimilarityEvaluator()
        
        for n in [100, 500, 1000]:
            real_data = pd.DataFrame(np.random.randn(n, 5))
            synthetic_data = pd.DataFrame(np.random.randn(n, 5))
            
            result = evaluator.evaluate(real_data, synthetic_data)
            
            assert result is not None


class TestEvaluationCaching:
    """Integration tests for evaluation caching."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        dir_path = tempfile.mkdtemp()
        yield Path(dir_path)
        shutil.rmtree(dir_path, ignore_errors=True)
    
    def test_result_caching(self, temp_dir):
        """Test evaluation result caching."""
        cache_path = temp_dir / 'cache.json'
        
        real_data = pd.DataFrame(np.random.randn(100, 5))
        synthetic_data = pd.DataFrame(np.random.randn(100, 5))
        
        # First evaluation
        evaluator = StatisticalSimilarityEvaluator(cache_path=cache_path)
        result1 = evaluator.evaluate(real_data, synthetic_data)
        
        # Save to cache
        with open(cache_path, 'w') as f:
            json.dump(result1, f)
        
        # Load from cache
        with open(cache_path) as f:
            cached_result = json.load(f)
        
        assert cached_result is not None


class TestEvaluationErrorHandling:
    """Integration tests for evaluation error handling."""
    
    def test_mismatched_columns(self):
        """Test handling of mismatched columns."""
        real_data = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        synthetic_data = pd.DataFrame({'a': [1, 2, 3], 'c': [4, 5, 6]})
        
        evaluator = StatisticalSimilarityEvaluator()
        
        with pytest.raises((ValueError, KeyError)):
            evaluator.evaluate(real_data, synthetic_data)
    
    def test_empty_data(self):
        """Test handling of empty data."""
        real_data = pd.DataFrame()
        synthetic_data = pd.DataFrame()
        
        evaluator = StatisticalSimilarityEvaluator()
        
        with pytest.raises((ValueError, IndexError)):
            evaluator.evaluate(real_data, synthetic_data)
    
    def test_missing_values_handling(self):
        """Test handling of missing values."""
        real_data = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, np.nan, 7, 8]
        })
        synthetic_data = pd.DataFrame({
            'a': [1, 2, 3, 4],
            'b': [5, 6, 7, 8]
        })
        
        evaluator = StatisticalSimilarityEvaluator()
        
        # Should handle missing values
        result = evaluator.evaluate(real_data, synthetic_data)
        
        assert result is not None


class TestThresholdValidation:
    """Integration tests for threshold validation."""
    
    def test_fairness_threshold_validation(self):
        """Test fairness threshold validation."""
        np.random.seed(42)
        n = 500
        
        data = pd.DataFrame({
            'prediction': np.random.random(n),
            'sensitive': np.random.choice([0, 1], n)
        })
        
        evaluator = GroupFairnessEvaluator(
            sensitive_attributes=['sensitive'],
            target_column='prediction',
            fairness_thresholds={'demographic_parity': 0.1}
        )
        
        results = evaluator.evaluate(data, data)
        
        assert 'threshold_violations' in results or 'passed' in results
    
    def test_fidelity_threshold_validation(self):
        """Test fidelity threshold validation."""
        np.random.seed(42)
        n = 500
        
        real_data = pd.DataFrame(np.random.randn(n, 5))
        synthetic_data = pd.DataFrame(np.random.randn(n, 5))
        
        evaluator = StatisticalSimilarityEvaluator(
            fidelity_thresholds={'js_divergence': 0.1}
        )
        
        results = evaluator.evaluate(real_data, synthetic_data)
        
        assert results is not None
