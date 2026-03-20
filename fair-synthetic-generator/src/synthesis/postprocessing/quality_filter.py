"""
Quality Filter
==============

Filters synthetic data based on quality criteria including:
- Statistical quality assessment
- Outlier detection
- Distribution matching
- Sample validity
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from scipy import stats
import warnings


class QualityMetric(ABC):
    """Abstract base class for quality metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get metric name."""
        pass
    
    @abstractmethod
    def compute(
        self,
        sample: np.ndarray,
        reference: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute quality score for a sample.
        
        Args:
            sample: Single sample to score
            reference: Optional reference data for comparison
            
        Returns:
            Quality score (higher is better)
        """
        pass


class DistanceQualityMetric(QualityMetric):
    """
    Quality metric based on distance to reference distribution.
    
    Samples closer to the reference distribution center are scored higher.
    """
    
    @property
    def name(self) -> str:
        return "distance"
    
    def compute(
        self,
        sample: np.ndarray,
        reference: Optional[np.ndarray] = None
    ) -> float:
        """Compute distance-based quality score."""
        if reference is None:
            return 1.0
        
        # Compute mean of reference
        ref_mean = np.mean(reference, axis=0)
        
        # Compute distance to reference mean
        if sample.ndim == 1:
            dist = np.linalg.norm(sample - ref_mean)
        else:
            dist = np.linalg.norm(sample - ref_mean, axis=1)
        
        # Normalize by reference std
        ref_std = np.std(reference, axis=0) + 1e-8
        normalized_dist = dist / np.mean(ref_std)
        
        # Convert to score (exponential decay)
        score = np.exp(-normalized_dist)
        
        return float(score) if np.isscalar(score) else score


class DensityQualityMetric(QualityMetric):
    """
    Quality metric based on density estimation.
    
    Samples in high-density regions are scored higher.
    """
    
    def __init__(self, bandwidth: float = 1.0):
        """
        Initialize density quality metric.
        
        Args:
            bandwidth: Bandwidth for kernel density estimation
        """
        self.bandwidth = bandwidth
        self._kde = None
    
    @property
    def name(self) -> str:
        return "density"
    
    def fit(self, reference: np.ndarray) -> None:
        """Fit KDE to reference data."""
        from sklearn.neighbors import KernelDensity
        
        if reference.ndim == 1:
            reference = reference.reshape(-1, 1)
        
        self._kde = KernelDensity(bandwidth=self.bandwidth)
        self._kde.fit(reference)
    
    def compute(
        self,
        sample: np.ndarray,
        reference: Optional[np.ndarray] = None
    ) -> float:
        """Compute density-based quality score."""
        if reference is not None:
            self.fit(reference)
        
        if self._kde is None:
            return 1.0
        
        if sample.ndim == 1:
            sample = sample.reshape(1, -1)
        
        # Compute log density
        log_density = self._kde.score_samples(sample)
        
        # Convert to score (exponential)
        score = np.exp(log_density / 10)  # Scale for better range
        
        return float(score[0]) if len(score) == 1 else score


class ReconstructionQualityMetric(QualityMetric):
    """
    Quality metric based on reconstruction error.
    
    Used with generative models to assess sample quality.
    """
    
    def __init__(self, model: Optional[Any] = None):
        """
        Initialize reconstruction quality metric.
        
        Args:
            model: Model with encode/decode methods
        """
        self.model = model
    
    @property
    def name(self) -> str:
        return "reconstruction"
    
    def compute(
        self,
        sample: np.ndarray,
        reference: Optional[np.ndarray] = None
    ) -> float:
        """Compute reconstruction-based quality score."""
        if self.model is None:
            return 1.0
        
        try:
            import torch
            
            with torch.no_grad():
                sample_tensor = torch.tensor(sample).float()
                
                if hasattr(self.model, 'encode'):
                    latent = self.model.encode(sample_tensor)
                elif hasattr(self.model, 'encoder'):
                    latent = self.model.encoder(sample_tensor)
                else:
                    return 1.0
                
                if hasattr(self.model, 'decode'):
                    reconstructed = self.model.decode(latent)
                elif hasattr(self.model, 'decoder'):
                    reconstructed = self.model.decoder(latent)
                else:
                    return 1.0
                
                # Compute reconstruction error
                error = torch.mean((sample_tensor - reconstructed) ** 2).item()
                
                # Convert to score
                score = 1 / (1 + error)
                
                return float(score)
        
        except Exception:
            return 1.0


class OutlierDetector:
    """
    Detects outliers in synthetic data.
    
    Supports multiple detection methods.
    """
    
    def __init__(
        self,
        method: str = "iqr",
        threshold: float = 1.5,
        contamination: float = 0.1
    ):
        """
        Initialize outlier detector.
        
        Args:
            method: Detection method ('iqr', 'zscore', 'isolation', 'lof')
            threshold: Threshold for outlier detection
            contamination: Expected proportion of outliers (for isolation/lof)
        """
        self.method = method
        self.threshold = threshold
        self.contamination = contamination
    
    def detect(
        self,
        data: np.ndarray,
        return_scores: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect outliers.
        
        Args:
            data: Data array
            return_scores: Whether to return outlier scores
            
        Returns:
            Boolean mask (True for non-outliers) or (mask, scores)
        """
        if self.method == "iqr":
            mask, scores = self._detect_iqr(data)
        elif self.method == "zscore":
            mask, scores = self._detect_zscore(data)
        elif self.method == "isolation":
            mask, scores = self._detect_isolation(data)
        elif self.method == "lof":
            mask, scores = self._detect_lof(data)
        else:
            mask = np.ones(len(data), dtype=bool)
            scores = np.zeros(len(data))
        
        if return_scores:
            return mask, scores
        return mask
    
    def _detect_iqr(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect outliers using IQR method."""
        scores = np.zeros(len(data))
        mask = np.ones(len(data), dtype=bool)
        
        for col in range(data.shape[1]):
            col_data = data[:, col]
            
            q1 = np.percentile(col_data, 25)
            q3 = np.percentile(col_data, 75)
            iqr = q3 - q1
            
            lower = q1 - self.threshold * iqr
            upper = q3 + self.threshold * iqr
            
            col_mask = (col_data >= lower) & (col_data <= upper)
            mask = mask & col_mask
            
            # Score based on distance from bounds
            col_scores = np.zeros(len(data))
            below = col_data < lower
            above = col_data > upper
            
            col_scores[below] = (lower - col_data[below]) / (iqr + 1e-8)
            col_scores[above] = (col_data[above] - upper) / (iqr + 1e-8)
            
            scores = np.maximum(scores, col_scores)
        
        return mask, scores
    
    def _detect_zscore(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect outliers using Z-score method."""
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0) + 1e-8
        
        z_scores = np.abs((data - means) / stds)
        
        mask = (z_scores < self.threshold).all(axis=1)
        scores = z_scores.max(axis=1)
        
        return mask, scores
    
    def _detect_isolation(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect outliers using Isolation Forest."""
        try:
            from sklearn.ensemble import IsolationForest
            
            clf = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
            predictions = clf.fit_predict(data)
            
            mask = predictions == 1
            scores = -clf.score_samples(data)  # Higher = more anomalous
            
            return mask, scores
        
        except ImportError:
            return np.ones(len(data), dtype=bool), np.zeros(len(data))
    
    def _detect_lof(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect outliers using Local Outlier Factor."""
        try:
            from sklearn.neighbors import LocalOutlierFactor
            
            clf = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=False
            )
            predictions = clf.fit_predict(data)
            
            mask = predictions == 1
            scores = -clf.negative_outlier_factor_
            
            return mask, scores
        
        except ImportError:
            return np.ones(len(data), dtype=bool), np.zeros(len(data))


class QualityFilter:
    """
    Comprehensive quality filter for synthetic data.
    
    Combines multiple quality assessment methods.
    """
    
    def __init__(
        self,
        quality_threshold: float = 0.5,
        outlier_method: str = "iqr",
        outlier_threshold: float = 1.5,
        use_reference: bool = True,
        quality_metrics: Optional[List[QualityMetric]] = None
    ):
        """
        Initialize quality filter.
        
        Args:
            quality_threshold: Minimum quality score threshold
            outlier_method: Method for outlier detection
            outlier_threshold: Threshold for outlier detection
            use_reference: Whether to use reference data for quality assessment
            quality_metrics: List of quality metrics to use
        """
        self.quality_threshold = quality_threshold
        self.use_reference = use_reference
        
        # Initialize outlier detector
        self.outlier_detector = OutlierDetector(
            method=outlier_method,
            threshold=outlier_threshold
        )
        
        # Initialize quality metrics
        self.quality_metrics = quality_metrics or [
            DistanceQualityMetric(),
        ]
    
    def filter(
        self,
        data: np.ndarray,
        reference: Optional[np.ndarray] = None,
        quality_scores: Optional[np.ndarray] = None,
        return_report: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Filter data based on quality criteria.
        
        Args:
            data: Data to filter
            reference: Optional reference data for comparison
            quality_scores: Optional pre-computed quality scores
            return_report: Whether to return detailed report
            
        Returns:
            Filtered data (and optionally report)
        """
        n_samples = len(data)
        
        report = {
            "input_samples": n_samples,
            "output_samples": 0,
            "filter_steps": [],
            "removal_reasons": {},
        }
        
        # Initialize mask
        mask = np.ones(n_samples, dtype=bool)
        
        # Step 1: Remove invalid samples
        validity_mask = self._check_validity(data)
        removed_invalid = (~validity_mask).sum()
        mask = mask & validity_mask
        
        if removed_invalid > 0:
            report["removal_reasons"]["invalid"] = removed_invalid
            report["filter_steps"].append({
                "step": "validity_check",
                "removed": removed_invalid,
                "remaining": mask.sum(),
            })
        
        # Step 2: Outlier detection
        outlier_mask, outlier_scores = self.outlier_detector.detect(
            data[mask], return_scores=True
        )
        
        # Map back to original indices
        full_outlier_mask = np.ones(n_samples, dtype=bool)
        full_outlier_mask[mask] = outlier_mask
        removed_outliers = (~full_outlier_mask).sum()
        mask = mask & full_outlier_mask
        
        if removed_outliers > 0:
            report["removal_reasons"]["outliers"] = removed_outliers
            report["filter_steps"].append({
                "step": "outlier_detection",
                "removed": removed_outliers,
                "remaining": mask.sum(),
                "outlier_scores": {
                    "mean": float(np.mean(outlier_scores)),
                    "max": float(np.max(outlier_scores)),
                },
            })
        
        # Step 3: Quality scoring
        if quality_scores is not None:
            quality_mask = quality_scores >= self.quality_threshold
        elif self.use_reference and reference is not None:
            quality_mask, quality_scores = self._compute_quality(
                data[mask], reference
            )
            full_quality_mask = np.ones(n_samples, dtype=bool)
            full_quality_mask[mask] = quality_mask
            quality_mask = full_quality_mask
        else:
            quality_mask = np.ones(n_samples, dtype=bool)
        
        removed_low_quality = (~quality_mask).sum()
        mask = mask & quality_mask
        
        if removed_low_quality > 0:
            report["removal_reasons"]["low_quality"] = removed_low_quality
            report["filter_steps"].append({
                "step": "quality_filter",
                "removed": removed_low_quality,
                "remaining": mask.sum(),
                "quality_threshold": self.quality_threshold,
            })
        
        # Apply mask
        filtered_data = data[mask]
        
        report["output_samples"] = len(filtered_data)
        report["retention_rate"] = len(filtered_data) / n_samples
        report["is_high_quality"] = report["retention_rate"] >= 0.8
        
        if return_report:
            return filtered_data, report
        return filtered_data
    
    def _check_validity(self, data: np.ndarray) -> np.ndarray:
        """Check sample validity."""
        mask = np.ones(len(data), dtype=bool)
        
        # Check for NaN
        if np.isnan(data).any():
            nan_mask = ~np.isnan(data).any(axis=1)
            mask = mask & nan_mask
        
        # Check for Inf
        if np.isinf(data).any():
            inf_mask = ~np.isinf(data).any(axis=1)
            mask = mask & inf_mask
        
        # Check for constant samples
        sample_std = np.std(data, axis=1)
        constant_mask = sample_std > 1e-10
        mask = mask & constant_mask
        
        return mask
    
    def _compute_quality(
        self,
        data: np.ndarray,
        reference: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute quality scores."""
        # Fit density metric to reference
        for metric in self.quality_metrics:
            if hasattr(metric, 'fit'):
                metric.fit(reference)
        
        # Compute combined quality score
        scores = np.ones(len(data))
        
        for metric in self.quality_metrics:
            metric_score = metric.compute(data, reference)
            
            if np.isscalar(metric_score):
                metric_score = np.array([metric_score])
            
            scores = scores * metric_score
        
        # Normalize to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        quality_mask = scores >= self.quality_threshold
        
        return quality_mask, scores
    
    def filter_with_replacement(
        self,
        data: np.ndarray,
        target_size: int,
        reference: Optional[np.ndarray] = None,
        max_attempts: int = 10
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Filter and replace removed samples.
        
        Args:
            data: Data to filter
            target_size: Target number of samples
            reference: Reference data
            max_attempts: Maximum regeneration attempts
            
        Returns:
            Tuple of (filtered/replaced data, report)
        """
        report = {
            "target_size": target_size,
            "attempts": 0,
            "final_size": 0,
        }
        
        current_data = data.copy()
        
        for attempt in range(max_attempts):
            report["attempts"] = attempt + 1
            
            # Filter
            filtered_data, filter_report = self.filter(
                current_data, reference, return_report=True
            )
            
            if len(filtered_data) >= target_size:
                report["final_size"] = len(filtered_data[:target_size])
                return filtered_data[:target_size], report
            
            # Need more samples
            needed = target_size - len(filtered_data)
            
            # Resample from remaining
            if len(filtered_data) > 0:
                indices = np.random.choice(
                    len(filtered_data),
                    size=needed,
                    replace=True
                )
                additional = filtered_data[indices]
                
                # Add noise for variation
                noise = np.random.normal(0, 0.01, additional.shape)
                additional = additional + noise
                
                current_data = np.vstack([filtered_data, additional])
            else:
                # All samples were filtered, return best effort
                report["final_size"] = len(data[:target_size])
                return data[:target_size], report
        
        # Max attempts reached
        filtered_data, _ = self.filter(current_data, reference, return_report=True)
        report["final_size"] = len(filtered_data)
        
        if len(filtered_data) < target_size:
            # Pad with available data
            if len(filtered_data) > 0:
                indices = np.random.choice(
                    len(filtered_data),
                    size=target_size - len(filtered_data),
                    replace=True
                )
                padding = filtered_data[indices]
                filtered_data = np.vstack([filtered_data, padding])
            else:
                filtered_data = data[:target_size]
        
        report["final_size"] = len(filtered_data)
        return filtered_data, report


class AdaptiveQualityFilter(QualityFilter):
    """
    Adaptive quality filter that adjusts thresholds based on data characteristics.
    """
    
    def __init__(
        self,
        initial_threshold: float = 0.5,
        min_threshold: float = 0.1,
        max_threshold: float = 0.9,
        target_retention: float = 0.8,
        adaptation_rate: float = 0.1
    ):
        """
        Initialize adaptive quality filter.
        
        Args:
            initial_threshold: Initial quality threshold
            min_threshold: Minimum threshold value
            max_threshold: Maximum threshold value
            target_retention: Target retention rate
            adaptation_rate: Rate of threshold adaptation
        """
        super().__init__(quality_threshold=initial_threshold)
        
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.target_retention = target_retention
        self.adaptation_rate = adaptation_rate
        
        self._threshold_history = [initial_threshold]
    
    def filter(
        self,
        data: np.ndarray,
        reference: Optional[np.ndarray] = None,
        quality_scores: Optional[np.ndarray] = None,
        return_report: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, Any]]]:
        """Filter with adaptive threshold adjustment."""
        result = super().filter(
            data, reference, quality_scores, return_report=True
        )
        
        filtered_data, report = result
        
        # Adjust threshold based on retention rate
        actual_retention = report["retention_rate"]
        
        if actual_retention < self.target_retention - 0.1:
            # Too many filtered, lower threshold
            self.quality_threshold = max(
                self.min_threshold,
                self.quality_threshold - self.adaptation_rate
            )
        elif actual_retention > self.target_retention + 0.1:
            # Too few filtered, raise threshold
            self.quality_threshold = min(
                self.max_threshold,
                self.quality_threshold + self.adaptation_rate
            )
        
        self._threshold_history.append(self.quality_threshold)
        report["adapted_threshold"] = self.quality_threshold
        report["threshold_history"] = self._threshold_history[-5:]
        
        if return_report:
            return filtered_data, report
        return filtered_data
    
    def get_threshold_history(self) -> List[float]:
        """Get threshold adaptation history."""
        return self._threshold_history


class StreamingQualityFilter:
    """
    Quality filter that processes data in a streaming fashion.
    
    Useful for real-time filtering or large datasets.
    """
    
    def __init__(
        self,
        quality_filter: QualityFilter,
        buffer_size: int = 10000,
        min_output_size: int = 1000
    ):
        """
        Initialize streaming quality filter.
        
        Args:
            quality_filter: Base quality filter to use
            buffer_size: Size of internal buffer
            min_output_size: Minimum samples before yielding output
        """
        self.quality_filter = quality_filter
        self.buffer_size = buffer_size
        self.min_output_size = min_output_size
        
        self._buffer = []
        self._total_input = 0
        self._total_output = 0
    
    def add(self, samples: np.ndarray) -> Optional[np.ndarray]:
        """
        Add samples to buffer and filter if buffer is full.
        
        Args:
            samples: Samples to add
            
        Returns:
            Filtered samples if buffer exceeded, None otherwise
        """
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        
        for i in range(len(samples)):
            self._buffer.append(samples[i])
            self._total_input += 1
            
            if len(self._buffer) >= self.buffer_size:
                result = self._filter_buffer()
                return result
        
        return None
    
    def _filter_buffer(self) -> np.ndarray:
        """Filter current buffer."""
        data = np.array(self._buffer)
        filtered, _ = self.quality_filter.filter(data, return_report=True)
        
        self._total_output += len(filtered)
        self._buffer = []
        
        return filtered
    
    def flush(self) -> np.ndarray:
        """Flush remaining buffer."""
        if not self._buffer:
            return np.array([])
        
        return self._filter_buffer()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        return {
            "total_input": self._total_input,
            "total_output": self._total_output,
            "overall_retention": (
                self._total_output / self._total_input 
                if self._total_input > 0 else 0
            ),
            "buffer_size": len(self._buffer),
        }
