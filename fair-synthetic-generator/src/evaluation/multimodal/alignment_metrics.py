"""
Multimodal Alignment Metrics
=============================

Metrics for evaluating alignment between different modalities
in synthetic data generation, particularly for fairness-aware
multimodal systems.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine, cdist
import warnings


class MultimodalAlignmentScore:
    """
    Score for evaluating alignment between modalities.
    
    Uses contrastive learning principles to measure how well
    corresponding samples from different modalities are aligned.
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize alignment score.
        
        Args:
            temperature: Temperature for softmax normalization
            similarity_threshold: Threshold for considering samples aligned
        """
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
    
    def compute(
        self,
        modality_a: np.ndarray,
        modality_b: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute multimodal alignment score.
        
        Args:
            modality_a: First modality embeddings (batch, features)
            modality_b: Second modality embeddings (batch, features)
            labels: Optional alignment labels
            
        Returns:
            Dictionary with alignment metrics
        """
        batch_size = min(len(modality_a), len(modality_b))
        
        # Normalize
        mod_a = modality_a[:batch_size]
        mod_b = modality_b[:batch_size]
        
        mod_a_norm = mod_a / (np.linalg.norm(mod_a, axis=1, keepdims=True) + 1e-8)
        mod_b_norm = mod_b / (np.linalg.norm(mod_b, axis=1, keepdims=True) + 1e-8)
        
        # Compute similarity matrix
        similarity = np.dot(mod_a_norm, mod_b_norm.T) / self.temperature
        
        # InfoNCE-style alignment
        # For each item in mod_a, the correct match should be at the same index in mod_b
        
        # Compute alignment accuracy
        pred_a = similarity.argmax(axis=1)  # For each in a, predicted match in b
        pred_b = similarity.argmax(axis=0)  # For each in b, predicted match in a
        
        correct_a = (pred_a == np.arange(batch_size)).mean()
        correct_b = (pred_b == np.arange(batch_size)).mean()
        
        alignment_accuracy = (correct_a + correct_b) / 2
        
        # Compute alignment loss (InfoNCE-style)
        def softmax(x, axis=-1):
            e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            return e_x / e_x.sum(axis=axis, keepdims=True)
        
        prob_a = softmax(similarity, axis=1)
        prob_b = softmax(similarity.T, axis=1)
        
        # Negative log likelihood of correct matches
        nll_a = -np.log(prob_a[np.arange(batch_size), np.arange(batch_size)] + 1e-10).mean()
        nll_b = -np.log(prob_b[np.arange(batch_size), np.arange(batch_size)] + 1e-10).mean()
        
        alignment_loss = (nll_a + nll_b) / 2
        
        # Average similarity of correct pairs
        correct_sim = np.mean([
            1 - cosine(mod_a[i], mod_b[i])
            for i in range(batch_size)
        ])
        
        return {
            "alignment_accuracy": float(alignment_accuracy),
            "alignment_loss": float(alignment_loss),
            "correct_pair_similarity": float(correct_sim),
            "modality_a_accuracy": float(correct_a),
            "modality_b_accuracy": float(correct_b),
        }


class FairMultimodalAlignment:
    """
    Fair Multimodal Alignment Metric.
    
    Evaluates alignment quality across different demographic groups,
    ensuring fair alignment for all groups.
    """
    
    def __init__(
        self,
        alignment_threshold: float = 0.1
    ):
        """
        Initialize fair alignment metric.
        
        Args:
            alignment_threshold: Maximum acceptable difference in alignment
        """
        self.alignment_threshold = alignment_threshold
        self.alignment_scorer = MultimodalAlignmentScore()
    
    def compute(
        self,
        modality_a: np.ndarray,
        modality_b: np.ndarray,
        sensitive_attribute: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute fair multimodal alignment.
        
        Args:
            modality_a: First modality embeddings
            modality_b: Second modality embeddings
            sensitive_attribute: Sensitive attribute values
            
        Returns:
            Dictionary with fair alignment metrics
        """
        unique_groups = np.unique(sensitive_attribute)
        
        # Compute alignment per group
        group_alignment = {}
        
        for group in unique_groups:
            mask = sensitive_attribute == group
            n = min(mask.sum(), len(modality_a), len(modality_b))
            
            if n > 0:
                indices = np.where(mask)[0][:n]
                alignment = self.alignment_scorer.compute(
                    modality_a[indices],
                    modality_b[indices]
                )
                group_alignment[f"group_{group}"] = alignment
        
        # Compute alignment differences
        accuracies = [
            g["alignment_accuracy"] 
            for g in group_alignment.values()
        ]
        
        if len(accuracies) >= 2:
            max_diff = max(accuracies) - min(accuracies)
        else:
            max_diff = 0.0
        
        # Overall alignment
        overall_alignment = self.alignment_scorer.compute(modality_a, modality_b)
        
        return {
            "group_alignment": group_alignment,
            "overall_alignment": overall_alignment,
            "alignment_difference": float(max_diff),
            "is_fair": max_diff <= self.alignment_threshold,
        }


class ModalityGapMetric:
    """
    Modality Gap Metric.
    
    Measures the gap between modality representation spaces
    and evaluates whether this gap is consistent across groups.
    """
    
    def __init__(
        self,
        gap_threshold: float = 0.5
    ):
        """
        Initialize modality gap metric.
        
        Args:
            gap_threshold: Threshold for acceptable gap
        """
        self.gap_threshold = gap_threshold
    
    def compute(
        self,
        modality_a: np.ndarray,
        modality_b: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute modality gap.
        
        Args:
            modality_a: First modality embeddings
            modality_b: Second modality embeddings
            
        Returns:
            Dictionary with gap metrics
        """
        # Compute centroids
        centroid_a = modality_a.mean(axis=0)
        centroid_b = modality_b.mean(axis=0)
        
        # Normalize centroids
        centroid_a_norm = centroid_a / (np.linalg.norm(centroid_a) + 1e-8)
        centroid_b_norm = centroid_b / (np.linalg.norm(centroid_b) + 1e-8)
        
        # Gap: distance between centroids
        gap_distance = np.linalg.norm(centroid_a - centroid_b)
        gap_cosine = 1 - np.dot(centroid_a_norm, centroid_b_norm)
        
        # Compute spread within each modality
        spread_a = np.mean(np.linalg.norm(modality_a - centroid_a, axis=1))
        spread_b = np.mean(np.linalg.norm(modality_b - centroid_b, axis=1))
        
        # Relative gap (compared to spread)
        avg_spread = (spread_a + spread_b) / 2
        relative_gap = gap_distance / (avg_spread + 1e-8)
        
        return {
            "gap_distance": float(gap_distance),
            "gap_cosine": float(gap_cosine),
            "spread_modality_a": float(spread_a),
            "spread_modality_b": float(spread_b),
            "relative_gap": float(relative_gap),
            "is_aligned": relative_gap <= self.gap_threshold,
        }
    
    def compute_by_group(
        self,
        modality_a: np.ndarray,
        modality_b: np.ndarray,
        sensitive_attribute: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute modality gap by demographic group.
        
        Args:
            modality_a: First modality embeddings
            modality_b: Second modality embeddings
            sensitive_attribute: Sensitive attribute values
            
        Returns:
            Dictionary with group-specific gap metrics
        """
        unique_groups = np.unique(sensitive_attribute)
        
        group_gaps = {}
        
        for group in unique_groups:
            mask = sensitive_attribute == group
            n = min(mask.sum(), len(modality_a), len(modality_b))
            
            if n > 0:
                indices = np.where(mask)[0][:n]
                gap = self.compute(
                    modality_a[indices],
                    modality_b[indices]
                )
                group_gaps[f"group_{group}"] = gap
        
        # Compare gaps across groups
        relative_gaps = [g["relative_gap"] for g in group_gaps.values()]
        
        return {
            "group_gaps": group_gaps,
            "overall_gap": self.compute(modality_a, modality_b),
            "gap_variance": float(np.var(relative_gaps)) if relative_gaps else 0.0,
            "gap_range": float(max(relative_gaps) - min(relative_gaps)) if relative_gaps else 0.0,
        }


class ContrastiveAlignmentMetric:
    """
    Contrastive Alignment Metric.
    
    Uses contrastive learning to evaluate multimodal alignment,
    measuring how well positive pairs are separated from negative pairs.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        temperature: float = 0.1
    ):
        """
        Initialize contrastive alignment metric.
        
        Args:
            margin: Margin for contrastive loss
            temperature: Temperature for softmax
        """
        self.margin = margin
        self.temperature = temperature
    
    def compute(
        self,
        modality_a: np.ndarray,
        modality_b: np.ndarray,
        negative_pairs: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Compute contrastive alignment.
        
        Args:
            modality_a: First modality embeddings
            modality_b: Second modality embeddings
            negative_pairs: Optional explicit negative pairs
            
        Returns:
            Dictionary with contrastive metrics
        """
        batch_size = min(len(modality_a), len(modality_b))
        
        # Normalize
        mod_a = modality_a[:batch_size]
        mod_b = modality_b[:batch_size]
        
        mod_a_norm = mod_a / (np.linalg.norm(mod_a, axis=1, keepdims=True) + 1e-8)
        mod_b_norm = mod_b / (np.linalg.norm(mod_b, axis=1, keepdims=True) + 1e-8)
        
        # Positive similarity (diagonal)
        positive_sim = np.array([
            np.dot(mod_a_norm[i], mod_b_norm[i])
            for i in range(batch_size)
        ])
        
        # Negative similarities (off-diagonal)
        if negative_pairs is not None:
            neg_a, neg_b = negative_pairs
            neg_a_norm = neg_a / (np.linalg.norm(neg_a, axis=1, keepdims=True) + 1e-8)
            neg_b_norm = neg_b / (np.linalg.norm(neg_b, axis=1, keepdims=True) + 1e-8)
            
            negative_sim = np.array([
                np.dot(neg_a_norm[i], neg_b_norm[i])
                for i in range(min(len(neg_a), len(neg_b)))
            ])
        else:
            # Use off-diagonal elements as negatives
            negative_sim = []
            for i in range(batch_size):
                j = (i + 1) % batch_size
                negative_sim.append(np.dot(mod_a_norm[i], mod_b_norm[j]))
            negative_sim = np.array(negative_sim)
        
        # Contrastive accuracy
        accuracy = (positive_sim > negative_sim).mean()
        
        # Contrastive loss (margin-based)
        contrastive_loss = np.maximum(0, self.margin - positive_sim + negative_sim).mean()
        
        # InfoNCE loss
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        
        # Combine positive and negative similarities
        infonce_losses = []
        for i in range(batch_size):
            logits = np.concatenate([[positive_sim[i]], negative_sim])
            probs = softmax(logits / self.temperature)
            infonce_losses.append(-np.log(probs[0] + 1e-10))
        
        infonce_loss = np.mean(infonce_losses)
        
        return {
            "positive_similarity": float(positive_sim.mean()),
            "negative_similarity": float(negative_sim.mean()),
            "similarity_margin": float(positive_sim.mean() - negative_sim.mean()),
            "contrastive_accuracy": float(accuracy),
            "contrastive_loss": float(contrastive_loss),
            "infonce_loss": float(infonce_loss),
        }


class MultimodalAlignmentEvaluator:
    """
    Comprehensive multimodal alignment evaluator.
    
    Combines multiple alignment metrics for thorough evaluation.
    """
    
    def __init__(
        self,
        fairness_threshold: float = 0.1
    ):
        """
        Initialize multimodal alignment evaluator.
        
        Args:
            fairness_threshold: Threshold for fair alignment
        """
        self.fairness_threshold = fairness_threshold
        
        self.alignment = MultimodalAlignmentScore()
        self.fair_alignment = FairMultimodalAlignment(fairness_threshold)
        self.gap = ModalityGapMetric()
        self.contrastive = ContrastiveAlignmentMetric()
    
    def evaluate(
        self,
        modality_a: np.ndarray,
        modality_b: np.ndarray,
        sensitive_attribute: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive multimodal alignment evaluation.
        
        Args:
            modality_a: First modality embeddings
            modality_b: Second modality embeddings
            sensitive_attribute: Optional sensitive attribute for fairness
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            "alignment": {},
            "gap": {},
            "fair_alignment": {},
            "contrastive": {},
            "overall": {},
        }
        
        # Basic alignment
        results["alignment"] = self.alignment.compute(modality_a, modality_b)
        
        # Modality gap
        results["gap"] = self.gap.compute(modality_a, modality_b)
        
        # Contrastive alignment
        results["contrastive"] = self.contrastive.compute(modality_a, modality_b)
        
        # Fair alignment (if sensitive attribute provided)
        if sensitive_attribute is not None:
            results["fair_alignment"] = self.fair_alignment.compute(
                modality_a, modality_b, sensitive_attribute
            )
        
        # Overall score
        scores = [
            results["alignment"]["alignment_accuracy"],
            1 - results["gap"]["relative_gap"],
            results["contrastive"]["contrastive_accuracy"],
        ]
        
        if results["fair_alignment"]:
            scores.append(1 - results["fair_alignment"]["alignment_difference"])
        
        results["overall"]["alignment_score"] = float(np.mean(scores))
        results["overall"]["is_well_aligned"] = results["overall"]["alignment_score"] > 0.7
        
        return results
