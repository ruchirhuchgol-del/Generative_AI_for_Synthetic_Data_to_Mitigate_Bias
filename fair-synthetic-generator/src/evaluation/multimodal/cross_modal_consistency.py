"""
Cross-Modal Consistency Metrics
================================

Metrics for evaluating consistency between different modalities
in multimodal synthetic data generation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.spatial.distance import cosine, cdist
from scipy.stats import spearmanr, pearsonr
import warnings


class CrossModalConsistencyMetric:
    """
    Metric for evaluating cross-modal consistency.
    
    Measures how well different modalities align with each other
    in the synthetic data, ensuring coherent multimodal outputs.
    """
    
    def __init__(
        self,
        modality_pairs: Optional[List[Tuple[str, str]]] = None,
        embedding_dim: int = 64
    ):
        """
        Initialize cross-modal consistency metric.
        
        Args:
            modality_pairs: List of modality pairs to evaluate
            embedding_dim: Dimension for joint embedding space
        """
        self.modality_pairs = modality_pairs or [
            ("tabular", "text"),
            ("tabular", "image"),
            ("text", "image"),
        ]
        self.embedding_dim = embedding_dim
    
    def compute(
        self,
        modalities: Dict[str, np.ndarray],
        embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Compute cross-modal consistency scores.
        
        Args:
            modalities: Dictionary of modality data
            embeddings: Optional pre-computed embeddings
            
        Returns:
            Dictionary of consistency scores
        """
        results = {}
        
        # Compute embeddings if not provided
        if embeddings is None:
            embeddings = self._compute_embeddings(modalities)
        
        # Evaluate each pair
        for mod_a, mod_b in self.modality_pairs:
            if mod_a in embeddings and mod_b in embeddings:
                score = self._compute_alignment(
                    embeddings[mod_a],
                    embeddings[mod_b]
                )
                results[f"{mod_a}_{mod_b}_consistency"] = score
        
        # Overall score
        if results:
            results["overall_consistency"] = float(np.mean(list(results.values())))
        
        return results
    
    def _compute_embeddings(
        self,
        modalities: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Compute embeddings for each modality.
        
        Args:
            modalities: Dictionary of modality data
            
        Returns:
            Dictionary of embeddings
        """
        embeddings = {}
        
        for mod_name, data in modalities.items():
            if data is None or len(data) == 0:
                continue
            
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            
            # Use PCA for dimensionality reduction
            try:
                from sklearn.decomposition import PCA
                
                n_components = min(
                    self.embedding_dim,
                    data.shape[1],
                    data.shape[0]
                )
                pca = PCA(n_components=n_components)
                embeddings[mod_name] = pca.fit_transform(data)
            except:
                # Fallback: use raw data truncated
                embeddings[mod_name] = data[:, :min(self.embedding_dim, data.shape[1])]
        
        return embeddings
    
    def _compute_alignment(
        self,
        emb_a: np.ndarray,
        emb_b: np.ndarray
    ) -> float:
        """
        Compute alignment between two embedding spaces.
        
        Args:
            emb_a: First embedding matrix
            emb_b: Second embedding matrix
            
        Returns:
            Alignment score between 0 and 1
        """
        # Ensure same dimensions
        min_dim = min(emb_a.shape[1], emb_b.shape[1])
        emb_a = emb_a[:, :min_dim]
        emb_b = emb_b[:, :min_dim]
        
        # Compute canonical correlation
        try:
            from sklearn.cross_decomposition import CCA
            
            n_components = min(min_dim, emb_a.shape[0] - 1)
            cca = CCA(n_components=n_components)
            cca.fit(emb_a, emb_b)
            
            # Average canonical correlation
            score = np.mean(cca.score(emb_a, emb_b))
            return float(max(0, min(1, (score + 1) / 2)))
            
        except:
            # Fallback: cosine similarity
            similarities = []
            for i in range(min(len(emb_a), len(emb_b))):
                sim = 1 - cosine(emb_a[i], emb_b[i])
                similarities.append(sim)
            
            return float(np.mean(similarities))


class CrossModalRetrievalMetric:
    """
    Cross-Modal Retrieval Metric.
    
    Evaluates how well items can be retrieved across modalities.
    Measures recall@k for cross-modal retrieval tasks.
    """
    
    def __init__(
        self,
        k_values: Optional[List[int]] = None
    ):
        """
        Initialize cross-modal retrieval metric.
        
        Args:
            k_values: Values of k for recall@k evaluation
        """
        self.k_values = k_values or [1, 5, 10, 20]
    
    def compute(
        self,
        modality_a: np.ndarray,
        modality_b: np.ndarray,
        labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute cross-modal retrieval metrics.
        
        Args:
            modality_a: First modality embeddings
            modality_b: Second modality embeddings
            labels: Optional ground truth alignment labels
            
        Returns:
            Dictionary with retrieval metrics
        """
        n = min(len(modality_a), len(modality_b))
        
        # Compute similarity matrix
        # Normalize
        modality_a = modality_a / (np.linalg.norm(modality_a, axis=1, keepdims=True) + 1e-8)
        modality_b = modality_b / (np.linalg.norm(modality_b, axis=1, keepdims=True) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(modality_a[:n], modality_b[:n].T)
        
        results = {}
        
        # Recall@k
        for k in self.k_values:
            if k <= n:
                recall = self._compute_recall_at_k(similarity, k)
                results[f"recall_at_{k}"] = float(recall)
        
        # Mean Reciprocal Rank
        mrr = self._compute_mrr(similarity)
        results["mrr"] = float(mrr)
        
        # Mean Average Precision
        map_score = self._compute_map(similarity)
        results["map"] = float(map_score)
        
        return results
    
    def _compute_recall_at_k(
        self,
        similarity: np.ndarray,
        k: int
    ) -> float:
        """Compute recall@k."""
        n = similarity.shape[0]
        correct = 0
        
        for i in range(n):
            # Get top-k indices
            top_k = np.argsort(similarity[i])[-k:]
            if i in top_k:
                correct += 1
        
        return correct / n
    
    def _compute_mrr(
        self,
        similarity: np.ndarray
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        n = similarity.shape[0]
        reciprocal_ranks = []
        
        for i in range(n):
            # Get ranking
            ranking = np.argsort(similarity[i])[::-1]
            rank = np.where(ranking == i)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)
        
        return np.mean(reciprocal_ranks)
    
    def _compute_map(
        self,
        similarity: np.ndarray
    ) -> float:
        """Compute Mean Average Precision."""
        n = similarity.shape[0]
        aps = []
        
        for i in range(n):
            ranking = np.argsort(similarity[i])[::-1]
            # Binary relevance (1 if correct item)
            relevance = (ranking == i).astype(int)
            
            # Cumulative precision
            precision_at_k = np.cumsum(relevance) / np.arange(1, len(relevance) + 1)
            
            # Average precision
            if relevance.sum() > 0:
                ap = (precision_at_k * relevance).sum() / relevance.sum()
            else:
                ap = 0.0
            aps.append(ap)
        
        return np.mean(aps)


class CrossModalSemanticConsistency:
    """
    Cross-Modal Semantic Consistency Metric.
    
    Evaluates whether semantically related content across modalities
    is properly aligned.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.5
    ):
        """
        Initialize semantic consistency metric.
        
        Args:
            similarity_threshold: Threshold for considering items similar
        """
        self.similarity_threshold = similarity_threshold
    
    def compute(
        self,
        real_modalities: Dict[str, np.ndarray],
        synthetic_modalities: Dict[str, np.ndarray],
        semantic_groups: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute semantic consistency.
        
        Args:
            real_modalities: Dictionary of real modality data
            synthetic_modalities: Dictionary of synthetic modality data
            semantic_groups: Optional group assignments for semantic clusters
            
        Returns:
            Dictionary with consistency metrics
        """
        results = {}
        
        # If no semantic groups provided, use clustering
        if semantic_groups is None:
            semantic_groups = self._infer_semantic_groups(real_modalities)
        
        # Evaluate consistency within semantic groups
        for mod_name in synthetic_modalities.keys():
            if mod_name in real_modalities:
                consistency = self._compute_modality_consistency(
                    real_modalities[mod_name],
                    synthetic_modalities[mod_name],
                    semantic_groups
                )
                results[f"{mod_name}_semantic_consistency"] = consistency
        
        # Cross-modal semantic alignment
        cross_modal_score = self._compute_cross_modal_alignment(
            synthetic_modalities, semantic_groups
        )
        results["cross_modal_semantic_alignment"] = cross_modal_score
        
        if results:
            results["overall_semantic_consistency"] = float(np.mean(list(results.values())))
        
        return results
    
    def _infer_semantic_groups(
        self,
        modalities: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Infer semantic groups using clustering."""
        # Concatenate modalities for clustering
        combined = np.hstack([
            data for data in modalities.values() 
            if data is not None and len(data) > 0
        ])
        
        # Use KMeans for clustering
        try:
            from sklearn.cluster import KMeans
            n_clusters = min(10, len(combined) // 10)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            return kmeans.fit_predict(combined)
        except:
            return np.zeros(len(combined), dtype=int)
    
    def _compute_modality_consistency(
        self,
        real: np.ndarray,
        synthetic: np.ndarray,
        groups: np.ndarray
    ) -> float:
        """Compute semantic consistency for a single modality."""
        unique_groups = np.unique(groups)
        consistencies = []
        
        for group in unique_groups:
            mask = groups == group
            if mask.sum() < 2:
                continue
            
            real_group = real[mask]
            synth_group = synthetic[mask[:len(synthetic)]]
            
            # Compare intra-group variance
            real_var = np.var(real_group, axis=0).mean()
            synth_var = np.var(synth_group, axis=0).mean()
            
            # Consistency: similar variance patterns
            if real_var > 0:
                consistency = 1 - abs(real_var - synth_var) / real_var
                consistencies.append(max(0, consistency))
        
        return float(np.mean(consistencies)) if consistencies else 0.5
    
    def _compute_cross_modal_alignment(
        self,
        modalities: Dict[str, np.ndarray],
        groups: np.ndarray
    ) -> float:
        """Compute cross-modal alignment within semantic groups."""
        mod_names = list(modalities.keys())
        
        if len(mod_names) < 2:
            return 1.0
        
        alignments = []
        
        for i, mod_a in enumerate(mod_names):
            for mod_b in mod_names[i+1:]:
                # Compute alignment between modalities within groups
                data_a = modalities[mod_a]
                data_b = modalities[mod_b]
                
                n = min(len(data_a), len(data_b), len(groups))
                
                # Correlation of group centroids
                unique_groups = np.unique(groups[:n])
                
                centroids_a = []
                centroids_b = []
                
                for group in unique_groups:
                    mask = groups[:n] == group
                    if mask.sum() > 0:
                        centroids_a.append(data_a[:n][mask].mean(axis=0))
                        centroids_b.append(data_b[:n][mask].mean(axis=0))
                
                if len(centroids_a) > 1:
                    centroids_a = np.array(centroids_a)
                    centroids_b = np.array(centroids_b)
                    
                    # Average correlation across dimensions
                    min_dim = min(centroids_a.shape[1], centroids_b.shape[1])
                    correlations = []
                    
                    for d in range(min_dim):
                        corr, _ = pearsonr(centroids_a[:, d], centroids_b[:, d])
                        if not np.isnan(corr):
                            correlations.append(abs(corr))
                    
                    if correlations:
                        alignments.append(np.mean(correlations))
        
        return float(np.mean(alignments)) if alignments else 0.5


class CrossModalCoherenceScore:
    """
    Cross-Modal Coherence Score.
    
    Evaluates how coherent the generated multimodal data is,
    ensuring that different modalities tell a consistent story.
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.7
    ):
        """
        Initialize coherence score.
        
        Args:
            coherence_threshold: Threshold for coherence determination
        """
        self.coherence_threshold = coherence_threshold
    
    def compute(
        self,
        modalities: Dict[str, np.ndarray],
        paired_indices: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute cross-modal coherence score.
        
        Args:
            modalities: Dictionary of modality data
            paired_indices: Optional pairing indices
            
        Returns:
            Dictionary with coherence metrics
        """
        results = {}
        
        mod_names = list(modalities.keys())
        
        if len(mod_names) < 2:
            return {"coherence": 1.0, "n_pairs": 0}
        
        # Compute pairwise coherence
        pairwise_coherence = {}
        
        for i, mod_a in enumerate(mod_names):
            for mod_b in mod_names[i+1:]:
                coherence = self._compute_pairwise_coherence(
                    modalities[mod_a],
                    modalities[mod_b]
                )
                pairwise_coherence[f"{mod_a}_{mod_b}"] = coherence
        
        results["pairwise_coherence"] = pairwise_coherence
        results["average_coherence"] = float(np.mean(list(pairwise_coherence.values())))
        results["min_coherence"] = float(min(pairwise_coherence.values()))
        results["is_coherent"] = results["average_coherence"] >= self.coherence_threshold
        
        # Identify problematic pairs
        results["low_coherence_pairs"] = [
            pair for pair, coh in pairwise_coherence.items()
            if coh < self.coherence_threshold
        ]
        
        return results
    
    def _compute_pairwise_coherence(
        self,
        mod_a: np.ndarray,
        mod_b: np.ndarray
    ) -> float:
        """Compute coherence between two modalities."""
        n = min(len(mod_a), len(mod_b))
        
        # Normalize
        mod_a_norm = mod_a[:n] / (np.linalg.norm(mod_a[:n], axis=1, keepdims=True) + 1e-8)
        mod_b_norm = mod_b[:n] / (np.linalg.norm(mod_b[:n], axis=1, keepdims=True) + 1e-8)
        
        # Compute alignment scores
        # For coherent data, corresponding samples should be more similar
        # than non-corresponding samples
        
        # Similarity with correct pair
        correct_sim = np.sum(mod_a_norm * mod_b_norm, axis=1)
        
        # Average similarity with incorrect pairs (sampled)
        incorrect_sims = []
        for i in range(min(n, 100)):  # Sample for efficiency
            j = (i + np.random.randint(1, n)) % n
            incorrect_sims.append(np.dot(mod_a_norm[i], mod_b_norm[j]))
        
        avg_incorrect = np.mean(incorrect_sims)
        
        # Coherence: how much better is correct pairing
        coherence = (correct_sim.mean() - avg_incorrect + 1) / 2
        
        return float(max(0, min(1, coherence)))


class CrossModalConsistencyEvaluator:
    """
    Comprehensive cross-modal consistency evaluator.
    
    Combines multiple metrics for thorough evaluation.
    """
    
    def __init__(
        self,
        modality_pairs: Optional[List[Tuple[str, str]]] = None
    ):
        """
        Initialize cross-modal consistency evaluator.
        
        Args:
            modality_pairs: List of modality pairs to evaluate
        """
        self.modality_pairs = modality_pairs
        
        self.consistency = CrossModalConsistencyMetric(modality_pairs)
        self.retrieval = CrossModalRetrievalMetric()
        self.semantic = CrossModalSemanticConsistency()
        self.coherence = CrossModalCoherenceScore()
    
    def evaluate(
        self,
        real_modalities: Dict[str, np.ndarray],
        synthetic_modalities: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Comprehensive cross-modal consistency evaluation.
        
        Args:
            real_modalities: Dictionary of real modality data
            synthetic_modalities: Dictionary of synthetic modality data
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            "consistency": {},
            "retrieval": {},
            "semantic": {},
            "coherence": {},
            "overall": {},
        }
        
        # Cross-modal consistency
        results["consistency"] = self.consistency.compute(synthetic_modalities)
        
        # Retrieval metrics (if multiple modalities)
        mod_names = list(synthetic_modalities.keys())
        if len(mod_names) >= 2:
            results["retrieval"] = self.retrieval.compute(
                synthetic_modalities[mod_names[0]],
                synthetic_modalities[mod_names[1]]
            )
        
        # Semantic consistency
        results["semantic"] = self.semantic.compute(
            real_modalities, synthetic_modalities
        )
        
        # Coherence
        results["coherence"] = self.coherence.compute(synthetic_modalities)
        
        # Overall score
        scores = []
        
        if "overall_consistency" in results["consistency"]:
            scores.append(results["consistency"]["overall_consistency"])
        
        if "mrr" in results["retrieval"]:
            scores.append(results["retrieval"]["mrr"])
        
        if "overall_semantic_consistency" in results["semantic"]:
            scores.append(results["semantic"]["overall_semantic_consistency"])
        
        if "average_coherence" in results["coherence"]:
            scores.append(results["coherence"]["average_coherence"])
        
        results["overall"]["cross_modal_score"] = float(np.mean(scores)) if scores else 0.5
        
        return results
