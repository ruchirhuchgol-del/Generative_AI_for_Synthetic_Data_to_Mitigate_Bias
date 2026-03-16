"""
Attribute Inference Attack Evaluation
======================================

Implementation of attribute inference attacks to evaluate
privacy risks associated with sensitive attribute disclosure.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings


class AttributeInferenceAttack:
    """
    Attribute Inference Attack (AIA) for synthetic data evaluation.
    
    Tests whether an attacker can infer sensitive attributes
    from other features in the synthetic data.
    """
    
    def __init__(
        self,
        attack_model: str = "rf",
        test_size: float = 0.3,
        random_state: int = 42
    ):
        """
        Initialize AIA evaluator.
        
        Args:
            attack_model: Attack model type ('rf', 'lr', 'mlp', 'gb')
            test_size: Fraction of data for testing
            random_state: Random seed
        """
        self.attack_model = attack_model
        self.test_size = test_size
        self.random_state = random_state
        self._attack_classifier = None
        self._scaler = None
    
    def _get_attack_model(self, n_classes: int = 2):
        """Get attack model instance."""
        if self.attack_model == "rf":
            return RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=self.random_state
            )
        elif self.attack_model == "lr":
            if n_classes > 2:
                return LogisticRegression(
                    max_iter=1000, 
                    multi_class='multinomial',
                    random_state=self.random_state
                )
            return LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state
            )
        elif self.attack_model == "mlp":
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=500,
                random_state=self.random_state
            )
        elif self.attack_model == "gb":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            )
        else:
            return RandomForestClassifier(
                n_estimators=100, 
                random_state=self.random_state
            )
    
    def compute(
        self,
        data: np.ndarray,
        sensitive_attribute_idx: int,
        real_data: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute attribute inference attack success rate.
        
        Args:
            data: Dataset to attack (synthetic or real)
            sensitive_attribute_idx: Index of sensitive attribute to infer
            real_data: Optional real data for baseline comparison
            
        Returns:
            Dictionary with attack metrics
        """
        n_features = data.shape[1]
        
        if sensitive_attribute_idx >= n_features:
            raise ValueError(
                f"Sensitive attribute index {sensitive_attribute_idx} "
                f"out of bounds for {n_features} features"
            )
        
        # Prepare features and target
        feature_indices = [i for i in range(n_features) if i != sensitive_attribute_idx]
        
        X = data[:, feature_indices]
        y = data[:, sensitive_attribute_idx]
        
        # Handle continuous target by discretization
        if len(np.unique(y)) > 10:
            y = self._discretize_target(y)
        
        # Encode categorical target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        
        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded,
            test_size=self.test_size,
            stratify=y_encoded if n_classes == 2 else None,
            random_state=self.random_state
        )
        
        # Train attack model
        self._attack_classifier = self._get_attack_model(n_classes)
        self._attack_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self._attack_classifier.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        if n_classes == 2:
            try:
                y_proba = self._attack_classifier.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except:
                auc = 0.5
            
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
        else:
            # Multiclass metrics
            try:
                y_proba = self._attack_classifier.predict_proba(X_test)
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            except:
                auc = 0.5
            
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Compute privacy risk
        if n_classes == 2:
            baseline = max(np.mean(y_test == 0), np.mean(y_test == 1))
        else:
            baseline = 1 / n_classes
        
        privacy_risk = max(0, (accuracy - baseline) / (1 - baseline)) if accuracy > baseline else 0
        
        return {
            "attack_accuracy": float(accuracy),
            "attack_auc": float(auc),
            "attack_precision": float(precision),
            "attack_recall": float(recall),
            "attack_f1": float(f1),
            "baseline_accuracy": float(baseline),
            "privacy_risk": float(privacy_risk),
            "n_classes": int(n_classes),
        }
    
    def _discretize_target(self, y: np.ndarray, n_bins: int = 5) -> np.ndarray:
        """Discretize continuous target variable."""
        percentiles = np.linspace(0, 100, n_bins + 1)
        bins = np.percentile(y, percentiles)
        return np.digitize(y, bins[1:-1])
    
    def compute_for_multiple_attributes(
        self,
        data: np.ndarray,
        sensitive_attribute_indices: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute AIA for multiple sensitive attributes.
        
        Args:
            data: Dataset to attack
            sensitive_attribute_indices: Indices of sensitive attributes
            
        Returns:
            Dictionary with results per attribute
        """
        results = {}
        
        for idx in sensitive_attribute_indices:
            attr_name = f"attribute_{idx}"
            results[attr_name] = self.compute(data, idx)
        
        return results


class CorrelationBasedAIA:
    """
    Correlation-based Attribute Inference Attack.
    
    Uses correlation patterns to infer sensitive attributes.
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.3
    ):
        """
        Initialize correlation-based AIA.
        
        Args:
            correlation_threshold: Threshold for significant correlation
        """
        self.correlation_threshold = correlation_threshold
        self._correlations = None
    
    def fit(
        self,
        data: np.ndarray,
        sensitive_attribute_idx: int
    ) -> None:
        """
        Fit correlation model.
        
        Args:
            data: Training data
            sensitive_attribute_idx: Index of sensitive attribute
        """
        n_features = data.shape[1]
        feature_indices = [i for i in range(n_features) if i != sensitive_attribute_idx]
        
        sensitive = data[:, sensitive_attribute_idx]
        
        self._correlations = {}
        for idx in feature_indices:
            corr = np.corrcoef(data[:, idx], sensitive)[0, 1]
            if abs(corr) >= self.correlation_threshold:
                self._correlations[idx] = corr
    
    def attack(
        self,
        data: np.ndarray,
        sensitive_attribute_idx: int
    ) -> np.ndarray:
        """
        Infer sensitive attribute using correlations.
        
        Args:
            data: Data to attack
            sensitive_attribute_idx: Index of sensitive attribute
            
        Returns:
            Inferred values
        """
        if self._correlations is None:
            self.fit(data, sensitive_attribute_idx)
        
        n_features = data.shape[1]
        
        # Use weighted average of correlated features
        weighted_sum = np.zeros(len(data))
        total_weight = 0
        
        for idx, corr in self._correlations.items():
            weighted_sum += data[:, idx] * corr
            total_weight += abs(corr)
        
        if total_weight > 0:
            inferred = weighted_sum / total_weight
            # Binarize if needed
            return (inferred > np.median(inferred)).astype(int)
        else:
            return np.zeros(len(data))


class ModelBasedAIA:
    """
    Model-based Attribute Inference Attack.
    
    Trains a model to predict sensitive attributes from non-sensitive features.
    More sophisticated than correlation-based approach.
    """
    
    def __init__(
        self,
        model_type: str = "ensemble",
        cross_validate: bool = True,
        cv_folds: int = 5
    ):
        """
        Initialize model-based AIA.
        
        Args:
            model_type: Model type ('rf', 'lr', 'ensemble')
            cross_validate: Whether to use cross-validation
            cv_folds: Number of CV folds
        """
        self.model_type = model_type
        self.cross_validate = cross_validate
        self.cv_folds = cv_folds
        self._model = None
        self._scaler = None
    
    def evaluate(
        self,
        data: np.ndarray,
        sensitive_attribute_idx: int,
        test_size: float = 0.3
    ) -> Dict[str, Any]:
        """
        Evaluate model-based attribute inference.
        
        Args:
            data: Dataset
            sensitive_attribute_idx: Index of sensitive attribute
            test_size: Test set fraction
            
        Returns:
            Evaluation results
        """
        n_features = data.shape[1]
        feature_indices = [i for i in range(n_features) if i != sensitive_attribute_idx]
        
        X = data[:, feature_indices]
        y = data[:, sensitive_attribute_idx]
        
        # Handle continuous target
        if len(np.unique(y)) > 10:
            y = (y > np.median(y)).astype(int)
        
        # Scale
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        results = {}
        
        if self.cross_validate:
            # Cross-validation
            if self.model_type == "ensemble":
                models = {
                    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
                    "lr": LogisticRegression(max_iter=1000, random_state=42),
                }
                
                for name, model in models.items():
                    scores = cross_val_score(model, X_scaled, y, cv=self.cv_folds, scoring='accuracy')
                    results[f"{name}_cv_accuracy"] = float(scores.mean())
                    results[f"{name}_cv_std"] = float(scores.std())
            else:
                if self.model_type == "rf":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    model = LogisticRegression(max_iter=1000, random_state=42)
                
                scores = cross_val_score(model, X_scaled, y, cv=self.cv_folds, scoring='accuracy')
                results["cv_accuracy"] = float(scores.mean())
                results["cv_std"] = float(scores.std())
        
        else:
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42, stratify=y
            )
            
            if self.model_type == "ensemble":
                models = {
                    "rf": RandomForestClassifier(n_estimators=100, random_state=42),
                    "lr": LogisticRegression(max_iter=1000, random_state=42),
                }
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    results[f"{name}_accuracy"] = float(accuracy_score(y_test, y_pred))
            else:
                if self.model_type == "rf":
                    self._model = RandomForestClassifier(n_estimators=100, random_state=42)
                else:
                    self._model = LogisticRegression(max_iter=1000, random_state=42)
                
                self._model.fit(X_train, y_train)
                y_pred = self._model.predict(X_test)
                results["accuracy"] = float(accuracy_score(y_test, y_pred))
        
        # Compute feature importance for inference
        results["inference_risk"] = self._compute_inference_risk(results)
        
        return results
    
    def _compute_inference_risk(self, results: Dict[str, float]) -> float:
        """Compute overall inference risk from results."""
        accuracies = [v for k, v in results.items() if 'accuracy' in k and 'baseline' not in k]
        
        if accuracies:
            max_accuracy = max(accuracies)
            # Risk is 0 at 50% accuracy, 1 at 100%
            return max(0, (max_accuracy - 0.5) * 2)
        
        return 0.0


class AttributeInferenceEvaluator:
    """
    Comprehensive attribute inference attack evaluator.
    
    Combines multiple attack strategies for thorough evaluation.
    """
    
    def __init__(
        self,
        attack_models: Optional[List[str]] = None
    ):
        """
        Initialize comprehensive AIA evaluator.
        
        Args:
            attack_models: List of attack model types
        """
        self.attack_models = attack_models or ["rf", "lr", "gb"]
    
    def evaluate(
        self,
        data: np.ndarray,
        sensitive_attribute_indices: List[int],
        compare_with_real: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive attribute inference evaluation.
        
        Args:
            data: Dataset to evaluate (typically synthetic)
            sensitive_attribute_indices: Indices of sensitive attributes
            compare_with_real: Optional real data for comparison
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            "per_attribute": {},
            "per_model": {},
            "summary": {},
        }
        
        # Evaluate per attribute
        for idx in sensitive_attribute_indices:
            attr_name = f"attribute_{idx}"
            results["per_attribute"][attr_name] = {}
            
            for model_type in self.attack_models:
                aia = AttributeInferenceAttack(attack_model=model_type)
                attack_result = aia.compute(data, idx)
                results["per_attribute"][attr_name][model_type] = attack_result
            
            # Also evaluate on real data if provided
            if compare_with_real is not None:
                aia_real = AttributeInferenceAttack(attack_model="rf")
                real_result = aia_real.compute(compare_with_real, idx)
                results["per_attribute"][attr_name]["real_data_baseline"] = real_result
        
        # Aggregate per model
        for model_type in self.attack_models:
            accuracies = []
            privacy_risks = []
            
            for attr_name in results["per_attribute"]:
                if model_type in results["per_attribute"][attr_name]:
                    accuracies.append(results["per_attribute"][attr_name][model_type]["attack_accuracy"])
                    privacy_risks.append(results["per_attribute"][attr_name][model_type]["privacy_risk"])
            
            results["per_model"][model_type] = {
                "avg_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
                "avg_privacy_risk": float(np.mean(privacy_risks)) if privacy_risks else 0.0,
            }
        
        # Summary
        all_privacy_risks = []
        for attr_name in results["per_attribute"]:
            for model_type in self.attack_models:
                if model_type in results["per_attribute"][attr_name]:
                    all_privacy_risks.append(
                        results["per_attribute"][attr_name][model_type]["privacy_risk"]
                    )
        
        results["summary"] = {
            "max_privacy_risk": float(max(all_privacy_risks)) if all_privacy_risks else 0.0,
            "avg_privacy_risk": float(np.mean(all_privacy_risks)) if all_privacy_risks else 0.0,
            "is_safe": max(all_privacy_risks) < 0.3 if all_privacy_risks else True,
        }
        
        return results
