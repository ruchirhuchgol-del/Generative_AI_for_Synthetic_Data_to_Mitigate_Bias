"""
Membership Inference Attack (MIA) Evaluation
=============================================

Implementation of membership inference attacks to evaluate
privacy preservation in synthetic data.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings


class MembershipInferenceAttack:
    """
    Membership Inference Attack (MIA) for synthetic data evaluation.
    
    Tests whether an attacker can determine if a sample
    was in the training data by examining synthetic data.
    
    A high attack success rate indicates privacy risk,
    while a rate close to random (50%) indicates good privacy.
    """
    
    def __init__(
        self,
        attack_model: str = "rf",
        n_shadow_models: int = 5,
        test_size: float = 0.3,
        random_state: int = 42
    ):
        """
        Initialize MIA evaluator.
        
        Args:
            attack_model: Attack model type ('rf', 'lr', 'mlp', 'gb')
            n_shadow_models: Number of shadow models (for shadow training)
            test_size: Fraction of data for testing
            random_state: Random seed
        """
        self.attack_model = attack_model
        self.n_shadow_models = n_shadow_models
        self.test_size = test_size
        self.random_state = random_state
        self._attack_classifier = None
        self._scaler = None
    
    def _get_attack_model(self):
        """Get attack model instance."""
        if self.attack_model == "rf":
            return RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                random_state=self.random_state
            )
        elif self.attack_model == "lr":
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
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        holdout_data: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute membership inference attack success rate.
        
        Args:
            real_data: Real training data (members)
            synthetic_data: Generated synthetic data
            holdout_data: Optional holdout set (non-members)
            
        Returns:
            Dictionary with attack metrics
        """
        # Ensure same number of samples
        n_samples = min(len(real_data), len(synthetic_data))
        
        # Prepare member and non-member data
        members = real_data[:n_samples]
        
        if holdout_data is not None:
            non_members = holdout_data[:n_samples]
        else:
            # Use synthetic data as non-members (worst case for privacy)
            non_members = synthetic_data[:n_samples]
        
        # Create attack dataset
        X = np.vstack([members, non_members])
        y = np.array([1] * n_samples + [0] * n_samples)  # 1 = member, 0 = non-member
        
        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        
        # Split for attack evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=self.test_size, 
            stratify=y,
            random_state=self.random_state
        )
        
        # Train attack model
        self._attack_classifier = self._get_attack_model()
        self._attack_classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self._attack_classifier.predict(X_test)
        y_proba = self._attack_classifier.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        
        # Privacy metrics
        attack_advantage = max(0, auc - 0.5) * 2
        privacy_score = 1 - attack_advantage
        
        return {
            "attack_accuracy": float(accuracy),
            "attack_auc": float(auc),
            "attack_precision": float(precision),
            "attack_recall": float(recall),
            "attack_advantage": float(attack_advantage),
            "privacy_score": float(privacy_score),
            "random_baseline": 0.5,
        }
    
    def compute_with_confidence(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        holdout_data: Optional[np.ndarray] = None,
        n_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Compute MIA with confidence intervals via multiple trials.
        
        Args:
            real_data: Real training data
            synthetic_data: Generated synthetic data
            holdout_data: Optional holdout set
            n_trials: Number of trials for confidence estimation
            
        Returns:
            Dictionary with confidence intervals
        """
        results = []
        
        for trial in range(n_trials):
            # Vary random state for each trial
            original_state = self.random_state
            self.random_state = self.random_state + trial * 100
            
            result = self.compute(real_data, synthetic_data, holdout_data)
            results.append(result)
            
            self.random_state = original_state
        
        # Aggregate results
        metrics = {}
        for key in results[0].keys():
            values = [r[key] for r in results]
            metrics[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "ci_lower": float(np.percentile(values, 2.5)),
                "ci_upper": float(np.percentile(values, 97.5)),
            }
        
        return metrics


class ShadowModelMIA:
    """
    Shadow Model-based Membership Inference Attack.
    
    Uses shadow models trained on similar data to learn
    the difference between member and non-member outputs.
    """
    
    def __init__(
        self,
        n_shadow_models: int = 5,
        shadow_model_type: str = "rf",
        attack_model_type: str = "rf"
    ):
        """
        Initialize shadow model MIA.
        
        Args:
            n_shadow_models: Number of shadow models
            shadow_model_type: Type of shadow models
            attack_model_type: Type of attack model
        """
        self.n_shadow_models = n_shadow_models
        self.shadow_model_type = shadow_model_type
        self.attack_model_type = attack_model_type
        self._shadow_models = []
        self._attack_model = None
    
    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.5
    ) -> None:
        """
        Train shadow models and attack model.
        
        Args:
            data: Full dataset
            labels: Labels for the data
            test_size: Fraction to use for each shadow model
        """
        self._shadow_models = []
        
        for i in range(self.n_shadow_models):
            # Split data for this shadow model
            X_train, X_out, y_train, y_out = train_test_split(
                data, labels, 
                test_size=test_size,
                random_state=i * 42
            )
            
            # Train shadow model
            if self.shadow_model_type == "rf":
                model = RandomForestClassifier(n_estimators=50, random_state=i)
            else:
                model = LogisticRegression(max_iter=1000, random_state=i)
            
            model.fit(X_train, y_train)
            self._shadow_models.append((model, X_train, X_out))
        
        # Create attack training data
        X_attack = []
        y_attack = []
        
        for model, X_train, X_out in self._shadow_models:
            # Get predictions for members (training data)
            member_proba = model.predict_proba(X_train)
            if member_proba.shape[1] == 2:
                member_proba = member_proba[:, 1]
            
            # Get predictions for non-members (held-out data)
            non_member_proba = model.predict_proba(X_out)
            if non_member_proba.shape[1] == 2:
                non_member_proba = non_member_proba[:, 1]
            
            # Combine
            for proba in member_proba:
                X_attack.append([np.max(member_proba), np.mean(member_proba)])
                y_attack.append(1)
            
            for proba in non_member_proba:
                X_attack.append([np.max(non_member_proba), np.mean(non_member_proba)])
                y_attack.append(0)
        
        X_attack = np.array(X_attack)
        y_attack = np.array(y_attack)
        
        # Train attack model
        if self.attack_model_type == "rf":
            self._attack_model = RandomForestClassifier(n_estimators=100)
        else:
            self._attack_model = LogisticRegression()
        
        self._attack_model.fit(X_attack, y_attack)
    
    def attack(
        self,
        target_predictions: np.ndarray,
        target_proba: np.ndarray
    ) -> np.ndarray:
        """
        Perform membership inference attack.
        
        Args:
            target_predictions: Target model predictions
            target_proba: Target model prediction probabilities
            
        Returns:
            Membership predictions (1 = member, 0 = non-member)
        """
        if self._attack_model is None:
            raise ValueError("Attack model not trained. Call fit() first.")
        
        # Create attack features
        if target_proba.ndim > 1 and target_proba.shape[1] == 2:
            target_proba = target_proba[:, 1]
        
        X_attack = np.column_stack([
            np.max(target_proba, axis=0) if target_proba.ndim > 1 else target_proba,
            np.mean(target_proba, axis=0) if target_proba.ndim > 1 else target_proba
        ])
        
        return self._attack_model.predict(X_attack)


class LossBasedMIA:
    """
    Loss-based Membership Inference Attack.
    
    Uses the loss values of samples to infer membership.
    Member samples typically have lower loss than non-members.
    """
    
    def __init__(
        self,
        threshold_percentile: float = 50,
        normalize: bool = True
    ):
        """
        Initialize loss-based MIA.
        
        Args:
            threshold_percentile: Percentile for threshold
            normalize: Whether to normalize losses
        """
        self.threshold_percentile = threshold_percentile
        self.normalize = normalize
        self._threshold = None
    
    def fit(
        self,
        member_losses: np.ndarray,
        non_member_losses: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit the attack by computing threshold.
        
        Args:
            member_losses: Loss values for member samples
            non_member_losses: Loss values for non-member samples (optional)
        """
        if self.normalize:
            member_losses = (member_losses - member_losses.mean()) / (member_losses.std() + 1e-8)
        
        self._threshold = np.percentile(member_losses, self.threshold_percentile)
    
    def attack(
        self,
        losses: np.ndarray
    ) -> np.ndarray:
        """
        Infer membership from loss values.
        
        Args:
            losses: Loss values for samples
            
        Returns:
            Membership predictions
        """
        if self._threshold is None:
            raise ValueError("Threshold not set. Call fit() first.")
        
        if self.normalize:
            losses = (losses - losses.mean()) / (losses.std() + 1e-8)
        
        # Lower loss = more likely to be member
        return (losses <= self._threshold).astype(int)


class MIADefenseEvaluator:
    """
    Evaluator for MIA defense mechanisms.
    
    Tests various defense strategies and their effectiveness.
    """
    
    def __init__(self):
        """Initialize MIA defense evaluator."""
        self.mia = MembershipInferenceAttack()
    
    def evaluate_defense(
        self,
        original_data: np.ndarray,
        defended_data: np.ndarray,
        synthetic_data: np.ndarray,
        holdout_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Evaluate effectiveness of MIA defense.
        
        Args:
            original_data: Original training data
            defended_data: Data after defense mechanism
            synthetic_data: Generated synthetic data
            holdout_data: Holdout set for non-members
            
        Returns:
            Defense evaluation results
        """
        # Attack on original data
        original_attack = self.mia.compute(original_data, synthetic_data, holdout_data)
        
        # Attack on defended data
        defended_attack = self.mia.compute(defended_data, synthetic_data, holdout_data)
        
        # Compute defense effectiveness
        privacy_improvement = (
            defended_attack["privacy_score"] - original_attack["privacy_score"]
        )
        attack_reduction = (
            original_attack["attack_auc"] - defended_attack["attack_auc"]
        )
        
        return {
            "original_attack": original_attack,
            "defended_attack": defended_attack,
            "privacy_improvement": float(privacy_improvement),
            "attack_reduction": float(attack_reduction),
            "defense_effective": privacy_improvement > 0.05,
        }


class MembershipInferenceEvaluator:
    """
    Comprehensive membership inference attack evaluator.
    
    Combines multiple attack strategies for thorough evaluation.
    """
    
    def __init__(
        self,
        attack_models: Optional[List[str]] = None,
        n_trials: int = 5
    ):
        """
        Initialize comprehensive MIA evaluator.
        
        Args:
            attack_models: List of attack model types to try
            n_trials: Number of trials for confidence estimation
        """
        self.attack_models = attack_models or ["rf", "lr"]
        self.n_trials = n_trials
    
    def evaluate(
        self,
        real_data: np.ndarray,
        synthetic_data: np.ndarray,
        holdout_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive MIA evaluation.
        
        Args:
            real_data: Real training data
            synthetic_data: Generated synthetic data
            holdout_data: Holdout set for non-members
            
        Returns:
            Comprehensive evaluation results
        """
        results = {
            "attacks": {},
            "summary": {},
        }
        
        attack_aucs = []
        privacy_scores = []
        
        for model_type in self.attack_models:
            mia = MembershipInferenceAttack(attack_model=model_type)
            
            # Run multiple trials
            trial_results = mia.compute_with_confidence(
                real_data, synthetic_data, holdout_data, self.n_trials
            )
            
            results["attacks"][model_type] = trial_results
            attack_aucs.append(trial_results["attack_auc"]["mean"])
            privacy_scores.append(trial_results["privacy_score"]["mean"])
        
        # Summary
        results["summary"] = {
            "avg_attack_auc": float(np.mean(attack_aucs)),
            "max_attack_auc": float(np.max(attack_aucs)),
            "avg_privacy_score": float(np.mean(privacy_scores)),
            "min_privacy_score": float(np.min(privacy_scores)),
            "is_private": np.max(attack_aucs) < 0.65,  # Threshold for privacy
        }
        
        return results
