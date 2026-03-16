"""
Differential Privacy Evaluation
================================

Implementation of differential privacy accounting and evaluation
for synthetic data generation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import warnings


class DifferentialPrivacyAccountant:
    """
    Differential Privacy Budget Accountant.
    
    Tracks privacy budget expenditure during training with differential privacy.
    Supports various composition methods for accurate budget tracking.
    """
    
    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize DP accountant.
        
        Args:
            target_epsilon: Target privacy budget (ε)
            target_delta: Target failure probability (δ)
            noise_multiplier: Noise multiplier for DP-SGD
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.steps = 0
        self.history = []
    
    def step(
        self,
        sample_rate: float,
        n_steps: int = 1
    ) -> Tuple[float, float]:
        """
        Account for training steps.
        
        Args:
            sample_rate: Batch sampling rate (batch_size / dataset_size)
            n_steps: Number of steps to account for
            
        Returns:
            Tuple of (epsilon, delta) spent in this step
        """
        try:
            from opacus.accountants import RDPAccountant
            
            accountant = RDPAccountant()
            
            for _ in range(n_steps):
                accountant.step(
                    noise_multiplier=self.noise_multiplier,
                    sample_rate=sample_rate
                )
            
            epsilon = accountant.get_epsilon(self.target_delta)
            
        except ImportError:
            # Simple approximation without opacus
            # Using moments accountant approximation
            epsilon = self._approximate_epsilon(sample_rate, n_steps)
        
        delta_spent = self.target_delta * sample_rate * n_steps
        
        self.spent_epsilon += epsilon
        self.spent_delta += delta_spent
        self.steps += n_steps
        
        self.history.append({
            "step": self.steps,
            "epsilon": epsilon,
            "delta": delta_spent,
            "cumulative_epsilon": self.spent_epsilon,
            "cumulative_delta": self.spent_delta,
        })
        
        return epsilon, delta_spent
    
    def _approximate_epsilon(
        self,
        sample_rate: float,
        n_steps: int
    ) -> float:
        """
        Approximate epsilon for DP-SGD.
        
        Uses analytical approximation from Abadi et al. (2016).
        """
        # Simplified approximation
        # ε ≈ q * sqrt(n_steps) / σ
        # where q = sample_rate, σ = noise_multiplier
        
        q = sample_rate
        sigma = self.noise_multiplier
        T = n_steps
        
        # Improved approximation
        epsilon = q * np.sqrt(T) / sigma
        
        return epsilon
    
    def remaining_budget(self) -> Dict[str, float]:
        """
        Get remaining privacy budget.
        
        Returns:
            Dictionary with budget information
        """
        return {
            "remaining_epsilon": max(0, self.target_epsilon - self.spent_epsilon),
            "remaining_delta": max(0, self.target_delta - self.spent_delta),
            "spent_epsilon": self.spent_epsilon,
            "spent_delta": self.spent_delta,
            "spent_ratio": self.spent_epsilon / self.target_epsilon,
            "is_exhausted": self.spent_epsilon >= self.target_epsilon,
            "steps": self.steps,
        }
    
    def reset(self) -> None:
        """Reset the accountant."""
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.steps = 0
        self.history = []
    
    def get_privacy_guarantee(self) -> Dict[str, Any]:
        """
        Get formal privacy guarantee statement.
        
        Returns:
            Dictionary with privacy guarantee details
        """
        return {
            "guarantee_type": "(ε, δ)-differential privacy",
            "epsilon": self.spent_epsilon,
            "delta": self.spent_delta,
            "noise_multiplier": self.noise_multiplier,
            "mechanism": "DP-SGD",
            "is_within_budget": (
                self.spent_epsilon <= self.target_epsilon and
                self.spent_delta <= self.target_delta
            ),
        }


class EpsilonDeltaCalculator:
    """
    Calculator for epsilon and delta values for DP mechanisms.
    
    Provides conversions between different DP parameterizations.
    """
    
    @staticmethod
    def gaussian_mechanism_epsilon(
        sensitivity: float,
        sigma: float,
        delta: float
    ) -> float:
        """
        Calculate epsilon for Gaussian mechanism.
        
        Args:
            sensitivity: Function sensitivity (L2)
            sigma: Noise standard deviation
            delta: Target delta
            
        Returns:
            Epsilon value
        """
        # Using the analytical formula
        # ε = sensitivity / sigma * sqrt(2 * ln(1.25 / delta))
        
        if delta <= 0 or delta >= 1:
            raise ValueError("delta must be in (0, 1)")
        
        epsilon = (sensitivity / sigma) * np.sqrt(2 * np.log(1.25 / delta))
        return float(epsilon)
    
    @staticmethod
    def gaussian_mechanism_sigma(
        sensitivity: float,
        epsilon: float,
        delta: float
    ) -> float:
        """
        Calculate required noise sigma for Gaussian mechanism.
        
        Args:
            sensitivity: Function sensitivity (L2)
            epsilon: Target epsilon
            delta: Target delta
            
        Returns:
            Required sigma
        """
        if epsilon <= 0 or delta <= 0 or delta >= 1:
            raise ValueError("epsilon > 0 and delta in (0, 1) required")
        
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        return float(sigma)
    
    @staticmethod
    def laplace_mechanism_epsilon(
        sensitivity: float,
        scale: float
    ) -> float:
        """
        Calculate epsilon for Laplace mechanism.
        
        Args:
            sensitivity: Function sensitivity (L1)
            scale: Laplace scale parameter
            
        Returns:
            Epsilon value
        """
        return sensitivity / scale
    
    @staticmethod
    def laplace_mechanism_scale(
        sensitivity: float,
        epsilon: float
    ) -> float:
        """
        Calculate required scale for Laplace mechanism.
        
        Args:
            sensitivity: Function sensitivity (L1)
            epsilon: Target epsilon
            
        Returns:
            Required scale
        """
        return sensitivity / epsilon
    
    @staticmethod
    def compose_epsilon(
        epsilons: List[float],
        method: str = "basic"
    ) -> float:
        """
        Compose multiple epsilon values.
        
        Args:
            epsilons: List of epsilon values
            method: Composition method ('basic', 'advanced')
            
        Returns:
            Composed epsilon
        """
        if method == "basic":
            # Basic composition: ε_total = Σ ε_i
            return sum(epsilons)
        elif method == "advanced":
            # Advanced composition (Dwork et al.)
            k = len(epsilons)
            if k == 0:
                return 0.0
            avg_eps = np.mean(epsilons)
            # Simplified advanced composition
            return avg_eps * np.sqrt(2 * k * np.log(1 / 1e-5))
        else:
            return sum(epsilons)


class PrivacyBudgetScheduler:
    """
    Scheduler for allocating privacy budget across training.
    
    Supports various budget allocation strategies.
    """
    
    def __init__(
        self,
        total_epsilon: float,
        total_delta: float,
        n_epochs: int,
        allocation_strategy: str = "uniform"
    ):
        """
        Initialize privacy budget scheduler.
        
        Args:
            total_epsilon: Total epsilon budget
            total_delta: Total delta budget
            n_epochs: Number of training epochs
            allocation_strategy: Budget allocation strategy
                ('uniform', 'decreasing', 'increasing')
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.n_epochs = n_epochs
        self.allocation_strategy = allocation_strategy
        
        self._compute_budget_schedule()
    
    def _compute_budget_schedule(self) -> None:
        """Compute budget schedule based on strategy."""
        if self.allocation_strategy == "uniform":
            self.epsilon_schedule = [self.total_epsilon / self.n_epochs] * self.n_epochs
            self.delta_schedule = [self.total_delta / self.n_epochs] * self.n_epochs
        
        elif self.allocation_strategy == "decreasing":
            # Decreasing budget (more privacy early)
            weights = np.exp(-np.linspace(0, 2, self.n_epochs))
            weights = weights / weights.sum()
            self.epsilon_schedule = list(self.total_epsilon * weights)
            self.delta_schedule = list(self.total_delta * weights)
        
        elif self.allocation_strategy == "increasing":
            # Increasing budget (more privacy later)
            weights = np.exp(np.linspace(0, 2, self.n_epochs))
            weights = weights / weights.sum()
            self.epsilon_schedule = list(self.total_epsilon * weights)
            self.delta_schedule = list(self.total_delta * weights)
        
        else:
            # Default to uniform
            self.epsilon_schedule = [self.total_epsilon / self.n_epochs] * self.n_epochs
            self.delta_schedule = [self.total_delta / self.n_epochs] * self.n_epochs
    
    def get_budget(self, epoch: int) -> Tuple[float, float]:
        """
        Get budget for a specific epoch.
        
        Args:
            epoch: Epoch number (0-indexed)
            
        Returns:
            Tuple of (epsilon, delta) for this epoch
        """
        if epoch >= self.n_epochs:
            return 0.0, 0.0
        
        return self.epsilon_schedule[epoch], self.delta_schedule[epoch]
    
    def get_schedule(self) -> Dict[str, List[float]]:
        """
        Get complete budget schedule.
        
        Returns:
            Dictionary with epsilon and delta schedules
        """
        return {
            "epsilon": self.epsilon_schedule,
            "delta": self.delta_schedule,
        }


class DifferentialPrivacyVerifier:
    """
    Verifier for differential privacy guarantees.
    
    Checks whether mechanisms and parameters provide valid DP guarantees.
    """
    
    @staticmethod
    def verify_parameters(
        epsilon: float,
        delta: float
    ) -> Dict[str, Any]:
        """
        Verify DP parameters are valid.
        
        Args:
            epsilon: Epsilon value
            delta: Delta value
            
        Returns:
            Verification results
        """
        issues = []
        
        if epsilon <= 0:
            issues.append("Epsilon must be positive")
        
        if epsilon > 10:
            issues.append("Epsilon > 10 provides very weak privacy")
        
        if epsilon > 1:
            issues.append("Epsilon > 1 may not provide meaningful privacy")
        
        if delta <= 0:
            issues.append("Delta must be positive")
        
        if delta >= 1:
            issues.append("Delta must be less than 1")
        
        if delta > 1e-5:
            issues.append("Delta > 1e-5 is typically too large")
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "epsilon": epsilon,
            "delta": delta,
            "privacy_level": DifferentialPrivacyVerifier._assess_privacy_level(epsilon),
        }
    
    @staticmethod
    def _assess_privacy_level(epsilon: float) -> str:
        """Assess privacy level based on epsilon."""
        if epsilon < 0.1:
            return "very_strong"
        elif epsilon < 1:
            return "strong"
        elif epsilon < 3:
            return "moderate"
        elif epsilon < 10:
            return "weak"
        else:
            return "very_weak"
    
    @staticmethod
    def compare_budgets(
        budgets: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Compare multiple privacy budgets.
        
        Args:
            budgets: List of budget dictionaries with 'epsilon' and 'delta'
            
        Returns:
            Comparison results
        """
        if not budgets:
            return {"error": "No budgets to compare"}
        
        epsilons = [b.get("epsilon", float('inf')) for b in budgets]
        deltas = [b.get("delta", 1.0) for b in budgets]
        
        # Find best (most private) budget
        # Lower is better for both epsilon and delta
        best_epsilon_idx = np.argmin(epsilons)
        best_delta_idx = np.argmin(deltas)
        
        return {
            "min_epsilon": min(epsilons),
            "max_epsilon": max(epsilons),
            "min_delta": min(deltas),
            "max_delta": max(deltas),
            "best_epsilon_idx": int(best_epsilon_idx),
            "best_delta_idx": int(best_delta_idx),
            "average_epsilon": float(np.mean(epsilons)),
            "average_delta": float(np.mean(deltas)),
        }


class DifferentialPrivacyEvaluator:
    """
    Comprehensive differential privacy evaluator.
    
    Combines accounting, calculation, and verification for DP evaluation.
    """
    
    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5
    ):
        """
        Initialize DP evaluator.
        
        Args:
            target_epsilon: Target privacy budget
            target_delta: Target delta
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.accountant = DifferentialPrivacyAccountant(
            target_epsilon, target_delta
        )
    
    def evaluate(
        self,
        spent_epsilon: float,
        spent_delta: float
    ) -> Dict[str, Any]:
        """
        Evaluate differential privacy expenditure.
        
        Args:
            spent_epsilon: Epsilon spent
            spent_delta: Delta spent
            
        Returns:
            Evaluation results
        """
        verification = DifferentialPrivacyVerifier.verify_parameters(
            spent_epsilon, spent_delta
        )
        
        remaining = {
            "remaining_epsilon": max(0, self.target_epsilon - spent_epsilon),
            "remaining_delta": max(0, self.target_delta - spent_delta),
        }
        
        # Privacy budget usage
        usage_ratio = spent_epsilon / self.target_epsilon
        
        return {
            "spent": {
                "epsilon": spent_epsilon,
                "delta": spent_delta,
            },
            "target": {
                "epsilon": self.target_epsilon,
                "delta": self.target_delta,
            },
            "remaining": remaining,
            "usage_ratio": float(usage_ratio),
            "is_within_budget": (
                spent_epsilon <= self.target_epsilon and
                spent_delta <= self.target_delta
            ),
            "verification": verification,
            "privacy_level": verification["privacy_level"],
        }
    
    def get_recommendations(
        self,
        current_epsilon: float
    ) -> List[str]:
        """
        Get recommendations for DP parameters.
        
        Args:
            current_epsilon: Current epsilon spent
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if current_epsilon > self.target_epsilon:
            recommendations.append(
                "Privacy budget exceeded. Consider reducing noise multiplier "
                "or training for fewer epochs."
            )
        
        if current_epsilon < 0.1 * self.target_epsilon:
            recommendations.append(
                "Very little privacy budget used. Could train longer or "
                "reduce noise for better utility."
            )
        
        if self.target_epsilon > 1:
            recommendations.append(
                "Consider using a smaller epsilon for stronger privacy guarantees."
            )
        
        if self.target_delta > 1e-5:
            recommendations.append(
                "Delta is relatively large. Consider using smaller delta."
            )
        
        return recommendations
