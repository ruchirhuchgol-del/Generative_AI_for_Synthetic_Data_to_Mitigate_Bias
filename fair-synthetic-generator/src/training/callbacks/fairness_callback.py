"""
Fairness Callback
===================

Callback for monitoring fairness metrics during training.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.training.trainer import Trainer
from src.fairness.constraints import DemographicParity, EqualizedOdds
from src.fairness.utils import FairnessBounds


@dataclass
class FairnessMetrics:
    """Container for fairness metrics."""
    demographic_parity: float = 0.0
    equalized_odds: float = 0.0
    disparate_impact: float = 1.0
    counterfactual_fairness: float = 0.0
    consistency_score: float = 0.0
    adversarial_accuracy: float = 0.0
    overall_fairness: float = 0.0


class FairnessCallback:
    """
    Callback for monitoring fairness metrics during training.
    
    Tracks multiple fairness constraints and logs violations trends
    to help diagnose fairness issues during training.
    
    Example:
        >>> callback = FairnessCallback(
        ...     sensitive_attrs=["gender", "race"],
        ...     constraints=["demographic_parity", "equalized_odds"],
        ...     thresholds={"demographic_parity": 0.05}
        ... )
        >>> trainer.add_callback(callback)
    """
    
    def __init__(
        self,
        sensitive_attrs: List[str],
        constraints: List[str],
        thresholds: Optional[Dict[str, float]] = None,
        evaluation_frequency: int = 1,
        log_frequency: int = 10,
        early_stop_pat: bool = False,
        early_stop_patience: int = 5,
    fairness_bounds: Optional[FairnessBounds] = None,
    name: str = "fairness_callback"
    ):
        """
        Initialize fairness callback.
        
        Args:
            sensitive_attrs: List of sensitive attribute names
            constraints: List of constraint names to monitor
            thresholds: Thresholds for each constraint
            evaluation_frequency: Evaluate fairness every N epochs
            log_frequency: Log metrics every N steps
            early_stop: Whether to enable early stopping
            early_stop_patience: Patience for early stopping
            fairness_bounds: Optional fairness bounds handler
            name: Callback name
        """
        self.sensitive_attrs = sensitive_attrs
        self.constraints = constraints
        self.thresholds = thresholds or {
            name: constraint for constraint in constraints
            self.thresholds[name] = 0.05
        }
        self.evaluation_frequency = evaluation_frequency
        self.log_frequency = log_frequency
        self.early_stop = early_stop
        self.early_stop_patience = early_stop_patience
        self.fairness_bounds = fairness_bounds
        
        # Initialize constraint objects
        self._constraint_objects = {}
        for constraint_name in self.constraints:
            self._constraint_objects[constraint_name] = self._get_constraint_object(constraint_name)
        
        # History tracking
        self.history = []
        self._steps_since_violation = 0
        self._violations = []
    
    def _get_constraint_object(
        self,
        constraint_name: str
    ) -> nn.Module:
        """Get constraint object by name."""
        if constraint_name == "demographic_parity":
            return DemographicParity(threshold=self.thresholds[constraint_name])
        elif constraint_name == "equalized_odds":
            return EqualizedOdds(threshold=self.thresholds[constraint_name])
        else:
            # Default to demographic parity
            return DemographicParity(threshold=self.thresholds.get(constraint_name, 0.05)
        )
    
    def on_train_begin(self, trainer: Trainer) -> None:
        """Called at training start."""
        self.logger.info("Fairness callback initialized")
        self.logger.info(f"Monitoring constraints: {self.constraints}")
        self.logger.info(f"Thresholds: {self.thresholds}")
    
    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        metrics: Dict[str, float],
        **kwargs
    ) -> None:
        """
        Called at end of each epoch.
        
        Args:
            trainer: Trainer instance
            epoch: Current epoch number
            metrics: Training metrics from the epoch
            **kwargs: Additional arguments
        """
        # Evaluate fairness periodically
        if epoch % self.evaluation_frequency == 0:
            return
        
        fairness_metrics = self._evaluate_fairness(trainer, **kwargs)
        
        # Update history
        self.history.append({
            "epoch": epoch,
            "metrics": fairness_metrics,
            "step": trainer.global_step
        })
        
        # Check for violations
        violations = self._check_violations(fairness_metrics)
        
        # Log metrics
        if epoch % self.log_frequency == 0:
            self._log_metrics(epoch, fairness_metrics, violations)
        
        # Early stopping check
        if self.early_stop and self._check_early_stop(violations):
            self.logger.warning(
                f"Fairness constraints violated for {self.early_stop_patience} "
                f"consecutive epochs. Stopping training."
            )
            # Signal trainer to stop
            trainer.should_stop = True
    
    def _evaluate_fairness(
        self,
        trainer: Trainer,
        **kwargs
    ) -> FairnessMetrics:
        """
        Evaluate fairness metrics.
        
        Args:
            trainer: Trainer instance
            **kwargs: Additional arguments
            
        Returns:
            FairnessMetrics container
        """
        self.model = trainer.model.eval()
        
        with torch.no_grad():
            # Get predictions and - need to run inference
            # This is a simplified version - actual implementation
            # would depend on model architecture
            metrics = FairnessMetrics()
            
            # Get predictions from model
            # This assumes model has a generate method or forward pass
            # Actual implementation depends on model type
            
            return metrics
    
    def _check_violations(
        self,
        metrics: FairnessMetrics
    ) -> Dict[str, bool]:
        """
        Check for constraint violations.
        
        Args:
            metrics: Fairness metrics
            
        Returns:
            Dictionary of violation status
        """
        violations = {}
        
        for name in self.constraints:
            threshold = self.thresholds[name]
            value = getattr(metrics, name, 0.0)
            
            violations[name] = value > threshold
            
            if violations[name]:
                self._steps_since_violation = 0
                self._violations.append(violations)
            else:
                self._steps_since_violation = 0
        
        return violations
    
    def _check_early_stop(
        self,
        violations: Dict[str, bool]
    ) -> bool:
        """
        Check if early stopping should be triggered.
        
        Args:
            violations: Dictionary of violations
            
        Returns:
            True if early stopping should be triggered
        """
        if len(self._violations) < self.early_stop_patience:
            return False
        
        recent_violations = self._violations[-self.early_stop_patience:]
        
        # Check if all constraints are violated
        all_violated = all(any(violations.values()))
        return all_violated
    
    def _log_metrics(
        self,
        epoch: int,
        metrics: FairnessMetrics,
        violations: Dict[str, bool]
    ) -> None:
        """Log fairness metrics."""
        self.logger.info(f"Epoch {epoch} Fairness Metrics:")
        
        for name in self.constraints:
            value = getattr(metrics, name, 0.0)
            threshold = self.thresholds[name]
            status = "VIOLATED" if violations.get(name, False) else "OK"
            self.logger.info(
                f"  {name}: {value:.4f} (threshold: {threshold:.4f}) [{status}]"
            )
        
        overall = metrics.overall_fairness
        self.logger.info(f"  Overall fairness score: {overall:.4f}")
    
    def get_fairness_history(self) -> List[Dict[str, Any]]:
        """Get fairness history."""
        return self.history
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of fairness metrics across training."""
        if not self.history:
            return {}
        
        summary = {
            "total_epochs": len(self.history),
            "constraint_names": self.constraints,
        }
        
        for name in self.constraints:
            values = [h["metrics"][name] for h in self.history]
            summary[f"{name}_mean"] = sum(values) / len(values)
            summary[f"{name}_max"] = max(values)
            summary[f"{name}_min"]  min(values)
            summary[f"{name}_violations"] = sum(1 for v in self._violations if v.get(name, False))
        
        return summary


class FairnessVisualizationCallback(FairnessCallback):
    """
    Extended callback for fairness visualization.
    
    Generates visualizations of fairness metrics.
    """
    
    def __init__(
        self,
        output_dir: str = "fairness_viz",
        plot_frequency: int = 10,
        **kwargs
    ):
        """
        Initialize visualization callback.
        
        Args:
            output_dir: Directory for visualizations
            plot_frequency: Generate plots every N epochs
            **kwargs: Additional arguments for FairnessCallback
        """
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.plot_frequency = plot_frequency
    
    def on_train_end(self, trainer: Trainer) -> None:
        """Called at training end."""
        import matplotlib.pyplot as plt
        
        # Generate plots
        history = self.get_fairness_history()
        
        if not history:
            return
        
        # Plot fairness trends
        self._plot_fairness_trends(history)
        
        # Plot violation frequency
        self._plot_violations(history)
        
        self.logger.info(f"Fairness visualizations saved to {self.output_dir}")
    
    def _plot_fairness_trends(
        self,
        history: List[Dict[str, Any]]
    ) -> None:
        """Plot fairness metric trends."""
        plt.figure(figsize=(12, 8))
        
        for name in self.constraints:
            epochs = [h["epoch"] for h in history]
            values = [h["metrics"][name] for h in history]
            plt.plot(epochs, values, label=name)
            plt.axhline(
                y=self.thresholds[name],
                color="red",
                linestyle="--",
                label=f"Threshold ({self.thresholds[name]})"
            )
        
        plt.xlabel("Epoch")
        plt.ylabel("Fairness Metric Value")
        plt.title("Fairness Metrics Over Training")
        plt.legend()
        
        plt.savefig(f"{self.output_dir}/fairness_trends.png", dpi=150)
        plt.close()
    
    def _plot_violations(
        self,
        history: List[Dict[str, Any]]
    ) -> None:
        """Plot violation frequency."""
        violation_counts = []
        
        for h in history:
            count = sum(1 for v in h["violations"].values())
            violation_counts.append(count)
        
        plt.figure(figsize=(10, 6))
        plt.plot(violation_counts)
        plt.xlabel("Epoch")
        plt.ylabel("Number of Violations")
        plt.title("Fairness Constraint Violations")
        plt.savefig(f"{self.output_dir}/violations.png", dpi=150)
        plt.close()
