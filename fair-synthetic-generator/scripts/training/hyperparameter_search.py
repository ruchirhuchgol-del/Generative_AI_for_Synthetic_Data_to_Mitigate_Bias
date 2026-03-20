#!/usr/bin/env python
"""
Hyperparameter Search Script
============================

Script for automated hyperparameter optimization using Optuna.
Supports various search strategies and fairness-aware optimization.

Usage:
    python hyperparameter_search.py [OPTIONS]

Options:
    --config PATH         Path to base configuration file
    --n-trials N          Number of optimization trials
    --search-space PATH   Path to search space configuration
    --sampler TYPE        Sampler type: tpe, random, cmaes, grid
    --direction DIR       Optimization direction: minimize, maximize
    --metric NAME         Metric to optimize
    --fairness-weight W   Weight for fairness in multi-objective optimization
    --output-dir PATH     Output directory for results
    --pruning             Enable pruning of unpromising trials
    -h, --help            Show this help message

Examples:
    python hyperparameter_search.py --n-trials 100 --metric val_loss
    python hyperparameter_search.py --sampler tpe --fairness-weight 0.3
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Search for Fair Synthetic Data Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default/training_config.yaml",
        help="Path to base configuration file"
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of optimization trials"
    )
    
    parser.add_argument(
        "--search-space",
        type=str,
        default=None,
        help="Path to search space configuration JSON"
    )
    
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["tpe", "random", "cmaes", "grid", "nsga2"],
        default="tpe",
        help="Sampler type for hyperparameter search"
    )
    
    parser.add_argument(
        "--direction",
        type=str,
        choices=["minimize", "maximize"],
        default="minimize",
        help="Optimization direction"
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="val_loss",
        help="Metric to optimize"
    )
    
    parser.add_argument(
        "--fairness-weight",
        type=float,
        default=0.0,
        help="Weight for fairness metric in multi-objective optimization"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--pruning",
        action="store_true",
        help="Enable pruning of unpromising trials"
    )
    
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name for the Optuna study"
    )
    
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Database URL for distributed optimization"
    )
    
    parser.add_argument(
        "--load-existing",
        action="store_true",
        help="Load existing study if it exists"
    )
    
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds for optimization"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--quick-eval",
        action="store_true",
        help="Use quick evaluation with fewer epochs"
    )
    
    return parser.parse_args()


# Default search space
DEFAULT_SEARCH_SPACE = {
    "learning_rate": {
        "type": "loguniform",
        "low": 1e-5,
        "high": 1e-2
    },
    "batch_size": {
        "type": "categorical",
        "choices": [32, 64, 128, 256]
    },
    "latent_dim": {
        "type": "int",
        "low": 64,
        "high": 512,
        "step": 64
    },
    "beta": {
        "type": "uniform",
        "low": 0.1,
        "high": 10.0
    },
    "adversary_weight": {
        "type": "uniform",
        "low": 0.0,
        "high": 1.0
    },
    "dropout": {
        "type": "uniform",
        "low": 0.0,
        "high": 0.5
    },
    "n_layers": {
        "type": "int",
        "low": 2,
        "high": 6
    },
    "hidden_dim": {
        "type": "int",
        "low": 128,
        "high": 1024,
        "step": 128
    },
    "weight_decay": {
        "type": "loguniform",
        "low": 1e-6,
        "high": 1e-2
    },
    "warmup_epochs": {
        "type": "int",
        "low": 0,
        "high": 10
    },
    "lr_scheduler": {
        "type": "categorical",
        "choices": ["none", "cosine", "step", "plateau"]
    },
    "fairness_lambda": {
        "type": "uniform",
        "low": 0.0,
        "high": 1.0
    }
}


class HyperparameterSearcher:
    """
    Hyperparameter optimization using Optuna.
    
    Supports:
    - Single-objective optimization
    - Multi-objective optimization (fidelity + fairness)
    - Various sampling strategies
    - Pruning of unpromising trials
    - Distributed optimization
    """
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        search_space: Dict[str, Any],
        metric: str = "val_loss",
        fairness_weight: float = 0.0,
        pruning: bool = True,
        quick_eval: bool = False,
        seed: int = 42
    ):
        """
        Initialize the hyperparameter searcher.
        
        Args:
            base_config: Base configuration dictionary
            search_space: Search space definition
            metric: Primary metric to optimize
            fairness_weight: Weight for fairness metric
            pruning: Whether to enable pruning
            quick_eval: Use quick evaluation mode
            seed: Random seed
        """
        self.base_config = base_config
        self.search_space = search_space
        self.metric = metric
        self.fairness_weight = fairness_weight
        self.pruning = pruning
        self.quick_eval = quick_eval
        self.seed = seed
        
        self.study = None
        self.best_params = None
        self.best_value = None
        
        # Import optuna
        try:
            import optuna
            self.optuna = optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter search. "
                            "Install it with: pip install optuna")
    
    def create_sampler(self, sampler_type: str):
        """
        Create an Optuna sampler.
        
        Args:
            sampler_type: Type of sampler
            
        Returns:
            Optuna sampler instance
        """
        if sampler_type == "tpe":
            return self.optuna.samplers.TPESampler(seed=self.seed)
        elif sampler_type == "random":
            return self.optuna.samplers.RandomSampler(seed=self.seed)
        elif sampler_type == "cmaes":
            return self.optuna.samplers.CmaEsSampler(seed=self.seed)
        elif sampler_type == "grid":
            # Grid sampler requires specification of grid points
            return self.optuna.samplers.GridSampler(
                self._create_grid_space()
            )
        elif sampler_type == "nsga2":
            return self.optuna.samplers.NSGAIISampler(seed=self.seed)
        else:
            return self.optuna.samplers.TPESampler(seed=self.seed)
    
    def _create_grid_space(self) -> Dict[str, List]:
        """Create grid search space."""
        grid_space = {}
        for param_name, param_config in self.search_space.items():
            if param_config["type"] == "categorical":
                grid_space[param_name] = param_config["choices"]
            elif param_config["type"] == "int":
                low = param_config["low"]
                high = param_config["high"]
                step = param_config.get("step", 1)
                grid_space[param_name] = list(range(low, high + 1, step))
            elif param_config["type"] in ["uniform", "loguniform"]:
                # Create 5 evenly spaced points
                low = param_config["low"]
                high = param_config["high"]
                grid_space[param_name] = np.linspace(low, high, 5).tolist()
        return grid_space
    
    def sample_params(self, trial) -> Dict[str, Any]:
        """
        Sample hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Dictionary of sampled hyperparameters
        """
        params = {}
        
        for param_name, param_config in self.search_space.items():
            param_type = param_config["type"]
            
            if param_type == "float" or param_type == "uniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"]
                )
            elif param_type == "loguniform":
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=True
                )
            elif param_type == "int":
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    step=param_config.get("step", 1)
                )
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config["choices"]
                )
        
        return params
    
    def create_objective(self) -> Callable:
        """
        Create the objective function for optimization.
        
        Returns:
            Objective function
        """
        def objective(trial) -> Union[float, Tuple[float, float]]:
            # Sample hyperparameters
            params = self.sample_params(trial)
            
            # Merge with base config
            config = self._merge_config(params)
            
            # Train model and get metrics
            try:
                metrics = self._train_and_evaluate(trial, config)
            except Exception as e:
                # Return worst value if training fails
                print(f"Trial {trial.number} failed: {e}")
                if self.fairness_weight > 0:
                    return float('inf'), float('inf')
                return float('inf')
            
            # Report metric
            primary_metric = metrics.get(self.metric, float('inf'))
            
            # Report intermediate values for pruning
            if self.pruning and 'intermediate_values' in metrics:
                for step, value in metrics['intermediate_values'].items():
                    trial.report(value, step)
                
                # Check for pruning
                if trial.should_prune():
                    raise self.optuna.TrialPruned()
            
            # Multi-objective optimization
            if self.fairness_weight > 0:
                fairness_metric = metrics.get('fairness_score', 0.0)
                # Combine metrics
                combined = primary_metric - self.fairness_weight * fairness_metric
                return combined
            
            return primary_metric
        
        return objective
    
    def _merge_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge sampled params with base config."""
        config = self.base_config.copy()
        
        # Map params to config structure
        param_mapping = {
            'learning_rate': ['optimizer', 'lr'],
            'batch_size': ['training', 'batch_size'],
            'latent_dim': ['model', 'latent_dim'],
            'beta': ['model', 'beta'],
            'adversary_weight': ['fairness', 'adversary_weight'],
            'dropout': ['model', 'dropout'],
            'n_layers': ['model', 'n_layers'],
            'hidden_dim': ['model', 'hidden_dim'],
            'weight_decay': ['optimizer', 'weight_decay'],
            'warmup_epochs': ['training', 'warmup_epochs'],
            'lr_scheduler': ['scheduler', 'name'],
            'fairness_lambda': ['fairness', 'lambda']
        }
        
        for param_name, value in params.items():
            if param_name in param_mapping:
                keys = param_mapping[param_name]
                current = config
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = {}
                    current = current[key]
                current[keys[-1]] = value
            else:
                # Direct assignment
                config[param_name] = value
        
        return config
    
    def _train_and_evaluate(
        self,
        trial,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train model and evaluate metrics.
        
        Args:
            trial: Optuna trial object
            config: Training configuration
            
        Returns:
            Dictionary of metrics
        """
        from src.core.utils import set_seed, get_device, get_logger
        from src.data.dataset import MultimodalDataset
        from src.data.dataloaders import create_dataloader
        from src.models.architectures import DebiasedVAE
        from src.training import Trainer, TrainingConfig
        
        # Set seed for reproducibility
        set_seed(self.seed + trial.number)
        
        device = get_device()
        logger = get_logger(f"trial_{trial.number}")
        
        # Quick eval settings
        n_epochs = 3 if self.quick_eval else config.get('training', {}).get('n_epochs', 10)
        
        # Create dataset (demo)
        dataset = MultimodalDataset.create_dummy(n_samples=2000 if self.quick_eval else 5000)
        train_dataset, val_dataset, _ = dataset.split(ratios=(0.7, 0.15, 0.15))
        
        batch_size = config.get('training', {}).get('batch_size', 128)
        
        train_loader = create_dataloader(train_dataset, mode="train", batch_size=batch_size)
        val_loader = create_dataloader(val_dataset, mode="val", batch_size=batch_size * 2)
        
        # Create model
        latent_dim = config.get('model', {}).get('latent_dim', 256)
        model = DebiasedVAE(
            modalities=['tabular'],
            latent_dim=latent_dim,
            num_sensitive_groups=2
        ).to(device)
        
        # Create optimizer
        lr = config.get('optimizer', {}).get('lr', 1e-4)
        weight_decay = config.get('optimizer', {}).get('weight_decay', 0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Create trainer
        training_config = TrainingConfig(
            n_epochs=n_epochs,
            batch_size=batch_size,
            learning_rate=lr,
            device=str(device)
        )
        
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            config=training_config,
            optimizer=optimizer
        )
        
        # Train with intermediate reporting
        intermediate_values = {}
        
        # Simple training loop
        model.train()
        global_step = 0
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass (simplified)
                if isinstance(batch, dict):
                    data = batch.get('tabular', batch.get('data'))
                else:
                    data = batch[0] if isinstance(batch, (list, tuple)) else batch
                
                data = data.to(device) if torch.is_tensor(data) else data
                data = torch.as_tensor(data, dtype=torch.float32, device=device)
                
                # VAE forward pass
                recon, mu, log_var = model(data)
                
                # Compute loss
                recon_loss = torch.nn.functional.mse_loss(recon, data)
                kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + config.get('model', {}).get('beta', 1.0) * kl_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
                global_step += 1
            
            # Report intermediate value
            avg_loss = epoch_loss / n_batches
            intermediate_values[epoch] = avg_loss
            
            # Report for pruning
            if self.pruning:
                trial.report(avg_loss, epoch)
                if trial.should_prune():
                    raise self.optuna.TrialPruned()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    data = batch.get('tabular', batch.get('data'))
                else:
                    data = batch[0] if isinstance(batch, (list, tuple)) else batch
                
                data = torch.as_tensor(data, dtype=torch.float32, device=device)
                recon, mu, log_var = model(data)
                loss = torch.nn.functional.mse_loss(recon, data)
                val_loss += loss.item()
                val_batches += 1
        
        val_loss /= val_batches
        
        # Compute fairness score (simplified)
        fairness_score = self._compute_fairness_score(model, val_loader, device)
        
        return {
            'val_loss': val_loss,
            'train_loss': avg_loss,
            'fairness_score': fairness_score,
            'intermediate_values': intermediate_values
        }
    
    def _compute_fairness_score(
        self,
        model,
        dataloader,
        device
    ) -> float:
        """
        Compute fairness score.
        
        Args:
            model: Trained model
            dataloader: Validation dataloader
            device: Device
            
        Returns:
            Fairness score (higher is better)
        """
        # Simplified fairness metric
        # In practice, this would compute demographic parity, equalized odds, etc.
        model.eval()
        
        group_losses = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    data = batch.get('tabular', batch.get('data'))
                    sensitive = batch.get('sensitive', None)
                else:
                    data = batch[0] if isinstance(batch, (list, tuple)) else batch
                    sensitive = None
                
                data = torch.as_tensor(data, dtype=torch.float32, device=device)
                recon, _, _ = model(data)
                loss = torch.nn.functional.mse_loss(recon, data, reduction='none')
                
                if sensitive is not None:
                    sensitive = torch.as_tensor(sensitive, device=device)
                    for group_id in sensitive.unique():
                        group_loss = loss[sensitive == group_id].mean().item()
                        group_losses.append(group_loss)
        
        if len(group_losses) >= 2:
            # Fairness as inverse of max disparity
            disparity = max(group_losses) - min(group_losses)
            fairness_score = 1.0 / (1.0 + disparity)
        else:
            fairness_score = 1.0
        
        return fairness_score
    
    def optimize(
        self,
        n_trials: int,
        sampler_type: str = "tpe",
        direction: str = "minimize",
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_existing: bool = False,
        n_jobs: int = 1,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials
            sampler_type: Type of sampler
            direction: Optimization direction
            study_name: Name for the study
            storage: Database URL for distributed optimization
            load_existing: Whether to load existing study
            n_jobs: Number of parallel jobs
            timeout: Timeout in seconds
            
        Returns:
            Optimization results
        """
        # Create sampler
        sampler = self.create_sampler(sampler_type)
        
        # Create study
        study_name = study_name or f"hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine directions for multi-objective
        directions = [direction]
        if self.fairness_weight > 0:
            directions = [direction, "maximize"]
        
        if load_existing and storage:
            self.study = self.optuna.load_study(
                study_name=study_name,
                storage=storage
            )
        else:
            self.study = self.optuna.create_study(
                study_name=study_name,
                sampler=sampler,
                direction=direction,
                storage=storage,
                pruner=self.optuna.pruners.MedianPruner() if self.pruning else None
            )
        
        # Run optimization
        self.study.optimize(
            self.create_objective(),
            n_trials=n_trials,
            n_jobs=n_jobs,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Get best results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value
        
        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "best_trial": self.study.best_trial.number,
            "n_trials": len(self.study.trials),
            "n_pruned": len([t for t in self.study.trials if t.state == self.optuna.trial.TrialState.PRUNED])
        }
    
    def get_importance(self) -> Dict[str, float]:
        """Get hyperparameter importance."""
        if self.study is None:
            return {}
        
        try:
            importance = self.optuna.importance.get_param_importances(self.study)
            return dict(importance)
        except Exception:
            return {}
    
    def save_results(self, output_dir: str):
        """
        Save optimization results.
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save best params
        with open(output_path / "best_params.json", 'w') as f:
            json.dump(self.best_params or {}, f, indent=2)
        
        # Save study results
        df = self.study.trials_dataframe()
        df.to_csv(output_path / "trials.csv", index=False)
        
        # Save importance
        importance = self.get_importance()
        with open(output_path / "param_importance.json", 'w') as f:
            json.dump(importance, f, indent=2)
        
        # Save optimization history
        history = {
            "study_name": self.study.study_name,
            "direction": self.study.direction.name,
            "best_trial": self.study.best_trial.number,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "n_trials": len(self.study.trials),
            "param_importance": importance
        }
        
        with open(output_path / "optimization_history.json", 'w') as f:
            json.dump(history, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file."""
    import yaml
    
    path = Path(config_path)
    if not path.exists():
        return {}
    
    with open(path) as f:
        if path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif path.suffix == '.json':
            return json.load(f)
    return {}


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Fair Synthetic Data Generator - Hyperparameter Search")
    print("=" * 60 + "\n")
    
    # Load base config
    base_config = load_config(args.config)
    
    # Load search space
    if args.search_space:
        with open(args.search_space) as f:
            search_space = json.load(f)
    else:
        search_space = DEFAULT_SEARCH_SPACE
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        output_dir = project_root / "checkpoints" / "hpo" / datetime.now().strftime('%Y%m%d_%H%M%S')
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create searcher
    searcher = HyperparameterSearcher(
        base_config=base_config,
        search_space=search_space,
        metric=args.metric,
        fairness_weight=args.fairness_weight,
        pruning=args.pruning,
        quick_eval=args.quick_eval,
        seed=args.seed
    )
    
    print(f"Starting hyperparameter search:")
    print(f"  Trials: {args.n_trials}")
    print(f"  Sampler: {args.sampler}")
    print(f"  Direction: {args.direction}")
    print(f"  Metric: {args.metric}")
    print(f"  Fairness weight: {args.fairness_weight}")
    print(f"  Output: {output_dir}\n")
    
    # Run optimization
    results = searcher.optimize(
        n_trials=args.n_trials,
        sampler_type=args.sampler,
        direction=args.direction,
        study_name=args.study_name,
        storage=args.storage,
        load_existing=args.load_existing,
        n_jobs=args.n_jobs,
        timeout=args.timeout
    )
    
    # Save results
    searcher.save_results(str(output_dir))
    
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print(f"Best value: {results['best_value']:.6f}")
    print(f"Best trial: {results['best_trial']}")
    print(f"\nBest parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    print(f"\nParameter importance:")
    importance = searcher.get_importance()
    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {imp:.4f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
