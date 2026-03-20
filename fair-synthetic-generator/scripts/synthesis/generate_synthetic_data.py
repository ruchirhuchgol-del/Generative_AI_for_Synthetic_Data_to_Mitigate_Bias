#!/usr/bin/env python
"""
Generate Synthetic Data
=======================

Comprehensive script for generating synthetic data using trained models.
Supports conditional generation, fairness constraints, multiple output formats,
and quality validation.

Usage:
    python generate_synthetic_data.py [OPTIONS]

Options:
    --checkpoint PATH     Path to model checkpoint (required)
    --n-samples N         Number of samples to generate
    --output PATH         Output file path
    --conditional         Enable conditional generation
    --condition FILE      Path to condition file for conditional generation
    --balance-sensitive   Balance output across sensitive groups
    --fairness-constraint Apply fairness constraints during generation
    --validate            Validate generated data quality
    --format FMT          Output format: csv, parquet, numpy, json
    --seed N              Random seed
    -h, --help            Show this help message

Examples:
    python generate_synthetic_data.py --checkpoint model.pt --n-samples 10000
    python generate_synthetic_data.py --checkpoint model.pt --conditional --balance-sensitive
    python generate_synthetic_data.py --checkpoint model.pt --validate --format parquet
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate Synthetic Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    
    # Generation parameters
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples to generate"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for generation"
    )
    
    # Output configuration
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "parquet", "numpy", "json", "hdf5"],
        default="csv",
        help="Output format"
    )
    
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress output file"
    )
    
    # Conditional generation
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Enable conditional generation"
    )
    
    parser.add_argument(
        "--condition",
        type=str,
        default=None,
        help="Path to condition file (JSON or CSV)"
    )
    
    parser.add_argument(
        "--condition-col",
        type=str,
        default=None,
        help="Column name for conditioning"
    )
    
    parser.add_argument(
        "--condition-values",
        type=str,
        default=None,
        help="Comma-separated condition values"
    )
    
    # Fairness options
    parser.add_argument(
        "--balance-sensitive",
        action="store_true",
        help="Balance output across sensitive groups"
    )
    
    parser.add_argument(
        "--sensitive-col",
        type=str,
        default=None,
        help="Sensitive attribute column name"
    )
    
    parser.add_argument(
        "--fairness-constraint",
        type=str,
        choices=["demographic_parity", "equalized_odds", "none"],
        default="none",
        help="Fairness constraint to apply"
    )
    
    parser.add_argument(
        "--fairness-threshold",
        type=float,
        default=0.8,
        help="Threshold for fairness constraint"
    )
    
    # Quality options
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated data quality"
    )
    
    parser.add_argument(
        "--reference-data",
        type=str,
        default=None,
        help="Reference data for validation comparison"
    )
    
    parser.add_argument(
        "--filter-outliers",
        action="store_true",
        help="Filter outliers from generated data"
    )
    
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for outlier filtering"
    )
    
    # Device and reproducibility
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cuda, cpu, auto"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    # Metadata
    parser.add_argument(
        "--metadata",
        type=str,
        default=None,
        help="Path to output metadata file"
    )
    
    parser.add_argument(
        "--include-original-cols",
        type=str,
        default=None,
        help="Path to reference file for column names"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    
    return parser.parse_args()


class SyntheticDataGenerator:
    """
    Comprehensive synthetic data generator.
    
    Features:
    - Multiple model architecture support
    - Conditional generation
    - Fairness-aware generation
    - Quality validation
    - Multiple output formats
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        seed: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize the generator.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device for generation
            seed: Random seed
            verbose: Print progress
        """
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Set seed
        if seed is not None:
            self.set_seed(seed)
        self.seed = seed
        
        # Load model
        self.model = None
        self.model_config = None
        self.model_type = None
        
        self._load_model()
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def log(self, msg: str):
        """Print log message if verbose."""
        if self.verbose:
            print(f"[Generator] {msg}")
    
    def _load_model(self):
        """Load model from checkpoint."""
        self.log(f"Loading checkpoint from {self.checkpoint_path}...")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract model config
        self.model_config = checkpoint.get('config', {})
        
        # Determine model type
        self.model_type = self._infer_model_type(checkpoint)
        self.log(f"Detected model type: {self.model_type}")
        
        # Create and load model
        self.model = self._create_model(checkpoint)
        self.model.eval()
        
        self.log(f"Model loaded on {self.device}")
    
    def _infer_model_type(self, checkpoint: Dict) -> str:
        """Infer model type from checkpoint."""
        # Check config
        if self.model_config:
            if 'model_type' in self.model_config:
                return self.model_config['model_type']
            if 'model' in self.model_config:
                return self.model_config['model'].get('type', 'vae')
        
        # Check state dict keys
        state_dict = checkpoint.get('model_state_dict', {})
        
        if 'fc_mu' in state_dict or 'encoder.0.weight' in state_dict:
            return 'vae'
        elif 'generator' in str(list(state_dict.keys())).lower():
            return 'gan'
        elif 'time_mlp' in state_dict:
            return 'diffusion'
        else:
            return 'vae'  # Default
    
    def _create_model(self, checkpoint: Dict) -> nn.Module:
        """Create model from checkpoint."""
        state_dict = checkpoint['model_state_dict']
        
        # Infer dimensions from state dict
        if self.model_type == 'vae':
            # Get dimensions from encoder
            encoder_keys = [k for k in state_dict.keys() if 'encoder' in k or 'fc_' in k]
            
            # Infer input dimension
            if 'encoder.0.weight' in state_dict:
                input_dim = state_dict['encoder.0.weight'].shape[1]
            else:
                input_dim = 20  # Default
            
            # Infer latent dimension
            if 'fc_mu.weight' in state_dict:
                latent_dim = state_dict['fc_mu.weight'].shape[0]
            else:
                latent_dim = 128
            
            # Infer hidden dimension
            if 'encoder.0.weight' in state_dict:
                hidden_dim = state_dict['encoder.0.weight'].shape[0]
            else:
                hidden_dim = 256
            
            # Count layers
            n_layers = sum(1 for k in state_dict.keys() if 'encoder' in k and 'weight' in k)
            
            # Count sensitive groups
            if 'adversary.4.weight' in state_dict:
                num_sensitive = state_dict['adversary.4.weight'].shape[0]
            else:
                num_sensitive = 2
            
            # Create VAE
            class LoadedVAE(nn.Module):
                def __init__(self, input_dim, latent_dim, hidden_dim, n_layers, num_sensitive):
                    super().__init__()
                    self.latent_dim = latent_dim
                    
                    self.encoder = nn.Sequential(
                        nn.Linear(input_dim, hidden_dim),
                        nn.ReLU(),
                        *([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] * (n_layers - 1))
                    )
                    self.fc_mu = nn.Linear(hidden_dim, latent_dim)
                    self.fc_var = nn.Linear(hidden_dim, latent_dim)
                    
                    self.decoder = nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, input_dim)
                    )
                    
                    self.adversary = nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, num_sensitive)
                    )
                
                def decode(self, z):
                    return self.decoder(z)
                
                def forward(self, x):
                    h = self.encoder(x)
                    mu, log_var = self.fc_mu(h), self.fc_var(h)
                    z = mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)
                    return self.decoder(z), mu, log_var, z
            
            model = LoadedVAE(input_dim, latent_dim, hidden_dim, max(n_layers, 2), num_sensitive)
            
        elif self.model_type == 'gan':
            # Infer dimensions for GAN
            if 'generator.0.weight' in state_dict:
                latent_dim = state_dict['generator.0.weight'].shape[1]
                hidden_dim = state_dict['generator.0.weight'].shape[0]
            else:
                latent_dim = 128
                hidden_dim = 256
            
            class LoadedGAN(nn.Module):
                def __init__(self, latent_dim, hidden_dim):
                    super().__init__()
                    self.latent_dim = latent_dim
                    
                    self.generator = nn.Sequential(
                        nn.Linear(latent_dim, hidden_dim),
                        nn.LeakyReLU(0.2),
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.LeakyReLU(0.2),
                        nn.Linear(hidden_dim * 2, 20)  # Default output dim
                    )
                
                def decode(self, z):
                    return self.generator(z)
            
            model = LoadedGAN(latent_dim, hidden_dim)
        else:
            # Diffusion model
            class SimpleDiffusion(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.latent_dim = 256
                    
                    self.time_mlp = nn.Sequential(
                        nn.Linear(1, 256),
                        nn.SiLU(),
                        nn.Linear(256, 256)
                    )
                    
                    self.net = nn.Sequential(
                        nn.Linear(20 + 256, 256),
                        nn.SiLU(),
                        nn.Linear(256, 256),
                        nn.SiLU(),
                        nn.Linear(256, 20)
                    )
                
                def decode(self, z):
                    # Simplified: just pass through network
                    t = torch.zeros(z.size(0), 1, device=z.device)
                    t_emb = self.time_mlp(t)
                    return self.net(torch.cat([z, t_emb], dim=-1))
            
            model = SimpleDiffusion()
        
        # Load state dict
        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            self.log(f"Warning: Could not load all weights: {e}")
        
        return model.to(self.device)
    
    def generate(
        self,
        n_samples: int,
        batch_size: int = 512,
        conditions: Optional[Dict] = None,
        balance_sensitive: bool = False,
        sensitive_groups: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate synthetic data.
        
        Args:
            n_samples: Number of samples to generate
            batch_size: Batch size for generation
            conditions: Optional conditions for conditional generation
            balance_sensitive: Balance across sensitive groups
            sensitive_groups: Number of sensitive groups
            
        Returns:
            Generated data as numpy array
        """
        self.log(f"Generating {n_samples} samples...")
        
        all_samples = []
        
        if balance_sensitive and sensitive_groups:
            # Generate balanced samples per group
            samples_per_group = n_samples // sensitive_groups
            remaining = n_samples % sensitive_groups
            
            for group_id in range(sensitive_groups):
                n_group = samples_per_group + (1 if group_id < remaining else 0)
                
                # Generate with group-specific latent codes
                group_samples = self._generate_batch(
                    n_group, 
                    batch_size,
                    group_condition=group_id if conditions else None
                )
                all_samples.append(group_samples)
        else:
            # Standard generation
            remaining = n_samples
            while remaining > 0:
                current_batch = min(batch_size, remaining)
                
                batch_samples = self._generate_batch(
                    current_batch,
                    current_batch,
                    conditions=conditions
                )
                all_samples.append(batch_samples)
                remaining -= current_batch
        
        # Concatenate all samples
        all_samples = np.vstack(all_samples)
        
        self.log(f"Generated {len(all_samples)} samples")
        return all_samples
    
    def _generate_batch(
        self,
        n_samples: int,
        batch_size: int,
        conditions: Optional[Dict] = None,
        group_condition: Optional[int] = None
    ) -> np.ndarray:
        """Generate a single batch."""
        with torch.no_grad():
            # Sample from prior
            latent_dim = getattr(self.model, 'latent_dim', 128)
            z = torch.randn(n_samples, latent_dim, device=self.device)
            
            # Apply conditions if provided
            if conditions is not None:
                z = self._apply_conditions(z, conditions)
            
            if group_condition is not None:
                z = self._apply_group_condition(z, group_condition)
            
            # Generate
            if hasattr(self.model, 'decode'):
                samples = self.model.decode(z)
            else:
                samples = self.model(z)
            
            if isinstance(samples, tuple):
                samples = samples[0]
            
            return samples.cpu().numpy()
    
    def _apply_conditions(self, z: torch.Tensor, conditions: Dict) -> torch.Tensor:
        """Apply conditions to latent codes."""
        # Simple conditioning: shift latent codes based on condition
        if 'shift' in conditions:
            z = z + conditions['shift'].to(z.device)
        return z
    
    def _apply_group_condition(self, z: torch.Tensor, group_id: int) -> torch.Tensor:
        """Apply group-specific conditioning."""
        # Shift latent space based on group
        shift = torch.randn_like(z) * 0.1 * group_id
        return z + shift
    
    def apply_fairness_constraint(
        self,
        data: np.ndarray,
        sensitive_col: Optional[int] = None,
        constraint: str = "demographic_parity",
        threshold: float = 0.8
    ) -> np.ndarray:
        """
        Apply fairness constraints to generated data.
        
        Args:
            data: Generated data
            sensitive_col: Column index for sensitive attribute
            constraint: Type of fairness constraint
            threshold: Fairness threshold
            
        Returns:
            Adjusted data
        """
        self.log(f"Applying {constraint} constraint...")
        
        if constraint == "none":
            return data
        
        # Simple approach: adjust distributions to achieve fairness
        if constraint == "demographic_parity":
            # Ensure similar distributions across groups
            # This is a simplified approach - real implementation would be more sophisticated
            adjusted_data = data.copy()
            
            if sensitive_col is not None:
                # Get unique groups
                groups = np.unique(adjusted_data[:, sensitive_col].round())
                
                if len(groups) >= 2:
                    # Calculate group means for other columns
                    for col in range(adjusted_data.shape[1]):
                        if col != sensitive_col:
                            group_means = [
                                adjusted_data[adjusted_data[:, sensitive_col].round() == g, col].mean()
                                for g in groups
                            ]
                            
                            # Adjust to balance
                            overall_mean = np.mean(group_means)
                            for g in groups:
                                mask = adjusted_data[:, sensitive_col].round() == g
                                adjustment = overall_mean - adjusted_data[mask, col].mean()
                                adjusted_data[mask, col] += adjustment
            
            return adjusted_data
        
        return data
    
    def filter_outliers(
        self,
        data: np.ndarray,
        threshold: float = 3.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter outliers from generated data.
        
        Args:
            data: Generated data
            threshold: Z-score threshold
            
        Returns:
            Tuple of (filtered_data, outlier_mask)
        """
        self.log("Filtering outliers...")
        
        # Calculate z-scores
        mean = data.mean(axis=0)
        std = data.std(axis=0) + 1e-8
        z_scores = np.abs((data - mean) / std)
        
        # Find outliers (any feature with z-score > threshold)
        outlier_mask = (z_scores > threshold).any(axis=1)
        
        filtered_data = data[~outlier_mask]
        
        self.log(f"Filtered {outlier_mask.sum()} outliers ({outlier_mask.mean():.1%})")
        
        return filtered_data, outlier_mask
    
    def validate(
        self,
        synthetic_data: np.ndarray,
        reference_data: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Validate generated data quality.
        
        Args:
            synthetic_data: Generated data
            reference_data: Reference real data for comparison
            
        Returns:
            Validation metrics
        """
        self.log("Validating generated data...")
        
        metrics = {
            "n_samples": len(synthetic_data),
            "n_features": synthetic_data.shape[1],
            "has_nans": bool(np.isnan(synthetic_data).any()),
            "has_infs": bool(np.isinf(synthetic_data).any()),
        }
        
        # Basic statistics
        metrics["mean"] = synthetic_data.mean(axis=0).tolist()
        metrics["std"] = synthetic_data.std(axis=0).tolist()
        metrics["min"] = synthetic_data.min(axis=0).tolist()
        metrics["max"] = synthetic_data.max(axis=0).tolist()
        
        if reference_data is not None:
            from scipy import stats
            
            # Distribution comparison
            ks_stats = []
            for col in range(min(synthetic_data.shape[1], reference_data.shape[1])):
                synth_col = synthetic_data[:, col]
                ref_col = reference_data[:, col]
                
                ks_stat, _ = stats.ks_2samp(synth_col, ref_col)
                ks_stats.append(ks_stat)
            
            metrics["avg_ks_statistic"] = float(np.mean(ks_stats))
            metrics["max_ks_statistic"] = float(np.max(ks_stats))
        
        self.log(f"Validation complete. Avg KS stat: {metrics.get('avg_ks_statistic', 'N/A')}")
        
        return metrics
    
    def save(
        self,
        data: np.ndarray,
        output_path: str,
        format: str = "csv",
        compress: bool = False,
        column_names: Optional[List[str]] = None
    ):
        """
        Save generated data to file.
        
        Args:
            data: Generated data
            output_path: Output file path
            format: Output format
            compress: Compress output
            column_names: Optional column names
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.log(f"Saving to {output_file} (format: {format})...")
        
        if column_names is None:
            column_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        if format == "csv":
            import pandas as pd
            df = pd.DataFrame(data, columns=column_names)
            if compress:
                df.to_csv(str(output_file) + '.gz', index=False, compression='gzip')
            else:
                df.to_csv(output_file, index=False)
            
        elif format == "parquet":
            import pandas as pd
            df = pd.DataFrame(data, columns=column_names)
            compression = 'gzip' if compress else None
            df.to_parquet(output_file, compression=compression)
            
        elif format == "numpy":
            if compress:
                np.savez_compressed(output_file, data=data)
            else:
                np.save(output_file, data)
                
        elif format == "json":
            import pandas as pd
            df = pd.DataFrame(data, columns=column_names)
            df.to_json(output_file, orient='records')
            
        elif format == "hdf5":
            import h5py
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('data', data=data, compression='gzip' if compress else None)
        
        self.log(f"Saved {len(data)} samples to {output_file}")
    
    def generate_metadata(
        self,
        data: np.ndarray,
        validation_metrics: Optional[Dict] = None,
        conditions: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate metadata for the synthetic data."""
        metadata = {
            "generation_timestamp": datetime.now().isoformat(),
            "checkpoint": str(self.checkpoint_path),
            "model_type": self.model_type,
            "device": str(self.device),
            "seed": self.seed,
            "n_samples": len(data),
            "n_features": data.shape[1],
            "statistics": {
                "mean": data.mean(axis=0).tolist(),
                "std": data.std(axis=0).tolist(),
                "min": data.min(axis=0).tolist(),
                "max": data.max(axis=0).tolist()
            },
            "validation": validation_metrics,
            "conditions": conditions
        }
        return metadata


def load_conditions(condition_path: str) -> Dict:
    """Load conditions from file."""
    path = Path(condition_path)
    
    if path.suffix == '.json':
        with open(path) as f:
            return json.load(f)
    elif path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(path)
        return {'conditions': df.to_dict('list')}
    else:
        raise ValueError(f"Unsupported condition file format: {path.suffix}")


def load_reference_data(path: str) -> np.ndarray:
    """Load reference data for validation."""
    path = Path(path)
    
    if path.suffix == '.csv':
        import pandas as pd
        return pd.read_csv(path).values
    elif path.suffix == '.npy':
        return np.load(path)
    elif path.suffix == '.parquet':
        import pandas as pd
        return pd.read_parquet(path).values
    else:
        raise ValueError(f"Unsupported reference data format: {path.suffix}")


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Fair Synthetic Data Generator - Data Generation")
    print("=" * 60 + "\n")
    
    # Initialize generator
    generator = SyntheticDataGenerator(
        checkpoint_path=args.checkpoint,
        device=args.device,
        seed=args.seed,
        verbose=args.verbose
    )
    
    # Load conditions if provided
    conditions = None
    if args.condition:
        conditions = load_conditions(args.condition)
    
    if args.condition_values:
        values = [float(v) for v in args.condition_values.split(',')]
        conditions = {'values': np.array(values)}
    
    # Generate data
    start_time = time.time()
    
    synthetic_data = generator.generate(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        conditions=conditions,
        balance_sensitive=args.balance_sensitive,
        sensitive_groups=2 if args.balance_sensitive else None
    )
    
    # Apply fairness constraint
    if args.fairness_constraint != "none":
        sensitive_col_idx = 0  # Default to first column
        if args.sensitive_col:
            # Find column index
            sensitive_col_idx = 0  # Would need column mapping
        
        synthetic_data = generator.apply_fairness_constraint(
            synthetic_data,
            sensitive_col=sensitive_col_idx,
            constraint=args.fairness_constraint,
            threshold=args.fairness_threshold
        )
    
    # Filter outliers
    if args.filter_outliers:
        synthetic_data, _ = generator.filter_outliers(
            synthetic_data,
            threshold=args.outlier_threshold
        )
    
    # Validate
    validation_metrics = None
    if args.validate:
        reference_data = None
        if args.reference_data:
            reference_data = load_reference_data(args.reference_data)
        
        validation_metrics = generator.validate(synthetic_data, reference_data)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        output_dir = project_root / "data" / "synthetic"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        ext = {"csv": "csv", "parquet": "parquet", "numpy": "npy", "json": "json", "hdf5": "h5"}[args.format]
        output_path = str(output_dir / f"synthetic_{timestamp}.{ext}")
    
    # Load column names if provided
    column_names = None
    if args.include_original_cols:
        ref_path = Path(args.include_original_cols)
        if ref_path.suffix == '.csv':
            import pandas as pd
            column_names = pd.read_csv(ref_path, nrows=0).columns.tolist()
    
    # Save
    generator.save(
        synthetic_data,
        output_path,
        format=args.format,
        compress=args.compress,
        column_names=column_names
    )
    
    # Generate and save metadata
    metadata = generator.generate_metadata(
        synthetic_data,
        validation_metrics,
        conditions
    )
    
    metadata_path = args.metadata or str(Path(output_path).with_suffix('.metadata.json'))
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)
    print(f"Generated: {len(synthetic_data):,} samples")
    print(f"Time: {elapsed_time:.2f} seconds")
    print(f"Rate: {len(synthetic_data) / elapsed_time:.0f} samples/second")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path}")
    
    if validation_metrics:
        print(f"\nValidation:")
        print(f"  Has NaNs: {validation_metrics['has_nans']}")
        print(f"  Has Infs: {validation_metrics['has_infs']}")
        if 'avg_ks_statistic' in validation_metrics:
            print(f"  Avg KS Stat: {validation_metrics['avg_ks_statistic']:.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()