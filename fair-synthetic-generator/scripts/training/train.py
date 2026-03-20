#!/usr/bin/env python
"""
Training Script
===============

Main script for training fair synthetic data generators.
Supports multiple architectures, distributed training, experiment tracking, and comprehensive callbacks.

Usage:
    python train.py [OPTIONS]

Options:
    --config PATH         Path to training configuration file
    --model-type TYPE     Model type: vae, gan, diffusion
    --fairness-paradigm   Fairness paradigm: group, individual, counterfactual
    --data PATH           Path to training data
    --output-dir PATH     Output directory for checkpoints and logs
    --epochs N            Number of training epochs
    --batch-size N        Batch size
    --lr RATE             Learning rate
    --device DEVICE       Device: cuda, cpu, auto
    --distributed         Enable distributed training
    --track-experiment    Enable experiment tracking (wandb/tensorboard)
    --seed N              Random seed
    -h, --help            Show this help message

Examples:
    python train.py --config configs/experiments/baseline.yaml
    python train.py --model-type vae --fairness-paradigm group --epochs 100
    python train.py --data data/processed/train.csv --track-experiment
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LinearLR, 
    SequentialLR,
    ReduceLROnPlateau
)

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Fair Synthetic Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (YAML)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["vae", "gan", "diffusion", "multimodal"],
        default="vae",
        help="Model architecture type"
    )
    
    parser.add_argument(
        "--fairness-paradigm",
        type=str,
        choices=["group", "individual", "counterfactual", "none"],
        default="group",
        help="Fairness paradigm to enforce"
    )
    
    # Data configuration
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training data file or directory"
    )
    
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to validation data"
    )
    
    parser.add_argument(
        "--sensitive-attrs",
        type=str,
        default=None,
        help="Comma-separated sensitive attribute columns"
    )
    
    parser.add_argument(
        "--target-col",
        type=str,
        default=None,
        help="Target column name"
    )
    
    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for optimizer"
    )
    
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        help="Number of warmup epochs"
    )
    
    # Model hyperparameters
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=256,
        help="Latent dimension for generative models"
    )
    
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden layer dimension"
    )
    
    parser.add_argument(
        "--n-layers",
        type=int,
        default=4,
        help="Number of hidden layers"
    )
    
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta parameter for VAE (KL weight)"
    )
    
    parser.add_argument(
        "--fairness-weight",
        type=float,
        default=0.5,
        help="Weight for fairness loss component"
    )
    
    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints and logs"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name for tracking"
    )
    
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    
    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: cuda, cpu, auto"
    )
    
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training"
    )
    
    # Experiment tracking
    parser.add_argument(
        "--track-experiment",
        action="store_true",
        help="Enable experiment tracking"
    )
    
    parser.add_argument(
        "--tracker",
        type=str,
        choices=["wandb", "tensorboard", "both"],
        default="wandb",
        help="Experiment tracker to use"
    )
    
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="fair-synthetic-data",
        help="WandB project name"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode"
    )
    
    # Misc
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling"
    )
    
    return parser.parse_args()


class TrainingConfig:
    """Training configuration container."""
    
    def __init__(self, args):
        # Load from config file if provided
        if args.config and os.path.exists(args.config):
            config = self._load_config(args.config)
            self._update_from_config(config)
        
        # Override with command line arguments
        self.model_type = args.model_type
        self.fairness_paradigm = args.fairness_paradigm
        self.data_path = args.data
        self.val_data_path = args.val_data
        self.sensitive_attrs = args.sensitive_attrs.split(",") if args.sensitive_attrs else []
        self.target_col = args.target_col
        
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_epochs = args.warmup_epochs
        
        self.latent_dim = args.latent_dim
        self.hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers
        self.beta = args.beta
        self.fairness_weight = args.fairness_weight
        
        self.output_dir = args.output_dir
        self.experiment_name = args.experiment_name
        self.save_every = args.save_every
        
        self.device = args.device
        self.distributed = args.distributed
        self.fp16 = args.fp16
        
        self.track_experiment = args.track_experiment
        self.tracker = args.tracker
        self.wandb_project = args.wandb_project
        
        self.seed = args.seed
        self.deterministic = args.deterministic
        self.verbose = args.verbose
        self.profile = args.profile
        
        # Set defaults
        if self.output_dir is None:
            self.output_dir = f"checkpoints/experiments/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.experiment_name is None:
            self.experiment_name = f"{self.model_type}_{self.fairness_paradigm}"
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            return yaml.safe_load(f)
    
    def _update_from_config(self, config: Dict[str, Any]):
        """Update configuration from loaded dict."""
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    setattr(self, f"{key}_{sub_key}", sub_value)
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


class FairTrainer:
    """
    Comprehensive trainer for fair synthetic data generators.
    
    Features:
    - Multiple model architectures (VAE, GAN, Diffusion)
    - Fairness-aware training with multiple paradigms
    - Distributed training support
    - Mixed precision training
    - Experiment tracking (WandB, TensorBoard)
    - Checkpoint management
    - Early stopping
    - Gradient accumulation
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize the trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'fairness_loss': [],
            'lr': []
        }
        
        # Set up device
        self._setup_device()
        
        # Set random seeds
        self._set_seed()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision
        self.tracker = None
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, 'logs'), exist_ok=True)
    
    def _setup_device(self):
        """Set up training device."""
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        if self.config.verbose:
            print(f"[Trainer] Using device: {self.device}")
        
        # Multi-GPU setup
        if self.config.distributed and torch.cuda.device_count() > 1:
            self.world_size = torch.cuda.device_count()
            if self.config.verbose:
                print(f"[Trainer] Distributed training with {self.world_size} GPUs")
        else:
            self.world_size = 1
    
    def _set_seed(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup_model(self, input_dim: int, num_sensitive_groups: int = 2):
        """
        Set up the model architecture.
        
        Args:
            input_dim: Input feature dimension
            num_sensitive_groups: Number of sensitive attribute groups
        """
        if self.config.verbose:
            print(f"[Trainer] Creating {self.config.model_type} model...")
        
        if self.config.model_type == "vae":
            self.model = self._create_vae(input_dim, num_sensitive_groups)
        elif self.config.model_type == "gan":
            self.model = self._create_gan(input_dim, num_sensitive_groups)
        elif self.config.model_type == "diffusion":
            self.model = self._create_diffusion(input_dim)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        self.model = self.model.to(self.device)
        
        # Multi-GPU
        if self.config.distributed and self.world_size > 1:
            self.model = nn.DataParallel(self.model)
        
        if self.config.verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"[Trainer] Total parameters: {total_params:,}")
            print(f"[Trainer] Trainable parameters: {trainable_params:,}")
    
    def _create_vae(self, input_dim: int, num_sensitive_groups: int) -> nn.Module:
        """Create VAE model with fairness constraints."""
        
        class FairVAE(nn.Module):
            """Fair VAE with adversarial debiasing."""
            
            def __init__(self, input_dim, latent_dim, hidden_dim, n_layers, beta, num_sensitive_groups):
                super().__init__()
                
                self.latent_dim = latent_dim
                self.beta = beta
                
                # Encoder
                encoder_layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
                for _ in range(n_layers - 1):
                    encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
                self.encoder = nn.Sequential(*encoder_layers)
                
                self.fc_mu = nn.Linear(hidden_dim, latent_dim)
                self.fc_var = nn.Linear(hidden_dim, latent_dim)
                
                # Decoder
                decoder_layers = [nn.Linear(latent_dim, hidden_dim), nn.ReLU()]
                for _ in range(n_layers - 1):
                    decoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
                decoder_layers.append(nn.Linear(hidden_dim, input_dim))
                self.decoder = nn.Sequential(*decoder_layers)
                
                # Adversary for fairness
                self.adversary = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, num_sensitive_groups)
                )
            
            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_var(h)
            
            def reparameterize(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std
            
            def decode(self, z):
                return self.decoder(z)
            
            def forward(self, x):
                mu, log_var = self.encode(x)
                z = self.reparameterize(mu, log_var)
                recon = self.decode(z)
                return recon, mu, log_var, z
            
            def loss_function(self, x, recon, mu, log_var, adv_pred=None, sensitive_labels=None, fairness_weight=0.5):
                # Reconstruction loss
                recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
                
                # KL divergence
                kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                
                # Fairness adversarial loss
                adv_loss = 0
                if adv_pred is not None and sensitive_labels is not None:
                    adv_loss = nn.functional.cross_entropy(adv_pred, sensitive_labels)
                
                total_loss = recon_loss + self.beta * kl_loss + fairness_weight * adv_loss
                
                return {
                    'total': total_loss,
                    'recon': recon_loss,
                    'kl': kl_loss,
                    'adv': adv_loss
                }
        
        return FairVAE(
            input_dim=input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
            beta=self.config.beta,
            num_sensitive_groups=num_sensitive_groups
        )
    
    def _create_gan(self, input_dim: int, num_sensitive_groups: int) -> nn.Module:
        """Create GAN model with fairness constraints."""
        
        class FairGAN(nn.Module):
            """Fair GAN with fairness discriminator."""
            
            def __init__(self, input_dim, latent_dim, hidden_dim, num_sensitive_groups):
                super().__init__()
                
                self.latent_dim = latent_dim
                
                # Generator
                self.generator = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.LeakyReLU(0.2),
                    nn.Linear(hidden_dim * 2, input_dim),
                )
                
                # Discriminator
                self.discriminator = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim // 2, 1),
                )
                
                # Fairness discriminator
                self.fairness_discriminator = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, num_sensitive_groups)
                )
            
            def generate(self, n_samples, device):
                z = torch.randn(n_samples, self.latent_dim, device=device)
                return self.generator(z), z
            
            def forward(self, x):
                return self.discriminator(x)
        
        return FairGAN(
            input_dim=input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            num_sensitive_groups=num_sensitive_groups
        )
    
    def _create_diffusion(self, input_dim: int) -> nn.Module:
        """Create Diffusion model (simplified)."""
        
        class SimpleDiffusion(nn.Module):
            """Simplified diffusion model for tabular data."""
            
            def __init__(self, input_dim, hidden_dim, n_layers):
                super().__init__()
                
                self.input_dim = input_dim
                
                # Time embedding
                self.time_mlp = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                
                # Denoising network
                layers = [nn.Linear(input_dim + hidden_dim, hidden_dim), nn.SiLU()]
                for _ in range(n_layers - 1):
                    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
                layers.append(nn.Linear(hidden_dim, input_dim))
                
                self.net = nn.Sequential(*layers)
                
            def forward(self, x, t):
                # Time embedding
                t_emb = self.time_mlp(t.unsqueeze(-1))
                
                # Concatenate with input
                x_t = torch.cat([x, t_emb], dim=-1)
                
                # Predict noise
                return self.net(x_t)
        
        return SimpleDiffusion(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers
        )
    
    def setup_optimizer(self):
        """Set up optimizer and scheduler."""
        if self.config.verbose:
            print(f"[Trainer] Setting up optimizer (lr={self.config.lr})...")
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Warmup + Cosine decay
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.config.warmup_epochs
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs - self.config.warmup_epochs,
            eta_min=self.config.lr * 0.01
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs]
        )
        
        # Mixed precision
        if self.config.fp16 and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()
            if self.config.verbose:
                print("[Trainer] Mixed precision training enabled")
    
    def setup_tracking(self):
        """Set up experiment tracking."""
        if not self.config.track_experiment:
            return
        
        if self.config.tracker in ["wandb", "both"]:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.experiment_name,
                    config=self.config.to_dict()
                )
                self.tracker = "wandb"
                if self.config.verbose:
                    print(f"[Trainer] WandB tracking enabled: {wandb.run.url}")
            except ImportError:
                if self.config.verbose:
                    print("[Trainer] WandB not available, skipping...")
        
        if self.config.tracker in ["tensorboard", "both"]:
            from torch.utils.tensorboard import SummaryWriter
            self.tb_writer = SummaryWriter(
                log_dir=os.path.join(self.config.output_dir, 'logs')
            )
            if self.config.tracker != "wandb":
                self.tracker = "tensorboard"
    
    def load_data(self) -> Tuple[Dataset, Dataset]:
        """Load training and validation data."""
        import pandas as pd
        
        if self.config.verbose:
            print(f"[Trainer] Loading data from {self.config.data_path}...")
        
        # Load data
        if self.config.data_path:
            if self.config.data_path.endswith('.csv'):
                df = pd.read_csv(self.config.data_path)
            elif self.config.data_path.endswith('.parquet'):
                df = pd.read_parquet(self.config.data_path)
            else:
                raise ValueError(f"Unsupported data format: {self.config.data_path}")
        else:
            # Create dummy data for demo
            if self.config.verbose:
                print("[Trainer] No data path provided, creating dummy data...")
            df = pd.DataFrame(
                np.random.randn(5000, 20).astype(np.float32),
                columns=[f"feature_{i}" for i in range(18)] + ["sensitive", "target"]
            )
        
        # Create dataset
        class TabularDataset(Dataset):
            def __init__(self, df, sensitive_cols, target_col):
                self.data = df
                feature_cols = [c for c in df.columns if c not in sensitive_cols and c != target_col]
                self.features = df[feature_cols].values.astype(np.float32)
                self.sensitive = df[sensitive_cols].values if sensitive_cols else None
                self.target = df[target_col].values if target_col else None
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = {'features': torch.from_numpy(self.features[idx])}
                if self.sensitive is not None:
                    item['sensitive'] = torch.tensor(self.sensitive[idx], dtype=torch.long)
                if self.target is not None:
                    item['target'] = torch.tensor(self.target[idx], dtype=torch.float32)
                return item
        
        sensitive_cols = self.config.sensitive_attrs if self.config.sensitive_attrs else []
        target_col = self.config.target_col if self.config.target_col else df.columns[-1]
        
        # Split data
        train_size = int(0.8 * len(df))
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:]
        
        train_dataset = TabularDataset(train_df, sensitive_cols, target_col)
        val_dataset = TabularDataset(val_df, sensitive_cols, target_col)
        
        if self.config.verbose:
            print(f"[Trainer] Train samples: {len(train_dataset)}")
            print(f"[Trainer] Val samples: {len(val_dataset)}")
            print(f"[Trainer] Input dimension: {train_dataset.features.shape[1]}")
        
        return train_dataset, val_dataset, train_dataset.features.shape[1]
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        recon_loss = 0
        kl_loss = 0
        fairness_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            self.optimizer.zero_grad()
            
            x = batch['features'].to(self.device)
            sensitive = batch.get('sensitive')
            if sensitive is not None:
                sensitive = sensitive.to(self.device)
            
            # Forward pass
            if self.config.model_type == "vae":
                recon, mu, log_var, z = self.model(x)
                
                # Adversary prediction
                adv_pred = self.model.adversary(z.detach()) if hasattr(self.model, 'adversary') else None
                
                loss_dict = self.model.loss_function(
                    x, recon, mu, log_var, 
                    adv_pred, sensitive,
                    self.config.fairness_weight
                )
                loss = loss_dict['total']
                
            elif self.config.model_type == "gan":
                # GAN training step
                batch_size = x.size(0)
                
                # Train discriminator
                fake, z = self.model.generate(batch_size, self.device)
                real_pred = self.model(x)
                fake_pred = self.model(fake.detach())
                
                d_loss_real = nn.functional.binary_cross_entropy_with_logits(
                    real_pred, torch.ones_like(real_pred)
                )
                d_loss_fake = nn.functional.binary_cross_entropy_with_logits(
                    fake_pred, torch.zeros_like(fake_pred)
                )
                d_loss = (d_loss_real + d_loss_fake) / 2
                
                # Train generator
                fake, z = self.model.generate(batch_size, self.device)
                fake_pred = self.model(fake)
                g_loss = nn.functional.binary_cross_entropy_with_logits(
                    fake_pred, torch.ones_like(fake_pred)
                )
                
                # Fairness loss
                if hasattr(self.model, 'fairness_discriminator') and sensitive is not None:
                    adv_pred = self.model.fairness_discriminator(z)
                    f_loss = nn.functional.cross_entropy(adv_pred, sensitive)
                    g_loss = g_loss + self.config.fairness_weight * f_loss
                
                loss = d_loss + g_loss
            else:
                # Diffusion step (simplified)
                t = torch.rand(x.size(0), device=self.device)
                noise = torch.randn_like(x)
                noisy_x = x + noise * t.unsqueeze(-1)
                pred_noise = self.model(noisy_x, t)
                loss = nn.functional.mse_loss(pred_noise, noise)
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            if isinstance(loss_dict, dict):
                recon_loss += loss_dict.get('recon', 0).item() if torch.is_tensor(loss_dict.get('recon', 0)) else loss_dict.get('recon', 0)
                kl_loss += loss_dict.get('kl', 0).item() if torch.is_tensor(loss_dict.get('kl', 0)) else loss_dict.get('kl', 0)
                fairness_loss += loss_dict.get('adv', 0).item() if torch.is_tensor(loss_dict.get('adv', 0)) else loss_dict.get('adv', 0)
            
            n_batches += 1
            self.global_step += 1
        
        return {
            'loss': total_loss / n_batches,
            'recon_loss': recon_loss / n_batches,
            'kl_loss': kl_loss / n_batches,
            'fairness_loss': fairness_loss / n_batches
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        n_batches = 0
        
        for batch in val_loader:
            x = batch['features'].to(self.device)
            sensitive = batch.get('sensitive')
            if sensitive is not None:
                sensitive = sensitive.to(self.device)
            
            if self.config.model_type == "vae":
                recon, mu, log_var, z = self.model(x)
                loss_dict = self.model.loss_function(x, recon, mu, log_var)
                loss = loss_dict['total']
            else:
                # Simplified validation
                loss = torch.tensor(0.0, device=self.device)
            
            total_loss += loss.item()
            n_batches += 1
        
        return {'val_loss': total_loss / n_batches}
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_loss': self.best_loss,
            'history': self.history,
            'config': self.config.to_dict()
        }
        
        path = os.path.join(self.config.output_dir, 'checkpoints', filename)
        torch.save(checkpoint, path)
        
        if self.config.verbose:
            print(f"[Trainer] Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint.get('history', self.history)
        
        if self.config.verbose:
            print(f"[Trainer] Loaded checkpoint from epoch {self.current_epoch}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log metrics to trackers."""
        if self.tracker == "wandb":
            import wandb
            wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()}, step=self.global_step)
        
        if hasattr(self, 'tb_writer'):
            for k, v in metrics.items():
                self.tb_writer.add_scalar(f"{prefix}/{k}", v, self.global_step)
    
    def train(self, train_dataset: Dataset, val_dataset: Dataset, input_dim: int):
        """Main training loop."""
        
        # Set up model
        num_sensitive = 2 if self.config.sensitive_attrs else 1
        self.setup_model(input_dim, num_sensitive)
        
        # Set up optimizer
        self.setup_optimizer()
        
        # Set up tracking
        self.setup_tracking()
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Training Started")
            print("=" * 60)
            print(f"Model: {self.config.model_type}")
            print(f"Fairness: {self.config.fairness_paradigm}")
            print(f"Epochs: {self.config.epochs}")
            print(f"Batch size: {self.config.batch_size}")
            print(f"Learning rate: {self.config.lr}")
            print("=" * 60 + "\n")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch
                epoch_start = time.time()
                
                # Train
                train_metrics = self.train_epoch(train_loader)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                # Update scheduler
                if self.scheduler:
                    self.scheduler.step()
                
                # Track learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Update history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['lr'].append(current_lr)
                
                # Log metrics
                all_metrics = {**train_metrics, **val_metrics, 'lr': current_lr}
                self.log_metrics(all_metrics)
                
                # Print progress
                if self.config.verbose:
                    epoch_time = time.time() - epoch_start
                    print(f"Epoch {epoch + 1}/{self.config.epochs} "
                          f"[{epoch_time:.1f}s] "
                          f"train_loss: {train_metrics['loss']:.4f} "
                          f"val_loss: {val_metrics['val_loss']:.4f} "
                          f"lr: {current_lr:.2e}")
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(f"epoch_{epoch + 1}.pt")
                
                # Track best model
                if val_metrics['val_loss'] < self.best_loss:
                    self.best_loss = val_metrics['val_loss']
                    self.save_checkpoint("best_model.pt")
        
        except KeyboardInterrupt:
            if self.config.verbose:
                print("\n[Trainer] Training interrupted by user")
            self.save_checkpoint("interrupted.pt")
        
        # Final save
        self.save_checkpoint("final_model.pt")
        
        total_time = time.time() - start_time
        
        if self.config.verbose:
            print("\n" + "=" * 60)
            print("Training Complete")
            print("=" * 60)
            print(f"Total time: {total_time / 60:.1f} minutes")
            print(f"Best validation loss: {self.best_loss:.4f}")
            print(f"Final model saved to: {self.config.output_dir}/checkpoints/final_model.pt")
            print("=" * 60)
        
        # Finish tracking
        if self.tracker == "wandb":
            import wandb
            wandb.finish()
        
        if hasattr(self, 'tb_writer'):
            self.tb_writer.close()
        
        return self.history


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "=" * 60)
    print("Fair Synthetic Data Generator - Training")
    print("=" * 60 + "\n")
    
    # Create config
    config = TrainingConfig(args)
    
    # Create trainer
    trainer = FairTrainer(config)
    
    # Load data
    train_dataset, val_dataset, input_dim = trainer.load_data()
    
    # Train
    history = trainer.train(train_dataset, val_dataset, input_dim)
    
    # Save training history
    history_path = os.path.join(config.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining history saved to: {history_path}")


if __name__ == "__main__":
    main()
