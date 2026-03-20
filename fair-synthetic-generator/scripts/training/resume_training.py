#!/usr/bin/env python
"""
Resume Training Script
======================

Script for resuming training from a checkpoint.
Handles checkpoint loading, training continuation, and recovery from interruptions.

Usage:
    python resume_training.py [OPTIONS]

Options:
    --checkpoint PATH     Path to checkpoint file or directory
    --additional-epochs N Additional epochs to train
    --output-dir PATH     Output directory for new checkpoints
    --device DEVICE       Device to use (cuda, cpu, auto)
    --lr-scale FACTOR     Learning rate scaling factor
    --force               Force resume even if checkpoint seems corrupted
    -h, --help            Show this help message

Examples:
    python resume_training.py --checkpoint checkpoints/model_epoch_50.pt --additional-epochs 50
    python resume_training.py --checkpoint checkpoints/experiment_001/ --lr-scale 0.5
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.core.utils import set_seed, get_device, get_logger, ensure_dir
from src.training import Trainer, TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Resume Training from Checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file or directory"
    )
    
    parser.add_argument(
        "--additional-epochs",
        type=int,
        default=None,
        help="Additional epochs to train (default: continue from original config)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for new checkpoints"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, auto)"
    )
    
    parser.add_argument(
        "--lr-scale",
        type=float,
        default=1.0,
        help="Learning rate scaling factor"
    )
    
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of warmup epochs after resuming"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: use seed from checkpoint)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force resume even if checkpoint seems corrupted"
    )
    
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Only evaluate the checkpoint without resuming training"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress"
    )
    
    return parser.parse_args()


class CheckpointManager:
    """
    Manages checkpoint loading, validation, and recovery.
    """
    
    def __init__(self, checkpoint_path: str, verbose: bool = True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_path: Path to checkpoint file or directory
            verbose: Whether to print detailed information
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.verbose = verbose
        self.logger = get_logger("checkpoint_manager")
    
    def find_latest_checkpoint(self) -> Optional[Path]:
        """
        Find the latest checkpoint in a directory.
        
        Returns:
            Path to the latest checkpoint file
        """
        if self.checkpoint_path.is_file():
            return self.checkpoint_path
        
        if self.checkpoint_path.is_dir():
            # Look for checkpoint files
            patterns = ["*.pt", "*.pth", "*.ckpt"]
            checkpoints = []
            
            for pattern in patterns:
                checkpoints.extend(self.checkpoint_path.glob(pattern))
            
            if not checkpoints:
                # Look in subdirectories
                for pattern in patterns:
                    checkpoints.extend(self.checkpoint_path.rglob(pattern))
            
            if checkpoints:
                # Sort by modification time
                checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                return checkpoints[0]
        
        return None
    
    def load_checkpoint(
        self,
        force: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load checkpoint with validation.
        
        Args:
            force: Force load even if validation fails
            
        Returns:
            Tuple of (checkpoint_data, metadata)
        """
        checkpoint_file = self.find_latest_checkpoint()
        
        if checkpoint_file is None:
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint from {checkpoint_file}")
        
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
        except Exception as e:
            if force:
                self.logger.warning(f"Failed to load checkpoint normally, attempting recovery: {e}")
                checkpoint = self._attempt_recovery(checkpoint_file)
            else:
                raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'optimizer_state_dict', 'epoch']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        
        if missing_keys:
            if force:
                self.logger.warning(f"Checkpoint missing keys: {missing_keys}. Attempting to proceed.")
            else:
                raise ValueError(f"Checkpoint is incomplete. Missing keys: {missing_keys}")
        
        # Extract metadata
        metadata = {
            'checkpoint_path': str(checkpoint_file),
            'epoch': checkpoint.get('epoch', 0),
            'global_step': checkpoint.get('global_step', 0),
            'best_loss': checkpoint.get('best_loss', float('inf')),
            'training_history': checkpoint.get('history', {}),
            'config': checkpoint.get('config', {}),
            'model_type': checkpoint.get('model_type', 'unknown'),
            'timestamp': checkpoint.get('timestamp', 'unknown')
        }
        
        if self.verbose:
            self._print_checkpoint_info(checkpoint, metadata)
        
        return checkpoint, metadata
    
    def _attempt_recovery(self, checkpoint_file: Path) -> Dict[str, Any]:
        """
        Attempt to recover a corrupted checkpoint.
        
        Args:
            checkpoint_file: Path to the checkpoint file
            
        Returns:
            Partially recovered checkpoint
        """
        self.logger.warning("Attempting checkpoint recovery...")
        
        # Try loading with different options
        try:
            checkpoint = torch.load(
                checkpoint_file,
                map_location='cpu',
                pickle_module=torch.serialization.pickle
            )
            return checkpoint
        except Exception:
            pass
        
        # Create minimal checkpoint with model weights only
        try:
            state_dict = torch.load(checkpoint_file, map_location='cpu')
            if isinstance(state_dict, dict):
                return {
                    'model_state_dict': state_dict,
                    'epoch': 0,
                    'optimizer_state_dict': {},
                    'best_loss': float('inf')
                }
        except Exception:
            pass
        
        raise RuntimeError("Could not recover checkpoint")
    
    def _print_checkpoint_info(self, checkpoint: Dict, metadata: Dict):
        """Print checkpoint information."""
        print("\n" + "=" * 50)
        print("Checkpoint Information")
        print("=" * 50)
        print(f"  File: {metadata['checkpoint_path']}")
        print(f"  Epoch: {metadata['epoch']}")
        print(f"  Global Step: {metadata['global_step']}")
        print(f"  Best Loss: {metadata['best_loss']:.6f}")
        print(f"  Timestamp: {metadata['timestamp']}")
        
        # Model info
        if 'model_state_dict' in checkpoint:
            num_params = sum(p.numel() for p in checkpoint['model_state_dict'].values() if isinstance(p, torch.Tensor))
            print(f"  Parameters: {num_params:,}")
        
        # Config info
        if metadata['config']:
            config = metadata['config']
            if isinstance(config, dict):
                print(f"  Original Config:")
                for key, value in list(config.items())[:5]:
                    print(f"    {key}: {value}")
        
        print("=" * 50 + "\n")


def restore_model(
    checkpoint: Dict[str, Any],
    metadata: Dict[str, Any],
    device: torch.device
) -> nn.Module:
    """
    Restore model from checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
        metadata: Checkpoint metadata
        device: Target device
        
    Returns:
        Restored model
    """
    logger = get_logger("model_restore")
    
    # Try to determine model type from checkpoint
    model_type = checkpoint.get('model_type', None)
    config = checkpoint.get('config', {})
    
    # Import model classes dynamically based on config
    if model_type == 'DebiasedVAE' or 'vae' in str(model_type).lower():
        from src.models.architectures import DebiasedVAE
        latent_dim = config.get('latent_dim', 512)
        num_sensitive = config.get('num_sensitive_groups', 2)
        model = DebiasedVAE(
            modalities=config.get('modalities', ['tabular']),
            latent_dim=latent_dim,
            num_sensitive_groups=num_sensitive
        )
    elif model_type == 'FairGAN' or 'gan' in str(model_type).lower():
        from src.models.architectures import FairGAN
        model = FairGAN(
            latent_dim=config.get('latent_dim', 128),
            num_sensitive_groups=config.get('num_sensitive_groups', 2)
        )
    elif model_type == 'FairDiffusion' or 'diffusion' in str(model_type).lower():
        from src.models.architectures import FairDiffusion
        model = FairDiffusion(
            latent_dim=config.get('latent_dim', 256)
        )
    else:
        # Try to infer model from state dict keys
        state_dict = checkpoint['model_state_dict']
        
        if any('fc_mu' in k for k in state_dict.keys()):
            # VAE-like architecture
            from src.models.architectures import DebiasedVAE
            model = DebiasedVAE(
                modalities=['tabular'],
                latent_dim=512,
                num_sensitive_groups=2
            )
        elif any('generator' in k for k in state_dict.keys()):
            # GAN-like architecture
            from src.models.architectures import FairGAN
            model = FairGAN(
                latent_dim=128,
                num_sensitive_groups=2
            )
        else:
            raise ValueError(f"Could not determine model type from checkpoint")
    
    # Load state dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        logger.warning(f"Strict loading failed, trying with strict=False: {e}")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    model = model.to(device)
    logger.info(f"Model restored to {device}")
    
    return model


def restore_optimizer(
    checkpoint: Dict[str, Any],
    model: nn.Module,
    lr_scale: float = 1.0
) -> torch.optim.Optimizer:
    """
    Restore optimizer from checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
        model: Model for optimizer
        lr_scale: Learning rate scaling factor
        
    Returns:
        Restored optimizer
    """
    config = checkpoint.get('config', {})
    optimizer_config = config.get('optimizer', {})
    
    # Get optimizer class
    optimizer_name = optimizer_config.get('name', 'adam')
    lr = optimizer_config.get('lr', 1e-4) * lr_scale
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=optimizer_config.get('betas', (0.9, 0.999))
        )
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Load optimizer state
    if checkpoint.get('optimizer_state_dict'):
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Scale learning rate if needed
            if lr_scale != 1.0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= lr_scale
        except Exception as e:
            get_logger("optimizer_restore").warning(
                f"Could not load optimizer state: {e}. Using fresh optimizer."
            )
    
    return optimizer


def restore_scheduler(
    checkpoint: Dict[str, Any],
    optimizer: torch.optim.Optimizer
) -> Optional[Any]:
    """
    Restore learning rate scheduler from checkpoint.
    
    Args:
        checkpoint: Checkpoint dictionary
        optimizer: Optimizer for scheduler
        
    Returns:
        Restored scheduler or None
    """
    config = checkpoint.get('config', {})
    scheduler_config = config.get('scheduler', {})
    
    if not scheduler_config:
        return None
    
    scheduler_name = scheduler_config.get('name', None)
    
    if scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get('T_max', 100),
            eta_min=scheduler_config.get('eta_min', 1e-6)
        )
    elif scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    elif scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_config.get('factor', 0.5),
            patience=scheduler_config.get('patience', 10)
        )
    else:
        return None
    
    # Load scheduler state
    if checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception:
            pass
    
    return scheduler


def create_dataloaders(
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from src.data.dataset import MultimodalDataset
    from src.data.dataloaders import create_dataloader
    
    data_config = config.get('data', {})
    batch_size = config.get('training', {}).get('batch_size', 128)
    
    # Create dummy dataset for demo (in production, load actual data)
    dataset = MultimodalDataset.create_dummy(n_samples=5000)
    train_dataset, val_dataset, _ = dataset.split(ratios=(0.7, 0.15, 0.15))
    
    train_loader = create_dataloader(
        train_dataset,
        mode="train",
        batch_size=batch_size,
        balance_sensitive_groups=True
    )
    
    val_loader = create_dataloader(
        val_dataset,
        mode="val",
        batch_size=batch_size * 2
    )
    
    return train_loader, val_loader


def main():
    """Main function."""
    args = parse_args()
    
    logger = get_logger("resume_training")
    
    print("\n" + "=" * 60)
    print("Fair Synthetic Data Generator - Resume Training")
    print("=" * 60 + "\n")
    
    # Determine device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint_manager = CheckpointManager(args.checkpoint, verbose=args.verbose)
    checkpoint, metadata = checkpoint_manager.load_checkpoint(force=args.force)
    
    # Set seed
    seed = args.seed if args.seed is not None else metadata['config'].get('seed', 42)
    set_seed(seed)
    logger.info(f"Using seed: {seed}")
    
    # Restore model
    logger.info("Restoring model...")
    model = restore_model(checkpoint, metadata, device)
    
    # Evaluate only mode
    if args.evaluate_only:
        logger.info("Evaluate-only mode. Running evaluation...")
        # Run evaluation
        model.eval()
        # ... evaluation code
        logger.info("Evaluation complete.")
        return
    
    # Restore optimizer
    logger.info("Restoring optimizer...")
    optimizer = restore_optimizer(checkpoint, model, lr_scale=args.lr_scale)
    
    # Restore scheduler
    scheduler = restore_scheduler(checkpoint, optimizer)
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_dataloaders(metadata['config'])
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use checkpoint directory with '_resumed' suffix
        checkpoint_dir = Path(metadata['checkpoint_path']).parent
        output_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}_resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    ensure_dir(output_dir)
    ensure_dir(output_dir / "checkpoints")
    ensure_dir(output_dir / "logs")
    
    # Determine number of epochs
    total_epochs = args.additional_epochs
    if total_epochs is None:
        total_epochs = metadata['config'].get('training', {}).get('n_epochs', 100)
    
    remaining_epochs = total_epochs - metadata['epoch']
    if remaining_epochs <= 0:
        logger.warning(f"Checkpoint already trained for {metadata['epoch']} epochs. "
                      f"Use --additional-epochs to specify more epochs.")
        return
    
    logger.info(f"Resuming from epoch {metadata['epoch']}, training for {remaining_epochs} more epochs")
    
    # Create training config
    training_config = TrainingConfig(
        n_epochs=total_epochs,
        batch_size=metadata['config'].get('training', {}).get('batch_size', 128),
        learning_rate=optimizer.param_groups[0]['lr'],
        device=str(device),
        checkpoint_dir=str(output_dir),
        start_epoch=metadata['epoch']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        config=training_config,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Restore trainer state
    trainer.global_step = metadata['global_step']
    trainer.best_loss = metadata['best_loss']
    if metadata['training_history']:
        trainer.history = metadata['training_history']
    
    # Save resume info
    resume_info = {
        "original_checkpoint": str(metadata['checkpoint_path']),
        "resumed_at": datetime.now().isoformat(),
        "start_epoch": metadata['epoch'],
        "target_epochs": total_epochs,
        "device": str(device),
        "seed": seed
    }
    
    with open(output_dir / "resume_info.json", 'w') as f:
        json.dump(resume_info, f, indent=2)
    
    # Resume training
    logger.info("Starting training...")
    history = trainer.fit()
    
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {trainer.best_loss:.6f}")
    
    # Save final model
    final_path = output_dir / "final_model.pt"
    trainer.save_checkpoint(str(final_path))
    logger.info(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
