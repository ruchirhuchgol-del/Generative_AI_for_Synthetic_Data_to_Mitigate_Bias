"""
Distributed Data Parallel (DDP) Trainer
==========================================

Implements distributed training using PyTorch DDP.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from src.training.trainer import Trainer, TrainingConfig
from src.core.utils import get_logger


@dataclass
class DDPConfig:
    """Configuration for DDP training."""
    backend: str = "nccl"
    init_method: str = "env://"
    world_size: int = -1  # Set from environment
    rank: int = -1  # Set from environment
    local_rank: int = -1
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    gradient_as_bucket_view: bool = True
    bucket_cap_mb: int = 25


def setup_ddp(config: DDPConfig) -> None:
    """
    Initialize distributed training environment.
    
    Args:
        config: DDP configuration
    """
    # Get world size and rank from environment
    if config.world_size == -1:
        config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    if config.rank == -1:
        config.rank = int(os.environ.get("RANK", 0))
    if config.local_rank == -1:
        config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(
            backend=config.backend,
            init_method=config.init_method,
            world_size=config.world_size,
            rank=config.rank
        )
    
    # Set device for this process
    torch.cuda.set_device(config.local_rank)


def cleanup_ddp() -> None:
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class DDPTrainer(Trainer):
    """
    Distributed Data Parallel Trainer.
    
    Extends base Trainer with DDP support for multi-GPU training.
    
    Example:
        >>> # Launch with: torchrun --nproc_per_node=4 train.py
        >>> trainer = DDPTrainer(
        ...     model=model,
        ...     train_dataloader=train_loader,
        ...     config=DDPConfig()
        ... )
        >>> trainer.fit()
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        config: Optional[TrainingConfig] = None,
        ddp_config: Optional[DDPConfig] = None,
        adversary: Optional[nn.Module] = None,
        adversary_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Initialize DDP trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training dataloader (will be wrapped)
            val_dataloader: Optional validation dataloader
            optimizer: Model optimizer
            scheduler: Learning rate scheduler
            config: Training configuration
            ddp_config: DDP-specific configuration
            adversary: Optional adversary network
            adversary_optimizer: Adversary optimizer
        """
        self.ddp_config = ddp_config or DDPConfig()
        
        # Setup DDP
        setup_ddp(self.ddp_config)
        
        # Set device
        self.config = config or TrainingConfig()
        self.device = torch.device(f"cuda:{self.ddp_config.local_rank}")
        
        # Move model to device
        model = model.to(self.device)
        
        # Wrap model with DDP
        self.model = DDP(
            model,
            device_ids=[self.ddp_config.local_rank],
            output_device=self.ddp_config.local_rank,
            find_unused_parameters=self.ddp_config.find_unused_parameters,
            broadcast_buffers=self.ddp_config.broadcast_buffers,
            gradient_as_bucket_view=self.ddp_config.gradient_as_bucket_view,
            bucket_cap_mb=self.ddp_config.bucket_cap_mb
        )
        
        # Store unwrapped model for saving
        self._unwrapped_model = model
        
        # Setup dataloaders with DistributedSampler
        self.train_dataloader = self._setup_dataloader(train_dataloader)
        if val_dataloader is not None:
            self.val_dataloader = self._setup_dataloader(val_dataloader, shuffle=False)
        else:
            self.val_dataloader = None
        
        # Optimizer
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = scheduler
        
        # Adversary (also needs DDP wrapping)
        if adversary is not None:
            self.adversary = DDP(
                adversary.to(self.device),
                device_ids=[self.ddp_config.local_rank],
                output_device=self.ddp_config.local_rank
            )
            self._unwrapped_adversary = adversary
        else:
            self.adversary = None
            self._unwrapped_adversary = None
        
        self.adversary_optimizer = adversary_optimizer
        
        # Logger
        self.logger = get_logger(f"DDPTrainer_rank{self.ddp_config.rank}")
        
        # State tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "learning_rate": []
        }
        
        # Callbacks
        self.callbacks = []
    
    def _setup_dataloader(
        self,
        dataloader: DataLoader,
        shuffle: bool = True
    ) -> DataLoader:
        """Wrap dataloader with DistributedSampler."""
        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=self.ddp_config.world_size,
            rank=self.ddp_config.rank,
            shuffle=shuffle
        )
        
        return DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            sampler=sampler,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last
        )
    
    def is_main_process(self) -> bool:
        """Check if this is the main process."""
        return self.ddp_config.rank == 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with DDP."""
        self.model.train()
        
        # Set epoch for sampler (ensures proper shuffling)
        self.train_dataloader.sampler.set_epoch(self.current_epoch)
        
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
            
            if isinstance(outputs, dict):
                loss = outputs.get("loss", outputs.get("losses", {}).get("total", 0))
            else:
                loss = outputs
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1
        
        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Average loss across all processes
        avg_loss = total_loss / max(n_batches, 1)
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / self.ddp_config.world_size
        
        self.history["train_loss"].append(avg_loss)
        self.history["learning_rate"].append(
            self.optimizer.param_groups[0]["lr"]
        )
        
        return {"train_loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Run validation with DDP."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                if isinstance(batch, dict):
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                
                outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
                
                if isinstance(outputs, dict):
                    loss = outputs.get("loss", outputs.get("losses", {}).get("total", 0))
                else:
                    loss = outputs
                
                total_loss += loss.item()
                n_batches += 1
        
        avg_loss = total_loss / max(n_batches, 1)
        
        # Average across processes
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / self.ddp_config.world_size
        
        self.history["val_loss"].append(avg_loss)
        
        return {"val_loss": avg_loss}
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save checkpoint (only on main process)."""
        if not self.is_main_process():
            return
        
        checkpoint = {
            "model_state_dict": self._unwrapped_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "history": self.history,
            "config": self.config,
            "ddp_config": self.ddp_config,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self._unwrapped_adversary is not None:
            checkpoint["adversary_state_dict"] = self._unwrapped_adversary.state_dict()
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self._unwrapped_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self._unwrapped_adversary is not None and "adversary_state_dict" in checkpoint:
            self._unwrapped_adversary.load_state_dict(checkpoint["adversary_state_dict"])
        
        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint["history"]
        
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def fit(self) -> Dict[str, List[float]]:
        """Run full training loop."""
        if self.is_main_process():
            self.logger.info(
                f"Starting DDP training for {self.config.n_epochs} epochs"
            )
            self.logger.info(
                f"World size: {self.ddp_config.world_size}, "
                f"Rank: {self.ddp_config.rank}"
            )
        
        for epoch in range(self.config.n_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            if self.is_main_process():
                self.logger.info(
                    f"Epoch {epoch}: train_loss = {train_metrics['train_loss']:.4f}"
                )
            
            # Validate
            if self.val_dataloader is not None and epoch % self.config.eval_interval == 0:
                val_metrics = self.validate()
                
                if self.is_main_process():
                    self.logger.info(
                        f"Epoch {epoch}: val_loss = {val_metrics.get('val_loss', 0):.4f}"
                    )
        
        if self.is_main_process():
            self.logger.info("Training completed")
        
        return self.history
    
    def __del__(self):
        """Cleanup on deletion."""
        cleanup_ddp()
