"""
Fully Sharded Data Parallel (FSDP) Trainer
============================================

Implements distributed training using PyTorch FSDP for memory efficiency.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.utils.data import DataLoader, DistributedSampler

from src.training.trainer import Trainer, TrainingConfig
from src.core.utils import get_logger


class ShardingType(Enum):
    """FSDP sharding strategies."""
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID_SHARD = "hybrid_shard"


@dataclass
class FSDPConfig:
    """Configuration for FSDP training."""
    sharding_strategy: str = "full_shard"
    mixed_precision: bool = True
    compute_dtype: str = "bf16"
    fp32_reduce: bool = True
    flatten_parameters: bool = True
    use_orig_params: bool = True
    sync_module_states: bool = True
    forward_prefetch: bool = False
    backward_prefetch: bool = True
    limit_all_gathers: bool = True
    
    # Wrapping policies
    min_params: int = 1_000_000
    transformer_wrap: bool = False
    transformer_layer_cls: Optional[str] = None
    
    # Checkpoint
    state_dict_type: str = "full"
    
    # DDP settings
    backend: str = "nccl"
    world_size: int = -1
    rank: int = -1
    local_rank: int = -1


def get_fsdp_wrapped_model(
    model: nn.Module,
    config: FSDPConfig,
    device: torch.device
) -> FSDP:
    """
    Wrap model with FSDP.
    
    Args:
        model: Model to wrap
        config: FSDP configuration
        device: Device
        
    Returns:
        FSDP-wrapped model
    """
    # Sharding strategy
    strategy_map = {
        "full_shard": ShardingStrategy.FULL_SHARD,
        "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
        "no_shard": ShardingStrategy.NO_SHARD,
        "hybrid_shard": ShardingStrategy.HYBRID_SHARD,
    }
    sharding_strategy = strategy_map.get(
        config.sharding_strategy,
        ShardingStrategy.FULL_SHARD
    )
    
    # Mixed precision
    mixed_precision = None
    if config.mixed_precision:
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        compute_dtype = dtype_map.get(config.compute_dtype, torch.bfloat16)
        
        mixed_precision = MixedPrecision(
            param_dtype=compute_dtype,
            reduce_dtype=torch.float32 if config.fp32_reduce else compute_dtype,
            buffer_dtype=torch.float32,
        )
    
    # Auto wrap policy
    auto_wrap_policy = None
    
    if config.transformer_wrap and config.transformer_layer_cls:
        # Import transformer layer class
        try:
            parts = config.transformer_layer_cls.rsplit(".", 1)
            module = __import__(parts[0], fromlist=[parts[1]])
            layer_cls = getattr(module, parts[1])
            
            auto_wrap_policy = transformer_auto_wrap_policy(
                transformer_layer_cls=layer_cls
            )
        except (ImportError, AttributeError):
            pass
    
    if auto_wrap_policy is None:
        auto_wrap_policy = size_based_auto_wrap_policy(
            min_num_params=config.min_params
        )
    
    # Wrap model
    fsdp_model = FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        use_orig_params=config.use_orig_params,
        sync_module_states=config.sync_module_states,
        forward_prefetch=config.forward_prefetch,
        backward_prefetch=config.backward_prefetch,
        limit_all_gathers=config.limit_all_gathers,
    )
    
    return fsdp_model


class FSDPTrainer(Trainer):
    """
    Fully Sharded Data Parallel Trainer.
    
    Uses FSDP for memory-efficient distributed training of large models.
    Automatically shards model parameters, gradients, and optimizer states.
    
    Example:
        >>> # Launch with: torchrun --nproc_per_node=4 train.py
        >>> trainer = FSDPTrainer(
        ...     model=large_model,
        ...     train_dataloader=train_loader,
        ...     config=FSDPConfig(mixed_precision=True)
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
        fsdp_config: Optional[FSDPConfig] = None,
        adversary: Optional[nn.Module] = None,
        adversary_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Initialize FSDP trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training dataloader
            val_dataloader: Optional validation dataloader
            optimizer: Model optimizer
            scheduler: Learning rate scheduler
            config: Training configuration
            fsdp_config: FSDP-specific configuration
            adversary: Optional adversary network
            adversary_optimizer: Adversary optimizer
        """
        self.fsdp_config = fsdp_config or FSDPConfig()
        
        # Initialize distributed
        self._setup_distributed()
        
        # Set device
        self.config = config or TrainingConfig()
        self.device = torch.device(f"cuda:{self.fsdp_config.local_rank}")
        
        # Wrap model with FSDP
        self._unwrapped_model = model
        self.model = get_fsdp_wrapped_model(model, self.fsdp_config, self.device)
        
        # Setup dataloaders
        self.train_dataloader = self._setup_dataloader(train_dataloader)
        if val_dataloader is not None:
            self.val_dataloader = self._setup_dataloader(val_dataloader, shuffle=False)
        else:
            self.val_dataloader = None
        
        # Optimizer (must be created after FSDP wrap)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = scheduler
        
        # Adversary (also needs FSDP wrapping)
        if adversary is not None:
            self._unwrapped_adversary = adversary
            self.adversary = get_fsdp_wrapped_model(
                adversary, self.fsdp_config, self.device
            )
        else:
            self.adversary = None
            self._unwrapped_adversary = None
        
        self.adversary_optimizer = adversary_optimizer
        
        # Logger
        self.logger = get_logger(f"FSDPTrainer_rank{self.fsdp_config.rank}")
        
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
    
    def _setup_distributed(self) -> None:
        """Initialize distributed environment."""
        if self.fsdp_config.world_size == -1:
            self.fsdp_config.world_size = int(os.environ.get("WORLD_SIZE", 1))
        if self.fsdp_config.rank == -1:
            self.fsdp_config.rank = int(os.environ.get("RANK", 0))
        if self.fsdp_config.local_rank == -1:
            self.fsdp_config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.fsdp_config.backend,
                world_size=self.fsdp_config.world_size,
                rank=self.fsdp_config.rank
            )
        
        torch.cuda.set_device(self.fsdp_config.local_rank)
    
    def _setup_dataloader(
        self,
        dataloader: DataLoader,
        shuffle: bool = True
    ) -> DataLoader:
        """Wrap dataloader with DistributedSampler."""
        sampler = DistributedSampler(
            dataloader.dataset,
            num_replicas=self.fsdp_config.world_size,
            rank=self.fsdp_config.rank,
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
        return self.fsdp_config.rank == 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with FSDP."""
        self.model.train()
        
        self.train_dataloader.sampler.set_epoch(self.current_epoch)
        
        total_loss = 0.0
        n_batches = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            if isinstance(batch, dict):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            
            self.optimizer.zero_grad()
            
            outputs = self.model(**batch) if isinstance(batch, dict) else self.model(batch)
            
            if isinstance(outputs, dict):
                loss = outputs.get("loss", outputs.get("losses", {}).get("total", 0))
            else:
                loss = outputs
            
            loss.backward()
            
            if self.config.grad_clip > 0:
                self.model.clip_grad_norm_(self.config.grad_clip)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        avg_loss = total_loss / max(n_batches, 1)
        
        # Average across processes
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / self.fsdp_config.world_size
        
        self.history["train_loss"].append(avg_loss)
        self.history["learning_rate"].append(
            self.optimizer.param_groups[0]["lr"]
        )
        
        return {"train_loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Run validation with FSDP."""
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
        
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / self.fsdp_config.world_size
        
        self.history["val_loss"].append(avg_loss)
        
        return {"val_loss": avg_loss}
    
    def save_checkpoint(self, path: str, **kwargs) -> None:
        """Save checkpoint with FSDP state dict handling."""
        if not self.is_main_process():
            return
        
        # Configure state dict saving
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            save_policy
        ):
            state_dict = self.model.state_dict()
        
        checkpoint = {
            "model_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "history": self.history,
            "config": self.config,
            "fsdp_config": self.fsdp_config,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load checkpoint with FSDP state dict handling."""
        checkpoint = torch.load(path, map_location="cpu")
        
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            load_policy
        ):
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint["history"]
        
        self.logger.info(f"Checkpoint loaded from {path}")
    
    def fit(self) -> Dict[str, List[float]]:
        """Run full training loop."""
        if self.is_main_process():
            self.logger.info(
                f"Starting FSDP training for {self.config.n_epochs} epochs"
            )
            self.logger.info(
                f"World size: {self.fsdp_config.world_size}, "
                f"Rank: {self.fsdp_config.rank}"
            )
        
        for epoch in range(self.config.n_epochs):
            self.current_epoch = epoch
            
            train_metrics = self.train_epoch()
            
            if self.is_main_process():
                self.logger.info(
                    f"Epoch {epoch}: train_loss = {train_metrics['train_loss']:.4f}"
                )
            
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
        if dist.is_initialized():
            dist.destroy_process_group()
