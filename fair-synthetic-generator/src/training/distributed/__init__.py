"""
Distributed Training Module
==========================

Distributed training implementations for scalability.

This module provides:
- DDPTrainer: Distributed Data Parallel trainer
- FSDPTrainer: Fully Sharded Data Parallel trainer
"""

from src.training.distributed.ddp_trainer import (
    DDPTrainer,
    DDPConfig,
    setup_ddp,
    cleanup_ddp,
)

from src.training.distributed.fsdp_trainer import (
    FSDPTrainer,
    FSDPConfig,
    get_fsdp_wrapped_model,
)


__all__ = [
    # DDP
    "DDPTrainer",
    "DDPConfig",
    "setup_ddp",
    "cleanup_ddp",
    
    # FSDP
    "FSDPTrainer",
    "FSDPConfig",
    "get_fsdp_wrapped_model",
]
