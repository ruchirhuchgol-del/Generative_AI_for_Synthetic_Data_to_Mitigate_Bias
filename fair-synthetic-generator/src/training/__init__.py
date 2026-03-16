"""
Training Module
================

Comprehensive training infrastructure for fair synthetic data generation.

This module provides:

## Core Components
- Trainer: Main training loop with adversarial debiasing
- TrainingConfig: Training configuration dataclass

## Training Strategies
- AdversarialTrainingStrategy: GRL-based adversarial debiasing
- MultiTaskTrainingStrategy: Multi-objective optimization
- CurriculumTrainingStrategy: Progressive difficulty training

## Optimizers
- MultiObjectiveOptimizer: MGDA-based optimization
- SchedulerFactory: Learning rate scheduler factory

## Callbacks
- FairnessCallback: Monitor fairness metrics
- CheckpointCallback: Save/load checkpoints
- LoggingCallback: Log training progress
- TensorBoardCallback: TensorBoard integration
- WandBCallback: Weights & Biases integration

## Distributed Training
- DDPTrainer: Distributed Data Parallel
- FSDPTrainer: Fully Sharded Data Parallel

## Architecture Overview

```
                    ┌─────────────────────────────────────┐
                    │           Trainer                   │
                    └─────────────────────────────────────┘
                                      │
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
    ┌──────────────┐          ┌──────────────┐          ┌──────────────┐
    │  Strategies  │          │  Optimizers  │          │   Callbacks  │
    │ - Adversarial│          │ - MGDA       │          │ - Fairness   │
    │ - Multi-Task │          │ - Schedulers │          │ - Checkpoint │
    │ - Curriculum │          │              │          │ - Logging    │
    └──────────────┘          └──────────────┘          └──────────────┘
           │                          │                          │
           └──────────────────────────┼──────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │     Distributed Training            │
                    │  - DDP (Distributed Data Parallel)  │
                    │  - FSDP (Fully Sharded)             │
                    └─────────────────────────────────────┘
```

## Example Usage

### Basic Training
```python
from src.training import Trainer, TrainingConfig

trainer = Trainer(
    model=model,
    train_dataloader=train_loader,
    config=TrainingConfig(n_epochs=100)
)

trainer.fit()
```

### With Fairness Callback
```python
from src.training import Trainer, FairnessCallback

trainer = Trainer(model=model, train_dataloader=train_loader)
trainer.add_callback(FairnessCallback(
    sensitive_attrs=["gender"],
    constraints=["demographic_parity"]
))

trainer.fit()
```

### Adversarial Training
```python
from src.training import AdversarialTrainingStrategy, AdversarialConfig

strategy = AdversarialTrainingStrategy(
    model=generator,
    adversary=adversary,
    config=AdversarialConfig(mode="gradient_reversal")
)

for epoch in range(num_epochs):
    metrics = strategy.train_epoch(dataloader, epoch)
```

### Distributed Training
```python
# Launch with: torchrun --nproc_per_node=4 train.py
from src.training import DDPTrainer

trainer = DDPTrainer(
    model=model,
    train_dataloader=train_loader
)

trainer.fit()
```
"""

# Core trainer
from src.training.trainer import Trainer, TrainingConfig

# Strategies
from src.training.strategies import (
    AdversarialTrainingStrategy,
    AdversarialConfig,
    AdversarialMode,
    MultiTaskTrainingStrategy,
    MultiTaskConfig,
    CurriculumTrainingStrategy,
    CurriculumConfig,
    get_strategy,
    STRATEGY_REGISTRY,
)

# Optimizers
from src.training.optimizers import (
    MultiObjectiveOptimizer,
    get_scheduler,
    get_linear_warmup,
    get_cosine_schedule,
    SchedulerFactory,
)

# Callbacks
from src.training.callbacks import (
    FairnessCallback,
    FairnessVisualizationCallback,
    FairnessMetrics,
    CheckpointCallback,
    ModelCheckpointCallback,
    BestModelCheckpointCallback,
    LoggingCallback,
    TensorBoardCallback,
    WandBCallback,
    MetricsLogger,
    get_callback,
    CALLBACK_REGISTRY,
)

# Distributed
from src.training.distributed import (
    DDPTrainer,
    DDPConfig,
    setup_ddp,
    cleanup_ddp,
    FSDPTrainer,
    FSDPConfig,
    get_fsdp_wrapped_model,
)


__all__ = [
    # Core
    "Trainer",
    "TrainingConfig",
    
    # Strategies
    "AdversarialTrainingStrategy",
    "AdversarialConfig",
    "AdversarialMode",
    "MultiTaskTrainingStrategy",
    "MultiTaskConfig",
    "CurriculumTrainingStrategy",
    "CurriculumConfig",
    "get_strategy",
    "STRATEGY_REGISTRY",
    
    # Optimizers
    "MultiObjectiveOptimizer",
    "get_scheduler",
    "get_linear_warmup",
    "get_cosine_schedule",
    "SchedulerFactory",
    
    # Callbacks
    "FairnessCallback",
    "FairnessVisualizationCallback",
    "FairnessMetrics",
    "CheckpointCallback",
    "ModelCheckpointCallback",
    "BestModelCheckpointCallback",
    "LoggingCallback",
    "TensorBoardCallback",
    "WandBCallback",
    "MetricsLogger",
    "get_callback",
    "CALLBACK_REGISTRY",
    
    # Distributed
    "DDPTrainer",
    "DDPConfig",
    "setup_ddp",
    "cleanup_ddp",
    "FSDPTrainer",
    "FSDPConfig",
    "get_fsdp_wrapped_model",
]
