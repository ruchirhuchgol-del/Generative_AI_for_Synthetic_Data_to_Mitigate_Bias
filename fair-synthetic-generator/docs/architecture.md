# Architecture Overview

The Fair Synthetic Data Generator is built on a modular architecture that separates data processing, model logic, and fairness enforcement.

## Core Principles

1.  **Decoupling**: Data modalities are handled by specialized encoders/decoders, but share a common fairness-aware latent space.
2.  **Adversarial Fairness**: We use an adversary to ensure the latent representation $Z$ is independent of sensitive attributes $S$.
3.  **Constraint-Based Optimization**: Explicit fairness metrics (e.g., Demographic Parity) are included as differentiable penalties in the loss function.
4.  **Extensibility**: New generators, encoders, or fairness constraints can be added by implementing simple interfaces.

## System Components

### 1. Data Layer (`src.data`)
- **Schema Management**: Defines the structure and sensitive attributes of the dataset.
- **Preprocessors**: Handles normalization, categorical encoding, and image resizing.

### 2. Model Layer (`src.models`)
- **Encoders**: Modality-specific networks (MLP, CNN, Transformer).
- **Generators**: High-level wrappers for VAE, GAN, or Diffusion processes.
- **Architectures**: End-to-end models like `FairGAN` or `FairDiffusion`.

### 3. Fairness Layer (`src.fairness`)
- **Constraints**: Mathematical formulations of fairness paradigms.
- **Losses**: Implementation of adversarial and constraint-based losses.

### 4. Training Layer (`src.training`)
- **Strategies**: Multi-task, Adversarial, and Curriculum training logic.
- **Callbacks**: Monitoring metrics and model checkpoints.

### 5. Evaluation Layer (`src.evaluation`)
- **Fidelity**: Statistical similarity between synthetic and real data.
- **Fairness**: Quantifying bias across protected groups.
- **Privacy**: Membership inference and differential privacy analysis.

## Workflow

1.  **Config**: User provides a YAML configuration.
2.  **Setup**: The `GeneratorPipeline` initializes the required modules.
3.  **Training**: The `Trainer` executes the chosen strategy (e.g., adversarial training).
4.  **Generation**: The trained model generates synthetic samples from the fair latent space.
5.  **Reporting**: The `Evaluator` produces a comprehensive HTML report.
