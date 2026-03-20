# Fair Synthetic Data Generator

A state-of-the-art framework for generating fair, high-fidelity synthetic data across multiple modalities (tabular, text, image).

## Key Features

- **Multi-Modal Generation**: Support for tabular, text, and image data.
- **Fairness-by-Design**: Integrated group, individual, and counterfactual fairness constraints.
- **SOTA Architectures**: Implementation of FairGAN, FairDiffusion, and DebiasedVAE.
- **Privacy Preservation**: Differential Privacy (DP) integration.
- **Comprehensive Evaluation**: Metrics for fidelity, fairness, and privacy.

## Project Structure

- `src/`: Core implementation (models, fairness, evaluation, etc.)
- `api/`: FastAPI-based REST API
- `notebooks/`: Tutorials and experiments
- `configs/`: Configuration management
- `tests/`: Comprehensive test suite
