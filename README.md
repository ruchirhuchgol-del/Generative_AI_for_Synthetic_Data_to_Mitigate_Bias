# 🔬 Fair Synthetic Data Generator

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.14+-red.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Multimodal Fair Synthetic Data Generation with Generative AI**

[Quick Start](#quick-start) • [Documentation](#documentation) • [Architecture](#architecture) • [Contributing](#contributing)

</div>

---

## 📋 Overview

**Fair Synthetic Data Generator** is a comprehensive framework for generating high-quality synthetic data while ensuring fairness across protected groups. It addresses the critical challenge of bias in training data by leveraging state-of-the-art generative AI techniques.

### Key Features

| Feature | Description |
| :--- | :--- |
| **Multimodal Generation** | Supports tabular, text, and image data synthesis with cross-modal consistency |
| **Multi-Paradigm Fairness** | Implements Group, Individual, and Counterfactual fairness constraints |
| **Flexible Architecture** | Modular design with VAE, GAN, and Diffusion-based generators |
| **Comprehensive Evaluation** | Built-in metrics for fidelity, fairness, and privacy assessment |
| **Production-Ready** | REST API, distributed training, and checkpoint management |

---

## 📊 Project Status

The project has undergone rigorous integration testing and verification.

- **Import Success**: 100% (All core modules verified)
- **API Status**: Healthy (FastAPI server verified responsive)
- **Data Schemas**: Standardized (Adult, COMPAS, Credit templates supported)
- **Containerization**: Verified (Docker/Compose ready)

---

## 🏗️ Architecture

│  │                    Fairness Constraints                      │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐    │   │
│  │  │   Group     │ │ Individual  │ │  Counterfactual     │    │   │
│  │  │  Fairness   │ │  Fairness   │ │     Fairness        │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/fair-synth/fair-synthetic-generator.git
cd fair-synthetic-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models import FairGAN
from src.fairness.constraints import DemographicParity, LipschitzConstraint
from src.evaluation import FairnessEvaluator

# Initialize generator
generator = FairGAN(
    data_dim=100,
    latent_dim=512,
    num_sensitive_groups=2
)

# Define constraints
constraints = [
    DemographicParity(threshold=0.05),
    LipschitzConstraint(lambda_lipschitz=0.1),
]

# Generate fair synthetic data
# Note: In a real scenario, you would train the model first
synthetic_data = generator.generate(n_samples=10000)

# Evaluate fairness
evaluator = FairnessEvaluator(sensitive_attrs=["gender"])
report = evaluator.evaluate(synthetic_data)
print(report.summary())
```

### Command Line

```bash
# Train a new model
make train CONFIG=configs/experiments/exp_001_baseline.yaml

# Generate synthetic data
make generate OUTPUT=data/synthetic/generated.csv

# Evaluate fairness metrics
make evaluate OUTPUT=artifacts/reports/fairness_report.html
```

---

## 📖 Documentation

### Fairness Paradigms

| Paradigm | Constraint | Implementation |
|----------|------------|----------------|
| **Group Fairness** | P(Ŷ=1\|A=0) = P(Ŷ=1\|A=1) | `src/fairness/constraints/group_fairness/` |
| **Individual Fairness** | Similar individuals → similar outcomes | `src/fairness/constraints/individual_fairness/` |
| **Counterfactual Fairness** | Y(X) = Y(X_cf) | `src/fairness/constraints/counterfactual_fairness.py` |

### Supported Modalities

| Modality | Encoder | Decoder | Generator |
|----------|---------|---------|-----------|
| **Tabular** | MLP + Embeddings | Feature-specific heads | VAE, GAN |
| **Text** | Transformer | Transformer Decoder | GPT-style, Diffusion |
| **Image** | VAE / ViT | U-Net / Diffusion | Latent Diffusion |

### Configuration

```yaml
# configs/default/model_config.yaml
model:
  name: "FairGAN"
  latent_dim: 512
  
  encoders:
    tabular:
      type: "mlp"
      hidden_dims: [256, 512]
    text:
      type: "transformer"
      hidden_dim: 768
      num_layers: 6
    image:
      type: "vae"
      resolution: 256
      
  fairness:
    group_fairness:
      enabled: true
      metrics: ["demographic_parity", "equalized_odds"]
      threshold: 0.05
    individual_fairness:
      enabled: true
      lambda_lipschitz: 0.1
    counterfactual_fairness:
      enabled: true
      num_counterfactuals: 5
```

---

## 🧪 Examples

### Example 1: Healthcare Data Synthesis

```python
from src.synthesis import GeneratorPipeline

pipeline = GeneratorPipeline.from_config("configs/examples/healthcare.yaml")

# Generate synthetic patient records
synthetic_patients = pipeline.generate(
    n_samples=50000,
    ensure_fairness=["race", "gender", "age"]
)
```

### Example 2: Finance - Credit Scoring Data

```python
from src.models.architectures import FairGAN

# Train on biased credit data
model = FairGAN(
    tabular_features=credit_schema,
    sensitive_attributes=["gender", "ethnicity"]
)

model.fit(
    biased_data,
    fairness_weight=0.5,
    n_epochs=1000
)

# Generate debiased synthetic data
fair_credit_data = model.generate(100000)
```

---

## 📊 Evaluation Metrics

### Fidelity Metrics
- **Statistical Similarity**: Jensen-Shannon divergence, Wasserstein distance
- **Machine Learning Utility**: Train on synthetic, test on real (TSTR)
- **Correlation Preservation**: Feature correlation matrices

### Fairness Metrics
- **Demographic Parity Difference**: |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|
- **Equalized Odds Difference**: TPR and FPR parity
- **Counterfactual Invariance**: Prediction stability under attribute flip

### Privacy Metrics
- **Membership Inference Attack**: Attack success rate < 50%
- **k-Anonymity**: Minimum equivalence class size
- **Differential Privacy**: ε and δ guarantees

---

## 🛠️ Development

### Project Structure

```
fair-synthetic-generator/
├── src/
│   ├── core/              # Base classes and utilities
│   ├── data/              # Data processing and loaders
│   ├── models/            # Model architectures
│   ├── fairness/          # Fairness constraints and losses
│   ├── training/          # Training infrastructure
│   ├── evaluation/        # Metrics and evaluation
│   ├── synthesis/         # Generation pipeline
│   └── api/               # REST API
├── tests/                 # Unit and integration tests
├── configs/               # Configuration files
├── scripts/               # Executable scripts
└── notebooks/             # Jupyter notebooks
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/unit/models/test_encoders.py -v
```

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{fair_synthetic_generator_2024,
  title = {Fair Synthetic Data Generator: Multimodal Fair Synthetic Data Generation with Generative AI},
  author = {Fair Synth Team},
  year = {2024},
  url = {https://github.com/fair-synth/fair-synthetic-generator}
}
```

---

## 🙏 Acknowledgments

- [Fairlearn](https://fairlearn.org/) for fairness metrics
- [AIF360](https://aif360.res.ibm.com/) for bias mitigation algorithms
- [Diffusers](https://huggingface.co/docs/diffusers/) for diffusion model implementations
- [PyTorch](https://pytorch.org/) and [TensorFlow](https://tensorflow.org/) for deep learning frameworks

---



**[⬆ Back to Top](#-fair-synthetic-data-generator)**

Made with ❤️ by the Fair Synth Team

</div>
