
# 🔬 Fair Synthetic Data Generator

**Generate privacy-safe, bias-free synthetic data across tabular, text, and image modalities — with fairness constraints baked into the model, not bolted on afterward.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[**Quick Start**](#-quick-start) · [**Architecture**](#-architecture) · [**Fairness Paradigms**](#-fairness-paradigms) · [**Evaluation**](#-evaluation-metrics) · [**Examples**](#-examples) · [**Contributing**](#-contributing)

---

### The problem

Real-world datasets carry bias baked in at collection time — race correlated with zip code, gender with job title, age with salary bands. Anonymization doesn't remove these correlations. And you often can't share sensitive data across teams, vendors, or borders anyway.

### The solution

A production-ready GAN pipeline that generates high-fidelity synthetic data while enforcing **Group**, **Individual**, and **Counterfactual** fairness simultaneously — across tabular records, text, and images — with cross-modal consistency so generated records are coherent across all three modalities.

</div>

---

## ✨ Key Features

| Feature | Details |
|:---|:---|
| **Multimodal generation** | Tabular, text, and image data with cross-modal consistency |
| **Three fairness paradigms** | Group · Individual · Counterfactual — enforced in the loss function, not post-hoc |
| **Flexible architecture** | Modular VAE, GAN, and Diffusion-based generators |
| **Pretrained models** | Ready-to-use on Adult Income, COMPAS Recidivism, German Credit |
| **Full evaluation suite** | Fidelity · Fairness · Privacy metrics out of the box |
| **Production-ready** | REST API · Docker Compose · Distributed training · Checkpoint management |

---

## 🚀 Quick Start

### 1. Clone and install

```bash
git clone https://github.com/ruchirhuchgol-del/Generative_AI_for_Synthetic_Data_to_Mitigate_Bias.git
cd Generative_AI_for_Synthetic_Data_to_Mitigate_Bias

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -e ".[dev]"
```

### 2. Generate fair synthetic data in 5 lines

```python
from src.models import FairGAN
from src.fairness.constraints import DemographicParity, LipschitzConstraint
from src.evaluation import FairnessEvaluator

# Load a pretrained model — no training required
from src.models import pretrained
df = pretrained.generate_synthetic_data('tabular_vae_adult', n_samples=1000)

# Evaluate fairness
evaluator = FairnessEvaluator(sensitive_attrs=["gender", "race"])
report = evaluator.evaluate(df)
print(report.summary())
```

### 3. Or use the CLI

```bash
make train    CONFIG=configs/experiments/exp_001_baseline.yaml
make generate OUTPUT=data/synthetic/generated.csv
make evaluate OUTPUT=artifacts/reports/fairness_report.html
```

### 4. Or spin up the REST API

```bash
docker compose up
# POST /generate  →  returns synthetic dataset
# POST /evaluate  →  returns fairness + privacy report
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Fair Synthetic Data Generator            │
│                                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │              Cross-Modal Consistency Layer           │  │
│   │   (shared latent vector across all three modalities) │  │
│   └──────────────────────────────────────────────────────┘  │
│                                                             │
│   ┌────────────┐   ┌────────────┐   ┌────────────────┐     │
│   │  Tabular   │   │    Text    │   │     Image      │     │
│   │ Generator  │   │ Generator  │   │   Generator    │     │
│   │ (CTGAN /   │   │ (Transformer│  │ (StyleGAN2 /   │     │
│   │   VAE)     │   │  Decoder)  │   │  Latent Diff.) │     │
│   └────────────┘   └────────────┘   └────────────────┘     │
│                                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │                  Fairness Constraints                │  │
│   │  ┌────────────┐ ┌─────────────┐ ┌─────────────────┐ │  │
│   │  │   Group    │ │ Individual  │ │ Counterfactual  │ │  │
│   │  │  Fairness  │ │  Fairness   │ │    Fairness     │ │  │
│   │  └────────────┘ └─────────────┘ └─────────────────┘ │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

**Why cross-modal consistency matters:** Most synthetic data research generates each modality independently. A synthetic patient's lab results, clinical notes, and scan are zipped together but belong to three different statistical entities. This pipeline conditions all three generators on a shared latent vector so every generated record is coherent across modalities.

---

## ⚖️ Fairness Paradigms

Three paradigms, all enforced inside the training loss — not as post-hoc filters.

| Paradigm | Constraint | Implementation |
|:---|:---|:---|
| **Group Fairness** | `P(Ŷ=1│A=0) = P(Ŷ=1│A=1)` | Fairness critic penalizes distribution divergence across protected groups |
| **Individual Fairness** | Similar individuals → similar outcomes | Lipschitz constraint on generator's latent-to-output mapping |
| **Counterfactual Fairness** | `Y(X) = Y(X_cf)` | Structural causal model ensures outputs are stable under attribute flip |

> **Why not post-hoc?** A generator trained without fairness constraints encodes protected-attribute proxies in the latent space. Post-processing corrects symptoms; embedding constraints in the loss corrects the cause.

---

## 📊 Supported Modalities

| Modality | Encoder | Generator |
|:---|:---|:---|
| **Tabular** | MLP + Embeddings | CTGAN / VAE with mode-specific normalization |
| **Text** | Transformer | GPT-style decoder with fairness-conditioned prefix tokens |
| **Image** | VAE / ViT | StyleGAN2 / Latent Diffusion with attribute disentanglement |

---

## 🧪 Examples

### Healthcare — synthetic patient records

```python
from src.synthesis import GeneratorPipeline

pipeline = GeneratorPipeline.from_config("configs/examples/healthcare.yaml")

synthetic_patients = pipeline.generate(
    n_samples=50000,
    ensure_fairness=["race", "gender", "age"]
)
```

### Finance — debiased credit scoring data

```python
from src.models.architectures import FairGAN

model = FairGAN(
    tabular_features=credit_schema,
    sensitive_attributes=["gender", "ethnicity"]
)

model.fit(biased_data, fairness_weight=0.5, n_epochs=1000)
fair_credit_data = model.generate(100000)
```

### Custom fairness configuration

```yaml
# configs/default/model_config.yaml
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

## 📈 Evaluation Metrics

Synthetic data quality cannot be measured on a single axis. This pipeline evaluates across three:

### Fidelity — how statistically close is synthetic to real?
- Jensen-Shannon divergence and Wasserstein distance
- Train on Synthetic, Test on Real (TSTR) utility score
- Pairwise feature correlation matrix similarity

### Fairness — how much bias was reduced?
- Demographic Parity Difference: `|P(Ŷ=1|A=0) − P(Ŷ=1|A=1)|`
- Equalized Odds Difference (TPR and FPR parity)
- Counterfactual Invariance score

### Privacy — how resistant is it to inference attacks?
- Membership Inference Attack success rate (target: < 50%)
- k-Anonymity equivalence class size
- Differential Privacy ε and δ guarantees

---

## 🗂️ Project Structure

```
fair-synthetic-generator/
├── src/
│   ├── core/          # Base classes and utilities
│   ├── data/          # Data loaders (Adult, COMPAS, Credit)
│   ├── models/        # GAN, VAE, Diffusion architectures
│   ├── fairness/      # Fairness constraints and loss functions
│   ├── training/      # Distributed training infrastructure
│   ├── evaluation/    # Fidelity, fairness, and privacy metrics
│   ├── synthesis/     # Generation pipeline
│   └── api/           # FastAPI REST server
├── tests/             # Unit and integration tests
├── configs/           # Experiment configuration files
├── scripts/           # Training and evaluation scripts
└── notebooks/         # Jupyter notebooks and demos
```

---

## 🛠️ Development

```bash
make test        # Run full test suite
make test-cov    # Run with coverage report
make format      # Auto-format with black + isort
make lint        # Ruff linting
make type-check  # Mypy type checking
```

---

## 🤝 Contributing

Contributions are welcome — especially in these areas:

- New fairness constraints or evaluation metrics
- Domain-specific pretrained models (healthcare, hiring, lending)
- Federated synthetic data generation
- LLM-integrated audit report generation

```bash
git checkout -b feature/your-feature
git commit -m 'Add your feature'
git push origin feature/your-feature
# Then open a Pull Request
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## 📚 Citation

If you use this framework in research, please cite:

```bibtex
@software{fair_synthetic_generator_2024,
  title  = {Fair Synthetic Data Generator: Multimodal Fair Synthetic Data Generation with Generative AI},
  author = {Huchgol, Ruchir},
  year   = {2024},
  url    = {https://github.com/ruchirhuchgol-del/Generative_AI_for_Synthetic_Data_to_Mitigate_Bias}
}
```

---

## 🙏 Acknowledgments

[Fairlearn](https://fairlearn.org/) · [AIF360](https://aif360.res.ibm.com/) · [Diffusers](https://huggingface.co/docs/diffusers/) · [PyTorch](https://pytorch.org/) · [SDV](https://sdv.dev/)

---

**If this project helps your work, please consider giving it a ⭐ — it helps others find it.**

[Report a bug](../../issues) 
[Request a feature](../../issues)
 [Read the Medium deep-dive]([https://medium.com](https://medium.com/@ruchirhuchgol/how-i-built-a-multimodal-gan-pipeline-that-enforces-all-three-fairness-paradigms-d75aaa3a8035))

Made with ❤️ by [Ruchir Huchgol](https://github.com/ruchirhuchgol-del)
insta DM ruchir__27
