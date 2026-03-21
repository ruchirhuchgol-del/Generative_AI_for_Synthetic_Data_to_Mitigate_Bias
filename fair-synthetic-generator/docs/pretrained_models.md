# Pretrained Models

The Fair Synthetic Data Generator provides a collection of pretrained models for rapid prototyping and baseline comparison. These models have been trained on standard datasets with fairness constraints.

## 📁 Overview

Pretrained models are available for three major architectures:

- **VAE**: $\beta$-VAE with fairness-aware representations.
- **GAN**: WGAN-GP with adversarial debiasing.
- **Diffusion**: DDPM/DDIM for high-fidelity synthesis.

## 🚀 Quick Usage

You can use the `src.models.pretrained` package to easily load and generate data.

```python
from src.models import pretrained

# List available models
models = pretrained.list_available_models()
print(f"Available: {models}")

# Load a specific model (e.g., VAE on Adult dataset)
model_id = 'tabular_vae_adult'
model = pretrained.load_model(model_id)

# Generate synthetic data
df = model.generate(n_samples=1000, seed=42)

# Check model metrics
metrics = model.get_metrics()
print(f"Fidelity: {metrics['fidelity']:.2f}, Fairness: {metrics['fairness']:.2f}")
```

## 📊 Available Models

| Model ID | Architecture | Dataset | Best For |
| :--- | :--- | :--- | :--- |
| `tabular_vae_adult` | VAE | Adult Income | General purpose, balanced fairness |
| `tabular_vae_credit` | VAE | Credit Card | Financial applications |
| `tabular_gan_adult` | GAN | Adult Income | High-fidelity synthesis |
| `tabular_gan_compas` | GAN | COMPAS | Fairness-critical applications |
| `tabular_diffusion_adult` | Diffusion | Adult Income | State-of-the-art quality |
| `tabular_diffusion_credit` | Diffusion | Credit Card | Enterprise synthesis |

## 🏋️ Training & Weights


Pretrained weights are stored in `models/pretrained/` as `.npz` and `.json` files. 

If you wish to re-generate weights or train models on your own datasets using these architectures, you can use the scripts in `models/training/`:

```bash
# Generate weights with NumPy (random initialization for testing)
python models/training/generate_weights.py --all

# Train actual weights with PyTorch (GPU recommended)
python models/training/train_pretrained.py --model vae --dataset adult --epochs 100
```

## 🧪 Evaluation Metrics

All pretrained models are evaluated on three axes:
- **Fidelity**: Statistical similarity to the original training data.
- **Fairness**: Mitigation of bias (e.g., Disparate Impact Ratio).
- **Privacy**: Resistance to membership inference attacks.

Metrics for each model can be retrieved using `model.get_metrics()`.

