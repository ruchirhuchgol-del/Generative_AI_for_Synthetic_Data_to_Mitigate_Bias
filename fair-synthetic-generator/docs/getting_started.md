# Getting Started with Fair Synthetic Data Generator

This guide will help you get up and running with the Fair Synthetic Data Generator.

## 1. Installation

### Requirements
- Python 3.10 or higher
- PyTorch 2.1.0+ or TensorFlow 2.15.0+
- (Optional) CUDA-compatible GPU for accelerated training

### Standard Setup
```bash
# Clone the repository
git clone https://github.com/fair-synth/fair-synthetic-generator.git
cd fair-synthetic-generator

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install package in editable mode
pip install -e .
```

### Developer Setup
```bash
# Install with dev dependencies (linting, testing, docs)
pip install -e ".[dev]"

# Install pre-commit hooks
make pre-commit-install
```

## 2. Basic Synthesis Workflow

### Step 1: Define a Data Schema
You can use one of our templates or define your own JSON schema.
```bash
python scripts/data/generate_synthetic_schema.py --template adult --output data/schemas/adult_schema.json
```

### Step 2: Preprocess Your Data
```python
from src.data import TabularPreprocessor
import pandas as pd

# Load your biased dataset
df = pd.read_csv("data/raw/adult.csv")

# Initialize and run preprocessor
preprocessor = TabularPreprocessor(schema_path="data/schemas/adult_schema.json")
processed_data = preprocessor.fit_transform(df)
```

### Step 3: Train Fair GAN
```python
from src.models import FairGAN
from src.training import Trainer

# Initialize model
model = FairGAN(
    data_dim=processed_data.shape[1],
    latent_dim=512,
    num_sensitive_groups=2
)

# Train with fairness constraints
trainer = Trainer(model=model, strategy="adversarial")
trainer.train(processed_data, epochs=100)
```

### Step 4: Generate Synthetic Samples
```python
# Generate 10,000 fair synthetic samples
synthetic_samples = model.generate(n_samples=10000)

# Save to CSV
synthetic_df = preprocessor.inverse_transform(synthetic_samples)
synthetic_df.to_csv("data/synthetic/fair_adult_data.csv", index=False)
```

## 3. Using the REST API

The framework includes a high-performance FastAPI server.

### Start the Server
```bash
make run-api
```

### Check Health
```bash
curl http://localhost:8000/api/v1/health
```

### API Documentation
Once the server is running, visit:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 4. Evaluation and Reporting

Assess the quality and fairness of your synthetic data.

```bash
# Run comprehensive evaluation pipeline
make evaluate
```

This will generate an HTML report in `artifacts/reports/` covering fidelity, group fairness, and privacy risk assessment.
