How to Use the Fair Synthetic Data Generation System
📦 Installation & Setup

1. Install Dependencies

# Clone the repository

git clone <https://github.com/your-org/Generative_AI_for_Synthetic_Data_to_Mitigate_Bias.git>
cd Generative_AI_for_Synthetic_Data_to_Mitigate_Bias

# Install dependencies (CPU version)

bash scripts/setup/install_dependencies.sh --device cpu

# Or with CUDA support

bash scripts/setup/install_dependencies.sh --device cuda

# Or with Apple Silicon MPS

bash scripts/setup/install_dependencies.sh --device mps

1. Download Pretrained Models (Optional)
bash
scripts/setup/download_pretrained.sh

🚀 Quick Start: Three Ways to Use
Method 1: Python SDK (Recommended for Developers)
from src.models.architectures import FairMultimodalGenerator
from src.data.preprocessing import DataProcessor
from src.evaluation import FidelityEvaluator, FairnessEvaluator

# 1. Load your data

processor = DataProcessor(config_path="configs/data/adult.yaml")
data = processor.load("data/raw/adult.csv")
processed_data = processor.preprocess(data)

# 2. Initialize generator with fairness constraints

generator = FairMultimodalGenerator(
    model_type="vae",  # Options: "vae", "gan", "diffusion"
    modality="tabular",
    fairness_config={
        "protected_attributes": ["sex", "race"],
        "fairness_type": "demographic_parity",
        "lambda_fairness": 0.5
    }
)

# 3. Train the model

generator.fit(
    data=processed_data,
    epochs=100,
    batch_size=256,
    checkpoint_dir="checkpoints/"
)

# 4. Generate synthetic data

synthetic_data = generator.generate(
    num_samples=10000,
    conditions={"race": "Black", "sex": "Female"}  # Optional conditional generation
)

# 5. Evaluate quality

fidelity = FidelityEvaluator().evaluate(processed_data, synthetic_data)
fairness = FairnessEvaluator().evaluate(processed_data, synthetic_data, protected_attrs=["sex", "race"])
print(f"Fidelity Score: {fidelity.overall_score:.3f}")
print(f"Fairness Score: {fairness.disparate_impact_ratio:.3f}")

# 6. Save results

synthetic_data.to_csv("output/synthetic_adult.csv")

Method 2: REST API (Recommended for Production)
Start the API Server

# Start API server

uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or with production settings

uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
API Endpoints Usage

# 1. Health check

curl <http://localhost:8000/health>

# 2. Generate synthetic data

curl -X POST <http://localhost:8000/api/v1/generate> \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "adult_vae_v1",
    "num_samples": 5000,
    "conditions": {"sex": "Female"},
    "fairness_constraints": {
      "protected_attributes": ["sex", "race"],
      "fairness_type": "demographic_parity"
    }
  }'

# 3. Train a new model

curl -X POST <http://localhost:8000/api/v1/train> \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/raw/custom_data.csv",
    "model_type": "diffusion",
    "modality": "tabular",
    "epochs": 200,
    "fairness_config": {
      "protected_attributes": ["gender", "age_group"],
      "lambda_fairness": 0.3
    }
  }'

# 4. Evaluate synthetic data

curl -X POST <http://localhost:8000/api/v1/evaluate> \
  -H "Content-Type: application/json" \
  -d '{
    "original_data_path": "data/raw/adult.csv",
    "synthetic_data_path": "output/synthetic_adult.csv",
    "evaluation_types": ["fidelity", "fairness", "privacy"]
  }'

# 5. Batch generation (async)

curl -X POST <http://localhost:8000/api/v1/batch/generate> \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "credit_gan_v1",
    "batch_configs": [
      {"num_samples": 10000, "conditions": {"income": "low"}},
      {"num_samples": 10000, "conditions": {"income": "high"}}
    ],
    "output_dir": "output/batch/"
  }'

Python API Client
import requests
BASE_URL = "<http://localhost:8000/api/v1>"

# Generate synthetic data

response = requests.post(f"{BASE_URL}/generate", json={
    "model_id": "adult_vae_v1",
    "num_samples": 10000,
    "fairness_constraints": {
        "protected_attributes": ["sex", "race"]
    }
})
synthetic_data_id = response.json()["generation_id"]
print(f"Generation ID: {synthetic_data_id}")

# Download results

download_url = response.json()["download_url"]
synthetic_df = pd.read_csv(download_url)

Method 3: Command-Line Scripts (Recommended for Experiments)
Training

# Basic training

python scripts/training/train.py \
  --config configs/training/adult_vae.yaml \
  --data data/raw/adult.csv \
  --output checkpoints/adult_vae/

# Training with fairness constraints

python scripts/training/train.py \
  --config configs/training/adult_diffusion.yaml \
  --data data/raw/adult.csv \
  --protected-attrs sex race \
  --fairness-type demographic_parity \
  --lambda-fairness 0.5 \
  --output checkpoints/adult_fair/

# Distributed training (multi-GPU)

torchrun --nproc_per_node=4 scripts/training/train.py \
  --config configs/training/credit_gan.yaml \
  --distributed \
  --output checkpoints/credit_gan/

# Resume from checkpoint

python scripts/training/resume_training.py \
  --checkpoint checkpoints/adult_vae/checkpoint_epoch_50.pt \
  --epochs 50
Hyperparameter Search
python scripts/training/hyperparameter_search.py \
  --config configs/hpo/adult_vae_hpo.yaml \
  --n-trials 50 \
  --output results/hpo/

Generation

# Generate synthetic data

python scripts/synthesis/generate_synthetic_data.py \
  --checkpoint checkpoints/adult_vae/best_model.pt \
  --num-samples 50000 \
  --output output/synthetic_adult.csv

# Conditional generation

python scripts/synthesis/generate_synthetic_data.py \
  --checkpoint checkpoints/adult_vae/best_model.pt \
  --num-samples 10000 \
  --conditions '{"sex": "Female", "race": "Black"}' \
  --output output/synthetic_female_black.csv

# Batch generation with parallel processing

python scripts/synthesis/batch_generation.py \
  --config configs/synthesis/batch_config.yaml \
  --output-dir output/batch/ \
  --num-workers 4

Evaluation

# Full evaluation

python scripts/evaluation/evaluate_fidelity.py \
  --original data/raw/adult.csv \
  --synthetic output/synthetic_adult.csv \
  --output reports/fidelity_report.json
python scripts/evaluation/evaluate_fairness.py \
  --original data/raw/adult.csv \
  --synthetic output/synthetic_adult.csv \
  --protected-attrs sex race \
  --output reports/fairness_report.json

# Generate comprehensive report

python scripts/evaluation/generate_report.py \
  --fidelity-report reports/fidelity_report.json \
  --fairness-report reports/fairness_report.json \
  --output reports/complete_report.html \
  --format html

⚙️ Configuration Files
Training Configuration (configs/training/adult_vae.yaml)
Data Schema Configuration (configs/data/adult.yaml)
dataset:
  name: "adult"
  target_column: "income"  
columns:
  numerical:
    - name: "age"
      min: 17
      max: 90
    - name: "education-num"
      min: 1
      max: 16
    - name: "capital-gain"
      min: 0
      max: 99999
    - name: "capital-loss"
      min: 0
      max: 4356
    - name: "hours-per-week"
      min: 1
      max: 99
  categorical:
    - name: "workclass"
      values: ["Private", "Self-emp-not-inc", "Local-gov", ...]
    - name: "education"
      values: ["Bachelors", "Some-college", "HS-grad", ...]
    - name: "sex"
      values: ["Male", "Female"]
    - name: "race"
      values: ["White", "Black", "Asian-Pac-Islander", ...]
  protected_attributes:
    - "sex"
    - "race"

📊 Example Use Cases
Use Case 1: Fair Credit Scoring Model
from src.models import FairMultimodalGenerator
from src.evaluation import ComprehensiveEvaluator

# Train fair synthetic data generator for credit data

generator = FairMultimodalGenerator(
    model_type="diffusion",
    modality="tabular",
    fairness_config={
        "protected_attributes": ["gender", "age_group", "race"],
        "fairness_type": "equalized_odds",
        "lambda_fairness": 0.7
    }
)

# Load and process credit data

generator.fit("data/raw/credit_applications.csv", epochs=300)

# Generate balanced synthetic dataset

synthetic = generator.generate(
    num_samples=100000,
    balance_strategy="demographic_balanced"
)

# Evaluate for regulatory compliance

evaluator = ComprehensiveEvaluator()
report = evaluator.evaluate(
    original="data/raw/credit_applications.csv",
    synthetic=synthetic,
    protected_attrs=["gender", "age_group", "race"]
)

# Generate compliance report for regulators

report.save_html("compliance/fair_lending_audit.html")

Use Case 2: Healthcare Data Sharing

# Generate HIPAA-compliant synthetic patient data

from src.data.preprocessing import HealthcareDataProcessor
from src.privacy import PrivacyAuditor
processor = HealthcareDataProcessor()
patient_data = processor.load("data/raw/patient_records.parquet")

# Configure for maximum privacy

generator = FairMultimodalGenerator(
    model_type="vae",
    privacy_config={
        "differential_privacy": True,
        "epsilon": 0.5,  # Strong privacy guarantee
        "delta": 1e-6
    },
    fairness_config={
        "protected_attributes": ["ethnicity", "gender", "age_decade"],
        "fairness_type": "counterfactual"
    }
)
generator.fit(patient_data, epochs=500)

# Generate synthetic data for research collaboration

synthetic_patients = generator.generate(num_samples=50000)

# Verify privacy guarantees

auditor = PrivacyAuditor()
audit_result = auditor.audit(
    original=patient_data,
    synthetic=synthetic_patients,
    attack_types=["membership_inference", "attribute_inference"]
)
print(f"Privacy Score: {audit_result.privacy_score:.3f}")
print(f"Re-identification Risk: {audit_result.reidentification_risk:.3%}")

Use Case 3: Multi-Modal AI Training

# Generate synthetic multimodal dataset

from src.models import MultimodalFairGenerator
generator = MultimodalFairGenerator(
    modalities=["tabular", "text", "image"],
    fairness_config={
        "protected_attributes": ["demographics"],
        "cross_modal_consistency": True
    }
)

# Train on mixed data

generator.fit(
    tabular_data="data/users.csv",
    text_data="data/user_descriptions/",
    image_data="data/user_images/",
    epochs=200
)

# Generate multimodal synthetic data

synthetic = generator.generate(
    num_samples=10000,
    output_dir="output/multimodal/"
)

# Use for foundation model training

# synthetic.tabular  -> structured features

# synthetic.text     -> synthetic descriptions

# synthetic.images   -> synthetic profile images

## 🔧 Advanced Features

### Distributed Training

from src.training import DistributedTrainer
trainer = DistributedTrainer(
    model_config=config,
    num_gpus=4,
    backend="nccl"
)
trainer.train(data_path="data/large_dataset.parquet")

### Experiment Tracking

from src.utils import ExperimentTracker
tracker = ExperimentTracker(
    backend="wandb",  # or "mlflow"
    project="fair-synthetic-data",
    tags=["production", "credit-scoring"]
)
with tracker.run("experiment_001"):
    tracker.log_params(config)
    for epoch in range(epochs):
        # training loop
        tracker.log_metrics({"loss": loss, "fairness": fairness})

### Custom Fairness Constraints

# Define custom fairness metric

from src.fairness import CustomFairnessConstraint
custom_constraint = CustomFairnessConstraint(
    metric_fn=lambda y_pred, y_true, groups: custom_metric(...),
    threshold=0.1
)
generator = FairMultimodalGenerator(
    fairness_config={"custom_constraint": custom_constraint}
)

| Task | Command/Code |
|------|-------------|
| **Install** | `bash scripts/setup/install_dependencies.sh` |
| **Start API** | `uvicorn api.main:app --port 8000` |
| **Train Model** | `python scripts/training/train.py --config ...` |
| **Generate Data** | `python scripts/synthesis/generate_synthetic_data.py ...` |
| **Evaluate** | `python scripts/evaluation/evaluate_fidelity.py ...` |
| **Python SDK** | `from src.models import FairMultimodalGenerator` |
| **API Docs** | `http://localhost:8000/docs` (Swagger UI) |

This system is designed to be flexible - use the Python SDK for research and development, the REST API for production integration, or CLI scripts for batch processing and experiments!
