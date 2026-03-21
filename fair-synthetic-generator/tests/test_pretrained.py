import os
import sys
from pathlib import Path

# Add fair-synthetic-generator to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import pretrained

def test_pretrained_integration():
    print("Testing Pretrained Models Integration...")
    print("-" * 50)
    
    # 1. List available models
    models = pretrained.list_available_models()
    print(f"Available models: {models}")
    
    if not models:
        print("Error: No models found!")
        return
    
    # 2. Get info for a model
    model_id = 'tabular_vae_adult'
    if model_id in models:
        info = pretrained.get_model_info(model_id)
        print(f"Model Info for {model_id}:")
        print(f"  Architecture: {info.architecture}")
        print(f"  Dataset: {info.dataset}")
        print(f"  Metrics: {info.metrics}")
    
    # 3. Load model and generate data
    print(f"\nLoading {model_id} and generating data...")
    try:
        df = pretrained.generate_synthetic_data(model_id, n_samples=100)
        print(f"Generated DataFrame shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nIntegration test PASSED!")
    except Exception as e:
        print(f"\nIntegration test FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pretrained_integration()
