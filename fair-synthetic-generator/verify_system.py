import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

def test_imports():
    modules = [
        "src.core",
        "src.data",
        "src.models",
        "src.fairness",
        "src.training",
        "src.evaluation",
        "src.synthesis",
        "src.api",
    ]
    
    success = True
    results = []
    for mod in modules:
        try:
            __import__(mod)
            results.append(f"[SUCCESS] Imported {mod}")
        except Exception as e:
            results.append(f"[FAILURE] Failed to import {mod}: {e}")
            success = False
            
    for res in results:
        print(res)
            
    if success:
        print("\nAll core modules are importable!")
        
        # Test specific critical exports
        try:
            from src.models import FairGAN
            from src.fairness import DemographicParity
            from src.data import TabularPreprocessor
            print("[SUCCESS] Critical exports verified.")
        except ImportError as e:
            print(f"[FAILURE] Critical export verification failed: {e}")
            success = False
            
    return success

if __name__ == "__main__":
    if test_imports():
        sys.exit(0)
    else:
        sys.exit(1)
