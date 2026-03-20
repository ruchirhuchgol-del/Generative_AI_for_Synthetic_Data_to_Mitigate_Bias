#!/bin/bash
#######################################################################
# Install Dependencies Script
# ===========================
# Installs all required dependencies for the Fair Synthetic Data Generator.
# Supports: CPU, CUDA (GPU), and MPS (Apple Silicon) backends.
#
# Usage:
#   ./install_dependencies.sh [OPTIONS]
#
# Options:
#   --cuda          Install with CUDA support (NVIDIA GPUs)
#   --mps           Install with MPS support (Apple Silicon)
#   --cpu           Install CPU-only version
#   --dev           Install development dependencies
#   --full          Install all optional dependencies
#   -h, --help      Show this help message
#
# Examples:
#   ./install_dependencies.sh --cuda --dev    # Full GPU dev setup
#   ./install_dependencies.sh --cpu           # Minimal CPU setup
#######################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default settings
BACKEND="auto"
DEV_MODE=false
FULL_INSTALL=false
PYTHON_CMD="python3"
PIP_CMD="pip3"

# Print colored message
print_msg() {
    local color=$1
    local msg=$2
    echo -e "${color}${msg}${NC}"
}

# Show help
show_help() {
    head -30 "$0" | tail -25 | sed 's/^#//' | sed 's/^ //'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            BACKEND="cuda"
            shift
            ;;
        --mps)
            BACKEND="mps"
            shift
            ;;
        --cpu)
            BACKEND="cpu"
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        --full)
            FULL_INSTALL=true
            DEV_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            print_msg $RED "Unknown option: $1"
            show_help
            ;;
    esac
done

# Print banner
print_msg $BLUE "╔══════════════════════════════════════════════════════════════╗"
print_msg $BLUE "║     Fair Synthetic Data Generator - Dependency Installer     ║"
print_msg $BLUE "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Detect Python
print_msg $YELLOW ">>> Detecting Python installation..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_msg $GREEN "Found Python $PYTHON_VERSION"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    PIP_CMD="pip"
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    print_msg $GREEN "Found Python $PYTHON_VERSION"
else
    print_msg $RED "ERROR: Python not found. Please install Python 3.9+ first."
    exit 1
fi

# Check Python version
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 9 ]]; then
    print_msg $RED "ERROR: Python 3.9+ required. Found $PYTHON_VERSION"
    exit 1
fi

# Detect backend if auto
if [[ $BACKEND == "auto" ]]; then
    print_msg $YELLOW ">>> Auto-detecting backend..."
    if command -v nvidia-smi &> /dev/null; then
        BACKEND="cuda"
        print_msg $GREEN "NVIDIA GPU detected. Using CUDA backend."
    elif [[ "$(uname)" == "Darwin" ]] && [[ $(sysctl -n machdep.cpu.brand_string) == *"Apple"* ]]; then
        BACKEND="mps"
        print_msg $GREEN "Apple Silicon detected. Using MPS backend."
    else
        BACKEND="cpu"
        print_msg $YELLOW "No GPU detected. Using CPU backend."
    fi
fi

# Create virtual environment
print_msg $YELLOW ">>> Setting up virtual environment..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
VENV_DIR="$PROJECT_ROOT/.venv"

if [[ ! -d "$VENV_DIR" ]]; then
    print_msg $BLUE "Creating virtual environment at $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    print_msg $GREEN "Virtual environment created."
else
    print_msg $GREEN "Virtual environment already exists."
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
print_msg $YELLOW ">>> Upgrading pip..."
$PIP_CMD install --upgrade pip setuptools wheel

# Install PyTorch based on backend
print_msg $YELLOW ">>> Installing PyTorch ($BACKEND backend)..."
case $BACKEND in
    cuda)
        # Detect CUDA version
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        print_msg $BLUE "CUDA Version: $CUDA_VERSION"
        
        # Install PyTorch with CUDA
        $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ;;
    mps)
        # MPS is supported in standard PyTorch for macOS
        $PIP_CMD install torch torchvision torchaudio
        ;;
    cpu)
        $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
esac
print_msg $GREEN "PyTorch installed successfully."

# Install TensorFlow
print_msg $YELLOW ">>> Installing TensorFlow..."
if [[ $BACKEND == "cuda" ]]; then
    $PIP_CMD install tensorflow[and-cuda]
else
    $PIP_CMD install tensorflow
fi
print_msg $GREEN "TensorFlow installed successfully."

# Install core requirements
print_msg $YELLOW ">>> Installing core dependencies..."
if [[ -f "$PROJECT_ROOT/requirements.txt" ]]; then
    $PIP_CMD install -r "$PROJECT_ROOT/requirements.txt"
    print_msg $GREEN "Core dependencies installed."
else
    print_msg $RED "WARNING: requirements.txt not found. Installing minimal dependencies..."
    $PIP_CMD install numpy pandas scipy scikit-learn matplotlib seaborn
    $PIP_CMD install pyyaml tqdm omegaconf hydra-core
    $PIP_CMD install fastapi uvicorn pydantic
    $PIP_CMD install fairlearn aif360
    $PIP_CMD install transformers datasets diffusers accelerate
    $PIP_CMD install tensorboard wandb
fi

# Install development dependencies
if [[ $DEV_MODE == true ]]; then
    print_msg $YELLOW ">>> Installing development dependencies..."
    if [[ -f "$PROJECT_ROOT/requirements-dev.txt" ]]; then
        $PIP_CMD install -r "$PROJECT_ROOT/requirements-dev.txt"
    else
        $PIP_CMD install pytest pytest-cov pytest-xdist pytest-timeout
        $PIP_CMD install black isort flake8 mypy pylint
        $PIP_CMD install pre-commit
        $PIP_CMD install jupyter jupyterlab ipykernel
        $PIP_CMD install sphinx sphinx-rtd-theme
    fi
    print_msg $GREEN "Development dependencies installed."
fi

# Install full optional dependencies
if [[ $FULL_INSTALL == true ]]; then
    print_msg $YELLOW ">>> Installing optional dependencies..."
    $PIP_CMD install optuna ray[tune]  # Hyperparameter optimization
    $PIP_CMD install onnx onnxruntime  # Model export
    $PIP_CMD install mlflow  # Experiment tracking alternative
    $PIP_CMD install streamlit  # Alternative dashboard
    $PIP_CMD install 'great_expectations'  # Data validation
    $PIP_CMD install sdv  # Synthetic data baseline
    print_msg $GREEN "Optional dependencies installed."
fi

# Verify installation
print_msg $YELLOW ">>> Verifying installation..."
VERIFY_SCRIPT=$(cat << 'EOF'
import sys
print(f"Python: {sys.version}")

# Check PyTorch
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("MPS available: Apple Silicon GPU")
    else:
        print("Running on CPU")
except ImportError:
    print("PyTorch: NOT INSTALLED")

# Check TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow: {tf.__version__}")
except ImportError:
    print("TensorFlow: NOT INSTALLED")

# Check other key packages
packages = ['numpy', 'pandas', 'sklearn', 'fairlearn', 'transformers', 'fastapi']
for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f"{pkg}: {version}")
    except ImportError:
        print(f"{pkg}: NOT INSTALLED")
EOF
)

$PYTHON_CMD -c "$VERIFY_SCRIPT"

# Install pre-commit hooks if in dev mode
if [[ $DEV_MODE == true ]] && [[ -f "$PROJECT_ROOT/.pre-commit-config.yaml" ]]; then
    print_msg $YELLOW ">>> Installing pre-commit hooks..."
    cd "$PROJECT_ROOT"
    pre-commit install
    print_msg $GREEN "Pre-commit hooks installed."
fi

# Summary
echo ""
print_msg $GREEN "╔══════════════════════════════════════════════════════════════╗"
print_msg $GREEN "║              Installation Complete!                          ║"
print_msg $GREEN "╠══════════════════════════════════════════════════════════════╣"
print_msg $GREEN "║  Backend: $BACKEND"
print_msg $GREEN "║  Virtual Environment: $VENV_DIR"
print_msg $GREEN "║  Dev Mode: $DEV_MODE"
print_msg $GREEN "╠══════════════════════════════════════════════════════════════╣"
print_msg $GREEN "║  To activate: source $VENV_DIR/bin/activate"
print_msg $GREEN "╚══════════════════════════════════════════════════════════════╝"

# Deactivate
deactivate 2>/dev/null || true

exit 0
