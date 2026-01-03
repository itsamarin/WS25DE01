#!/bin/bash
# Setup Python 3.12 virtual environment for SHAP
# This script creates a dedicated Python 3.12 environment to avoid dependency conflicts with SHAP

echo "======================================================================"
echo "Setting up Python 3.12 virtual environment for SHAP"
echo "======================================================================"

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "ERROR: Python 3.12 is not installed or not in PATH"
    echo ""
    echo "Please install Python 3.12 first:"
    echo "  - macOS: brew install python@3.12"
    echo "  - Ubuntu/Debian: sudo apt install python3.12 python3.12-venv"
    echo "  - Or download from: https://www.python.org/downloads/"
    exit 1
fi

echo "Found Python 3.12: $(python3.12 --version)"
echo ""

# Create virtual environment
echo "Creating virtual environment at src/.venv_py312_shap..."
python3.12 -m venv src/.venv_py312_shap

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "✓ Virtual environment created"
echo ""

# Activate and install dependencies
echo "Installing SHAP and dependencies..."
src/.venv_py312_shap/bin/pip install --upgrade pip
src/.venv_py312_shap/bin/pip install shap==0.50.0 pandas scikit-learn matplotlib seaborn joblib numpy

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓ SHAP environment setup complete!"
echo "======================================================================"
echo ""
echo "Virtual environment location: src/.venv_py312_shap/"
echo "Python version: $(src/.venv_py312_shap/bin/python --version)"
echo "SHAP version: $(src/.venv_py312_shap/bin/python -c 'import shap; print(shap.__version__)')"
echo ""
echo "You can now run SHAP visualizations with:"
echo "  ./src/mac_linux_generate_shap.sh"
echo ""
