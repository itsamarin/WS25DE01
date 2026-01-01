#!/bin/bash
# Generate RQ4_Fig6 using SHAP with Python 3.12 virtual environment

echo "======================================================================"
echo "Generating SHAP Feature Importance Visualization for RQ4_Fig6"
echo "Using Python 3.12 virtual environment (src/.venv_py312_shap)"
echo "======================================================================"

# Use Python 3.12 virtual environment directly
src/.venv_py312_shap/bin/python src/generate_shap_fig6.py

echo ""
echo "======================================================================"
echo "SHAP figure generation complete!"
echo "======================================================================"
