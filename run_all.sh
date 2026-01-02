#!/bin/bash
# Complete End-to-End Workflow Script
# Runs the entire project pipeline and generates all figures and tables
#
# Usage: ./run_all.sh
#
# Expected outputs:
#   - 19 PDF figures in figures/
#   - 2 XLSX tables in tables/
#   - 2 trained models in src/modeling/models/
#   - Processed datasets in data/

echo "================================================================================"
echo "ACADEMIC PERFORMANCE PREDICTION - COMPLETE WORKFLOW"
echo "================================================================================"
echo ""
echo "This script will:"
echo "  1. Download and process student performance data"
echo "  2. Train machine learning models"
echo "  3. Generate all 19 figures and 2 tables"
echo "  4. Generate SHAP visualization for RQ4_Fig6"
echo ""
echo "Estimated runtime: 2-3 minutes"
echo ""
echo "================================================================================"
echo ""

# Ensure output directories exist
mkdir -p figures tables

# Step 1: Run the main pipeline
echo "[STEP 1/3] Running main pipeline (data ingestion, cleaning, training)..."
echo "--------------------------------------------------------------------------------"
python3 main.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Main pipeline failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "================================================================================"
echo ""

# Step 2: Generate all figures and tables
echo "[STEP 2/3] Generating all figures and tables..."
echo "--------------------------------------------------------------------------------"
python3 src/run_simple_analysis.py

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Figure generation failed. Please check the error messages above."
    exit 1
fi

echo ""
echo "================================================================================"
echo ""

# Step 3: Generate SHAP visualization for RQ4_Fig6
echo "[STEP 3/3] Generating SHAP visualization for RQ4_Fig6..."
echo "--------------------------------------------------------------------------------"
./src/generate_shap_with_py312.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "WARNING: SHAP generation failed. RQ4_Fig6 will use permutation importance fallback."
    echo "This is not critical - all other figures were generated successfully."
fi

echo ""
echo "================================================================================"
echo "SUCCESS! All outputs generated successfully!"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  - $(ls figures/*.pdf 2>/dev/null | wc -l | tr -d ' ') PDF figures in figures/"
echo "  - $(ls tables/*.xlsx 2>/dev/null | wc -l | tr -d ' ') XLSX tables in tables/"
echo "  - $(ls src/modeling/models/*.pkl 2>/dev/null | wc -l | tr -d ' ') trained models in src/modeling/models/"
echo ""
echo "================================================================================"
