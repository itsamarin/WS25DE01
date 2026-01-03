@echo off
REM ================================================================================
REM ACADEMIC PERFORMANCE PREDICTION - COMPLETE WORKFLOW (Windows)
REM ================================================================================

echo ================================================================================
echo ACADEMIC PERFORMANCE PREDICTION - COMPLETE WORKFLOW
echo ================================================================================
echo.
echo This script will:
echo   1. Download and process student performance data
echo   2. Train machine learning models
echo   3. Generate all 19 figures and 2 tables
echo   4. Generate SHAP visualization for RQ4_Fig6
echo.
echo Estimated runtime: 2-3 minutes
echo.
echo ================================================================================
echo.

REM Step 1: Run main pipeline
echo [STEP 1/3] Running main pipeline (data ingestion, cleaning, training)...
echo --------------------------------------------------------------------------------
python main.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Main pipeline failed!
    exit /b 1
)

echo.
echo ================================================================================
echo.

REM Step 2: Generate figures and tables
echo [STEP 2/3] Generating all figures and tables...
echo --------------------------------------------------------------------------------
python src/generate_all_figures.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Figure generation failed!
    exit /b 1
)

echo.
echo ================================================================================
echo.

REM Step 3: Generate SHAP visualization
echo [STEP 3/3] Generating SHAP visualization for RQ4_Fig6...
echo --------------------------------------------------------------------------------

REM Check if Python 3.12 venv exists
if not exist "src\.venv_py312_shap\Scripts\python.exe" (
    echo.
    echo ========================================================================
    echo   PYTHON 3.12 VIRTUAL ENVIRONMENT REQUIRED FOR RQ4_Fig6 (SHAP)
    echo ========================================================================
    echo.
    echo   RQ4_Fig6 requires SHAP visualization, which needs Python 3.12.
    echo.
    echo   To set up the environment (one-time setup^):
    echo     python -m venv src\.venv_py312_shap
    echo     src\.venv_py312_shap\Scripts\pip install shap==0.50.0 pandas scikit-learn matplotlib seaborn joblib numpy
    echo.
    echo   Current status: RQ4_Fig6 uses permutation importance (temporary^)
    echo   For the full SHAP beeswarm plot, please run the setup above.
    echo.
    echo ========================================================================
) else (
    src\.venv_py312_shap\Scripts\python src/generate_shap_rq4fig6.py
    if %ERRORLEVEL% neq 0 (
        echo.
        echo ========================================================================
        echo   WARNING: SHAP GENERATION FAILED
        echo ========================================================================
        echo.
        echo   RQ4_Fig6 will use permutation importance fallback.
        echo.
        echo ========================================================================
    )
)

echo.
echo ================================================================================
echo SUCCESS! All outputs generated successfully!
echo ================================================================================
echo.
echo Generated files:
echo   - 19 PDF figures in figures/
echo   - 2 XLSX tables in tables/
echo   - 2 trained models in src/modeling/models/
echo.
echo ================================================================================
