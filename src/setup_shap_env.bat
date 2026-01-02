@echo off
REM Setup Python 3.12 virtual environment for SHAP (Windows)

echo ======================================================================
echo Setting up Python 3.12 virtual environment for SHAP
echo ======================================================================

REM Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.12 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    exit /b 1
)

echo Found Python:
python --version
echo.

REM Create virtual environment
echo Creating virtual environment at src\.venv_py312_shap...
python -m venv src\.venv_py312_shap

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create virtual environment
    exit /b 1
)

echo Virtual environment created
echo.

REM Activate and install dependencies
echo Installing SHAP and dependencies...
src\.venv_py312_shap\Scripts\python -m pip install --upgrade pip
src\.venv_py312_shap\Scripts\pip install shap==0.50.0 pandas scikit-learn matplotlib seaborn joblib numpy

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

echo.
echo ======================================================================
echo SHAP environment setup complete!
echo ======================================================================
echo.
echo Virtual environment location: src\.venv_py312_shap\
for /f "tokens=*" %%i in ('src\.venv_py312_shap\Scripts\python --version') do set PYVER=%%i
echo Python version: %PYVER%
for /f "tokens=*" %%i in ('src\.venv_py312_shap\Scripts\python -c "import shap; print(shap.__version__)"') do set SHAPVER=%%i
echo SHAP version: %SHAPVER%
echo.
echo You can now run SHAP visualizations with:
echo   run_all_windows.bat
echo.
