# Academic Performance Prediction via Data Fusion

## Project overview

End-to-end ML pipeline predicting student performance using multi-source data fusion (academic, demographic, behavioral features) from Portuguese secondary schools.

**Key Features:**
- Multi-source data fusion with Random Forest, Logistic Regression, Linear Regression
- SHAP (SHapley Additive exPlanations) for model interpretability
- Fairness analysis across demographic subgroups
- Automated Airflow orchestration
- Programmatic figure and table generation

## Dataset

**Source:** [Student Performance Dataset](https://www.kaggle.com/datasets/whenamancodes/student-performance)
**Records:** 1,044 students (395 Math, 649 Portuguese) from two Portuguese secondary schools

**Features (33):** Demographic (age, sex, address), Parental (education, jobs), School (support, failures, study time), Behavioral (activities, relationships, absences), Academic (G1, G2, G3 grades)

**Targets:**
- Classification: `target_pass` (pass if G3 ≥ 10)
- Regression: `G3` (final grade, 0-20 scale)

## Research Questions

**RQ1:** Can integrating multiple data sources (academic, behavioral, demographic, LMS) improve the predictive accuracy of academic performance compared to single-source models? 

**RQ2:** How do parental education and family support impact student outcomes?

**RQ3:** How fair is the prediction model across student demographic subgroups (gender, parental education, and support level)?

**RQ4:** Which features (attendance, demographics, study habits) most strongly predict final academic performance? 

## Installation

**Prerequisites:** Python 3.8+, pip

```bash
# 1. Clone and install dependencies
git clone <repository-url>
cd WS25DE01
pip install -r requirements.txt

# 2. Set up Kaggle API (required for data download)
# - Get API token from Kaggle → Account Settings → API → Create New Token
# - Place kaggle.json in ~/.kaggle/ and set permissions
chmod 600 ~/.kaggle/kaggle.json
```

## Quick Start

**Step 1: First-time setup (one-time only)**
```bash
# Linux/macOS:
./src/setup_shap_env.sh

# Windows:
src\setup_shap_env.bat
```

**Step 2: Run complete workflow**
```bash
# Linux/macOS:
./run_all_mac_linux.sh

# Windows:
run_all_windows.bat
```

**Outputs:** 19 PDF figures, 2 XLSX tables, 2 trained models
**Runtime:** ~2-3 minutes

For detailed options and Airflow setup, see [How to Run](#how-to-run-detailed-options) below.

---

## How to Run (Detailed Options)

**All three options produce identical outputs:**
- 19 PDF figures (RQ1: 4, RQ2: 5, RQ3: 4, RQ4: 6)
- 2 XLSX tables (RQ1_Table1.xlsx, RQ3_Table1.xlsx)
- 2 trained models (rf_pass_prediction.pkl, linear_regression_model.pkl)

| Option | Method | Steps | Runtime | Python Version | Status |
|--------|--------|-------|---------|----------------|--------|
| **Option 1** | Automated script | 3 steps | 2-3 min | 3.8+ | ✓ Easiest |
| **Option 2** | Manual step-by-step | 7 steps | 2-3 min | 3.8+ | ✓ Working |
| **Option 3** | Airflow DAG | 9 tasks | 2-3 min | **3.12 or earlier** | ✓ Configured |

### Option 1: Standalone Pipeline (Easiest)

**Step 1: First-time setup (one-time only)**
```bash
# Linux/macOS:
./src/setup_shap_env.sh

# Windows:
src\setup_shap_env.bat
```

**Step 2: Run complete workflow**
```bash
# Linux/macOS:
./run_all_mac_linux.sh

# Windows:
run_all_windows.bat
```

**What it does:**
1. Runs main pipeline (data ingestion, cleaning, feature engineering, model training, evaluation)
2. Generates all 19 figures and 2 tables
3. Generates SHAP visualization for RQ4_Fig6

**Note:** For manual control, run: `python main.py` (pipeline only) or `python src/run_simple_analysis.py` (figures only)

### Option 2: Individual Modules (Step-by-step)

**Steps 1-6: Run pipeline modules**
```bash
python -m src.data_ingestion.loader        # Step 1: Download data
python -m src.data_cleaning.cleaner        # Step 2: Clean data
python -m src.feature_engineering.features # Step 3: Create features
python -m src.modeling.train               # Step 4: Train models
python -m src.evaluation.metrics           # Step 5: Evaluate models
python src/run_simple_analysis.py          # Step 6: Generate all 19 figures and 2 tables
```

**Step 7: Generate SHAP visualization for RQ4_Fig6**
```bash
# Linux/macOS:
./src/generate_shap_with_py312.sh

# Windows:
src\.venv_py312_shap\Scripts\python src\generate_shap_fig6.py
```

**When to use:** Fine-grained control over each pipeline stage, debugging, or learning the workflow.

**Note:** Use `python` or `python3` depending on your system. Windows typically uses `python`, Linux/macOS may use `python3`.

### Option 3: Airflow DAG (Advanced)

**Prerequisites:** Docker (recommended) or Python 3.12 or earlier, Airflow 2.x, PostgreSQL 12+ (optional)

**When to use:** Production workflows, scheduling, monitoring, or data warehouse integration

**Choose your approach based on your platform:**

#### Approach A: Docker (Recommended - All Platforms)

```bash
# Step 1: Download docker-compose.yml
# Get from: https://airflow.apache.org/docs/apache-airflow/stable/docker-compose.yaml

# Step 2: Initialize Airflow
docker-compose up airflow-init

# Step 3: Start Airflow
docker-compose up -d

# Step 4: Access Airflow Web UI
# Open http://localhost:8080 (username/password: airflow/airflow)
# Copy dags/student_performance_pipeline_dag.py to ./dags folder
# Trigger the DAG from the UI

# Step 5: Stop Airflow
docker-compose down
```

#### Approach B: Native Installation (Linux/macOS/WSL2)

```bash
# Step 1: Setup PostgreSQL (optional - for data warehouse integration)
docker run --name postgres-student -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=student_performance -p 5432:5432 -d postgres:15

# Step 2: Initialize Airflow (first-time only)
export AIRFLOW_HOME=$(pwd)/airflow
airflow db migrate
airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin --email admin@example.com

# Step 3: Link DAG file
mkdir -p $AIRFLOW_HOME/dags
ln -s $(pwd)/dags/student_performance_pipeline_dag.py $AIRFLOW_HOME/dags/

# Step 4: Start Airflow (requires 2 terminals)
# Terminal 1:
airflow webserver --port 8080

# Terminal 2:
airflow scheduler

# Step 5: Run the DAG
# Option A - Web UI: Open http://localhost:8080 → Toggle DAG ON → Click "Trigger DAG"
# Option B - CLI: airflow dags trigger student_performance_prediction_pipeline
```

**Windows users:** Install WSL2 with `wsl --install`, then follow Approach B inside WSL2

**What it does:**
- Executes all pipeline stages as Airflow tasks
- Loads data to PostgreSQL in parallel (optional data warehouse integration)
- Provides web UI for monitoring and scheduling
- Enables production-ready workflow orchestration

**DAG Structure (9 tasks):**
- **Main Pipeline:** data_ingestion → data_cleaning → feature_engineering → model_training → model_evaluation → generate_figures → generate_shap → completion
- **Parallel Branch:** feature_engineering → load_to_postgres (data warehouse, runs independently)

**Outputs:** Same as Options 1 & 2 (19 figures, 2 tables, 2 models) + PostgreSQL tables (if configured)

**Note:** Python 3.14+ users should use Option 1 or 2 instead due to Airflow 2.x compatibility.

---

## Additional Information

### Model Configuration
- **Models:** Random Forest (n_estimators=300), Logistic Regression, Linear Regression
- **Split:** 80/20 train-test, stratified, random_state=42
- **Feature Importance:** SHAP (Python 3.12 venv) + Permutation importance

### SHAP Setup (One-Time)
```bash
# Linux/macOS:
./src/setup_shap_env.sh

# Windows:
src\setup_shap_env.bat
```
Run this once before using the complete workflow. If Python 3.12 unavailable, RQ4_Fig6 uses permutation importance.

### PostgreSQL (Optional)
Airflow DAG loads data to PostgreSQL as parallel branch. Tables: `student_performance_cleaned`, `student_performance_abt`
```bash
# Standalone loading (without Airflow)
python3 -m src.data_ingestion.postgres_loader
```

## Folder Structure Explanation

```
WS25DE01/
│
├── dags/                                    # Airflow DAGs
│   └── student_performance_pipeline_dag.py # Main pipeline orchestration
│
├── src/                                     # Core code modules
│   ├── data_ingestion/                     # Data downloading and loading
│   │   ├── __init__.py
│   │   └── loader.py                       # Kaggle data download and CSV conversion
│   │
│   ├── data_cleaning/                      # Data preprocessing and cleaning
│   │   ├── __init__.py
│   │   └── cleaner.py                      # Missing value handling, data combination
│   │
│   ├── feature_engineering/                # Feature creation and ABT
│   │   ├── __init__.py
│   │   └── features.py                     # Derived features and target creation
│   │
│   ├── modeling/                           # Model training and pipelines
│   │   ├── __init__.py
│   │   ├── preprocessing.py                # Preprocessing transformers
│   │   ├── train.py                        # Model training (RF, LR, Linear Regression)
│   │   └── models/                         # Saved trained models
│   │       ├── rf_pass_prediction.pkl
│   │       └── linear_regression_model.pkl
│   │
│   ├── evaluation/                         # Model evaluation and fairness
│   │   ├── __init__.py
│   │   ├── metrics.py                      # Performance metrics, feature importance, fairness
│   │   └── visualizations.py               # All RQ figure generation code (RQ1-RQ4)
│   │
│   ├── .venv_py312_shap/                   # Python 3.12 virtual environment for SHAP (git-ignored)
│   ├── generate_shap_fig6.py               # SHAP beeswarm visualization for RQ4_Fig6
│   ├── generate_enhanced_fig6.py           # Permutation importance fallback for RQ4_Fig6
│   ├── generate_shap_with_py312.sh         # Shell wrapper to run SHAP (Linux/macOS)
│   ├── setup_shap_env.sh                   # SHAP environment setup (Linux/macOS)
│   ├── setup_shap_env.bat                  # SHAP environment setup (Windows)
│   └── run_simple_analysis.py              # Main script to generate all figures/tables
│
├── data/                                   # Data storage (NO large raw datasets in Git)
│   └── sample/                             # Sample data files only
│
├── figures/                                # Auto-generated visualizations (PDF format)
│   ├── RQ1_Fig*.pdf                        # Model comparison figures
│   ├── RQ2_Fig*.pdf                        # Parental education impact
│   ├── RQ3_Fig*.pdf                        # Fairness analysis
│   └── RQ4_Fig*.pdf                        # Feature importance
│
├── tables/                                 # Auto-generated tables (XLSX format)
│   ├── RQ1_Table1.xlsx                     # Model performance metrics
│   └── RQ3_Table1.xlsx                     # Fairness metrics
│
├── main.py                                 # Standalone pipeline runner (no Airflow)
├── run_all_mac_linux.sh                    # Complete workflow script (macOS/Linux)
├── run_all_windows.bat                     # Complete workflow script (Windows)
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git ignore configuration
└── README.md                               # This file
```

## License & Contact

**Course:** WS25DE01 Data Engineering
**Contact:** amrin.yanya@gmail.com, David.Joyson@gmail.com
**Dataset:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance) via Kaggle
**Reference:** P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance". 2008.
