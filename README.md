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

## Quick Start - Complete Workflow

**Fastest way to run the entire project and generate all outputs (19 figures + 2 tables):**

```bash
# Option 1: Use convenience script (recommended - includes SHAP)
# Automatically creates all necessary directories and handles dependencies
./run_all.sh

# Option 2: One-liner command (without SHAP)
python3 main.py && python3 src/run_simple_analysis.py
```

Or run step-by-step:
```bash
python3 main.py                      # Step 1: Run pipeline (download, clean, train, evaluate)
python3 src/run_simple_analysis.py   # Step 2: Generate all figures and tables
```

**What you get:**
- 19 PDF figures in `figures/` (RQ1: 4 figs, RQ2: 5 figs, RQ3: 4 figs, RQ4: 6 figs)
  - RQ4_Fig6 includes SHAP beeswarm plot (auto-generated with Python 3.12 venv)
- 2 XLSX tables in `tables/` (RQ1_Table1.xlsx, RQ3_Table1.xlsx)
- Trained models in `src/modeling/models/`
- Processed datasets in `data/`

**Estimated runtime:** ~2-3 minutes (depending on hardware)

**Note:** The workflow automatically creates all necessary output directories (`figures/`, `tables/`) - no manual setup required.

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
```bash
# Complete workflow with all outputs (includes SHAP)
./run_all.sh

# Alternative: Run core pipeline only (without figures/SHAP)
python3 main.py  # Outputs: 2 trained models, processed data
```

**What it does:**
1. Runs main pipeline (data ingestion, cleaning, feature engineering, model training, evaluation)
2. Generates all 19 figures and 2 tables
3. Generates SHAP visualization for RQ4_Fig6

**Note:** For full output (19 figures + 2 tables), use `./run_all.sh` or add figure generation steps manually.

### Option 2: Individual Modules (Step-by-step)
```bash
python3 -m src.data_ingestion.loader        # Step 1: Download data
python3 -m src.data_cleaning.cleaner        # Step 2: Clean data
python3 -m src.feature_engineering.features # Step 3: Create features
python3 -m src.modeling.train               # Step 4: Train models
python3 -m src.evaluation.metrics           # Step 5: Evaluate models
python3 src/run_simple_analysis.py          # Step 6: Generate all 19 figures and 2 tables
./generate_shap_with_py312.sh               # Step 7: Generate SHAP visualization for RQ4_Fig6
```

**When to use:** Fine-grained control over each pipeline stage, debugging, or learning the workflow.

### Option 3: Airflow DAG (Advanced - Python 3.12 or earlier required)
See [Airflow Setup](#how-to-run-the-airflow-dag) below for automated orchestration.

**Prerequisites:** Python 3.12 or earlier (Airflow 2.x incompatible with Python 3.14+)

**When to use:** Production workflows, scheduling, monitoring, or integration with existing Airflow infrastructure.

**Important:** The DAG is fully configured with all 9 tasks including PostgreSQL loading, figure generation, and SHAP. It will produce identical outputs to Options 1 & 2 when run with Python 3.12 or earlier. For Python 3.14+ users, use Option 1 or Option 2 instead.

## How to Run the Airflow DAG

### Setup

**Prerequisites:** Python 3.12 or earlier, Airflow 2.x, PostgreSQL 12+

```bash
# 1. Set up PostgreSQL (if using PostgreSQL integration)
# Option A: Using Docker
docker run --name postgres-student -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=student_performance -p 5432:5432 -d postgres:15

# Option B: Using local PostgreSQL
createdb student_performance

# Configure PostgreSQL connection (optional - uses defaults if not set)
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=student_performance
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres

# 2. Initialize Airflow (first time only)
export AIRFLOW_HOME=$(pwd)/airflow
airflow db migrate  # Updated command for Airflow 2.x (use 'init' for older versions)
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com

# 3. Link DAG file
mkdir -p $AIRFLOW_HOME/dags
ln -s $(pwd)/dags/student_performance_pipeline_dag.py $AIRFLOW_HOME/dags/

# 4. Start Airflow (two terminals)
airflow webserver --port 8080  # Terminal 1
airflow scheduler              # Terminal 2
```

### Run
- **UI:** Open `http://localhost:8080` → Toggle DAG ON → Click Trigger
- **CLI:** `airflow dags trigger student_performance_prediction_pipeline`

### DAG Tasks (Parallel Execution)

**Main Pipeline (uses CSV files):**
1. data_ingestion → 2. data_cleaning → 3. feature_engineering → 4. model_training → 5. model_evaluation → 6. generate_figures → 7. generate_shap → 8. pipeline_completion

**Parallel Branch (data warehouse):**
3. feature_engineering → 3.5. load_to_postgres

The PostgreSQL loading runs independently as a parallel branch after feature engineering. This allows data to be ingested into the database for BI tools and reporting without blocking the ML pipeline, which continues using CSV files.

**Outputs:**
- PostgreSQL tables: `student_performance_cleaned`, `student_performance_abt` (parallel branch)
- 19 PDF figures + 2 XLSX tables + 2 trained models (same as Options 1 & 2)

## PostgreSQL Database Integration

The Airflow DAG includes PostgreSQL integration as a parallel branch for data warehouse functionality. This allows the ML pipeline to continue using CSV files while simultaneously loading data to PostgreSQL for BI tools, reporting, and data warehouse integration.

### Database Schema

**Tables Created:**
1. `student_performance_cleaned` - Cleaned student data (1,048 rows, 34 columns)
2. `student_performance_abt` - Analytical Base Table with engineered features (1,048 rows, 38 columns)

**Key Features:**
- Automatic table creation with proper data types
- Indexes on frequently queried columns (course, target_pass, G3)
- Timestamp tracking (`created_at` column)
- Support for both Docker and local PostgreSQL installations

### Standalone PostgreSQL Loading

You can also load data to PostgreSQL independently of Airflow:

```bash
# Ensure PostgreSQL is running and configured
export POSTGRES_HOST=localhost
export POSTGRES_DB=student_performance
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres

# Load data to PostgreSQL
python3 -m src.data_ingestion.postgres_loader
```

This loads both cleaned data and ABT tables.

### Querying the Data

```sql
-- Connect to database
psql -U postgres -d student_performance

-- View total students and pass rate
SELECT
    COUNT(*) as total_students,
    SUM(CASE WHEN target_pass = 1 THEN 1 ELSE 0 END) as passed,
    AVG(G3) as avg_final_grade
FROM student_performance_abt;

-- View performance by course
SELECT
    course,
    COUNT(*) as students,
    AVG(G3) as avg_grade,
    SUM(CASE WHEN target_pass = 1 THEN 1 ELSE 0 END)::FLOAT / COUNT(*) as pass_rate
FROM student_performance_abt
GROUP BY course;
```

## Model Configuration

**Models:** Random Forest (n_estimators=300), Logistic Regression (max_iter=1000), Linear Regression
**Split:** 80/20 train-test, stratified, random_state=42
**Feature Importance:**
- SHAP (SHapley Additive exPlanations) for interpretable AI explanations
- Permutation importance (n_repeats=10) with confidence intervals for robust feature ranking

> **Note on SHAP:** SHAP visualizations are automatically generated when using [run_all.sh](run_all.sh). SHAP requires a dedicated Python 3.12 virtual environment at [src/.venv_py312_shap](src/.venv_py312_shap) to resolve dependency compatibility issues. You can also run SHAP separately using [generate_shap_with_py312.sh](generate_shap_with_py312.sh).

## SHAP Integration for Model Interpretability

This project includes SHAP (SHapley Additive exPlanations) support for advanced model interpretability. Due to dependency compatibility issues between SHAP and Python 3.14+, we maintain a separate Python 3.12 virtual environment specifically for SHAP visualizations.

**SHAP is automatically included when using `./run_all.sh`** - no additional steps needed!

### Manual SHAP Generation

If you need to regenerate only the SHAP visualization:
```bash
# Generate SHAP-based RQ4_Fig6 visualization
./generate_shap_with_py312.sh
```

**Available SHAP Scripts:**
- [src/generate_shap_fig6.py](src/generate_shap_fig6.py) - Main SHAP visualization (recommended)
- [src/generate_shap_figure.py](src/generate_shap_figure.py) - Alternative SHAP generator
- [src/generate_shap_kernel.py](src/generate_shap_kernel.py) - SHAP KernelExplainer version

**Environment:**
- Python 3.12 virtual environment: [src/.venv_py312_shap](src/.venv_py312_shap)
- Isolated from main project dependencies to avoid conflicts

**Note:** When running [src/run_simple_analysis.py](src/run_simple_analysis.py) standalone, RQ4_Fig6 uses permutation importance. The [run_all.sh](run_all.sh) script automatically replaces this with the SHAP beeswarm plot in Step 3.

## Reproducibility

All 19 figures and 2 tables are programmatically generated by [src/run_simple_analysis.py](src/run_simple_analysis.py):

**Figure Generation:**
- RQ1 (4 figures): Model comparison, grade scatter, improvement analysis, study time analysis
- RQ2 (5 figures): Parental education impact on grades and pass rates
- RQ3 (4 figures): Fairness analysis across demographic groups
- RQ4 (6 figures): Feature importance using Gini, permutation, and SHAP methods

**Table Generation:**
- RQ1_Table1.xlsx: Model performance metrics (accuracy, precision, recall, F1)
- RQ3_Table1.xlsx: Fairness metrics (demographic parity, equal opportunity)

**Regenerate all outputs:**
```bash
# Clean previous outputs and regenerate everything
rm -rf figures/ tables/ data/ src/modeling/models/
./run_all.sh

# Or clean only figures and tables (keeps trained models and data)
rm -rf figures/*.pdf tables/*.xlsx
./run_all.sh

# Or manually run individual steps (SHAP is auto-included in run_all.sh)
python3 src/run_simple_analysis.py
./generate_shap_with_py312.sh
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
│   ├── .venv_py312_shap/                   # Python 3.12 virtual environment for SHAP
│   ├── generate_shap_fig6.py               # SHAP-based RQ4_Fig6 generator
│   ├── generate_shap_figure.py             # Alternative SHAP generator
│   ├── generate_shap_kernel.py             # SHAP KernelExplainer version
│   ├── generate_enhanced_fig6.py           # Enhanced permutation importance fallback
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
├── run_all.sh                              # Convenience script to run complete workflow
├── generate_shap_with_py312.sh             # Shell script to generate SHAP with Python 3.12
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git ignore configuration
└── README.md                               # This file
```

## Pipeline Stages

### 1. Data Ingestion
Downloads datasets from Kaggle → `data/raw/{maths,portuguese}.csv`

### 2. Data Cleaning
Combines datasets, imputes missing values (median/mode) → `data/cleaned/student_performance_clean.csv`

### 3. Feature Engineering
Creates derived features (avg_prev_grade, grade_trend, high_absence, target_pass) → `data/processed/abt_student_performance.csv`

### 4. Modeling
**Preprocessing:** StandardScaler + OneHotEncoder
**Models:** Multi-source (all features) vs Single-source (G1, G2 only) - Random Forest, Logistic Regression, Linear Regression
**Output:** `src/modeling/models/*.pkl`

### 5. Evaluation
**Metrics:** Accuracy, precision, recall, F1, permutation importance
**Fairness:** Demographic parity & equal opportunity across sex, Medu, schoolsup, famsup
**Visualizations:** 19 figures (RQ1-RQ4) using matplotlib/seaborn
**Feature Importance:**
- SHAP (SHapley Additive exPlanations) for model interpretability
- Permutation importance with uncertainty quantification (default)
**Output:** `figures/*.pdf`, `tables/*.xlsx`

## License & Contact

**Course:** WS25DE01 Data Engineering
**Contact:** amrin.yanya@gmail.com, David.Joyson@gmail.com
**Dataset:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance) via Kaggle
**Reference:** P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance". 2008.
