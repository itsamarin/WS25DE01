# Academic Performance Prediction via Data Fusion

This project predicts student academic performance using multi-source data fusion, combining academic, demographic, and behavioral features.

## Project Overview

This project implements an end-to-end machine learning pipeline for predicting student academic performance using data from Portuguese secondary schools. The pipeline leverages multi-source data fusion, combining academic grades, demographic information, and behavioral features to predict student pass/fail outcomes and final grades.

**Key Features:**
- Complete data pipeline from ingestion to model evaluation
- Multi-source data fusion combining academic, demographic, and behavioral data
- Multiple ML models: Random Forest, Logistic Regression, Linear Regression
- Fairness analysis across demographic subgroups
- Automated Airflow orchestration for reproducibility
- All figures and tables generated programmatically from code

## Dataset

**Source:** [Student Performance Dataset](https://www.kaggle.com/datasets/whenamancodes/student-performance) on Kaggle

**Direct Download Link:** `https://www.kaggle.com/datasets/whenamancodes/student-performance`

**Description:** This dataset contains student achievement records in Math and Portuguese courses from two Portuguese secondary schools. It includes demographic, social, and academic information about students.

**Dataset Size:**
- Math students: 395 records
- Portuguese students: 649 records
- Combined: 1,044 student records

**Features (33 attributes):**
- **Demographic:** age, sex, address, family size, parent cohabitation status
- **Parental:** mother/father education level, mother/father job, guardian
- **School-related:** school name, reason for choosing school, travel time, study time, past class failures, extra educational support
- **Behavioral:** extracurricular activities, attended nursery school, wants higher education, internet access, romantic relationship, family relationship quality, free time, going out with friends, workday/weekend alcohol consumption, health status, absences
- **Academic Performance:** G1 (first period grade), G2 (second period grade), G3 (final grade, 0-20 scale)

**Target Variables:**
- **Classification:** `target_pass` (binary: 1 if G3 >= 10, 0 otherwise)
- **Regression:** `G3` (final grade, continuous 0-20)

## Research Questions

This project addresses four key research questions:

**RQ1: Multi-Source vs Single-Source Performance**
- Does combining demographic and behavioral features with academic data improve prediction accuracy compared to using only academic features (G1, G2)?
- Methodology: Compare Random Forest and Logistic Regression models trained on all features vs. academic-only features

**RQ2: Parental Education & Family Support Impact**
- How do parental education levels and family support systems affect student academic outcomes?
- Methodology: Analyze correlation between parental education (Medu, Fedu), family support (famsup), school support (schoolsup), and student performance

**RQ3: Model Fairness Across Demographics**
- Are our predictive models fair across different demographic subgroups (gender, parental education, support systems)?
- Methodology: Calculate demographic parity and equal opportunity metrics for sensitive attributes (sex, Medu, schoolsup, famsup)

**RQ4: Feature Importance Analysis**
- Which features are most influential in predicting student success?
- Methodology: Use permutation importance and SHAP values to identify top predictive features

## Project Structure

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
│   │   └── train.py                        # Model training (RF, LR, Linear Regression)
│   │
│   └── evaluation/                         # Model evaluation and fairness
│       ├── __init__.py
│       ├── metrics.py                      # Performance metrics, feature importance, fairness
│       └── visualizations.py               # All RQ figure generation code (RQ1-RQ4)
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
├── tables/                                 # Auto-generated tables (CSV format)
│   ├── RQ1_Table1.csv                      # Model performance metrics
│   └── RQ3_Table1.csv                      # Fairness metrics
│
├── models/                                 # Saved trained models
│   ├── rf_pass_prediction.pkl
│   └── linear_regression_model.pkl
│
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git ignore configuration
└── README.md                               # This file
```

**Folder Structure Explanation:**
- **dags/**: Contains Airflow DAG definitions for pipeline orchestration
- **src/**: Modular Python code organized by pipeline stage (ingestion → cleaning → features → modeling → evaluation)
- **data/sample/**: Only small sample data files (full datasets downloaded at runtime)
- **figures/**: All visualizations automatically generated as PDFs from code
- **tables/**: All data tables automatically generated as CSV/Excel from code
- **models/**: Serialized trained model artifacts

## Module Overview

### 1. Data Ingestion (`src/data_ingestion/`)
**Purpose:** Download and load student performance datasets from Kaggle.

**Key Functions:**
- `create_directory_structure()` - Creates project folder structure
- `download_student_performance_data()` - Downloads dataset from Kaggle
- `load_and_save_datasets()` - Loads Excel files and saves as CSV
- `ingest_data()` - Complete ingestion pipeline

**Outputs:**
- `data/raw/maths/Maths.csv`
- `data/raw/portuguese/Portuguese.csv`

### 2. Data Cleaning (`src/data_cleaning/`)
**Purpose:** Combine, clean, and preprocess student performance datasets.

**Key Functions:**
- `add_course_labels()` - Adds course identifier (math/portuguese)
- `combine_datasets()` - Merges Math and Portuguese datasets
- `handle_empty_strings()` - Replaces empty strings with pd.NA
- `impute_numeric_columns()` - Fills missing numeric values with median
- `impute_categorical_columns()` - Fills missing categorical values with "unknown"
- `clean_student_performance_data()` - Complete cleaning pipeline

**Outputs:**
- `data/cleaned/student_performance_clean.csv`

### 3. Feature Engineering (`src/feature_engineering/`)
**Purpose:** Create derived features and analytical base table (ABT).

**Key Functions:**
- `create_average_previous_grade()` - Averages G1 and G2
- `create_grade_trend()` - Calculates grade change (G3 - G1)
- `create_high_absence_indicator()` - Binary indicator for high absences
- `create_target_pass()` - Binary classification target (pass/fail)
- `build_analytical_base_table()` - Complete feature engineering pipeline

**Features Created:**
- `avg_prev_grade` - Mean of G1 and G2
- `grade_trend` - Grade change from G1 to G3
- `high_absence` - Binary indicator (1 if absences > median)
- `target_pass` - Binary target (1 if G3 >= 10)

**Outputs:**
- `data/processed/abt_student_performance.csv`

### 4. Modeling (`src/modeling/`)
**Purpose:** Define preprocessing pipelines and train classification/regression models.

#### 4.1 Preprocessing (`preprocessing.py`)
**Key Functions:**
- `get_numeric_categorical_features()` - Identifies feature types
- `create_multi_source_preprocessor()` - StandardScaler + OneHotEncoder for all features
- `create_academic_only_preprocessor()` - StandardScaler for G1, G2 only
- `create_regression_preprocessor()` - Preprocessing for regression models

#### 4.2 Training (`train.py`)
**Key Functions:**
- `load_abt()` - Loads analytical base table
- `prepare_data()` - Separates features and target
- `split_data()` - Train-test split with stratification
- `train_multi_source_models()` - Trains LR and RF with all features
- `train_academic_only_models()` - Trains LR and RF with G1, G2 only
- `train_regression_model()` - Trains linear regression for G3 prediction
- `save_model()` / `load_model()` - Model persistence

**Models Trained:**
- **Multi-Source Logistic Regression** - All features
- **Multi-Source Random Forest** - All features (n_estimators=300)
- **Single-Source Logistic Regression** - G1, G2 only
- **Single-Source Random Forest** - G1, G2 only
- **Linear Regression** - Predicts final grade (G3)

**Outputs:**
- `models/rf_pass_prediction.pkl`
- `models/linear_regression_model.pkl`

### 5. Evaluation (`src/evaluation/`)
**Purpose:** Evaluate model performance, feature importance, and fairness.

#### 5.1 Metrics (`metrics.py`)
**Key Functions:**
- `evaluate_model()` - Calculates accuracy, precision, recall, F1
- `compare_models()` - Compares multiple models
- `calculate_permutation_importance()` - Feature importance via permutation
- `subgroup_metrics()` - Performance metrics by demographic subgroup
- `calculate_fairness_metrics()` - Fairness analysis for sensitive attributes
- `demographic_parity_difference()` - Demographic parity metric
- `equal_opportunity_difference()` - Equal opportunity (TPR) metric

**Fairness Analysis:**
- Analyzes model performance across demographic subgroups:
  - `sex` - Gender (M/F)
  - `Medu` - Mother's education level (0-4)
  - `schoolsup` - School support (yes/no)
  - `famsup` - Family support (yes/no)

**Outputs:**
- `figures/model_comparison.csv`
- `figures/feature_importance_rf_full.csv`
- `figures/feature_importance_rf_top15.csv`
- `figures/fairness_by_sex.csv`
- `figures/fairness_by_Medu.csv`
- `figures/fairness_by_schoolsup.csv`
- `figures/fairness_by_famsup.csv`

#### 5.2 Visualizations (`visualizations.py`)
**Purpose:** Generate all research question figures programmatically.

**RQ1 Figures (Multi-Source vs Single-Source):**
- `plot_rq1_fig1_model_comparison()` - Performance comparison bar plot
- `plot_rq1_fig2_grade_scatter()` - G1 vs G2 scatter plot by pass/fail
- `plot_rq1_fig3_improvement()` - Improvement percentage comparison
- `plot_rq1_fig4_studytime_boxplot()` - Grades by study time

**RQ2 Figures (Parental Education & Support):**
- `plot_rq2_fig1_parental_education()` - Mean grade by parental education
- `plot_rq2_fig2_resilience_drivers()` - Key drivers for low education group
- `plot_rq2_fig3_grade_improvement_trend()` - Education-location interaction
- `plot_rq2_fig4_improvement_heatmap()` - Improvement probability heatmap
- `plot_rq2_fig5_parental_ed_by_address()` - Education level by location trends

**RQ3 Figures (Model Fairness):**
- `plot_rq3_fig1_fairness_gap()` - F1-score gaps from baseline
- `plot_rq3_fig2_subgroup_heatmap()` - Subgroup performance heatmap
- `plot_rq3_fig3_subgroup_performance()` - Performance by demographics
- `plot_rq3_fig4_fairness_metrics()` - Demographic parity & equal opportunity

**RQ4 Figures (Feature Importance):**
- `plot_rq4_fig1_feature_stability()` - Cross-validation feature stability
- `plot_rq4_fig2_model_comparison()` - LR vs RF performance
- `plot_rq4_fig3_confusion_matrices()` - Confusion matrices for both models
- `plot_rq4_fig4_runtime_comparison()` - Training/prediction time comparison
- `plot_rq4_fig5_feature_importance()` - Top predictive features bar plot
- `plot_rq4_fig6_shap_importance()` - SHAP global feature importance analysis

**Outputs:**
- All 19 PDF figures in `figures/` directory (RQ1_Fig1.pdf through RQ4_Fig6.pdf)

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd amarin-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Kaggle API credentials (required for data download):
   - Create an account on [Kaggle](https://www.kaggle.com/)
   - Go to Account Settings → API → Create New API Token
   - Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

## How to Run the Code

### Option 1: Run Complete Pipeline Manually

Run the complete pipeline step-by-step from data ingestion to model training:

```python
# 1. Data Ingestion
from src.data_ingestion.loader import ingest_data
mat, por = ingest_data()

# 2. Data Cleaning
from src.data_cleaning.cleaner import clean_student_performance_data
cleaned_df = clean_student_performance_data(mat, por)

# 3. Feature Engineering
from src.feature_engineering.features import build_analytical_base_table
abt = build_analytical_base_table()

# 4. Model Training
from src.modeling.train import load_abt, prepare_data, split_data, train_multi_source_models
abt = load_abt()
X, y = prepare_data(abt)
X_train, X_test, y_train, y_test = split_data(X, y)
models = train_multi_source_models(X_train, X_test, y_train, y_test)

# 5. Evaluation
from src.evaluation.metrics import calculate_permutation_importance, calculate_fairness_metrics
fi_df = calculate_permutation_importance(models['random_forest'], X_test, y_test)
fairness = calculate_fairness_metrics(models['random_forest'], X_test, y_test)
```

### Option 2: Run Individual Modules

Each module can be run standalone for testing or debugging:

```bash
# Step 1: Data ingestion
python -m src.data_ingestion.loader

# Step 2: Data cleaning
python -m src.data_cleaning.cleaner

# Step 3: Feature engineering
python -m src.feature_engineering.features

# Step 4: Model training
python -m src.modeling.train

# Step 5: Evaluation
python -m src.evaluation.metrics
```

### Option 3: Run via Python Script

Create a main script to run the entire pipeline:

```python
# main.py
from src.data_ingestion.loader import ingest_data
from src.data_cleaning.cleaner import clean_student_performance_data
from src.feature_engineering.features import build_analytical_base_table
from src.modeling.train import load_abt, prepare_data, split_data, train_multi_source_models
from src.evaluation.metrics import calculate_permutation_importance, calculate_fairness_metrics

def run_pipeline():
    print("Starting pipeline...")

    # 1. Data Ingestion
    print("Step 1: Data Ingestion")
    mat, por = ingest_data()

    # 2. Data Cleaning
    print("Step 2: Data Cleaning")
    cleaned_df = clean_student_performance_data(mat, por)

    # 3. Feature Engineering
    print("Step 3: Feature Engineering")
    abt = build_analytical_base_table()

    # 4. Model Training
    print("Step 4: Model Training")
    X, y = prepare_data(abt)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = train_multi_source_models(X_train, X_test, y_train, y_test)

    # 5. Evaluation
    print("Step 5: Model Evaluation")
    fi_df = calculate_permutation_importance(models['random_forest'], X_test, y_test)
    fairness = calculate_fairness_metrics(models['random_forest'], X_test, y_test)

    print("Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()
```

Then run:
```bash
python main.py
```

## How to Run the Airflow DAG

### Prerequisites
- Apache Airflow installed (included in requirements.txt)
- Kaggle API credentials configured
- All dependencies installed

### Setup Airflow

1. **Initialize Airflow Database** (first time only):
```bash
# Set Airflow home directory (optional, defaults to ~/airflow)
export AIRFLOW_HOME=$(pwd)/airflow

# Initialize the database
airflow db init

# Create an admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

2. **Configure DAG Directory**:
```bash
# Update airflow.cfg to point to your dags folder
# Or create a symlink
mkdir -p $AIRFLOW_HOME/dags
ln -s $(pwd)/dags/student_performance_pipeline_dag.py $AIRFLOW_HOME/dags/
```

3. **Start Airflow Services**:

Open two terminal windows:

Terminal 1 - Start the Airflow webserver:
```bash
airflow webserver --port 8080
```

Terminal 2 - Start the Airflow scheduler:
```bash
airflow scheduler
```

4. **Access Airflow UI**:
- Open your browser and navigate to `http://localhost:8080`
- Login with credentials (admin/admin)
- Find the DAG named `student_performance_prediction_pipeline`

5. **Run the DAG**:
- Toggle the DAG to "ON" state
- Click the "Trigger DAG" button (play icon) to manually execute
- Monitor progress in the Graph View or Tree View

### Run DAG from Command Line

Alternatively, run the DAG without the UI:

```bash
# Test a specific task
airflow tasks test student_performance_prediction_pipeline data_ingestion 2025-01-01

# Run the entire DAG
airflow dags test student_performance_prediction_pipeline 2025-01-01

# Or trigger via CLI
airflow dags trigger student_performance_prediction_pipeline
```

### DAG Structure

The DAG consists of 7 sequential tasks:

1. **data_ingestion**: Downloads datasets from Kaggle
2. **data_cleaning**: Cleans and combines datasets
3. **feature_engineering**: Creates derived features and ABT
4. **model_training**: Trains ML models (RF, LR, Linear Regression)
5. **model_evaluation**: Evaluates models and calculates fairness metrics
6. **generate_figures**: Creates visualizations and tables
7. **pipeline_completion**: Prints summary statistics

Each task must complete successfully before the next task begins.

### Monitoring and Logs

- **View Logs**: Click on any task in the Airflow UI → View Logs
- **Check Status**: Monitor task colors (green=success, red=failed, yellow=running)
- **XCom Values**: View inter-task communication data in the XCom tab

### Troubleshooting

**DAG not appearing:**
- Verify the DAG file is in the correct dags folder
- Check for Python syntax errors: `python dags/student_performance_pipeline_dag.py`
- Refresh the DAG list in the UI

**Import errors:**
- Ensure project root is in PYTHONPATH: `export PYTHONPATH=$(pwd):$PYTHONPATH`
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Kaggle API errors:**
- Confirm `~/.kaggle/kaggle.json` exists and has correct permissions (600)
- Test Kaggle API: `kaggle datasets list`

## Model Hyperparameters

### Random Forest
- `n_estimators`: 300
- `random_state`: 42
- `n_jobs`: -1 (use all CPU cores)

### Logistic Regression
- `max_iter`: 1000

### Train-Test Split
- `test_size`: 0.2 (20% test set)
- `random_state`: 42
- `stratify`: True (maintain class distribution)

### Permutation Importance
- `n_repeats`: 5
- `sample_size`: 300
- `random_state`: 42

## Reproducibility

All figures and tables in this project are **automatically generated from code**. No manually created visualizations or tables are included.

**Figure Generation:**
- All 19 PDF figures in [figures/](figures/) are programmatically created using matplotlib/seaborn/SHAP
- Figure generation code is in [src/evaluation/visualizations.py](src/evaluation/visualizations.py)
- Each research question (RQ1-RQ4) has dedicated plotting functions
- RQ4_Fig6 uses SHAP (SHapley Additive exPlanations) for interpretable feature importance
- Running the pipeline or calling individual plot functions automatically saves figures to figures/

**Table Generation:**
- All CSV tables in [tables/](tables/) are programmatically created using pandas
- Table generation code is in [src/evaluation/metrics.py](src/evaluation/metrics.py)
- Tables are saved automatically during model evaluation
- Tables include model performance metrics and fairness analysis results

**Verification:**
To verify reproducibility, delete all figures and tables and re-run the evaluation:
```bash
# Delete existing outputs
rm -rf figures/*.pdf tables/*.csv

# Regenerate all outputs
python -m src.evaluation.metrics
python -m src.evaluation.visualizations
```

Alternatively, run the complete Airflow DAG which will regenerate all figures and tables automatically.

All outputs will be regenerated identically from the code, ensuring full reproducibility.

## License

This project is for educational purposes as part of the **WS25DE01 Data Engineering** course.

## Contributors

- ES25DE01 Project Team
- Institution: [Your University Name]
- Course: Data Engineering WS 2025

## Contact

For questions or issues about this project:
- Open an issue in this repository
- Contact: amrin.yanya@gmail.com

## Acknowledgments

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance) via Kaggle
- Original Paper: P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance". 2008.
