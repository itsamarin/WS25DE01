# Academic Performance Prediction via Data Fusion

## Project overview

End-to-end ML pipeline predicting student performance using multi-source data fusion (academic, demographic, behavioral features) from Portuguese secondary schools.

**Key Features:**
- Multi-source data fusion with Random Forest, Logistic Regression, Linear Regression
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
cd amarin-project
pip install -r requirements.txt

# 2. Set up Kaggle API (required for data download)
# - Get API token from Kaggle → Account Settings → API → Create New Token
# - Place kaggle.json in ~/.kaggle/ and set permissions
chmod 600 ~/.kaggle/kaggle.json
```

## How to Run

### Option 1: Individual Modules
```bash
python -m src.data_ingestion.loader        # Download data
python -m src.data_cleaning.cleaner        # Clean data
python -m src.feature_engineering.features # Create features
python -m src.modeling.train               # Train models
python -m src.evaluation.metrics           # Evaluate
```

### Option 2: Airflow DAG (Recommended)
See [Airflow Setup](#how-to-run-the-airflow-dag) below for automated orchestration.

## How to Run the Airflow DAG

### Setup
```bash
# 1. Initialize Airflow (first time only)
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com

# 2. Link DAG file
mkdir -p $AIRFLOW_HOME/dags
ln -s $(pwd)/dags/student_performance_pipeline_dag.py $AIRFLOW_HOME/dags/

# 3. Start Airflow (two terminals)
airflow webserver --port 8080  # Terminal 1
airflow scheduler              # Terminal 2
```

### Run
- **UI:** Open `http://localhost:8080` → Toggle DAG ON → Click Trigger
- **CLI:** `airflow dags trigger student_performance_prediction_pipeline`

### DAG Tasks (Sequential)
1. data_ingestion → 2. data_cleaning → 3. feature_engineering → 4. model_training → 5. model_evaluation → 6. generate_figures → 7. pipeline_completion

## Model Configuration

**Models:** Random Forest (n_estimators=300), Logistic Regression (max_iter=1000), Linear Regression
**Split:** 80/20 train-test, stratified, random_state=42
**Feature Importance:** Permutation (n_repeats=5, sample_size=300)

## Reproducibility

All figures (19 PDFs) and tables (2 XLSX) are programmatically generated:
- **Figures:** [src/evaluation/visualizations.py](src/evaluation/visualizations.py) (matplotlib/seaborn/SHAP)
- **Tables:** [src/evaluation/metrics.py](src/evaluation/metrics.py) (pandas)

**Regenerate all outputs:**
```bash
rm -rf figures/*.pdf tables/*.xlsx
python3 run_simple_analysis.py
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
├── tables/                                 # Auto-generated tables (XLSX format)
│   ├── RQ1_Table1.xlsx                     # Model performance metrics
│   └── RQ3_Table1.xlsx                     # Fairness metrics
│
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git ignore configuration
├── README.md                               # This file
│
└── models/                                 # Saved trained models
    ├── rf_pass_prediction.pkl
    └── linear_regression_model.pkl
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
**Output:** `models/*.pkl`

### 5. Evaluation
**Metrics:** Accuracy, precision, recall, F1, permutation importance
**Fairness:** Demographic parity & equal opportunity across sex, Medu, schoolsup, famsup
**Visualizations:** 19 figures (RQ1-RQ4) using matplotlib/seaborn/SHAP
**Output:** `figures/*.pdf`, `tables/*.xlsx`

## License & Contact

**Course:** WS25DE01 Data Engineering
**Contact:** amrin.yanya@gmail.com, David.Joyson@gmail.com
**Dataset:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance) via Kaggle
**Reference:** P. Cortez and A. Silva. "Using Data Mining to Predict Secondary School Student Performance". 2008.
