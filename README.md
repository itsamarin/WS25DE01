# Academic Performance Prediction via Data Fusion

This project predicts student academic performance using multi-source data fusion, combining academic, demographic, and behavioral features.

## Project Structure

```
project/
├── src/                              # Core code modules
│   ├── data_ingestion/              # Data downloading and loading
│   │   ├── __init__.py
│   │   └── loader.py                # Kaggle data download and CSV conversion
│   │
│   ├── data_cleaning/               # Data preprocessing and cleaning
│   │   ├── __init__.py
│   │   └── cleaner.py               # Missing value handling, data combination
│   │
│   ├── feature_engineering/         # Feature creation and ABT
│   │   ├── __init__.py
│   │   └── features.py              # Derived features and target creation
│   │
│   ├── modeling/                    # Model training and pipelines
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Preprocessing transformers
│   │   └── train.py                 # Model training (RF, LR, Linear Regression)
│   │
│   └── evaluation/                  # Model evaluation and fairness
│       ├── __init__.py
│       └── metrics.py               # Performance metrics, feature importance, fairness
│
├── data/                            # Data storage (created by pipeline)
│   ├── raw/                         # Raw datasets
│   ├── cleaned/                     # Cleaned datasets
│   └── processed/                   # Analytical Base Tables (ABT)
│
├── models/                          # Saved model artifacts
├── figures/                         # Generated visualizations and tables
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

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

## Usage

### Complete Pipeline

Run the complete pipeline from data ingestion to model training:

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

### Individual Modules

Each module can be run standalone:

```bash
# Data ingestion
python -m src.data_ingestion.loader

# Data cleaning
python -m src.data_cleaning.cleaner

# Feature engineering
python -m src.feature_engineering.features

# Model training
python -m src.modeling.train

# Evaluation
python -m src.evaluation.metrics
```

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

## Research Questions

This project addresses four key research questions:

1. **RQ1: Multi-Source vs Single-Source Performance**
   - Compares prediction accuracy using all features vs. academic features only (G1, G2)

2. **RQ2: Parental Education & Family Support**
   - Analyzes impact of parental education level and family support on student outcomes

3. **RQ3: Model Fairness**
   - Evaluates fairness across demographic subgroups (sex, maternal education, support systems)

4. **RQ4: Feature Importance**
   - Identifies most influential features using permutation importance and SHAP values

## Dataset

**Source:** [Student Performance Dataset](https://www.kaggle.com/datasets/whenamancodes/student-performance) on Kaggle

**Description:** Student achievement in Math and Portuguese courses from two Portuguese schools.

**Features:**
- **Demographic:** age, sex, address, family size, parent status
- **Parental:** mother/father education, jobs, guardian
- **School:** school, reason, travel time, study time, failures, support
- **Behavioral:** activities, absences, alcohol consumption
- **Academic:** G1 (first period grade), G2 (second period grade), G3 (final grade)

**Target Variables:**
- **Regression:** G3 (final grade, 0-20)
- **Classification:** target_pass (1 if G3 >= 10, 0 otherwise)

## License

This project is for educational purposes as part of the WS25DE01 course.

## Contact

For questions or issues, please contact the project maintainer.
