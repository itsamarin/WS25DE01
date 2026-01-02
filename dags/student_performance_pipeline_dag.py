"""
Student Performance Prediction Pipeline DAG

This Airflow DAG orchestrates the complete student performance prediction pipeline,
including data ingestion, cleaning, feature engineering, model training, and evaluation.

Author: ES25DE01 Project Team
Date: 2025-12-25
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Default arguments for the DAG
default_args = {
    'owner': 'es25de01-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 1),
}

# Define the DAG
dag = DAG(
    'student_performance_prediction_pipeline',
    default_args=default_args,
    description='End-to-end pipeline for student performance prediction using multi-source data fusion',
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=['student-performance', 'machine-learning', 'data-fusion'],
)


def task_data_ingestion(**context):
    """
    Task 1: Data Ingestion
    Downloads student performance datasets from Kaggle and saves them as CSV files.

    Inputs: None (downloads from Kaggle API)
    Outputs:
        - data/raw/maths/Maths.csv
        - data/raw/portuguese/Portuguese.csv
    """
    from src.data_ingestion.loader import ingest_data

    print("Starting data ingestion from Kaggle...")
    mat, por = ingest_data()

    # Push metadata to XCom for downstream tasks
    context['ti'].xcom_push(key='mat_shape', value=mat.shape)
    context['ti'].xcom_push(key='por_shape', value=por.shape)

    print(f"Data ingestion completed. Math dataset: {mat.shape}, Portuguese dataset: {por.shape}")
    return "Data ingestion successful"


def task_data_cleaning(**context):
    """
    Task 2: Data Cleaning and Preprocessing
    Combines Math and Portuguese datasets, handles missing values, and cleans data.

    Inputs:
        - data/raw/maths/Maths.csv
        - data/raw/portuguese/Portuguese.csv
    Outputs:
        - data/cleaned/student_performance_clean.csv
    """
    from src.data_cleaning.cleaner import clean_student_performance_data
    from src.data_ingestion.loader import ingest_data

    print("Starting data cleaning...")

    # Load raw data
    mat, por = ingest_data()

    # Clean and combine datasets
    cleaned_df = clean_student_performance_data(mat, por)

    # Push metadata to XCom
    context['ti'].xcom_push(key='cleaned_shape', value=cleaned_df.shape)
    context['ti'].xcom_push(key='missing_values', value=cleaned_df.isnull().sum().sum())

    print(f"Data cleaning completed. Cleaned dataset shape: {cleaned_df.shape}")
    return "Data cleaning successful"


def task_feature_engineering(**context):
    """
    Task 3: Feature Engineering and ABT Creation
    Creates derived features (avg_prev_grade, grade_trend, high_absence) and target variable.

    Inputs:
        - data/cleaned/student_performance_clean.csv
    Outputs:
        - data/processed/abt_student_performance.csv
    Features Created:
        - avg_prev_grade: Mean of G1 and G2
        - grade_trend: Grade change from G1 to G3
        - high_absence: Binary indicator for high absences
        - target_pass: Binary classification target (1 if G3 >= 10)
    """
    from src.feature_engineering.features import build_analytical_base_table

    print("Starting feature engineering...")

    abt = build_analytical_base_table()

    # Push metadata to XCom
    context['ti'].xcom_push(key='abt_shape', value=abt.shape)
    context['ti'].xcom_push(key='abt_columns', value=list(abt.columns))

    print(f"Feature engineering completed. ABT shape: {abt.shape}")
    return "Feature engineering successful"


def task_load_to_postgres(**context):
    """
    Task 3.5: Load Data to PostgreSQL
    Loads both cleaned data and ABT into PostgreSQL database for data warehouse integration.

    Inputs:
        - data/cleaned/student_performance_clean.csv
        - data/processed/abt_student_performance.csv
    Outputs:
        - PostgreSQL tables: student_performance_cleaned, student_performance_abt
    """
    from src.data_ingestion.postgres_loader import load_cleaned_data_to_postgres, load_abt_to_postgres

    print("="*70)
    print("Task 3.5: Loading data to PostgreSQL...")
    print("="*70)

    # Load cleaned data
    rows_cleaned = load_cleaned_data_to_postgres()

    # Load ABT
    rows_abt = load_abt_to_postgres()

    # Push metadata to XCom
    context['ti'].xcom_push(key='postgres_cleaned_rows', value=rows_cleaned)
    context['ti'].xcom_push(key='postgres_abt_rows', value=rows_abt)

    print(f"\n✓ PostgreSQL loading completed")
    print(f"  - Cleaned data: {rows_cleaned} rows")
    print(f"  - ABT data: {rows_abt} rows")

    return "PostgreSQL loading successful"


def task_model_training(**context):
    """
    Task 4: Model Training
    Trains multiple models including:
        - Multi-source Random Forest (all features)
        - Multi-source Logistic Regression (all features)
        - Single-source Random Forest (G1, G2 only)
        - Single-source Logistic Regression (G1, G2 only)
        - Linear Regression for G3 prediction

    Inputs:
        - data/processed/abt_student_performance.csv
    Outputs:
        - src/modeling/models/rf_pass_prediction.pkl
        - src/modeling/models/linear_regression_model.pkl
    """
    from src.modeling.train import (
        load_abt, prepare_data, split_data,
        train_multi_source_models, train_academic_only_models,
        train_regression_model
    )

    print("Starting model training...")

    # Load and prepare data
    abt = load_abt()
    X, y = prepare_data(abt)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train multi-source models (all features)
    print("Training multi-source models...")
    multi_models = train_multi_source_models(X_train, X_test, y_train, y_test)

    # Train academic-only models (G1, G2 only)
    print("Training academic-only models...")
    academic_models = train_academic_only_models(X_train, X_test, y_train, y_test)

    # Train regression model
    print("Training regression model...")
    reg_model, reg_metrics = train_regression_model(X_train, X_test, y_train, y_test)

    # Push model performance to XCom
    context['ti'].xcom_push(key='rf_accuracy', value=multi_models['random_forest']['test_accuracy'])
    context['ti'].xcom_push(key='lr_accuracy', value=multi_models['logistic_regression']['test_accuracy'])

    print("Model training completed successfully")
    return "Model training successful"


def task_model_evaluation(**context):
    """
    Task 5: Model Evaluation and Fairness Analysis
    Evaluates model performance, calculates feature importance, and analyzes fairness.

    Inputs:
        - src/modeling/models/rf_pass_prediction.pkl
        - data/processed/abt_student_performance.csv
    Outputs:
        - figures/model_comparison.csv
        - figures/feature_importance_rf_full.csv
        - figures/fairness_by_sex.csv
        - figures/fairness_by_Medu.csv
        - Plus various PDF visualizations
    Fairness Metrics:
        - Analyzes performance across sex, maternal education, school support, family support
        - Calculates demographic parity and equal opportunity differences
    """
    from src.evaluation.metrics import (
        calculate_permutation_importance,
        calculate_fairness_metrics,
        compare_models
    )
    from src.modeling.train import load_abt, prepare_data, split_data, load_model

    print("Starting model evaluation...")

    # Load data and models
    abt = load_abt()
    X, y = prepare_data(abt)
    X_train, X_test, y_train, y_test = split_data(X, y)

    rf_model = load_model('src/modeling/models/rf_pass_prediction.pkl')

    # Calculate permutation importance
    print("Calculating feature importance...")
    fi_df = calculate_permutation_importance(rf_model, X_test, y_test)

    # Calculate fairness metrics
    print("Calculating fairness metrics...")
    fairness = calculate_fairness_metrics(rf_model, X_test, y_test)

    # Push evaluation results to XCom
    context['ti'].xcom_push(key='top_features', value=fi_df.head(5)['feature'].tolist())
    context['ti'].xcom_push(key='fairness_analysis', value='Completed')

    print("Model evaluation completed successfully")
    return "Model evaluation successful"


def task_generate_figures(**context):
    """
    Task 6: Generate Figures and Tables
    Generates all required visualizations and tables for the research questions.

    Outputs:
        - figures/RQ1_Fig*.pdf (Model comparison visualizations - 4 figures)
        - figures/RQ2_Fig*.pdf (Parental education impact - 5 figures)
        - figures/RQ3_Fig*.pdf (Fairness analysis - 4 figures)
        - figures/RQ4_Fig*.pdf (Feature importance - 6 figures)
        - tables/RQ1_Table1.xlsx (Model performance metrics)
        - tables/RQ3_Table1.xlsx (Fairness metrics)
    """
    import os

    print("="*70)
    print("Task 6: Generating all 19 figures and 2 tables...")
    print("="*70)

    # Ensure output directories exist
    os.makedirs('figures', exist_ok=True)
    os.makedirs('tables', exist_ok=True)

    # Import and run the main figure generation script
    from src.run_simple_analysis import main as generate_all_figures
    generate_all_figures()

    print("\n" + "="*70)
    print("All 19 figures and 2 tables generated successfully!")
    print("="*70)

    return "Figure generation successful"


def task_generate_shap(**context):
    """
    Task 7: Generate SHAP Visualization
    Generates SHAP-based feature importance visualization for RQ4_Fig6.

    Outputs:
        - figures/RQ4_Fig6.pdf (SHAP beeswarm plot, replacing permutation importance)
    """
    import os
    import subprocess

    print("="*70)
    print("Task 7: Generating SHAP visualization for RQ4_Fig6...")
    print("="*70)

    # Ensure figures directory exists
    os.makedirs('figures', exist_ok=True)

    # Run SHAP generation using the dedicated Python 3.12 virtual environment
    script_path = os.path.join(project_root, 'generate_shap_with_py312.sh')
    result = subprocess.run([script_path], cwd=project_root, capture_output=True, text=True)

    if result.returncode != 0:
        print("\nWARNING: SHAP generation failed. RQ4_Fig6 will use permutation importance fallback.")
        print(f"Error: {result.stderr}")
    else:
        print("\n" + "="*70)
        print("SHAP visualization generated successfully!")
        print("="*70)

    return "SHAP generation completed"


def task_pipeline_completion(**context):
    """
    Task 8: Pipeline Completion and Summary
    Summarizes pipeline execution and prints key metrics.
    """
    ti = context['ti']

    # Pull results from previous tasks
    mat_shape = ti.xcom_pull(task_ids='data_ingestion', key='mat_shape')
    por_shape = ti.xcom_pull(task_ids='data_ingestion', key='por_shape')
    cleaned_shape = ti.xcom_pull(task_ids='data_cleaning', key='cleaned_shape')
    abt_shape = ti.xcom_pull(task_ids='feature_engineering', key='abt_shape')
    rf_accuracy = ti.xcom_pull(task_ids='model_training', key='rf_accuracy')
    top_features = ti.xcom_pull(task_ids='model_evaluation', key='top_features')

    print("\n" + "="*60)
    print("STUDENT PERFORMANCE PREDICTION PIPELINE - EXECUTION SUMMARY")
    print("="*60)
    print(f"\n1. Data Ingestion:")
    print(f"   - Math dataset shape: {mat_shape}")
    print(f"   - Portuguese dataset shape: {por_shape}")
    print(f"\n2. Data Cleaning:")
    print(f"   - Cleaned dataset shape: {cleaned_shape}")
    print(f"\n3. Feature Engineering:")
    print(f"   - ABT shape: {abt_shape}")
    print(f"\n4. Model Training:")
    print(f"   - Random Forest test accuracy: {rf_accuracy:.4f}")
    print(f"\n5. Model Evaluation:")
    print(f"   - Top 5 features: {top_features}")
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60 + "\n")

    return "Pipeline completed successfully"


# Define tasks using PythonOperator
t1_ingestion = PythonOperator(
    task_id='data_ingestion',
    python_callable=task_data_ingestion,
    dag=dag,
    doc_md="""
    ### Data Ingestion Task
    Downloads student performance datasets from Kaggle.
    - Source: Student Performance Dataset (Kaggle)
    - Outputs: Math and Portuguese CSV files
    """
)

t2_cleaning = PythonOperator(
    task_id='data_cleaning',
    python_callable=task_data_cleaning,
    dag=dag,
    doc_md="""
    ### Data Cleaning Task
    Combines datasets and handles missing values.
    - Merges Math and Portuguese datasets
    - Imputes missing values
    """
)

t3_features = PythonOperator(
    task_id='feature_engineering',
    python_callable=task_feature_engineering,
    dag=dag,
    doc_md="""
    ### Feature Engineering Task
    Creates derived features and analytical base table.
    - Creates: avg_prev_grade, grade_trend, high_absence, target_pass
    """
)

t3_5_postgres = PythonOperator(
    task_id='load_to_postgres',
    python_callable=task_load_to_postgres,
    dag=dag,
    doc_md="""
    ### Load to PostgreSQL Task
    Loads cleaned data and ABT into PostgreSQL database.
    - Tables: student_performance_cleaned, student_performance_abt
    - Enables data warehouse integration
    """
)

t4_training = PythonOperator(
    task_id='model_training',
    python_callable=task_model_training,
    dag=dag,
    doc_md="""
    ### Model Training Task
    Trains multiple classification and regression models.
    - Random Forest, Logistic Regression, Linear Regression
    """
)

t5_evaluation = PythonOperator(
    task_id='model_evaluation',
    python_callable=task_model_evaluation,
    dag=dag,
    doc_md="""
    ### Model Evaluation Task
    Evaluates models and calculates fairness metrics.
    - Feature importance analysis
    - Fairness across demographic groups
    """
)

t6_figures = PythonOperator(
    task_id='generate_figures',
    python_callable=task_generate_figures,
    dag=dag,
    doc_md="""
    ### Generate Figures Task
    Creates all visualizations and tables for research questions.
    - Generates 19 PDF figures (RQ1-RQ4)
    - Generates 2 XLSX tables
    """
)

t7_shap = PythonOperator(
    task_id='generate_shap',
    python_callable=task_generate_shap,
    dag=dag,
    doc_md="""
    ### Generate SHAP Visualization Task
    Creates SHAP-based feature importance visualization.
    - Replaces RQ4_Fig6 with SHAP beeswarm plot
    - Uses Python 3.12 virtual environment
    """
)

t8_completion = PythonOperator(
    task_id='pipeline_completion',
    python_callable=task_pipeline_completion,
    dag=dag,
    doc_md="""
    ### Pipeline Completion Task
    Summarizes pipeline execution and prints metrics.
    """
)

# Define task dependencies
# Main pipeline (uses CSV files for ML pipeline)
t1_ingestion >> t2_cleaning >> t3_features >> t4_training >> t5_evaluation >> t6_figures >> t7_shap >> t8_completion

# Parallel branch: Load data to PostgreSQL for data warehouse (independent, doesn't block main pipeline)
t3_features >> t3_5_postgres

"""
DAG Structure:

    [Data Ingestion]
           ↓
    [Data Cleaning]
           ↓
    [Feature Engineering]
           ↓                    ↘
    [Model Training]          [Load to PostgreSQL] ← Parallel branch for data warehouse
           ↓
    [Model Evaluation]
           ↓
    [Generate Figures]
           ↓
    [Generate SHAP]
           ↓
    [Pipeline Completion]

Main Pipeline (left branch):
- Uses CSV files throughout
- Each stage completes successfully before moving to the next
- Maintains data quality and model reproducibility

Parallel Branch (right branch):
- PostgreSQL loading runs independently after feature engineering
- Loads data to warehouse without blocking ML pipeline
- Useful for BI tools, reporting, and data warehouse integration

Outputs:
- PostgreSQL Database: student_performance_cleaned, student_performance_abt tables (parallel branch)
- 19 PDF figures in figures/ (RQ1: 4, RQ2: 5, RQ3: 4, RQ4: 6)
- 2 XLSX tables in tables/ (RQ1_Table1.xlsx, RQ3_Table1.xlsx)
- 2 trained models in src/modeling/models/
"""
