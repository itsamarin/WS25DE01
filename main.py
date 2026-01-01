"""
Student Performance Prediction - Main Pipeline
Standalone execution script (no Airflow required)

This script runs the complete student performance prediction pipeline:
1. Data Ingestion
2. Data Cleaning
3. Feature Engineering
4. Model Training
5. Model Evaluation

Author: ES25DE01 Project Team
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """
    Execute the complete student performance prediction pipeline.
    """
    print("="*70)
    print("STUDENT PERFORMANCE PREDICTION PIPELINE - STANDALONE EXECUTION")
    print("="*70)

    # Step 1: Data Ingestion
    print("\n[STEP 1/5] Data Ingestion")
    print("-" * 70)
    try:
        from src.data_ingestion.loader import ingest_data
        mat, por = ingest_data()
        print(f"✓ Data ingestion completed")
        print(f"  - Math dataset: {mat.shape}")
        print(f"  - Portuguese dataset: {por.shape}")
    except Exception as e:
        print(f"✗ Error during data ingestion: {e}")
        return

    # Step 2: Data Cleaning
    print("\n[STEP 2/5] Data Cleaning")
    print("-" * 70)
    try:
        from src.data_cleaning.cleaner import clean_student_performance_data
        cleaned_df = clean_student_performance_data(mat, por)
        print(f"✓ Data cleaning completed")
        print(f"  - Cleaned dataset shape: {cleaned_df.shape}")
        print(f"  - Missing values: {cleaned_df.isnull().sum().sum()}")
    except Exception as e:
        print(f"✗ Error during data cleaning: {e}")
        return

    # Step 3: Feature Engineering
    print("\n[STEP 3/5] Feature Engineering")
    print("-" * 70)
    try:
        from src.feature_engineering.features import build_analytical_base_table
        abt = build_analytical_base_table()
        print(f"✓ Feature engineering completed")
        print(f"  - ABT shape: {abt.shape}")
        print(f"  - Features created: avg_prev_grade, grade_trend, high_absence, target_pass")
    except Exception as e:
        print(f"✗ Error during feature engineering: {e}")
        return

    # Step 4: Model Training
    print("\n[STEP 4/5] Model Training")
    print("-" * 70)
    try:
        from src.modeling.train import (
            load_abt, prepare_data, split_data,
            train_multi_source_models, train_academic_only_models,
            train_regression_model, save_model
        )

        # Load ABT
        abt = load_abt()

        # 4a. Train classification models (multi-source)
        print("\n  [4a] Training multi-source classification models...")
        X, y = prepare_data(abt, target_col="target_pass", drop_cols=["G3", "target_pass"])
        X_train, X_test, y_train, y_test = split_data(X, y)
        multi_models = train_multi_source_models(X_train, X_test, y_train, y_test)

        # 4b. Train academic-only models
        print("\n  [4b] Training academic-only models (G1, G2 only)...")
        X_academic = abt[['G1', 'G2']]
        y_academic = abt['target_pass']
        X_academic_train, X_academic_test, y_academic_train, y_academic_test = split_data(
            X_academic, y_academic
        )
        academic_models = train_academic_only_models(
            X_academic_train, X_academic_test, y_academic_train, y_academic_test
        )

        # 4c. Train regression model
        print("\n  [4c] Training regression model (predict G3)...")
        X_reg, y_reg = prepare_data(abt, target_col="G3", drop_cols=["G3", "target_pass"])
        X_reg_train, X_reg_test, y_reg_train, y_reg_test = split_data(X_reg, y_reg, stratify=False)
        reg_model = train_regression_model(X_reg_train, X_reg_test, y_reg_train, y_reg_test)

        # Save models
        print("\n  Saving models...")
        os.makedirs("src/modeling/models", exist_ok=True)
        save_model(multi_models["random_forest"], "src/modeling/models/rf_pass_prediction.pkl")
        save_model(reg_model, "src/modeling/models/linear_regression_model.pkl")

        print("✓ Model training completed")

    except Exception as e:
        print(f"✗ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Model Evaluation
    print("\n[STEP 5/5] Model Evaluation")
    print("-" * 70)
    try:
        from src.evaluation.metrics import (
            calculate_permutation_importance,
            calculate_fairness_metrics
        )
        from src.modeling.train import load_model

        # Load the best model
        rf_model = load_model('src/modeling/models/rf_pass_prediction.pkl')

        # Calculate feature importance
        print("  Calculating permutation importance...")
        fi_df = calculate_permutation_importance(rf_model, X_test, y_test)
        print(f"  - Top 5 features: {fi_df.head(5)['feature'].tolist()}")

        # Calculate fairness metrics
        print("  Calculating fairness metrics...")
        fairness = calculate_fairness_metrics(rf_model, X_test, y_test)

        print("✓ Model evaluation completed")

    except Exception as e:
        print(f"✗ Error during model evaluation: {e}")
        import traceback
        traceback.print_exc()
        return

    # Pipeline Summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nOutputs:")
    print("  - data/processed/abt_student_performance.csv")
    print("  - src/modeling/models/rf_pass_prediction.pkl")
    print("  - src/modeling/models/linear_regression_model.pkl")
    print("  - figures/model_comparison.csv")
    print("  - figures/feature_importance_rf_full.csv")
    print("  - figures/fairness_by_*.csv")
    print("="*70)


if __name__ == "__main__":
    main()
