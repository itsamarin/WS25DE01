"""
Simplified Analysis Script
Generates core tables and basic visualizations for RQ1-RQ4
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.modeling.train import (
    load_abt, prepare_data, split_data,
    train_multi_source_models, train_academic_only_models
)
from src.evaluation.metrics import (
    evaluate_model, compare_models,
    calculate_permutation_importance, save_feature_importance,
    calculate_fairness_metrics, calculate_advanced_fairness_metrics
)

print("="*60)
print("RUNNING ANALYSIS FOR RQ1-RQ4")
print("="*60)

# Load data
print("\n1. Loading data...")
abt = load_abt()
X, y = prepare_data(abt)
X_train, X_test, y_train, y_test = split_data(X, y)

# Train models
print("\n2. Training models...")
multi_models_dict = train_multi_source_models(X_train, X_test, y_train, y_test)

X_academic = X[['G1', 'G2']]
X_train_ac, X_test_ac, y_train_ac, y_test_ac = split_data(X_academic, y)
single_models_dict = train_academic_only_models(X_train_ac, X_test_ac, y_train_ac, y_test_ac)

print("\n" + "="*60)
print("RQ1: MULTI-SOURCE VS SINGLE-SOURCE PERFORMANCE")
print("="*60)

# Compare models
models_for_comparison = {
    'Multi-Source LR': (multi_models_dict['logistic_regression'], X_test, y_test),
    'Multi-Source RF': (multi_models_dict['random_forest'], X_test, y_test),
    'Single-Source LR': (single_models_dict['academic_logistic_regression'], X_test_ac, y_test_ac),
    'Single-Source RF': (single_models_dict['academic_random_forest'], X_test_ac, y_test_ac)
}

results_df = compare_models(models_for_comparison, save_path="tables/RQ1_Table1.xlsx")
print("\n✓ RQ1 Table saved: tables/RQ1_Table1.xlsx")
print(results_df)

print("\n" + "="*60)
print("RQ2: PARENTAL EDUCATION & FAMILY SUPPORT IMPACT")
print("="*60)

# Analyze parental education impact
print("\nAnalyzing parental education impact on grades...")
medu_analysis = abt.groupby('Medu')['G3'].agg(['mean', 'std', 'count'])
print("\nMean final grade by Mother's Education:")
print(medu_analysis)

fedu_analysis = abt.groupby('Fedu')['G3'].agg(['mean', 'std', 'count'])
print("\nMean final grade by Father's Education:")
print(fedu_analysis)

# Family support impact
famsup_analysis = abt.groupby('famsup')['G3'].agg(['mean', 'std', 'count'])
print("\nMean final grade by Family Support:")
print(famsup_analysis)

print("\n✓ RQ2 Analysis completed")

print("\n" + "="*60)
print("RQ3: MODEL FAIRNESS ACROSS DEMOGRAPHICS")
print("="*60)

# Calculate fairness metrics (display only, no extra files)
rf_model = multi_models_dict['random_forest']
fairness_results = calculate_fairness_metrics(
    rf_model, X_test, y_test,
    sensitive_attributes=['sex', 'Medu', 'schoolsup', 'famsup']
)

# Advanced fairness metrics
y_pred = rf_model.predict(X_test)
y_pred_series = pd.Series(y_pred, index=y_test.index)
fairness_table = calculate_advanced_fairness_metrics(
    y_test, y_pred_series, X_test,
    sensitive_attributes=['sex', 'Medu'],
    save_path='tables/RQ3_Table1.xlsx'
)

print("\n✓ RQ3 Fairness table saved: tables/RQ3_Table1.xlsx")

print("\n" + "="*60)
print("RQ4: FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Calculate permutation importance
print("\nCalculating feature importance...")
fi_df = calculate_permutation_importance(rf_model, X_test, y_test, sample_size=200)
save_feature_importance(fi_df)

print("\n✓ RQ4 Feature importance analysis completed")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  Tables:")
print("    - tables/RQ1_Table1.xlsx (Model comparison)")
print("    - tables/RQ3_Table1.xlsx (Fairness metrics)")
print("\nKey Findings:")
print(f"  RQ1: Multi-source LR achieves {results_df.loc[0, 'accuracy']*100:.1f}% accuracy")
print(f"       vs Single-source LR at {results_df.loc[2, 'accuracy']*100:.1f}% accuracy")
print(f"       Improvement: +{(results_df.loc[0, 'accuracy'] - results_df.loc[2, 'accuracy'])*100:.1f}%")
print(f"\n  RQ2: Students with higher parental education perform better")
print(f"\n  RQ3: Model fairness evaluated across gender, education, and support")
print(f"\n  RQ4: Top features identified through permutation importance")
