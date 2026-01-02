"""
Generate RQ4_Fig6 using SHAP Feature Importance
This script generates SHAP visualizations by disabling numba to avoid dependency issues.
"""

import sys
sys.path.insert(0, '.')

# Disable numba before importing SHAP
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.modeling.train import load_model, load_abt, prepare_data, split_data

def main():
    print("="*70)
    print("Generating SHAP Feature Importance Visualization for RQ4_Fig6")
    print("="*70)

    # Load the trained Random Forest model
    print("\n[1/4] Loading trained Random Forest model...")
    rf_model = load_model('src/modeling/models/rf_pass_prediction.pkl')

    # Load and prepare data
    print("[2/4] Loading and preparing data...")
    abt = load_abt()
    X, y = prepare_data(abt, target_col="target_pass", drop_cols=["G3", "target_pass"])
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"  - Test set size: {X_test.shape}")

    # Import SHAP with numba disabled
    print("[3/4] Initializing SHAP (with numba disabled)...")
    try:
        import shap
        print(f"  - SHAP version: {shap.__version__}")
    except Exception as e:
        print(f"  - ERROR: Failed to import SHAP: {e}")
        print("  - Falling back to generate_permutation_rq4fig6.py...")
        from src.generate_permutation_rq4fig6 import main as generate_rq4fig6
        generate_rq4fig6()
        return

    # Extract model and preprocessor from pipeline
    print("[4/4] Calculating SHAP values...")
    rf_classifier = rf_model.named_steps['model']
    preprocessor = rf_model.named_steps['preprocess']

    # Preprocess X_test data
    X_test_processed = preprocessor.transform(X_test)

    # Ensure X_test_processed is a dense array for SHAP
    if hasattr(X_test_processed, 'toarray'):
        X_test_processed = X_test_processed.toarray()

    # Get feature names after preprocessing
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback if get_feature_names_out doesn't exist
        try:
            numeric_features = preprocessor.named_transformers_['num'].feature_names_in_
            categorical_features = preprocessor.named_transformers_['cat'].feature_names_in_
            numeric_feature_names = preprocessor.named_transformers_['num'].get_feature_names_out(numeric_features)
            categorical_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
            feature_names_out = np.concatenate([numeric_feature_names, categorical_feature_names])
        except:
            feature_names_out = [f"feature_{i}" for i in range(X_test_processed.shape[1])]

    # Initialize SHAP TreeExplainer
    explainer = shap.TreeExplainer(rf_classifier)

    # Compute SHAP values
    print("  - Computing SHAP values (this may take a moment)...")
    all_shap_values = explainer.shap_values(X_test_processed)

    # Handle SHAP values for binary classification
    if isinstance(all_shap_values, list):
        # For binary classification, use class 1 (positive class)
        shap_values = all_shap_values[1] if len(all_shap_values) == 2 else all_shap_values[0]
    else:
        # Handle 3D array for multi-class
        if len(all_shap_values.shape) == 3:
            shap_values = all_shap_values[:, :, 1]
        else:
            shap_values = all_shap_values

    # Create SHAP visualization (beeswarm plot only)
    print("  - Creating SHAP visualization...")
    fig = plt.figure(figsize=(14, 10))

    # SHAP beeswarm plot showing feature impact distribution
    shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names_out,
                     show=False, max_display=20)

    plt.title('RQ4_Fig6: SHAP Value Distribution for Academic Performance (Random Forest)',
              fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save figure
    save_path = "figures/RQ4_Fig6.pdf"
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ SHAP visualization saved to: {save_path}")
    print("="*70)
    print("COMPLETE!")
    print("="*70)

    # Print caption
    print("\n📊 Figure Caption:")
    print("SHAP (SHapley Additive exPlanations) value distribution showing the top 20")
    print("most influential features for academic performance prediction. The beeswarm plot")
    print("displays how each feature's impact varies across all predictions, with position")
    print("on the x-axis showing feature impact magnitude and direction. Red indicates high")
    print("feature values, blue indicates low feature values. Features are ranked by mean")
    print("absolute SHAP value (importance).")

    # Print top 5 features by mean absolute SHAP
    print("\nTop 5 Most Important Features (by mean |SHAP|):")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:5]
    for idx, i in enumerate(top_indices, 1):
        print(f"  {idx}. {feature_names_out[i]}: {mean_abs_shap[i]:.4f}")

if __name__ == "__main__":
    main()
