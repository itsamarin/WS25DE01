"""
Generate SHAP visualization for RQ4_Fig6
This script replaces the current RQ4_Fig6.pdf with SHAP feature importance
"""

import sys
sys.path.insert(0, '.')

from src.modeling.train import load_model, load_abt, prepare_data, split_data
from src.evaluation.visualizations import plot_rq4_fig6_shap_importance

def main():
    print("="*70)
    print("Generating SHAP Visualization for RQ4_Fig6")
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

    # Generate SHAP visualization
    print("[3/4] Generating SHAP values and visualization...")
    print("  (This may take a minute...)")
    plot_rq4_fig6_shap_importance(rf_model, X_test, save_path="figures/RQ4_Fig6_SHAP.pdf")

    print("\n[4/4] Complete!")
    print("="*70)
    print("SHAP visualization saved to: figures/RQ4_Fig6_SHAP.pdf")
    print("="*70)

    # Also save to RQ4_Fig6.pdf to replace the old one
    print("\nReplacing RQ4_Fig6.pdf with SHAP version...")
    import shutil
    shutil.copy("figures/RQ4_Fig6_SHAP.pdf", "figures/RQ4_Fig6.pdf")
    print("✓ RQ4_Fig6.pdf now contains SHAP visualization")

if __name__ == "__main__":
    main()
