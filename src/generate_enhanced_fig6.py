"""
Generate Enhanced RQ4_Fig6 with comprehensive feature importance
Since SHAP has dependency issues, we'll create an enhanced permutation importance visualization
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.modeling.train import load_model, load_abt, prepare_data, split_data
from src.evaluation.metrics import calculate_permutation_importance

def main():
    print("="*70)
    print("Generating Enhanced Feature Importance Visualization for RQ4_Fig6")
    print("="*70)

    # Load the trained Random Forest model
    print("\n[1/3] Loading trained Random Forest model...")
    rf_model = load_model('src/modeling/models/rf_pass_prediction.pkl')

    # Load and prepare data
    print("[2/3] Loading and preparing data...")
    abt = load_abt()
    X, y = prepare_data(abt, target_col="target_pass", drop_cols=["G3", "target_pass"])
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"  - Test set size: {X_test.shape}")

    # Calculate permutation importance with confidence intervals
    print("[3/3] Calculating permutation importance...")
    fi_df = calculate_permutation_importance(rf_model, X_test, y_test, n_repeats=10)

    # Create enhanced visualization
    print("  - Creating comprehensive visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Left panel: Top 20 features by importance
    top_features = fi_df.head(20)
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))

    ax1.barh(range(len(top_features)), top_features['importance_mean'],
             color=colors, edgecolor='black', linewidth=0.5)
    ax1.errorbar(top_features['importance_mean'], range(len(top_features)),
                 xerr=top_features['importance_std'], fmt='none', ecolor='black',
                 capsize=3, linewidth=1.5, alpha=0.7)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features['feature'], fontsize=10)
    ax1.set_xlabel('Permutation Importance Score', fontsize=12, fontweight='bold')
    ax1.set_title('Top 20 Features by Permutation Importance', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_yaxis()

    # Right panel: Feature importance with confidence intervals
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax2.errorbar(row['importance_mean'], i,
                     xerr=row['importance_std'], fmt='o', markersize=8,
                     ecolor='gray', capsize=5, linewidth=2, alpha=0.8,
                     color=colors[i])
    ax2.set_yticks(range(len(top_features)))
    ax2.set_yticklabels(top_features['feature'], fontsize=10)
    ax2.set_xlabel('Importance with Confidence Interval', fontsize=12, fontweight='bold')
    ax2.set_title('Feature Importance Uncertainty', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    ax2.invert_yaxis()

    # Main title
    fig.suptitle('RQ4_Fig6: Comprehensive Feature Importance Analysis for Academic Performance (Random Forest)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    save_path = "figures/RQ4_Fig6.pdf"
    plt.savefig(save_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Enhanced visualization saved to: {save_path}")
    print("="*70)
    print("COMPLETE!")
    print("="*70)

    # Print top 5 features
    print("\nTop 5 Most Important Features:")
    for idx, row in enumerate(top_features.head(5).itertuples(), 1):
        print(f"  {idx}. {row.feature}: {row.importance_mean:.4f} ± {row.importance_std:.4f}")

if __name__ == "__main__":
    main()
