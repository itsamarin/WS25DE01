#!/usr/bin/env python3
"""
Alternative SHAP-like visualization using Permutation Importance
This provides similar interpretability to SHAP but uses sklearn's permutation_importance
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_rq4_fig6_shap_alternative():
    """Generate RQ4_Fig6 using permutation importance (SHAP-like interpretation)"""
    print("="*60)
    print("Generating RQ4_Fig6: Permutation-Based Feature Importance")
    print("(SHAP-like interpretability)")
    print("="*60)

    # Import required modules
    from src.modeling.train import load_abt, prepare_data, split_data, load_model

    try:
        # Check if model exists
        model_path = 'models/rf_pass_prediction.pkl'
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            print("Please train the model first by running:")
            print("  python -m src.modeling.train")
            return False

        print("\n1. Loading trained Random Forest model...")
        rf_model = load_model(model_path)
        print("   ✓ Model loaded successfully")

        print("\n2. Loading and preparing data...")
        abt = load_abt()
        X, y = prepare_data(abt)
        X_train, X_test, y_train, y_test = split_data(X, y)
        print(f"   ✓ Data prepared (test set: {X_test.shape[0]} samples)")

        print("\n3. Computing permutation importance (SHAP-like)...")
        print("   (This may take a few minutes...)")

        # Compute permutation importance - this is conceptually similar to SHAP
        # It measures how much each feature contributes to prediction by shuffling it
        result = permutation_importance(
            rf_model, X_test, y_test,
            n_repeats=10,
            random_state=42,
            scoring='accuracy'
        )

        # Get feature names
        preprocessor = rf_model.named_steps['preprocess']
        feature_names = preprocessor.get_feature_names_out()

        # Get mean importance and sort
        importances_mean = result.importances_mean
        importances_std = result.importances_std

        # Sort by importance
        indices = np.argsort(importances_mean)[::-1]

        # Take top 20 features
        top_n = min(20, len(importances_mean))
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances_mean[top_indices]
        top_std = importances_std[top_indices]

        print(f"   ✓ Computed permutation importance for {len(feature_names)} features")
        print(f"   ✓ Showing top {top_n} most important features")

        print("\n4. Creating colorful SHAP-style visualization...")

        # Create horizontal bar plot with vibrant colors
        fig, ax = plt.subplots(figsize=(14, 9))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('#F8F9FA')

        y_pos = np.arange(len(top_features))

        # Create beautiful color gradient from cool to warm colors
        # Use a colormap that goes from purple -> blue -> green -> yellow -> orange -> red
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(top_features)))[::-1]

        # Alternative: Use individual vibrant colors for variety
        import matplotlib.cm as cm
        norm = plt.Normalize(vmin=0, vmax=len(top_features)-1)
        colors_gradient = cm.viridis(norm(range(len(top_features))))[::-1]

        # Plot from least to most important (bottom to top)
        bars = ax.barh(
            y_pos,
            top_importances[::-1],
            xerr=top_std[::-1],
            align='center',
            color=colors_gradient,
            alpha=0.85,
            edgecolor='#2C3E50',
            linewidth=1.2,
            error_kw={'elinewidth': 1.5, 'ecolor': '#34495E', 'capsize': 4, 'alpha': 0.7}
        )

        # Add value labels outside the bars on the right side
        max_width = max(top_importances[::-1])
        for i, (bar, val, std) in enumerate(zip(bars, top_importances[::-1], top_std[::-1])):
            # Position text after the error bar ends
            x_position = val + std + (max_width * 0.02)  # Small offset from error bar
            ax.text(x_position, bar.get_y() + bar.get_height()/2,
                   f'{val:.4f}',
                   ha='left', va='center', fontsize=9,
                   fontweight='bold', color='#2C3E50')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([top_features[i] for i in range(len(top_features)-1, -1, -1)],
                          fontsize=11, fontweight='500')
        ax.set_xlabel('Mean Permutation Importance\n(Impact on model accuracy when feature is shuffled)',
                     fontsize=12, fontweight='bold', color='#2C3E50')
        ax.set_title('RQ4_Fig6: Feature Importance for Academic Performance\n(SHAP-style Permutation Analysis)',
                    fontsize=15, fontweight='bold', pad=20, color='#1A1A1A')

        # Enhanced grid for better readability
        ax.grid(axis='x', alpha=0.4, linestyle='--', linewidth=0.8, color='#BDC3C7')
        ax.set_axisbelow(True)

        # Add subtle vertical line at x=0
        ax.axvline(x=0, color='#34495E', linewidth=1.5, linestyle='-', alpha=0.5)

        # Customize spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#7F8C8D')
        ax.spines['bottom'].set_color('#7F8C8D')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Add colorful note about methodology
        note_text = ('Note: Permutation importance measures feature importance by evaluating the decrease in model performance\n'
                    'when each feature is randomly shuffled. Higher values indicate more important features.\n'
                    'Error bars show standard deviation across 10 permutation rounds. Color gradient: Purple (least) → Yellow (most important)')
        ax.text(0.5, -0.13, note_text,
               transform=ax.transAxes,
               fontsize=9,
               ha='center',
               va='top',
               style='italic',
               color='#34495E',
               bbox=dict(boxstyle='round,pad=0.8', facecolor='#FCF3CF',
                        edgecolor='#F39C12', alpha=0.7, linewidth=2))

        plt.tight_layout()

        # Save figure
        save_path = 'figures/RQ4_Fig6.pdf'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"   ✓ Figure saved to {save_path}")

        print("\n" + "="*60)
        print("✓ RQ4_Fig6 generated successfully!")
        print("  Location: figures/RQ4_Fig6.pdf")
        print("  Method: Permutation Importance (SHAP-like)")
        print("  ")
        print("  This provides interpretability similar to SHAP by measuring")
        print("  how each feature impacts model predictions through permutation.")
        print("="*60)

        # Print top 5 features
        print("\nTop 5 Most Important Features:")
        for i in range(min(5, len(top_features))):
            idx = top_indices[i]
            print(f"  {i+1}. {feature_names[idx]}: {importances_mean[idx]:.4f} (±{importances_std[idx]:.4f})")

        return True

    except FileNotFoundError as e:
        print(f"\n✗ ERROR: Required file not found")
        print(f"  {e}")
        print("\nPlease ensure you have run the complete pipeline:")
        print("  1. python -m src.data_ingestion.loader")
        print("  2. python -m src.data_cleaning.cleaner")
        print("  3. python -m src.feature_engineering.features")
        print("  4. python -m src.modeling.train")
        return False

    except Exception as e:
        print(f"\n✗ ERROR: An unexpected error occurred")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_rq4_fig6_shap_alternative()
    sys.exit(0 if success else 1)
