#!/usr/bin/env python3
"""
Fallback script to generate RQ4_Fig6 using sklearn feature importance
(when SHAP has compatibility issues)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_rq4_fig6_fallback():
    """Generate RQ4_Fig6 using scikit-learn feature importance as fallback"""
    print("="*60)
    print("Generating RQ4_Fig6: Feature Importance (Fallback Method)")
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

        print("\n3. Extracting feature importance from Random Forest...")

        # Extract the RandomForest model from the pipeline
        rf = rf_model.named_steps['model']
        preprocessor = rf_model.named_steps['preprocess']

        # Get feature names
        feature_names = preprocessor.get_feature_names_out()

        # Get feature importances
        importances = rf.feature_importances_

        # Sort by importance
        indices = np.argsort(importances)[::-1]

        # Take top 20 features for better visualization
        top_n = min(20, len(importances))
        top_indices = indices[:top_n]
        top_features = [feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]

        print(f"   ✓ Extracted importance for {len(feature_names)} features")
        print(f"   ✓ Showing top {top_n} most important features")

        print("\n4. Creating visualization...")
        plt.figure(figsize=(12, 8))

        # Create bar plot (sorted from least to most important for better readability)
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_importances[::-1], align='center', color='steelblue')
        plt.yticks(y_pos, [top_features[i] for i in range(len(top_features)-1, -1, -1)])
        plt.xlabel('Feature Importance (MDI)', fontsize=12)
        plt.title('RQ4_Fig6: Feature Importance for Academic Performance (Random Forest)',
                  fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Save figure
        save_path = 'figures/RQ4_Fig6.pdf'
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

        print(f"   ✓ Figure saved to {save_path}")

        print("\n" + "="*60)
        print("✓ RQ4_Fig6 generated successfully!")
        print("  Location: figures/RQ4_Fig6.pdf")
        print("  Method: RandomForest Mean Decrease in Impurity (MDI)")
        print("  Note: Using sklearn feature importance (SHAP unavailable)")
        print("="*60)

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
    success = generate_rq4_fig6_fallback()
    sys.exit(0 if success else 1)
