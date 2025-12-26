#!/usr/bin/env python3
"""
Quick script to generate RQ4_Fig6 (SHAP feature importance)
"""

import sys
import os

# Ensure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_rq4_fig6():
    """Generate RQ4_Fig6 SHAP plot"""
    print("="*60)
    print("Generating RQ4_Fig6: SHAP Feature Importance")
    print("="*60)

    # Import required modules
    from src.modeling.train import load_abt, prepare_data, split_data, load_model
    from src.evaluation.visualizations import plot_rq4_fig6_shap_importance

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

        print("\n3. Generating SHAP values and plot...")
        print("   (This may take a few minutes...)")
        plot_rq4_fig6_shap_importance(rf_model, X_test)

        print("\n" + "="*60)
        print("✓ RQ4_Fig6 generated successfully!")
        print("  Location: figures/RQ4_Fig6.pdf")
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

    except ImportError as e:
        print(f"\n✗ ERROR: Missing required package")
        print(f"  {e}")
        print("\nPlease install SHAP:")
        print("  pip install shap")
        return False

    except Exception as e:
        print(f"\n✗ ERROR: An unexpected error occurred")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = generate_rq4_fig6()
    sys.exit(0 if success else 1)
