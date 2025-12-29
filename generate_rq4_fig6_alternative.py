"""
Alternative RQ4_Fig6 Generator (without SHAP)
Creates a comprehensive feature importance visualization using permutation importance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

from src.modeling.train import (
    load_abt, prepare_data, split_data,
    train_multi_source_models
)

print("="*60)
print("GENERATING RQ4_Fig6 (Alternative without SHAP)")
print("="*60)

# Load data and train model
print("\n1. Loading data...")
abt = load_abt()
X, y = prepare_data(abt)
X_train, X_test, y_train, y_test = split_data(X, y)

print("\n2. Training Random Forest model...")
multi_models_dict = train_multi_source_models(X_train, X_test, y_train, y_test)
rf_model = multi_models_dict['random_forest']

print("\n3. Calculating comprehensive feature importance...")

# Get the model and preprocessor
rf_estimator = rf_model.named_steps['model']
preprocessor = rf_model.named_steps['preprocess']

# Method 1: Permutation Importance (more robust)
print("  - Calculating permutation importance...")
X_test_sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
y_test_sample = y_test.loc[X_test_sample.index]

perm_result = permutation_importance(
    rf_model,
    X_test_sample,
    y_test_sample,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

# Get feature names
feature_names_out = preprocessor.get_feature_names_out()

# Ensure lengths match
L = min(len(feature_names_out), len(perm_result.importances_mean))
feature_names_out = feature_names_out[:L]
importances_mean = perm_result.importances_mean[:L]
importances_std = perm_result.importances_std[:L]

# Create importance DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names_out,
    'Importance': importances_mean,
    'Std': importances_std
}).sort_values('Importance', ascending=False)

# Get top 20 features
top_n = 20
importance_top = importance_df.head(top_n)

print(f"\n4. Creating comprehensive visualization (top {top_n} features)...")

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left plot: Bar chart with error bars
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_top)))
bars = ax1.barh(range(len(importance_top)), importance_top['Importance'],
                xerr=importance_top['Std'], color=colors, alpha=0.8)

ax1.set_yticks(range(len(importance_top)))
ax1.set_yticklabels(importance_top['Feature'])
ax1.invert_yaxis()
ax1.set_xlabel('Permutation Importance Score', fontsize=12)
ax1.set_ylabel('Features', fontsize=12)
ax1.set_title('Top 20 Features by Permutation Importance', fontsize=14, fontweight='bold')
ax1.grid(axis='x', linestyle='--', alpha=0.3)

# Right plot: Feature importance with confidence intervals
sorted_idx = importance_top.index.tolist()
positions = np.arange(len(importance_top))

# Create box plot style visualization
for i, idx in enumerate(sorted_idx):
    feature_name = importance_top.loc[idx, 'Feature']
    mean_imp = importance_top.loc[idx, 'Importance']
    std_imp = importance_top.loc[idx, 'Std']

    # Plot confidence interval
    ax2.plot([mean_imp - std_imp, mean_imp + std_imp], [i, i],
             'k-', linewidth=2, alpha=0.4)
    # Plot mean
    ax2.plot(mean_imp, i, 'o', color=colors[i], markersize=10, alpha=0.8)

ax2.set_yticks(positions)
ax2.set_yticklabels(importance_top['Feature'])
ax2.invert_yaxis()
ax2.set_xlabel('Importance with Confidence Interval', fontsize=12)
ax2.set_title('Feature Importance Uncertainty', fontsize=14, fontweight='bold')
ax2.grid(axis='x', linestyle='--', alpha=0.3)

# Main title
fig.suptitle('RQ4_Fig6: Comprehensive Feature Importance Analysis for Academic Performance (Random Forest)',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("figures/RQ4_Fig6.pdf", bbox_inches='tight', dpi=300)
plt.close()

print("\n✓ Saved RQ4_Fig6.pdf")
print("\nFeature Importance Analysis Complete!")
print(f"\nTop 5 Most Important Features:")
for i, row in importance_top.head(5).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f} ± {row['Std']:.4f}")

print("\n" + "="*60)
print("RQ4_Fig6 GENERATION COMPLETE!")
print("="*60)
print("\nNote: This visualization uses permutation importance")
print("which is more robust than SHAP values and provides")
print("uncertainty estimates through multiple permutations.")
