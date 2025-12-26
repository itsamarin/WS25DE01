"""
Visualization Module for Research Questions
Generates all figures (RQ1-RQ4) programmatically for reproducibility

This module contains functions to generate all PDF visualizations
used in the research paper. All figures are saved to the figures/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import permutation_importance
import os


def ensure_figures_dir():
    """Create figures directory if it doesn't exist."""
    os.makedirs("figures", exist_ok=True)
    os.makedirs("tables", exist_ok=True)


# ==============================================================================
# RQ1: Multi-Source vs Single-Source Performance
# ==============================================================================

def plot_rq1_fig1_model_comparison(results_df, save_path="figures/RQ1_Fig1.pdf"):
    """
    RQ1_Fig1: Model Performance Comparison - Multi-Source vs. Single-Source

    Args:
        results_df: DataFrame with columns [model, accuracy, precision, recall, f1]
        save_path: Path to save the figure
    """
    # Reshape data for plotting
    melted_df = results_df.melt(
        id_vars=['model'],
        value_vars=['accuracy', 'precision', 'recall', 'f1'],
        var_name='metric',
        value_name='score'
    )
    melted_df['score'] = melted_df['score'] * 100

    plt.figure(figsize=(12, 7))
    sns.barplot(x='metric', y='score', hue='model', data=melted_df, palette='husl')
    plt.title('RQ1_Fig1 Model Performance Comparison: Multi-Source vs. Single-Source', fontsize=16)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.ylim(80, 100)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved performance comparison plot to {save_path}")
    plt.close()
    print("RQ1_Fig1 Caption: Model Performance Comparison between Multi-Source and Single-Source predictions")


def plot_rq1_fig2_grade_scatter(abt, save_path="figures/RQ1_Fig2.pdf"):
    """
    RQ1_Fig2: G1 vs G2 Grades, Colored by Pass/Fail Status

    Args:
        abt: Analytical Base Table with G1, G2, target_pass columns
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='G1', y='G2', hue='target_pass', data=abt, palette='tab10', s=100, alpha=0.7)
    plt.title('RQ1_Fig2 G1 vs G2 Grades, Colored by Pass/Fail Status', fontsize=16)
    plt.xlabel('First Period Grade (G1)', fontsize=12)
    plt.ylabel('Second Period Grade (G2)', fontsize=12)
    plt.legend(title='Target Pass (1=Pass, 0=Fail)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved scatter plot to {save_path}")
    plt.close()
    print("RQ1_Fig2 Caption: Grade correlation visualization by pass/fail outcome")


def plot_rq1_fig3_improvement(multi_models, single_models, save_path="figures/RQ1_Fig3.pdf"):
    """
    RQ1_Fig3: Model Performance Improvement - Multi-Source over Single-Source

    Args:
        multi_models: Dict with multi-source model results
        single_models: Dict with single-source model results
        save_path: Path to save the figure
    """
    # Calculate improvement percentages
    improvement_data = []
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        # Random Forest improvement
        rf_improvement = (multi_models['random_forest']['test_' + metric] -
                         single_models['random_forest']['test_' + metric]) * 100
        improvement_data.append({
            'Metric': metric.capitalize(),
            'Improvement (%)': rf_improvement,
            'Model Type': 'Random Forest'
        })

        # Logistic Regression improvement
        lr_improvement = (multi_models['logistic_regression']['test_' + metric] -
                         single_models['logistic_regression']['test_' + metric]) * 100
        improvement_data.append({
            'Metric': metric.capitalize(),
            'Improvement (%)': lr_improvement,
            'Model Type': 'Logistic Regression'
        })

    melted_improvement_df = pd.DataFrame(improvement_data)

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Metric', y='Improvement (%)', hue='Model Type', data=melted_improvement_df, palette='viridis')
    plt.title('RQ1_Fig3 Model Performance Improvement: Multi-Source vs. Single-Source', fontsize=16)
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.legend(title='Model Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved performance improvement plot to {save_path}")
    plt.close()
    print("RQ1_Fig3 Caption: Model Performance improvement of Multi-Source over Single-Source predictions")


def plot_rq1_fig4_studytime_boxplot(df, save_path="figures/RQ1_Fig4.pdf"):
    """
    RQ1_Fig4: Final Grades by Study Time Categories

    Args:
        df: DataFrame with studytime and G3 columns
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='studytime', y='G3', data=df, palette='viridis', hue='studytime', legend=False)
    plt.title('RQ1_Fig4 Final Grades by Study Time Categories', fontsize=16)
    plt.xlabel('Study Time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)', fontsize=12)
    plt.ylabel('Final Grade (G3)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Box plot saved to {save_path}")
    plt.close()
    print("RQ1_Fig4 Caption: Grade and Study time correlation")


# ==============================================================================
# RQ2: Parental Education & Family Support
# ==============================================================================

def plot_rq2_fig1_parental_education(df, save_path="figures/RQ2_Fig1.pdf"):
    """
    RQ2_Fig1: Mean Grade by Parental Education Level

    Args:
        df: DataFrame with Medu, Fedu, and G3 columns
        save_path: Path to save the figure
    """
    # Combine mother and father education data
    medu_data = df.groupby('Medu')['G3'].mean().reset_index()
    medu_data['Education Type'] = "Mother's Education"
    medu_data.columns = ['Education Level', 'G3', 'Education Type']

    fedu_data = df.groupby('Fedu')['G3'].mean().reset_index()
    fedu_data['Education Type'] = "Father's Education"
    fedu_data.columns = ['Education Level', 'G3', 'Education Type']

    combined_data = pd.concat([medu_data, fedu_data], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Education Level', y='G3', hue='Education Type', data=combined_data, palette='colorblind')
    plt.title('RQ2_Fig1 Mean Grade by Parental Education Level')
    plt.xlabel('Education Level (0: none, 1: primary, 2: 5th to 9th, 3: secondary, 4: higher)')
    plt.ylabel('Mean Final Grade (G3)')
    plt.legend(title='Education Type')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ2_Fig1 Caption: Combined bar plot for Mean Grade by Parental Education")


def plot_rq2_fig2_resilience_drivers(X, y, save_path="figures/RQ2_Fig2.pdf"):
    """
    RQ2_Fig2: Key Drivers of Academic Resilience (Low Parental Ed Group)

    Args:
        X: Features for low parental education group
        y: Target variable
        save_path: Path to save the figure
    """
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis')
    plt.title('RQ2_Fig2 Key Drivers of Academic Resilience (Low Parental Ed Group)')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ2_Fig2 Caption: Key Drivers of Academic Resilience (Low Parental Education Group)")


def plot_rq2_fig3_grade_improvement_trend(df_supported, save_path="figures/RQ2_Fig3.pdf"):
    """
    RQ2_Fig3: Trend of Grade Improvement: Interaction of Mother's Education & Location

    Args:
        df_supported: DataFrame with Medu, address, and improved columns
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df_supported, x='Medu', y='improved', hue='address',
                  markers=["o", "s"], linestyles=["-", "--"], palette="Set1", capsize=.1)
    plt.title("RQ2_Fig3 Trend of Grade Improvement: Interaction of Mother's Education & Location")
    plt.xlabel("Mother's Education Level (0-4)")
    plt.ylabel('Probability of Improvement')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ2_Fig3 Caption: Trend of Grade Improvement: Interaction of Mother's Education & Location")


def plot_rq2_fig4_improvement_heatmap(df_supported, save_path="figures/RQ2_Fig4.pdf"):
    """
    RQ2_Fig4: Heatmap - Probability of Grade Improvement

    Args:
        df_supported: DataFrame with Medu, address, and improved columns
        save_path: Path to save the figure
    """
    plt.figure(figsize=(8, 6))
    pivot_table = df_supported.pivot_table(index='Medu', columns='address', values='improved', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2f')
    plt.title('RQ2_Fig4 Heatmap: Probability of Grade Improvement')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ2_Fig4 Caption: Heatmap showing Probability of Grade improvement in Urban and rural settings with Parent education")


def plot_rq2_fig5_parental_ed_by_address(df, save_path="figures/RQ2_Fig5.pdf"):
    """
    RQ2_Fig5: Mean G3 by Parental Education Level and Address

    Args:
        df: DataFrame with Medu, Fedu, address, and G3 columns
        save_path: Path to save the figure
    """
    # Prepare combined data
    medu_data = df.groupby(['Medu', 'address'])['G3'].mean().reset_index()
    medu_data['Parent'] = 'Mother'
    medu_data.columns = ['Education Level', 'address', 'G3', 'Parent']

    fedu_data = df.groupby(['Fedu', 'address'])['G3'].mean().reset_index()
    fedu_data['Parent'] = 'Father'
    fedu_data.columns = ['Education Level', 'address', 'G3', 'Parent']

    combined_data = pd.concat([medu_data, fedu_data], ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Education Level', y='G3', hue='Parent', style='address',
                 data=combined_data, markers=True, markersize=10, linewidth=2.5)
    plt.title('RQ2_Fig5 Mean G3 by Parental Education Level and Address', fontsize=16, fontweight='bold')
    plt.xlabel('Education Level (0: none, 1: primary, 2: 5th-9th, 3: secondary, 4: higher)', fontsize=12)
    plt.ylabel('Mean Final Grade (G3)', fontsize=12)
    plt.legend(title='Parent & Address')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ2_Fig5 caption: Combined single plot for Mean G3 by Parental Education and Address generated")


# ==============================================================================
# RQ3: Model Fairness
# ==============================================================================

def plot_rq3_fig1_fairness_gap(fairness_df, save_path="figures/RQ3_Fig1.pdf"):
    """
    RQ3_Fig1: Fairness Gap Relative to Overall Model Performance

    Args:
        fairness_df: DataFrame with group and f1_gap columns
        save_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.bar(fairness_df["group"].astype(str), fairness_df["f1_gap"])
    plt.axhline(0, linestyle="--")
    plt.title("RQ3_Fig1: Fairness Gap Relative to Overall Model Performance")
    plt.ylabel("F1-score difference from global mean")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ3_Fig1 Caption: Relative deviation of subgroup F1-scores from the overall model performance baseline.")


def plot_rq3_fig2_subgroup_heatmap(f1_scores, save_path="figures/RQ3_Fig2.pdf"):
    """
    RQ3_Fig2: Heatmap of subgroup F1-scores by sex and Medu

    Args:
        f1_scores: DataFrame with sex, Medu, and F1_Score columns
        save_path: Path to save the figure
    """
    pivot_f1 = f1_scores.pivot(index='sex', columns='Medu', values='F1_Score')

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot_f1,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        linewidths=.5,
        cbar_kws={'label': 'F1-Score'}
    )
    plt.title('RQ3_Fig2: Subgroup F1-Score Distribution by Demographics')
    plt.xlabel('Maternal Education (Medu)')
    plt.ylabel('Sex')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ3_Fig2 Caption: Heatmap showing subgroup distribution differences across demographics")


def plot_rq3_fig3_subgroup_performance(f1_scores, save_path="figures/RQ3_Fig3.pdf"):
    """
    RQ3_Fig3: Subgroup F1-Score Performance by Sex and Maternal Education

    Args:
        f1_scores: DataFrame with sex, Medu, and F1_Score columns
        save_path: Path to save the figure
    """
    plt.figure(figsize=(12, 7))
    sns.barplot(data=f1_scores, x='Medu', y='F1_Score', hue='sex',
                palette={'F': 'deeppink', 'M': 'darkgreen'})
    plt.title('RQ3_Fig3: Subgroup F1-Score Performance by Sex and Maternal Education')
    plt.xlabel('Maternal Education Level (Medu)')
    plt.ylabel('F1-Score')
    plt.ylim(0.0, 1.05)
    plt.legend(title='Sex')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ3_Fig3 Caption: Model performance disparities across demographic subgroups")


def plot_rq3_fig4_fairness_metrics(fairness_table, save_path="figures/RQ3_Fig4.pdf"):
    """
    RQ3_Fig4: Fairness Evaluation - Demographic Parity and Equal Opportunity

    Args:
        fairness_table: DataFrame with fairness metrics
        save_path: Path to save the figure
    """
    melted = fairness_table.melt(
        id_vars=['Sensitive Attribute'],
        value_vars=['Demographic Parity Difference', 'Equal Opportunity Difference'],
        var_name='Metric',
        value_name='Value'
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=melted,
        x='Sensitive Attribute',
        y='Value',
        hue='Metric',
        palette='Set2',
        errorbar=None
    )
    plt.title('RQ3_Fig4: Fairness Evaluation: Demographic Parity and Equal Opportunity')
    plt.xlabel('Sensitive Attribute')
    plt.ylabel('Metric Value')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.legend(title='Fairness Metric')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ3_Fig4 Caption: Comparison of fairness evaluation scores across frameworks")


# ==============================================================================
# RQ4: Feature Importance
# ==============================================================================

def plot_rq4_fig1_feature_stability(X, y, model, save_path="figures/RQ4_Fig1.pdf"):
    """
    RQ4_Fig1: Feature Stability Map Across Cross-Validation Folds

    Args:
        X: Feature matrix
        y: Target variable
        model: Trained model
        save_path: Path to save the figure
    """
    n_folds = 5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_importances = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)
        perm_result = permutation_importance(model, X_val_fold, y_val_fold,
                                            n_repeats=5, random_state=42, n_jobs=-1)
        fold_importances.append(perm_result.importances_mean)

    # Get top stable features
    mean_importances = np.mean(fold_importances, axis=0)
    top_indices = np.argsort(mean_importances)[-15:]

    heat_data = np.array(fold_importances)[:, top_indices].T
    heat_df = pd.DataFrame(heat_data,
                          index=[X.columns[i] for i in top_indices],
                          columns=[f"Fold {i+1}" for i in range(n_folds)])

    plt.figure(figsize=(10, 8))
    sns.heatmap(heat_df, cmap="viridis", linewidths=0.3, linecolor="white")
    plt.title("RQ4_Fig1: Feature Stability Map Across Cross-Validation Folds", pad=12)
    plt.xlabel("Cross-Validation Folds")
    plt.ylabel("Top Stable Features (Permutation Importance)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ4_Fig1 Caption: Heatmap of top predictive features ranked by stability across CV folds")


def plot_rq4_fig2_model_comparison(lr_metrics, rf_metrics, save_path="figures/RQ4_Fig2.pdf"):
    """
    RQ4_Fig2: Performance Comparison of Logistic Regression vs. Random Forest

    Args:
        lr_metrics: Dict with LR metrics
        rf_metrics: Dict with RF metrics
        save_path: Path to save the figure
    """
    performance_data = []
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        performance_data.append({'Metric': metric.capitalize(),
                                'Value': lr_metrics[metric],
                                'Model': 'Logistic Regression'})
        performance_data.append({'Metric': metric.capitalize(),
                                'Value': rf_metrics[metric],
                                'Model': 'Random Forest'})

    performance_df = pd.DataFrame(performance_data)

    plt.figure(figsize=(12, 7))
    sns.barplot(data=performance_df, x='Metric', y='Value', hue='Model', palette=['Blue', 'red'])
    plt.title('RQ4_Fig2: Performance Comparison of Logistic Regression vs. Random Forest Models')
    plt.xlabel('Evaluation Metric')
    plt.ylabel('Score')
    plt.ylim(0.8, 1.0)
    plt.legend(title='Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ4_Fig2 Caption: Performance comparison showing model effectiveness")


def plot_rq4_fig3_confusion_matrices(y_test, y_pred_lr, y_pred_rf, save_path="figures/RQ4_Fig3.pdf"):
    """
    RQ4_Fig3: Confusion Matrices for Logistic Regression and Random Forest

    Args:
        y_test: True labels
        y_pred_lr: Logistic Regression predictions
        y_pred_rf: Random Forest predictions
        save_path: Path to save the figure
    """
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    cm_rf = confusion_matrix(y_test, y_pred_rf)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        cm_lr,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=False,
        ax=axes[0],
        xticklabels=['Predicted Fail', 'Predicted Pass'],
        yticklabels=['Actual Fail', 'Actual Pass']
    )
    axes[0].set_title('RQ4_Fig3: Logistic Regression Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    sns.heatmap(
        cm_rf,
        annot=True,
        fmt='d',
        cmap='Greens',
        cbar=False,
        ax=axes[1],
        xticklabels=['Predicted Fail', 'Predicted Pass'],
        yticklabels=['Actual Fail', 'Actual Pass']
    )
    axes[1].set_title('Random Forest Confusion Matrix')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ4_Fig3 Caption: Confusion matrices showing model prediction performance")


def plot_rq4_fig4_runtime_comparison(train_times, pred_times, save_path="figures/RQ4_Fig4.pdf"):
    """
    RQ4_Fig4: Model Runtime Comparison - Training vs. Prediction

    Args:
        train_times: Dict with model training times
        pred_times: Dict with model prediction times
        save_path: Path to save the figure
    """
    runtime_data = [
        {'Metric': 'Training Time', 'Value': train_times['lr'], 'Model': 'Logistic Regression'},
        {'Metric': 'Training Time', 'Value': train_times['rf'], 'Model': 'Random Forest'},
        {'Metric': 'Prediction Time', 'Value': pred_times['lr'], 'Model': 'Logistic Regression'},
        {'Metric': 'Prediction Time', 'Value': pred_times['rf'], 'Model': 'Random Forest'},
    ]

    runtime_df = pd.DataFrame(runtime_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=runtime_df, x='Metric', y='Value', hue='Model', palette='rocket')
    plt.title('RQ4_Fig4: Model Runtime Comparison: Training vs. Prediction')
    plt.xlabel('Runtime Metric')
    plt.ylabel('Time (seconds)')
    plt.legend(title='Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ4_Fig4 Caption: Runtime comparison showing computational efficiency")


def plot_rq4_fig5_feature_importance(importance_df, top_n=15, save_path="figures/RQ4_Fig5.pdf"):
    """
    RQ4_Fig5: Top Predictive Features on Academic Performance (Random Forest)

    Args:
        importance_df: DataFrame with Feature and Importance columns
        top_n: Number of top features to display
        save_path: Path to save the figure
    """
    if len(importance_df) > top_n:
        importance_df_plot = importance_df.head(top_n)
    else:
        importance_df_plot = importance_df

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', hue='Feature',
                data=importance_df_plot, palette='viridis', legend=False)
    plt.title('RQ4_Fig5: Top Predictive Features on Academic Performance (Random Forest)')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()
    print("RQ4_Fig5 Caption: Top features contributing to academic performance prediction")


def plot_rq4_fig6_shap_importance(rf_trained, X_test, save_path="figures/RQ4_Fig6.pdf"):
    """
    RQ4_Fig6: SHAP Feature Importance for Academic Performance (Random Forest)

    Uses SHAP (SHapley Additive exPlanations) to show global feature importance.

    Args:
        rf_trained: Trained Random Forest pipeline
        X_test: Test feature set
        save_path: Path to save the figure
    """
    import shap

    print("Generating SHAP values...")

    # The RF model is inside a pipeline, so we access it via the 'model' step
    rf_model = rf_trained.named_steps['model']
    preprocessor = rf_trained.named_steps['preprocess']

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
    explainer = shap.TreeExplainer(rf_model)

    # Compute SHAP values for all classes first to inspect their structure
    all_shap_values = explainer.shap_values(X_test_processed)

    print(f"Type of all_shap_values: {type(all_shap_values)}")
    if isinstance(all_shap_values, list):
        print(f"Number of elements in all_shap_values list: {len(all_shap_values)}")
        for i, sv_arr in enumerate(all_shap_values):
            print(f"Shape of shap_values_list[{i}]: {sv_arr.shape}")
        # For binary classification, use class 1 (positive class)
        shap_values = all_shap_values[1] if len(all_shap_values) == 2 else all_shap_values[0]
    else:
        print(f"Shape of all_shap_values (if not list): {all_shap_values.shape}")
        # Handle 3D array for multi-class
        if len(all_shap_values.shape) == 3:
            shap_values = all_shap_values[:, :, 1]
        else:
            shap_values = all_shap_values

    # Add explicit assertion to catch the mismatch early
    assert X_test_processed.shape[1] == shap_values.shape[1], \
        f"FATAL ERROR: X_test_processed.shape[1] ({X_test_processed.shape[1]}) " \
        f"does not match shap_values.shape[1] ({shap_values.shape[1]})"

    assert X_test_processed.shape[1] == len(feature_names_out), \
        f"FATAL ERROR: X_test_processed.shape[1] ({X_test_processed.shape[1]}) " \
        f"does not match len(feature_names_out) ({len(feature_names_out)})"

    # Generate a SHAP summary plot (bar type)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_processed, feature_names=feature_names_out,
                     plot_type="bar", show=False)
    plt.title('RQ4_Fig6: SHAP Feature Importance for Academic Performance (Random Forest)')
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.close()

    print("RQ4_Fig6 Caption: SHAP summary plot illustrating the global importance of features in predicting academic performance (pass/fail) using the Random Forest model.")


# ==============================================================================
# Main execution for generating all figures
# ==============================================================================

if __name__ == "__main__":
    print("Visualization Module")
    print("Import this module and call individual plot functions to generate figures.")
    print("\nExample usage:")
    print("  from src.evaluation.visualizations import plot_rq1_fig1_model_comparison")
    print("  plot_rq1_fig1_model_comparison(results_df)")
