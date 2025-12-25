"""
Evaluation Metrics Module
Functions for model evaluation, comparison, and fairness analysis
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.inspection import permutation_importance
from typing import Dict, List, Tuple
from sklearn.pipeline import Pipeline


def evaluate_model(name: str, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Evaluate model performance with standard classification metrics.

    Args:
        name: Model name
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        dict: Dictionary with model name and metrics
    """
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def compare_models(
    models: Dict[str, Tuple[Pipeline, pd.DataFrame, pd.Series]],
    save_path: str = None
) -> pd.DataFrame:
    """
    Compare multiple models and optionally save results.

    Args:
        models: Dictionary mapping model name to (trained_model, X_test, y_test)
        save_path: Optional path to save comparison table

    Returns:
        pd.DataFrame: Comparison results
    """
    results = []

    for name, (model, X_test, y_test) in models.items():
        y_pred = model.predict(X_test)
        metrics = evaluate_model(name, y_test, y_pred)
        results.append(metrics)

    results_df = pd.DataFrame(results)

    if save_path:
        results_df.to_csv(save_path, index=False)
        print(f"Saved model comparison table to {save_path}")

    return results_df


def calculate_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sample_size: int = 300,
    n_repeats: int = 5,
    random_state: int = 42,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Calculate permutation feature importance.

    Args:
        model: Trained model pipeline
        X_test: Test features
        y_test: Test target
        sample_size: Number of samples to use (default: 300)
        n_repeats: Number of permutation repeats (default: 5)
        random_state: Random seed (default: 42)
        n_jobs: Number of parallel jobs (default: -1)

    Returns:
        pd.DataFrame: Feature importance DataFrame sorted by importance
    """
    print("=== Permutation Feature Importance ===")

    # Sample for speed
    sample_size = min(sample_size, len(X_test))
    X_test_sample = X_test.sample(n=sample_size, random_state=random_state)
    y_test_sample = y_test.loc[X_test_sample.index]

    # Calculate permutation importance
    perm_result = permutation_importance(
        model,
        X_test_sample,
        y_test_sample,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs
    )

    # Get feature names
    preprocess = model.named_steps["preprocess"]
    try:
        all_feature_names = preprocess.get_feature_names_out()
    except AttributeError:
        all_feature_names = np.array(
            [f"feat_{i}" for i in range(perm_result.importances_mean.shape[0])]
        )

    # Ensure lengths match
    L = min(len(all_feature_names), len(perm_result.importances_mean))
    all_feature_names = all_feature_names[:L]
    importance_mean = perm_result.importances_mean[:L]
    importance_std = perm_result.importances_std[:L]

    # Create DataFrame
    fi_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance_mean": importance_mean,
        "importance_std": importance_std
    }).sort_values("importance_mean", ascending=False)

    return fi_df


def save_feature_importance(
    fi_df: pd.DataFrame,
    full_path: str = "figures/feature_importance_rf_full.csv",
    top_n: int = 15,
    top_path: str = "figures/feature_importance_rf_top15.csv"
) -> None:
    """
    Save feature importance tables (full and top N).

    Args:
        fi_df: Feature importance DataFrame
        full_path: Path to save full importance table (default: "figures/feature_importance_rf_full.csv")
        top_n: Number of top features to save separately (default: 15)
        top_path: Path to save top N features (default: "figures/feature_importance_rf_top15.csv")
    """
    # Save full table
    fi_df.to_csv(full_path, index=False)

    # Save top N
    fi_top = fi_df.head(top_n)
    fi_top.to_csv(top_path, index=False)

    print(f"Saved feature importance tables to {full_path} and {top_path}")
    print(f"\nTop {top_n} features:")
    print(fi_top)


def subgroup_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    group_series: pd.Series,
    group_name: str
) -> pd.DataFrame:
    """
    Calculate fairness metrics for each subgroup.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        group_series: Series containing subgroup labels
        group_name: Name of the grouping attribute

    Returns:
        pd.DataFrame: Metrics for each subgroup
    """
    rows = []
    for g in sorted(group_series.dropna().unique()):
        mask = (group_series == g)
        if mask.sum() == 0:
            continue

        yt = y_true[mask]
        yp = y_pred[mask]

        rows.append({
            group_name: g,
            "n_samples": int(mask.sum()),
            "accuracy": accuracy_score(yt, yp),
            "precision": precision_score(yt, yp, zero_division=0),
            "recall": recall_score(yt, yp, zero_division=0),
            "f1": f1_score(yt, yp, zero_division=0),
        })

    return pd.DataFrame(rows)


def calculate_fairness_metrics(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    sensitive_attributes: List[str] = None,
    save_dir: str = "figures"
) -> Dict[str, pd.DataFrame]:
    """
    Calculate and save fairness metrics for sensitive attributes.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        sensitive_attributes: List of sensitive attribute names (default: ["sex", "Medu", "schoolsup", "famsup"])
        save_dir: Directory to save fairness tables (default: "figures")

    Returns:
        dict: Dictionary mapping attribute name to fairness DataFrame
    """
    if sensitive_attributes is None:
        sensitive_attributes = ["sex", "Medu", "schoolsup", "famsup"]

    # Get predictions
    y_pred = model.predict(X_test)

    # Calculate fairness for each attribute
    fairness_results = {}

    for attr in sensitive_attributes:
        if attr in X_test.columns:
            fair_df = subgroup_metrics(y_test, y_pred, X_test[attr], attr)
            fairness_results[attr] = fair_df

            # Save to file
            save_path = f"{save_dir}/fairness_by_{attr}.csv"
            fair_df.to_csv(save_path, index=False)
            print(f"Saved fairness metrics for '{attr}' to {save_path}")

            # Display
            print(f"\nFairness metrics for '{attr}':")
            print(fair_df)

    return fairness_results


def demographic_parity_difference(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_attribute: pd.Series,
    group_a,
    group_b
) -> float:
    """
    Calculate demographic parity difference between two groups.

    Demographic Parity: P(Y_pred=1 | A=group_a) - P(Y_pred=1 | A=group_b)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attribute: Series with group labels
        group_a: First group identifier
        group_b: Second group identifier

    Returns:
        float: Demographic parity difference (or np.nan if groups are empty)
    """
    pred_a = y_pred[sensitive_attribute == group_a]
    pred_b = y_pred[sensitive_attribute == group_b]

    if not pred_a.empty and not pred_b.empty:
        return pred_a.mean() - pred_b.mean()
    else:
        return np.nan


def equal_opportunity_difference(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_attribute: pd.Series,
    group_a,
    group_b
) -> float:
    """
    Calculate equal opportunity difference (TPR difference) between two groups.

    Equal Opportunity: P(Y_pred=1 | Y_true=1, A=group_a) - P(Y_pred=1 | Y_true=1, A=group_b)

    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive_attribute: Series with group labels
        group_a: First group identifier
        group_b: Second group identifier

    Returns:
        float: Equal opportunity difference (or np.nan if groups are empty)
    """
    # True positives for group A
    true_pos_a = y_pred[(y_true == 1) & (sensitive_attribute == group_a)]
    actual_pos_a = y_true[(y_true == 1) & (sensitive_attribute == group_a)]

    # True positives for group B
    true_pos_b = y_pred[(y_true == 1) & (sensitive_attribute == group_b)]
    actual_pos_b = y_true[(y_true == 1) & (sensitive_attribute == group_b)]

    # Calculate TPR for each group
    tpr_a = true_pos_a.mean() if not actual_pos_a.empty else np.nan
    tpr_b = true_pos_b.mean() if not actual_pos_b.empty else np.nan

    # Return difference
    if not np.isnan(tpr_a) and not np.isnan(tpr_b):
        return tpr_a - tpr_b
    else:
        return np.nan


def calculate_advanced_fairness_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    subgroup_data: pd.DataFrame,
    sensitive_attributes: List[str] = None,
    save_path: str = "figures/RQ3_Table1.csv"
) -> pd.DataFrame:
    """
    Calculate advanced fairness metrics (demographic parity, equal opportunity).

    Args:
        y_true: True labels
        y_pred: Predicted labels
        subgroup_data: DataFrame with sensitive attributes and predictions
        sensitive_attributes: List of attributes to analyze (default: ["sex", "Medu"])
        save_path: Path to save fairness table (default: "figures/RQ3_Table1.csv")

    Returns:
        pd.DataFrame: Fairness metrics table
    """
    if sensitive_attributes is None:
        sensitive_attributes = ["sex", "Medu"]

    fairness_results = []

    # Fairness across 'sex'
    if "sex" in sensitive_attributes and "sex" in subgroup_data.columns:
        sex_groups = subgroup_data['sex'].unique()
        if len(sex_groups) >= 2:
            group_a_sex = sex_groups[0]
            group_b_sex = sex_groups[1]

            dp_sex = demographic_parity_difference(
                y_true, y_pred, subgroup_data['sex'], group_a_sex, group_b_sex
            )
            eo_sex = equal_opportunity_difference(
                y_true, y_pred, subgroup_data['sex'], group_a_sex, group_b_sex
            )

            fairness_results.append({
                'Sensitive Attribute': 'sex',
                'Group Comparison': f'{group_a_sex} vs {group_b_sex}',
                'Demographic Parity Difference': dp_sex,
                'Equal Opportunity Difference': eo_sex
            })

    # Fairness across 'Medu'
    if "Medu" in sensitive_attributes and "Medu" in subgroup_data.columns:
        medu_groups = sorted(subgroup_data['Medu'].unique())
        if len(medu_groups) >= 2:
            group_a_medu = medu_groups[0]
            group_b_medu = medu_groups[-1]

            dp_medu = demographic_parity_difference(
                y_true, y_pred, subgroup_data['Medu'], group_a_medu, group_b_medu
            )
            eo_medu = equal_opportunity_difference(
                y_true, y_pred, subgroup_data['Medu'], group_a_medu, group_b_medu
            )

            fairness_results.append({
                'Sensitive Attribute': 'Medu',
                'Group Comparison': f'{group_a_medu} vs {group_b_medu}',
                'Demographic Parity Difference': dp_medu,
                'Equal Opportunity Difference': eo_medu
            })

    # Create DataFrame
    fairness_df = pd.DataFrame(fairness_results)

    # Save
    if save_path:
        fairness_df.to_csv(save_path, index=False)
        print(f"Saved advanced fairness metrics to {save_path}")

    print("\nAdvanced Fairness Metrics:")
    print(fairness_df)

    return fairness_df


if __name__ == "__main__":
    print("Evaluation Metrics Module")
    print("Use this module to evaluate models, calculate feature importance, and assess fairness.")
