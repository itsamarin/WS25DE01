"""
Feature Engineering Module
Creates derived features and analytical base table (ABT) for modeling
"""

import pandas as pd
from typing import List


# Configuration constants
PASS_THRESHOLD = 10  # G3 >= 10 is considered passing
TARGET_COLS = ["G3", "target_pass"]


def create_average_previous_grade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create average of previous grades (G1 and G2).

    Args:
        df: DataFrame with G1 and G2 columns

    Returns:
        pd.DataFrame: DataFrame with avg_prev_grade column added
    """
    df["avg_prev_grade"] = df[["G1", "G2"]].mean(axis=1)
    return df


def create_grade_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create grade trend feature (change from G1 to G3).

    Args:
        df: DataFrame with G1 and G3 columns

    Returns:
        pd.DataFrame: DataFrame with grade_trend column added
    """
    df["grade_trend"] = df["G3"] - df["G1"]
    return df


def create_high_absence_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary indicator for high absence (above median).

    Args:
        df: DataFrame with absences column

    Returns:
        pd.DataFrame: DataFrame with high_absence column added
    """
    df["high_absence"] = (df["absences"] > df["absences"].median()).astype(int)
    return df


def create_target_pass(df: pd.DataFrame, pass_threshold: int = PASS_THRESHOLD) -> pd.DataFrame:
    """
    Create binary classification target for pass/fail.

    Args:
        df: DataFrame with G3 column
        pass_threshold: Minimum grade to be considered passing (default: 10)

    Returns:
        pd.DataFrame: DataFrame with target_pass column added
    """
    df["target_pass"] = (df["G3"] >= pass_threshold).astype(int)
    return df


def engineer_features(df: pd.DataFrame, pass_threshold: int = PASS_THRESHOLD) -> pd.DataFrame:
    """
    Apply all feature engineering transformations.

    Args:
        df: Cleaned DataFrame
        pass_threshold: Minimum grade to be considered passing (default: 10)

    Returns:
        pd.DataFrame: DataFrame with all engineered features
    """
    # Create derived features
    df = create_average_previous_grade(df)
    df = create_grade_trend(df)
    df = create_high_absence_indicator(df)

    # Create target variable
    df = create_target_pass(df, pass_threshold)

    print(f"\nFeature engineering completed.")
    print(f"New features created: avg_prev_grade, grade_trend, high_absence, target_pass")

    return df


def build_analytical_base_table(
    input_path: str = "data/cleaned/student_performance_clean.csv",
    output_path: str = "data/processed/abt_student_performance.csv",
    pass_threshold: int = PASS_THRESHOLD
) -> pd.DataFrame:
    """
    Build complete Analytical Base Table (ABT) from cleaned data.

    Args:
        input_path: Path to cleaned dataset (default: "data/cleaned/student_performance_clean.csv")
        output_path: Path to save ABT (default: "data/processed/abt_student_performance.csv")
        pass_threshold: Minimum grade to be considered passing (default: 10)

    Returns:
        pd.DataFrame: Complete ABT with all features and targets
    """
    # Load cleaned data
    print(f"Loading cleaned data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Cleaned data shape: {df.shape}")

    # Engineer features
    abt = engineer_features(df, pass_threshold)

    # Save ABT
    abt.to_csv(output_path, index=False)
    print(f"\nABT saved to: {output_path}")
    print(f"ABT shape: {abt.shape}")

    # Display basic statistics
    print(f"\nTarget distribution:")
    print(f"  Pass rate: {abt['target_pass'].mean():.2%}")
    print(f"  Average final grade (G3): {abt['G3'].mean():.2f}")
    print(f"  Average grade trend: {abt['grade_trend'].mean():.2f}")
    print(f"  High absence rate: {abt['high_absence'].mean():.2%}")

    return abt


def get_feature_groups() -> dict:
    """
    Get feature groups for different types of features.

    Returns:
        dict: Dictionary mapping feature group names to lists of column names
    """
    return {
        'grade_features': ['G1', 'G2', 'G3'],
        'derived_grade_features': ['avg_prev_grade', 'grade_trend'],
        'demographic_features': ['age', 'sex', 'address', 'famsize', 'Pstatus'],
        'parental_features': ['Medu', 'Fedu', 'Mjob', 'Fjob', 'guardian', 'famrel', 'famsup'],
        'school_features': ['school', 'reason', 'traveltime', 'studytime', 'failures', 'schoolsup', 'higher'],
        'behavioral_features': ['activities', 'absences', 'high_absence'],
        'target_features': ['target_pass'],
        'course_feature': ['course']
    }


def get_academic_only_features() -> List[str]:
    """
    Get list of academic-only features (G1, G2) for single-source modeling.

    Returns:
        List[str]: List of academic-only feature names
    """
    return ['G1', 'G2']


if __name__ == "__main__":
    # Build ABT
    abt = build_analytical_base_table()

    print("\n" + "="*60)
    print("Feature Engineering Summary")
    print("="*60)

    feature_groups = get_feature_groups()
    for group_name, features in feature_groups.items():
        available_features = [f for f in features if f in abt.columns]
        print(f"\n{group_name}: {len(available_features)} features")
        print(f"  {available_features}")

    print("\nAnalytical Base Table creation completed successfully!")
