"""
Data Cleaning Module
Combines, cleans, and preprocesses student performance datasets
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def add_course_labels(mat: pd.DataFrame, por: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add course labels to distinguish between Math and Portuguese datasets.

    Args:
        mat: Math course DataFrame
        por: Portuguese course DataFrame

    Returns:
        tuple: (math_df, portuguese_df) with course labels added
    """
    mat["course"] = "math"
    por["course"] = "portuguese"
    return mat, por


def combine_datasets(mat: pd.DataFrame, por: pd.DataFrame) -> pd.DataFrame:
    """
    Combine Math and Portuguese datasets into a single DataFrame.

    Args:
        mat: Math course DataFrame
        por: Portuguese course DataFrame

    Returns:
        pd.DataFrame: Combined dataset
    """
    df = pd.concat([mat, por], ignore_index=True)
    print(f"Combined shape: {df.shape}")
    return df


def inspect_data(df: pd.DataFrame, verbose: bool = True) -> None:
    """
    Perform basic data inspection.

    Args:
        df: DataFrame to inspect
        verbose: Whether to print detailed information (default: True)
    """
    if verbose:
        print("\nDataset Head:")
        print(df.head())
        print("\nDataset Info:")
        print(df.info())
        print("\nMissing values per column:")
        print(df.isna().sum())


def handle_empty_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace empty strings with pd.NA.

    Args:
        df: DataFrame to process

    Returns:
        pd.DataFrame: DataFrame with empty strings replaced
    """
    df = df.replace("", pd.NA)
    return df


def separate_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separate numeric and categorical columns.

    Args:
        df: DataFrame to analyze

    Returns:
        tuple: (numeric_columns, categorical_columns) lists
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")

    return numeric_cols, categorical_cols


def impute_numeric_columns(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Ensure numeric columns are numeric and impute missing values with median.

    Args:
        df: DataFrame to process
        numeric_cols: List of numeric column names

    Returns:
        pd.DataFrame: DataFrame with imputed numeric columns
    """
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def impute_categorical_columns(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Impute missing categorical values with 'unknown' placeholder.

    Args:
        df: DataFrame to process
        categorical_cols: List of categorical column names

    Returns:
        pd.DataFrame: DataFrame with imputed categorical columns
    """
    df[categorical_cols] = df[categorical_cols].fillna("unknown")
    return df


def save_cleaned_data(df: pd.DataFrame, output_path: str = "data/cleaned/student_performance_clean.csv") -> None:
    """
    Save cleaned dataset to CSV file.

    Args:
        df: Cleaned DataFrame
        output_path: Path to save the cleaned data (default: "data/cleaned/student_performance_clean.csv")
    """
    df.to_csv(output_path, index=False)
    print(f"\nCleaned dataset saved to: {output_path}")
    print(f"Final shape: {df.shape}")


def clean_student_performance_data(
    mat: pd.DataFrame,
    por: pd.DataFrame,
    output_path: str = "data/cleaned/student_performance_clean.csv",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Complete data cleaning pipeline.

    Args:
        mat: Math course DataFrame
        por: Portuguese course DataFrame
        output_path: Path to save cleaned data (default: "data/cleaned/student_performance_clean.csv")
        verbose: Whether to print detailed information (default: True)

    Returns:
        pd.DataFrame: Cleaned and combined dataset
    """
    # 1) Add course labels
    mat, por = add_course_labels(mat, por)

    # 2) Combine datasets
    df = combine_datasets(mat, por)

    # 3) Basic inspection
    if verbose:
        inspect_data(df, verbose=True)

    # 4) Handle empty strings
    df = handle_empty_strings(df)

    # 5) Separate column types
    numeric_cols, categorical_cols = separate_column_types(df)

    # 6) Impute numeric columns
    df = impute_numeric_columns(df, numeric_cols)

    # 7) Impute categorical columns
    df = impute_categorical_columns(df, categorical_cols)

    # 8) Save cleaned data
    save_cleaned_data(df, output_path)

    return df


if __name__ == "__main__":
    # Example usage
    import os

    # Load raw data
    mat_path = "data/raw/maths/Maths.csv"
    por_path = "data/raw/portuguese/Portuguese.csv"

    if os.path.exists(mat_path) and os.path.exists(por_path):
        mat = pd.read_csv(mat_path)
        por = pd.read_csv(por_path)

        # Clean data
        cleaned_df = clean_student_performance_data(mat, por)
        print("\nData cleaning completed successfully!")
    else:
        print("Raw data files not found. Please run data ingestion first.")
