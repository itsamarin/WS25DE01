"""
Preprocessing Module
Defines preprocessing pipelines for different model types
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import List, Tuple
import pandas as pd


def get_numeric_categorical_features(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identify numeric and categorical features from DataFrame.

    Args:
        X: Feature DataFrame

    Returns:
        tuple: (numeric_features, categorical_features) lists
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    return numeric_features, categorical_features


def create_multi_source_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Create preprocessing pipeline for multi-source data (all features).

    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names

    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    print("Multi-source preprocessor created (StandardScaler + OneHotEncoder)")
    return preprocessor


def create_academic_only_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing pipeline for academic-only data (G1, G2).

    Returns:
        ColumnTransformer: Preprocessing pipeline for academic features
    """
    academic_features_scaler = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[("num", academic_features_scaler, ['G1', 'G2'])],
        remainder='passthrough'
    )

    print("Academic-only preprocessor created (StandardScaler for G1, G2)")
    return preprocessor


def create_regression_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create preprocessing pipeline for regression models.

    Args:
        X: Feature DataFrame

    Returns:
        ColumnTransformer: Preprocessing pipeline for regression
    """
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    print(f"Regression preprocessor - Categorical columns: {list(categorical_cols)}")
    print(f"Regression preprocessor - Numeric columns: {list(numeric_cols)}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols),
        ]
    )

    return preprocessor
