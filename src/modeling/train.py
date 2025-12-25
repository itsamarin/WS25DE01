"""
Model Training Module
Defines and trains classification and regression models
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_squared_error, r2_score
)
from typing import Tuple, Dict, Any

from .preprocessing import (
    get_numeric_categorical_features,
    create_multi_source_preprocessor,
    create_academic_only_preprocessor,
    create_regression_preprocessor
)


# Model hyperparameters
RF_PARAMS = {
    'n_estimators': 300,
    'random_state': 42,
    'n_jobs': -1,
}

LR_PARAMS = {
    'max_iter': 1000,
}

SPLIT_PARAMS = {
    'test_size': 0.2,
    'random_state': 42,
}


def load_abt(abt_path: str = "data/processed/abt_student_performance.csv") -> pd.DataFrame:
    """
    Load Analytical Base Table.

    Args:
        abt_path: Path to ABT file

    Returns:
        pd.DataFrame: Loaded ABT
    """
    abt = pd.read_csv(abt_path)
    print(f"ABT shape: {abt.shape}")
    return abt


def prepare_data(
    abt: pd.DataFrame,
    target_col: str = "target_pass",
    drop_cols: list = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features (X) and target (y) from ABT.

    Args:
        abt: Analytical Base Table
        target_col: Name of target column (default: "target_pass")
        drop_cols: Columns to drop from features (default: ["G3", "target_pass"])

    Returns:
        tuple: (X, y) features and target
    """
    if drop_cols is None:
        drop_cols = ["G3", "target_pass"]

    X = abt.drop(columns=drop_cols)
    y = abt[target_col]

    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = SPLIT_PARAMS['test_size'],
    random_state: int = SPLIT_PARAMS['random_state'],
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Args:
        X: Features
        y: Target
        test_size: Proportion of test set (default: 0.2)
        random_state: Random seed (default: 42)
        stratify: Whether to stratify split (default: True)

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    stratify_param = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    return X_train, X_test, y_train, y_test


def create_classification_pipeline(
    preprocessor,
    model_type: str = "random_forest"
) -> Pipeline:
    """
    Create classification pipeline with preprocessor and model.

    Args:
        preprocessor: Preprocessing pipeline
        model_type: Type of model ("random_forest" or "logistic_regression")

    Returns:
        Pipeline: Complete classification pipeline
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(**RF_PARAMS)
    elif model_type == "logistic_regression":
        model = LogisticRegression(**LR_PARAMS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])

    return pipeline


def create_regression_pipeline(preprocessor) -> Pipeline:
    """
    Create regression pipeline with preprocessor and LinearRegression.

    Args:
        preprocessor: Preprocessing pipeline

    Returns:
        Pipeline: Complete regression pipeline
    """
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("regressor", LinearRegression())
    ])

    return pipeline


def eval_model(
    name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Pipeline:
    """
    Train and evaluate a classification model.

    Args:
        name: Model name for display
        model: Model pipeline to train
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target

    Returns:
        Pipeline: Trained model
    """
    # Train model
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print results
    print(f"\n{'='*16} {name} {'='*16}")
    print(f"Accuracy : {round(acc, 3)}")
    print(f"Precision: {round(prec, 3)}")
    print(f"Recall   : {round(rec, 3)}")
    print(f"F1-score : {round(f1, 3)}")
    print(f"\nClassification report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    return model


def train_multi_source_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict[str, Pipeline]:
    """
    Train multi-source classification models (Logistic Regression and Random Forest).

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        dict: Dictionary of trained models
    """
    # Get feature types
    numeric_features, categorical_features = get_numeric_categorical_features(X_train)

    # Create preprocessor
    preprocessor = create_multi_source_preprocessor(numeric_features, categorical_features)

    # Create pipelines
    log_reg_clf = create_classification_pipeline(preprocessor, "logistic_regression")
    rf_clf = create_classification_pipeline(preprocessor, "random_forest")

    # Train and evaluate
    log_reg_trained = eval_model("Logistic Regression", log_reg_clf, X_train, y_train, X_test, y_test)
    rf_trained = eval_model("Random Forest", rf_clf, X_train, y_train, X_test, y_test)

    return {
        "logistic_regression": log_reg_trained,
        "random_forest": rf_trained
    }


def train_academic_only_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Dict[str, Pipeline]:
    """
    Train academic-only classification models (G1, G2 features only).

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target

    Returns:
        dict: Dictionary of trained academic-only models
    """
    # Create preprocessor
    preprocessor = create_academic_only_preprocessor()

    # Create pipelines
    log_reg_clf = create_classification_pipeline(preprocessor, "logistic_regression")
    rf_clf = create_classification_pipeline(preprocessor, "random_forest")

    # Train and evaluate
    log_reg_trained = eval_model("Academic-Only Logistic Regression", log_reg_clf, X_train, y_train, X_test, y_test)
    rf_trained = eval_model("Academic-Only Random Forest", rf_clf, X_train, y_train, X_test, y_test)

    return {
        "academic_logistic_regression": log_reg_trained,
        "academic_random_forest": rf_trained
    }


def train_regression_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Pipeline:
    """
    Train regression model to predict final grade (G3).

    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training target (G3 values)
        y_test: Test target (G3 values)

    Returns:
        Pipeline: Trained regression pipeline
    """
    # Create preprocessor
    preprocessor = create_regression_preprocessor(X_train)

    # Build regression pipeline
    reg_pipe = create_regression_pipeline(preprocessor)

    # Train
    reg_pipe.fit(X_train, y_train)
    print("Regression model pipeline trained successfully.")

    # Evaluate
    y_pred_reg = reg_pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_reg)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred_reg)

    print("\nRegression Model Performance:")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    return reg_pipe


def save_model(model: Pipeline, filepath: str) -> None:
    """
    Save trained model to disk.

    Args:
        model: Trained model pipeline
        filepath: Path to save the model
    """
    joblib.dump(model, filepath)
    print(f"Saved model to {filepath}")


def load_model(filepath: str) -> Pipeline:
    """
    Load trained model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        Pipeline: Loaded model
    """
    model = joblib.load(filepath)
    print(f"Loaded model from {filepath}")
    return model


if __name__ == "__main__":
    # Example: Train all models
    print("="*60)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*60)

    # Load data
    abt = load_abt()

    # Prepare multi-source data
    X, y = prepare_data(abt, target_col="target_pass", drop_cols=["G3", "target_pass"])
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train multi-source models
    multi_models = train_multi_source_models(X_train, X_test, y_train, y_test)

    # Prepare academic-only data
    X_academic = abt[['G1', 'G2']]
    y_academic = abt['target_pass']
    X_academic_train, X_academic_test, y_academic_train, y_academic_test = split_data(
        X_academic, y_academic
    )
    print(f"Academic-only Train size: {X_academic_train.shape}, Test size: {X_academic_test.shape}")

    # Train academic-only models
    academic_models = train_academic_only_models(
        X_academic_train, X_academic_test, y_academic_train, y_academic_test
    )

    # Save best model
    save_model(multi_models["random_forest"], "models/rf_pass_prediction.pkl")

    print("\n" + "="*60)
    print("TRAINING REGRESSION MODEL")
    print("="*60)

    # Prepare regression data (predict G3)
    X_reg, y_reg = prepare_data(abt, target_col="G3", drop_cols=["G3", "target_pass"])
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = split_data(X_reg, y_reg, stratify=False)

    # Train regression model
    reg_model = train_regression_model(X_reg_train, X_reg_test, y_reg_train, y_reg_test)

    # Save regression model
    save_model(reg_model, "models/linear_regression_model.pkl")

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETED")
    print("="*60)
