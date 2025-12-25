"""
Data Ingestion Module
Downloads and loads student performance datasets from Kaggle
"""

import os
import pandas as pd
import kagglehub
from pathlib import Path


def create_directory_structure(base_dir: str = ".") -> None:
    """
    Create the project folder structure for data, models, and figures.

    Args:
        base_dir: Base directory for the project (default: current directory)
    """
    directories = [
        "data/raw",
        "data/raw/maths",
        "data/raw/portuguese",
        "data/cleaned",
        "data/processed",
        "models",
        "figures"
    ]

    for directory in directories:
        path = Path(base_dir) / directory
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")


def download_student_performance_data() -> str:
    """
    Download student performance dataset from Kaggle.

    Returns:
        str: Path to the downloaded dataset directory
    """
    print("Downloading student performance dataset from Kaggle...")
    download_dir = kagglehub.dataset_download("whenamancodes/student-performance")
    print(f"Dataset downloaded to: {download_dir}")
    return download_dir


def load_and_save_datasets(download_dir: str, output_dir: str = "data/raw") -> tuple:
    """
    Load Excel datasets and save as CSV files.

    Args:
        download_dir: Directory containing the downloaded Excel files
        output_dir: Directory to save the CSV files (default: "data/raw")

    Returns:
        tuple: (math_df, portuguese_df) DataFrames
    """
    # Load Excel files
    print("Loading Excel files...")
    mat = pd.read_excel(os.path.join(download_dir, "Maths.csv"))
    por = pd.read_excel(os.path.join(download_dir, "Portuguese.csv"))

    # Save as CSV
    mat_path = os.path.join(output_dir, "maths", "Maths.csv")
    por_path = os.path.join(output_dir, "portuguese", "Portuguese.csv")

    mat.to_csv(mat_path, index=False)
    por.to_csv(por_path, index=False)

    print(f"Math dataset saved to: {mat_path}")
    print(f"Portuguese dataset saved to: {por_path}")
    print(f"Math dataset shape: {mat.shape}")
    print(f"Portuguese dataset shape: {por.shape}")

    return mat, por


def ingest_data(base_dir: str = ".") -> tuple:
    """
    Complete data ingestion pipeline: create directories, download, and load data.

    Args:
        base_dir: Base directory for the project (default: current directory)

    Returns:
        tuple: (math_df, portuguese_df) DataFrames
    """
    # Create directory structure
    create_directory_structure(base_dir)

    # Download dataset
    download_dir = download_student_performance_data()

    # Load and save datasets
    output_dir = os.path.join(base_dir, "data/raw")
    mat, por = load_and_save_datasets(download_dir, output_dir)

    return mat, por


if __name__ == "__main__":
    # Run the complete ingestion pipeline
    mat_df, por_df = ingest_data()
    print("\nData ingestion completed successfully!")
    print(f"Math dataset: {mat_df.shape[0]} rows, {mat_df.shape[1]} columns")
    print(f"Portuguese dataset: {por_df.shape[0]} rows, {por_df.shape[1]} columns")
