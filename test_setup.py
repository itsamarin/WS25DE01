#!/usr/bin/env python3
"""
Test script to verify ES25DE01 project setup and compliance
Run this before submission to ensure everything is working correctly.
"""

import os
import sys
from pathlib import Path

def test_directory_structure():
    """Test that all required directories exist."""
    print("Testing directory structure...")
    required_dirs = [
        "dags",
        "src/data_ingestion",
        "src/data_cleaning",
        "src/feature_engineering",
        "src/modeling",
        "src/evaluation",
        "data/sample",
        "figures",
        "tables",
        "models"
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
            print(f"  ❌ Missing: {dir_path}")
        else:
            print(f"  ✅ Found: {dir_path}")

    if missing_dirs:
        print(f"\n❌ Missing {len(missing_dirs)} required directories")
        return False
    print("✅ All required directories exist\n")
    return True


def test_required_files():
    """Test that all required files exist."""
    print("Testing required files...")
    required_files = [
        "dags/student_performance_pipeline_dag.py",
        "src/__init__.py",
        "src/data_ingestion/__init__.py",
        "src/data_ingestion/loader.py",
        "src/data_cleaning/__init__.py",
        "src/data_cleaning/cleaner.py",
        "src/feature_engineering/__init__.py",
        "src/feature_engineering/features.py",
        "src/modeling/__init__.py",
        "src/modeling/preprocessing.py",
        "src/modeling/train.py",
        "src/evaluation/__init__.py",
        "src/evaluation/metrics.py",
        "src/evaluation/visualizations.py",
        "requirements.txt",
        "README.md",
        ".gitignore"
    ]

    missing_files = []
    for file_path in required_files:
        if not os.path.isfile(file_path):
            missing_files.append(file_path)
            print(f"  ❌ Missing: {file_path}")
        else:
            print(f"  ✅ Found: {file_path}")

    if missing_files:
        print(f"\n❌ Missing {len(missing_files)} required files")
        return False
    print("✅ All required files exist\n")
    return True


def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("Testing Python syntax...")
    py_files = []
    for root, dirs, files in os.walk('.'):
        # Skip __pycache__ and hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                py_files.append(os.path.join(root, file))

    syntax_errors = []
    for py_file in py_files:
        try:
            with open(py_file, 'r') as f:
                compile(f.read(), py_file, 'exec')
            print(f"  ✅ Valid syntax: {py_file}")
        except SyntaxError as e:
            syntax_errors.append((py_file, str(e)))
            print(f"  ❌ Syntax error in {py_file}: {e}")

    if syntax_errors:
        print(f"\n❌ Found {len(syntax_errors)} files with syntax errors")
        return False
    print(f"✅ All {len(py_files)} Python files have valid syntax\n")
    return True


def test_readme_content():
    """Test that README contains all mandatory sections."""
    print("Testing README.md content...")
    required_sections = [
        "Project Overview",
        "Dataset",
        "Research Questions",
        "Project Structure",
        "How to Run the Code",
        "How to Run the Airflow DAG",
        "Reproducibility"
    ]

    with open('README.md', 'r') as f:
        readme_content = f.read()

    missing_sections = []
    for section in required_sections:
        if section not in readme_content:
            missing_sections.append(section)
            print(f"  ❌ Missing section: {section}")
        else:
            print(f"  ✅ Found section: {section}")

    if missing_sections:
        print(f"\n❌ Missing {len(missing_sections)} required sections in README")
        return False
    print("✅ All mandatory README sections present\n")
    return True


def test_requirements():
    """Test that requirements.txt contains key dependencies."""
    print("Testing requirements.txt...")
    required_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "apache-airflow"
    ]

    with open('requirements.txt', 'r') as f:
        requirements = f.read().lower()

    missing_packages = []
    for package in required_packages:
        if package.lower() not in requirements:
            missing_packages.append(package)
            print(f"  ❌ Missing: {package}")
        else:
            print(f"  ✅ Found: {package}")

    if missing_packages:
        print(f"\n❌ Missing {len(missing_packages)} required packages")
        return False
    print("✅ All required packages in requirements.txt\n")
    return True


def test_gitignore():
    """Test that .gitignore contains essential patterns."""
    print("Testing .gitignore...")
    required_patterns = [
        "__pycache__",
        "*.pyc",
        ".DS_Store",
        "data/raw",
        "data/cleaned",
        "data/processed"
    ]

    with open('.gitignore', 'r') as f:
        gitignore = f.read()

    missing_patterns = []
    for pattern in required_patterns:
        if pattern not in gitignore:
            missing_patterns.append(pattern)
            print(f"  ❌ Missing pattern: {pattern}")
        else:
            print(f"  ✅ Found pattern: {pattern}")

    if missing_patterns:
        print(f"\n⚠️  Missing {len(missing_patterns)} recommended .gitignore patterns")
        print("   (This is a warning, not a critical error)\n")
        return True
    print("✅ All recommended .gitignore patterns present\n")
    return True


def test_figures_exist():
    """Test that figure files exist."""
    print("Testing figures...")
    expected_figures = 18  # RQ1: 4, RQ2: 5, RQ3: 4, RQ4: 5

    if not os.path.exists('figures'):
        print(f"  ⚠️  figures/ directory exists but no PDF files found")
        print(f"   (Figures should be generated by running the pipeline)\n")
        return True

    pdf_files = [f for f in os.listdir('figures') if f.endswith('.pdf')]
    print(f"  Found {len(pdf_files)} PDF figures")

    if len(pdf_files) < expected_figures:
        print(f"  ⚠️  Expected {expected_figures} figures, found {len(pdf_files)}")
        print(f"   (Run the pipeline to generate all figures)\n")
    else:
        print(f"  ✅ All expected figures present\n")

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("ES25DE01 PROJECT SETUP VERIFICATION")
    print("="*60 + "\n")

    tests = [
        ("Directory Structure", test_directory_structure),
        ("Required Files", test_required_files),
        ("Python Syntax", test_python_syntax),
        ("README Content", test_readme_content),
        ("Requirements.txt", test_requirements),
        (".gitignore", test_gitignore),
        ("Figures", test_figures_exist)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error running {test_name} test: {e}\n")
            results.append((test_name, False))

    # Summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")

    print("\n" + "="*60)
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("="*60)
        print("\n🎉 Your project is ready for submission!")
        return 0
    else:
        print(f"⚠️  {total - passed} TEST(S) FAILED ({passed}/{total} passed)")
        print("="*60)
        print("\n⚠️  Please fix the issues above before submission.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
