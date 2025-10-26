#!/usr/bin/env python3
"""
Quick Start Setup Script
Validates environment and prepares for pipeline execution
"""

import sys
import subprocess
from pathlib import Path
import importlib.util

def check_python_version():
    """Check if Python version is adequate."""
    print("1. Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"   ✗ Python 3.7+ required. Found: {version.major}.{version.minor}")
        return False
    print(f"   ✓ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    print("\n2. Checking dependencies...")
    required = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn',
        'lightgbm', 'xgboost', 'imblearn', 'flask', 'catboost'
    ]
    
    missing = []
    for package in required:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(package)
            print(f"   ✗ {package}")
        else:
            print(f"   ✓ {package}")
    
    if missing:
        print(f"\n   Missing packages: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False
    return True

def check_directory_structure():
    """Verify required directories exist."""
    print("\n3. Checking directory structure...")
    required_dirs = [
        'data/raw',
        'src/preprocessing',
        'src/models',
        'src/api'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✓ {dir_path}/")
        else:
            print(f"   ✗ {dir_path}/ (missing)")
            all_exist = False
    
    return all_exist

def check_datasets():
    """Check which datasets are available."""
    print("\n4. Checking datasets...")
    datasets = {
        'diabetes.csv': False,
        'liver.csv': False,
        'mental_health.csv': False,
        'heart.csv': False
    }
    
    for dataset in datasets.keys():
        path = Path(f'data/raw/{dataset}')
        datasets[dataset] = path.exists()
        status = "✓" if datasets[dataset] else "✗"
        print(f"   {status} {dataset}")
    
    if not datasets['heart.csv']:
        print("\n   ⚠️  Note: heart.csv is missing")
        print("      Option 1: Add your own heart.csv to data/raw/")
        print("      Option 2: Run 'python generate_heart_data.py' for synthetic data")
        print("      Option 3: Pipeline will skip heart disease automatically")
    
    return datasets

def create_output_directories():
    """Create output directories if they don't exist."""
    print("\n5. Creating output directories...")
    output_dirs = [
        'models',
        'scalers',
        'results',
        'data/processed',
        'data/splits'
    ]
    
    for dir_path in output_dirs:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {dir_path}/")
    
    return True

def display_next_steps(datasets):
    """Display next steps for the user."""
    print("\n" + "="*70)
    print("SETUP COMPLETE")
    print("="*70)
    
    available = sum(1 for v in datasets.values() if v)
    print(f"\nDatasets available: {available}/4")
    
    if available >= 3:
        print("\n✓ You're ready to run the pipeline!")
        print("\nNext steps:")
        print("  1. Run complete pipeline:")
        print("     python main.py")
        print("\n  2. Or run individual pipelines:")
        if datasets['diabetes.csv']:
            print("     python -m src.models.diabetes_model")
        if datasets['liver.csv']:
            print("     python -m src.models.liver_model")
        if datasets['heart.csv']:
            print("     python -m src.models.heart_model")
        if datasets['mental_health.csv']:
            print("     python -m src.models.mental_health_model")
        
        print("\n  3. After training, start the API:")
        print("     python src/api/app.py")
    else:
        print("\n⚠️  Not enough datasets found")
        print("\nPlease add at least 3 datasets to data/raw/ to proceed")
    
    print("\n" + "="*70)

def main():
    """Run all setup checks."""
    print("="*70)
    print("DISEASE PREDICTION ML PIPELINE - SETUP VERIFICATION")
    print("="*70)
    
    checks = {
        'python': check_python_version(),
        'dependencies': check_dependencies(),
        'structure': check_directory_structure(),
    }
    
    datasets = check_datasets()
    checks['outputs'] = create_output_directories()
    
    print("\n" + "="*70)
    print("SETUP SUMMARY")
    print("="*70)
    for check, status in checks.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {check.title()}: {'PASS' if status else 'FAIL'}")
    
    all_passed = all(checks.values())
    
    if all_passed:
        display_next_steps(datasets)
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("Common fixes:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Ensure you're in the project root directory")
        print("  - Check that all source files are present")

if __name__ == "__main__":
    main()