import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is included
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.liver_model import train_liver_model
from src.models.heart_model import train_heart_model
from src.models.diabetes_model import train_diabetes_model
from src.models.mental_health_model import train_all_mental_health_targets

def main():
    print("=" * 80)
    print("DISEASE PREDICTION ML PIPELINE - COMPLETE TRAINING")
    print("=" * 80)
    print("\nThis script will train models for 4 disease datasets:")
    print("  1. Liver Disease")
    print("  2. Heart Disease")
    print("  3. Diabetes")
    print("  4. Mental Health (3 targets: Depression, Anxiety, Sleepiness)")
    print("\nEach pipeline includes:")
    print("  - Data visualization and exploration")
    print("  - Preprocessing and feature engineering")
    print("  - Train-test split")
    print("  - Model training and evaluation")
    print("  - Best model selection and saving")
    print("=" * 80)
    
    results = {
        "liver": None,
        "heart": None,
        "diabetes": None,
        "mental_health": None
    }
    
    # 1. Train Liver Disease Model
    print("\n" + "=" * 80)
    print("[1/4] LIVER DISEASE PREDICTION PIPELINE")
    print("=" * 80)
    try:
        liver_trainer = train_liver_model()
        results["liver"] = "SUCCESS"
        print("\n✓ Liver model training completed successfully!")
    except Exception as e:
        results["liver"] = f"FAILED: {str(e)}"
        print(f"\n✗ Liver model training failed: {str(e)}")
        print("Continuing with next pipeline...")

    # 2. Train Heart Disease Model
    print("\n" + "=" * 80)
    print("[2/4] HEART DISEASE PREDICTION PIPELINE")
    print("=" * 80)
    try:
        heart_trainer = train_heart_model()
        results["heart"] = "SUCCESS"
        print("\n✓ Heart model training completed successfully!")
    except FileNotFoundError:
        results["heart"] = "SKIPPED: Dataset not found (heart.csv)"
        print("\n⊘ Heart dataset not found. Skipping...")
    except Exception as e:
        results["heart"] = f"FAILED: {str(e)}"
        print(f"\n✗ Heart model training failed: {str(e)}")
        print("Continuing with next pipeline...")

    # 3. Train Diabetes Model
    print("\n" + "=" * 80)
    print("[3/4] DIABETES PREDICTION PIPELINE")
    print("=" * 80)
    try:
        diabetes_trainer = train_diabetes_model()
        results["diabetes"] = "SUCCESS"
        print("\n✓ Diabetes model training completed successfully!")
    except Exception as e:
        results["diabetes"] = f"FAILED: {str(e)}"
        print(f"\n✗ Diabetes model training failed: {str(e)}")
        print("Continuing with next pipeline...")

    # 4. Train Mental Health Models
    print("\n" + "=" * 80)
    print("[4/4] MENTAL HEALTH PREDICTION PIPELINE")
    print("=" * 80)
    try:
        mental_health_trainers = train_all_mental_health_targets()
        results["mental_health"] = "SUCCESS"
        print("\n✓ Mental health models training completed successfully!")
    except Exception as e:
        results["mental_health"] = f"FAILED: {str(e)}"
        print(f"\n✗ Mental health models training failed: {str(e)}")

    # Summary Report
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print("\nPipeline Results:")
    for pipeline, result in results.items():
        status_symbol = "✓" if result == "SUCCESS" else "⊘" if "SKIPPED" in result else "✗"
        print(f"  {status_symbol} {pipeline.upper()}: {result}")
    
    print("\n" + "=" * 80)
    print("Output Directories:")
    print("=" * 80)
    print("  - Trained Models: ./models/")
    print("  - Scalers & Encoders: ./scalers/")
    print("  - Results & Metrics: ./results/")
    print("  - Train/Test Splits: ./data/splits/")
    print("  - Processed Data: ./data/processed/")
    
    success_count = sum(1 for r in results.values() if r == "SUCCESS")
    total_count = len(results)
    
    print("\n" + "=" * 80)
    if success_count == total_count:
        print("ALL PIPELINES COMPLETED SUCCESSFULLY! 🎉")
    elif success_count > 0:
        print(f"PARTIAL SUCCESS: {success_count}/{total_count} pipelines completed")
    else:
        print("ALL PIPELINES FAILED. Please check error messages above.")
    print("=" * 80)

if __name__ == "__main__":
    main()