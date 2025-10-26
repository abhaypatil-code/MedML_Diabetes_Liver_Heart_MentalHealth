import sys
from pathlib import Path

# Ensure project root is included
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.liver_model import train_liver_model
from src.models.heart_model import train_heart_model
from src.models.diabetes_model import train_diabetes_model
from src.models.mental_health_model import train_all_mental_health_targets

def main():
    print("Starting all model training pipelines...\n")

    print("Training Liver model pipeline...")
    liver_trainer = train_liver_model()

    print("\nTraining Heart model pipeline...")
    heart_trainer = train_heart_model()

    print("\nTraining Diabetes model pipeline...")
    diabetes_trainer = train_diabetes_model()

    print("\nTraining Mental Health models pipeline...")
    mental_health_trainers = train_all_mental_health_targets()

    print("\nAll model training completed successfully.")

if __name__ == "__main__":
    main()
