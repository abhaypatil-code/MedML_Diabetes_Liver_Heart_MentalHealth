""" Mental Health ML Model Training Pipeline """

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.mental_health_preprocessing import preprocess_mental_health_data
from src.models.advanced_trainer import AdvancedTrainer

def train_mental_health_target(target, data_path="data/raw/mental_health.csv"):
    print(f"\n--- Training for Target: {target.upper()} ---")
    
    # 1. Data Preprocessing
    print("1. Data Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_mental_health_data(data_path, target=target)
    
    # 2. Model Training
    print("2. Initializing Advanced Trainer...")
    trainer = AdvancedTrainer(f"mental_health_{target}", X_train, X_test, y_train, y_test)
    
    print("3. Running Training Pipeline...")
    best_model = trainer.run()
    
    return trainer

def train_all_mental_health_targets(data_path="data/raw/mental_health.csv"):
    print("="*70)
    print("MENTAL HEALTH PREDICTION - MODEL TRAINING PIPELINE")
    print("="*70)
    
    targets = ["depressiveness", "anxiousness", "sleepiness"]
    trainers = {}
    
    for target in targets:
        try:
            trainers[target] = train_mental_health_target(target, data_path)
            print(f"✓ {target} model trained successfully.")
        except Exception as e:
            print(f"✗ Failed to train {target} model: {e}")
            
    print("="*70)
    print("MENTAL HEALTH MODELS TRAINING COMPLETED!")
    print("="*70)
    return trainers

if __name__ == "__main__":
    train_all_mental_health_targets()