""" Diabetes ML Model Training Pipeline """

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.diabetes_preprocessing import preprocess_diabetes_data
from src.models.advanced_trainer import AdvancedTrainer

def train_diabetes_model(data_path="data/raw/diabetes.csv"):
    print("="*70)
    print("DIABETES PREDICTION - MODEL TRAINING PIPELINE")
    print("="*70)
    
    # 1. Data Preprocessing
    print("1. Data Preprocessing...")
    X_train, X_test, y_train, y_test = preprocess_diabetes_data(data_path)
    
    # 2. Model Training
    print("2. Initializing Advanced Trainer...")
    trainer = AdvancedTrainer("diabetes", X_train, X_test, y_train, y_test)
    
    print("3. Running Training Pipeline...")
    best_model = trainer.run()
    
    print("="*70)
    print("DIABETES MODEL TRAINING COMPLETED!")
    print("="*70)
    return trainer

if __name__ == "__main__":
    train_diabetes_model()
