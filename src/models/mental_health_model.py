"""
Mental Health Prediction - Model Training Script

Multi-label classification for Depression, Anxiety, and Sleepiness
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set up project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import preprocessing and utility functions
from src.preprocessing.mental_health_preprocessing import preprocess_mental_health_data
from src.models.utils import (
    evaluate_model, cross_validate_model, plot_confusion_matrix,
    plot_roc_curve, save_model, save_results, compare_models
)

# Import ML models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Multi-label and metrics
from sklearn.multioutput import MultiOutputClassifier

# Imbalanced data handling
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

class MentalHealthModelTrainer:
    """Comprehensive model training for Mental Health prediction.

    Supports both single-label and multi-label classification.
    """

    def __init__(self, X_train, X_test, y_train, y_test, target='depressiveness'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target = target
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None

    def check_class_distribution(self):
        print("-" * 70)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("-" * 70)
        if isinstance(self.y_train, pd.DataFrame):
            print("Multi-label Classification")
            print(f"Targets: {list(self.y_train.columns)}")
            for col in self.y_train.columns:
                dist = self.y_train[col].value_counts()
                print(f"{col}: Class 0: {dist.get(0,0)} ({dist.get(0,0) / len(self.y_train):.2f}), "
                      f"Class 1: {dist.get(1,0)} ({dist.get(1,0) / len(self.y_train):.2f})")
        else:
            dist = self.y_train.value_counts()
            print(f"Single-label Classification: {self.target}")
            print(f"Class 0: {dist.get(0,0)} ({dist.get(0,0) / len(self.y_train):.2f}), "
                  f"Class 1: {dist.get(1,0)} ({dist.get(1,0) / len(self.y_train):.2f})")

    def train_baseline_models(self):
        print("-" * 70)
        print("TRAINING BASELINE MODELS")
        print("-" * 70)
        base_models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=200, class_weight='balanced'),
            "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=200),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1, n_estimators=200, class_weight='balanced'),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=200),
            "SVM": SVC(probability=True, random_state=42, class_weight='balanced'),
            "Naive Bayes": GaussianNB(),
        }
        for model_name, model in base_models.items():
            print("-" * 70)
            print(f"Training {model_name}")
            print("-" * 70)
            try:
                if isinstance(self.y_train, pd.DataFrame):
                    # Multi-label: wrap with MultiOutputClassifier
                    multi_model = MultiOutputClassifier(model)
                    multi_model.fit(self.X_train, self.y_train)
                    self.models[model_name] = multi_model
                    results = evaluate_model(multi_model, self.X_test, self.y_test, model_name)
                else:
                    model.fit(self.X_train, self.y_train)
                    self.models[model_name] = model
                    results = evaluate_model(model, self.X_test, self.y_test, model_name)
                self.results.append(results)
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        print("Baseline model training completed!")

    def train_with_smote(self):
        print("-" * 70)
        print("TRAINING MODELS WITH SMOTE")
        print("-" * 70)
        smote = SMOTE(random_state=42)
        if isinstance(self.y_train, pd.DataFrame):
            # For multi-label, apply SMOTE on each label independently (simple approach)
            X_resampled_list = []
            y_resampled_list = []
            for col in self.y_train.columns:
                X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train[col])
                X_resampled_list.append(X_resampled)
                y_resampled_list.append(y_resampled)
            # For this implementation, use original data as SMOTE for multilabel is complex
            print("SMOTE for multi-label targets: skipped advanced resampling due to complexity.")
            X_train_resampled = self.X_train
            y_train_resampled = self.y_train
        else:
            X_train_resampled, y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
            print(f"Original training size: {len(self.y_train)}, After SMOTE: {len(y_train_resampled)}")

        smote_models = {
            "Random Forest SMOTE": RandomForestClassifier(random_state=42, n_estimators=200),
            "XGBoost SMOTE": XGBClassifier(random_state=42, eval_metric='logloss', n_estimators=200),
            "LightGBM SMOTE": LGBMClassifier(random_state=42, verbose=-1, n_estimators=200),
        }

        for model_name, model in smote_models.items():
            print("-" * 70)
            print(f"Training {model_name}")
            print("-" * 70)
            if isinstance(self.y_train, pd.DataFrame):
                # Multi-label: wrap with MultiOutputClassifier
                multi_model = MultiOutputClassifier(model)
                multi_model.fit(self.X_train, self.y_train)  # Using original due to complexity
                self.models[model_name] = multi_model
                results = evaluate_model(multi_model, self.X_test, self.y_test, model_name)
            else:
                model.fit(X_train_resampled, y_train_resampled)
                self.models[model_name] = model
                results = evaluate_model(model, self.X_test, self.y_test, model_name)
            self.results.append(results)

        print("SMOTE-based training completed!")

    def tune_best_models(self):
        print("-" * 70)
        print("HYPERPARAMETER TUNING")
        print("-" * 70)
        sorted_results = sorted(self.results, key=lambda x: x['f1_score'], reverse=True)
        top3_models = [r['model_name'] for r in sorted_results[:3]]
        print("Top 3 models for tuning:", top3_models)

        param_grids = {
            "Random Forest": {
                'n_estimators': [200, 300],
                'max_depth': [15, 20, None],
                'min_samples_split': [2, 5],
            },
            "Random Forest SMOTE": {
                'n_estimators': [200, 300],
                'max_depth': [15, 20, None],
                'min_samples_split': [2, 5],
            },
            "XGBoost": {
                'n_estimators': [200, 300],
                'max_depth': [5, 7],
                'learning_rate': [0.01, 0.1],
            },
            "XGBoost SMOTE": {
                'n_estimators': [200, 300],
                'max_depth': [5, 7],
                'learning_rate': [0.01, 0.1],
            },
            "LightGBM": {
                'n_estimators': [200, 300],
                'max_depth': [10, 15],
                'learning_rate': [0.01, 0.1],
            },
            "LightGBM SMOTE": {
                'n_estimators': [200, 300],
                'max_depth': [10, 15],
                'learning_rate': [0.01, 0.1],
            },
        }

        for model_name in top3_models:
            if model_name in param_grids:
                print("-" * 70)
                print(f"Tuning {model_name}")
                print("-" * 70)
                base_model = self.models[model_name]
                param_grid = param_grids[model_name]

                if isinstance(self.y_train, pd.DataFrame):
                    print(f"Skipping tuning for multi-label model {model_name} due to complexity.")
                    continue

                grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
                grid_search.fit(self.X_train, self.y_train)

                print("Best parameters:", grid_search.best_params_)
                print(f"Best CV F1-score: {grid_search.best_score_:.4f}")

                tuned_model = grid_search.best_estimator_
                tuned_name = f"{model_name} Tuned"
                results = evaluate_model(tuned_model, self.X_test, self.y_test, tuned_name)
                self.models[tuned_name] = tuned_model
                self.results.append(results)

        print("Hyperparameter tuning completed!")

    def select_best_model(self):
        print("-" * 70)
        print("MODEL SELECTION")
        print("-" * 70)
        comparison_df = compare_models(self.results)
        best_result = max(self.results, key=lambda x: x['f1_score'])
        self.best_model_name = best_result['model_name']
        self.best_model = self.models[self.best_model_name]
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"F1-Score: {best_result['f1_score']:.4f}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"Precision: {best_result['precision']:.4f}")
        print(f"Recall: {best_result['recall']:.4f}")
        print(f"ROC-AUC: {best_result['roc_auc']:.4f}")
        return self.best_model, self.best_model_name, comparison_df

    def save_visualizations(self):
        print("-" * 70)
        print("GENERATING VISUALIZATIONS")
        print("-" * 70)
        results_dir = Path(f"results/mental_health/{self.target}")
        results_dir.mkdir(parents=True, exist_ok=True)
        best_result = next(r for r in self.results if r['model_name'] == self.best_model_name)
        cm = np.array(best_result['confusion_matrix'])
        plot_confusion_matrix(cm, self.best_model_name, save_path=results_dir / "confusion_matrix.png")
        if best_result.get('probabilities') is not None:
            plot_roc_curve(self.y_test, best_result['probabilities'], self.best_model_name, save_path=results_dir / "roc_curve.png")
        print("Visualizations saved!")

    def save_best_model(self):
        print("-" * 70)
        print("SAVING BEST MODEL")
        print("-" * 70)
        model_name_with_target = f"{self.best_model_name}_{self.target}"
        model_path = save_model(self.best_model, model_name_with_target, "mental_health")
        best_result = next(r for r in self.results if r['model_name'] == self.best_model_name)
        best_result_clean = {k: v for k, v in best_result.items() if k not in ["predictions", "probabilities"]}
        best_result_clean["target"] = self.target
        results_path = save_results(best_result_clean, f"mental_health_{self.target}")
        print("Model and results saved!")
        return model_path, results_path

def train_mental_health_model(data_path="data/raw/mental_health.csv", target="depressiveness"):
    """
    Complete training pipeline for mental health prediction.

    Parameters
    ----------
    data_path : str
        Path to mental health dataset
    target : str
        Target to predict ('depressiveness', 'anxiousness', or 'sleepiness')

    Returns
    -------
    trainer : MentalHealthModelTrainer
        Trained model trainer object
    """
    print("=" * 70)
    print(f"MENTAL HEALTH PREDICTION - {target.upper()}")
    print("=" * 70)
    # 1. Data Preprocessing
    X_train, X_test, y_train, y_test, preprocessor = preprocess_mental_health_data(data_path, target=target)
    # 2. Initialize Model Trainer
    trainer = MentalHealthModelTrainer(X_train, X_test, y_train, y_test, target=target)
    # 3. Check Class Distribution
    trainer.check_class_distribution()
    # 4. Train Baseline Models
    trainer.train_baseline_models()
    # 5. Train with SMOTE
    trainer.train_with_smote()
    # 6. Hyperparameter Tuning
    trainer.tune_best_models()
    # 7. Model Selection
    best_model, best_name, comparison = trainer.select_best_model()
    # 8. Generate Visualizations
    trainer.save_visualizations()
    # 9. Save Best Model
    trainer.save_best_model()
    print("=" * 70)
    print(f"MENTAL HEALTH {target.upper()} MODEL TRAINING COMPLETED!")
    print("=" * 70)
    return trainer

def train_all_mental_health_targets(data_path="data/raw/mental_health.csv"):
    """
    Train models for all three mental health targets.
    """
    targets = ["depressiveness", "anxiousness", "sleepiness"]
    trainers = {}
    print("=" * 70)
    print("TRAINING ALL MENTAL HEALTH TARGETS")
    print("=" * 70)
    for target in targets:
        print("=" * 70)
        print(f"TARGET: {target.upper()}")
        print("=" * 70)
        trainer = train_mental_health_model(data_path, target=target)
        trainers[target] = trainer
    print("=" * 70)
    print("ALL MENTAL HEALTH MODELS TRAINED!")
    print("=" * 70)
    return trainers

if __name__ == "__main__":
    trainers = train_all_mental_health_targets()
