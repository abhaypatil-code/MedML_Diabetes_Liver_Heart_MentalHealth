""" Liver Disease ML Model Training Pipeline """

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.liver_preprocessing import preprocess_liver_data
from src.models.utils import (
    evaluate_model, compare_models,
    plot_confusion_matrix, plot_roc_curve,
    save_model, save_results
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              AdaBoostClassifier,
                              ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV

class LiverModelTrainer:
    """Handles model training and evaluation for Liver Disease dataset."""

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X_train_resampled = None
        self.y_train_resampled = None
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None

    def check_class_imbalance(self):
        print("="*70)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*70)
        train_dist = self.y_train.value_counts()
        test_dist = self.y_test.value_counts()
        print(f"Train class distribution:\n{train_dist}")
        print(f"Test class distribution:\n{test_dist}")
        print(f"Imbalance Ratio Train: {train_dist.min()/(train_dist.max() + 1e-6):.2f}")

    def apply_smote(self):
        print("="*70)
        print("APPLYING SMOTE - SYNTHETIC MINORITY OVERSAMPLING")
        print("="*70)
        smote = SMOTE(random_state=42, k_neighbors=5)
        self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
        print(f"Training samples after SMOTE: {len(self.y_train_resampled)}")
        print(f"Class distribution: {pd.Series(self.y_train_resampled).value_counts().to_dict()}")
        print("SMOTE applied successfully! Classes are now balanced.")

    def train_baseline_models_with_smote(self):
        print("="*70)
        print("TRAINING MODELS WITH SMOTE-BALANCED DATA")
        print("="*70)
        models_dict = {
            "Logistic Regression SMOTE": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest SMOTE": RandomForestClassifier(random_state=42, n_estimators=100),
            "XGBoost SMOTE": XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False),
            "LightGBM SMOTE": LGBMClassifier(random_state=42, verbose=-1),
            "Gradient Boosting SMOTE": GradientBoostingClassifier(random_state=42)
        }
        for model_name, model in models_dict.items():
            print("-"*70)
            print(f"Training {model_name}")
            model.fit(self.X_train_resampled, self.y_train_resampled)
            self.models[model_name] = model
            results = evaluate_model(model, self.X_test, self.y_test, model_name)
            self.results.append(results)
        print("SMOTE-based model training completed!")

    def train_baseline_models_with_class_weights(self):
        print("="*70)
        print("TRAINING MODELS WITH CLASS WEIGHTS")
        print("="*70)
        class_weight = 'balanced'
        models_dict = {
            "Logistic Regression Weighted": LogisticRegression(random_state=42, max_iter=1000, class_weight=class_weight),
            "Random Forest Weighted": RandomForestClassifier(random_state=42, n_estimators=100, class_weight=class_weight),
            "XGBoost Weighted": XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=(len(self.y_train) / (np.sum(self.y_train)+1e-6)), use_label_encoder=False),
            "SVM Weighted": SVC(probability=True, random_state=42, class_weight=class_weight)
        }
        for model_name, model in models_dict.items():
            print("-"*70)
            print(f"Training {model_name}")
            model.fit(self.X_train, self.y_train)
            self.models[model_name] = model
            results = evaluate_model(model, self.X_test, self.y_test, model_name)
            self.results.append(results)
        print("Class-weighted model training completed!")

    def train_other_baseline_models(self):
        print("="*70)
        print("TRAINING ADDITIONAL BASELINE MODELS")
        print("="*70)
        models_dict = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Extra Trees": ExtraTreesClassifier(random_state=42),
        }
        for model_name, model in models_dict.items():
            print("-"*70)
            print(f"Training {model_name}")
            model.fit(self.X_train, self.y_train)
            self.models[model_name] = model
            results = evaluate_model(model, self.X_test, self.y_test, model_name)
            self.results.append(results)
        print("Additional baseline models trained!")

    def tune_best_models(self):
        print("="*70)
        print("HYPERPARAMETER TUNING")
        print("="*70)
        sorted_results = sorted(self.results, key=lambda x: x['f1_score'], reverse=True)
        top3_models = [r['model_name'] for r in sorted_results[:3]]
        print(f"Top 3 models for tuning: {', '.join(top3_models)}")
        param_grids = {
            "Random Forest SMOTE": {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5],
                'class_weight': ['balanced', 'balanced_subsample']
            },
            "Random Forest Weighted": {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5]
            },
            "XGBoost SMOTE": {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            },
            "XGBoost Weighted": {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            },
            "LightGBM SMOTE": {
                'n_estimators': [100, 200],
                'max_depth': [5, 10],
                'learning_rate': [0.01, 0.1]
            },
            "Logistic Regression SMOTE": {
                'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear', 'saga']
            },
            "Logistic Regression Weighted": {
                'C': [0.1, 1, 10], 'penalty': ['l2'], 'solver': ['liblinear']
            }
        }
        tuned_models = {}
        for model_name in top3_models:
            if model_name in param_grids:
                print("-"*70)
                print(f"Tuning {model_name}")
                base_model = self.models[model_name]
                param_grid = param_grids[model_name]
                if 'SMOTE' in model_name:
                    X_train_use = self.X_train_resampled
                    y_train_use = self.y_train_resampled
                else:
                    X_train_use = self.X_train
                    y_train_use = self.y_train
                grid_search = GridSearchCV(
                    base_model, param_grid, cv=5,
                    scoring='f1', n_jobs=-1, verbose=1
                )
                grid_search.fit(X_train_use, y_train_use)
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best CV F1-score: {grid_search.best_score_:.4f}")
                tuned_model = grid_search.best_estimator_
                tuned_name = f"{model_name} Tuned"
                tuned_results = evaluate_model(tuned_model, self.X_test, self.y_test, tuned_name)
                tuned_models[tuned_name] = tuned_model
                self.models[tuned_name] = tuned_model
                self.results.append(tuned_results)
        print("Hyperparameter tuning completed!")

    def select_best_model(self):
        print("="*70)
        print("MODEL SELECTION")
        print("="*70)
        comparison_df = compare_models(self.results)
        best_result = max(self.results, key=lambda x: x['f1_score'])
        self.best_model_name = best_result['model_name']
        self.best_model = self.models[self.best_model_name]
        print(f"BEST MODEL: {self.best_model_name}")
        print(f"F1-Score: {best_result['f1_score']:.4f}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"Recall: {best_result['recall']:.4f}")
        print(f"ROC-AUC: {best_result['roc_auc']:.4f}")
        return self.best_model, self.best_model_name, comparison_df

    def save_visualizations(self):
        print("="*70)
        print("GENERATING VISUALIZATIONS")
        print("="*70)
        results_dir = Path("results/liver")
        results_dir.mkdir(parents=True, exist_ok=True)
        best_result = next(r for r in self.results if r['model_name'] == self.best_model_name)
        cm = np.array(best_result['confusion_matrix'])
        plot_confusion_matrix(cm, self.best_model_name, save_path=results_dir / "confusion_matrix.png")
        if best_result.get('probabilities') is not None:
            plot_roc_curve(self.y_test, best_result['probabilities'], self.best_model_name, save_path=results_dir / "roc_curve.png")
        print("Visualizations saved!")

    def save_best_model(self):
        print("="*70)
        print("SAVING BEST MODEL")
        print("="*70)
        model_path = save_model(self.best_model, self.best_model_name, "liver")
        best_result = next(r for r in self.results if r['model_name'] == self.best_model_name)
        best_result_clean = {k: v for k, v in best_result.items() if k not in ['predictions', 'probabilities']}
        results_path = save_results(best_result_clean, "liver")
        print("Model and results saved successfully!")
        return model_path, results_path

def train_liver_model(data_path="data/raw/liver.csv"):
    print("="*70)
    print("LIVER DISEASE PREDICTION - MODEL TRAINING PIPELINE")
    print("="*70)
    print("1. Data Preprocessing")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_liver_data(data_path)
    print("2. Initializing Model Trainer")
    trainer = LiverModelTrainer(X_train, X_test, y_train, y_test)
    print("3. Checking Class Imbalance")
    trainer.check_class_imbalance()
    print("4. Applying SMOTE")
    trainer.apply_smote()
    print("5. Training SMOTE-Based Models")
    trainer.train_baseline_models_with_smote()
    print("6. Training Class-Weighted Models")
    trainer.train_baseline_models_with_class_weights()
    print("7. Training Additional Baselines")
    trainer.train_other_baseline_models()
    print("8. Hyperparameter Tuning")
    trainer.tune_best_models()
    print("9. Model Selection")
    best_model, best_name, comparison = trainer.select_best_model()
    print("10. Generating Visualizations")
    trainer.save_visualizations()
    print("11. Saving Best Model")
    trainer.save_best_model()
    print("="*70)
    print("LIVER MODEL TRAINING COMPLETED!")
    print("="*70)
    return trainer

if __name__ == "__main__":
    trainer = train_liver_model()
