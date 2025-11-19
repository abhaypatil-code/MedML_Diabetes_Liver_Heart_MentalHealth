import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from src.models.utils import (evaluate_model, compare_models, 
                              plot_confusion_matrix, plot_roc_curve, 
                              save_model, save_results)
from src.utils.common import setup_logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class AdvancedTrainer:
    """
    A generic, robust trainer class for classification problems.
    Handles:
    - Class imbalance check
    - SMOTE application
    - Training multiple baselines (Standard, Weighted, SMOTE-based)
    - Hyperparameter tuning
    - Model selection
    - Visualization and Saving
    """

    def __init__(self, dataset_name, X_train, X_test, y_train, y_test):
        self.dataset_name = dataset_name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.logger = setup_logging(f"{dataset_name}_trainer")
        
        self.X_train_resampled = None
        self.y_train_resampled = None
        
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None
        
        self.results_dir = Path(f"results/{dataset_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def check_class_imbalance(self):
        """Checks and logs class distribution."""
        self.logger.info("Checking class imbalance...")
        train_dist = self.y_train.value_counts()
        imbalance_ratio = train_dist.min() / (train_dist.max() + 1e-6)
        self.logger.info(f"Train class distribution:\n{train_dist}")
        self.logger.info(f"Imbalance Ratio: {imbalance_ratio:.2f}")
        return imbalance_ratio

    def apply_smote(self):
        """Applies SMOTE to the training data."""
        self.logger.info("Applying SMOTE...")
        try:
            smote = SMOTE(random_state=42, k_neighbors=5)
            self.X_train_resampled, self.y_train_resampled = smote.fit_resample(self.X_train, self.y_train)
            self.logger.info(f"SMOTE applied. New shape: {self.X_train_resampled.shape}")
        except Exception as e:
            self.logger.error(f"SMOTE failed: {e}. Fallback to original data.")
            self.X_train_resampled, self.y_train_resampled = self.X_train, self.y_train

    def train_model(self, name, model, use_smote=False):
        """Helper to train and evaluate a single model."""
        self.logger.info(f"Training {name}...")
        
        X_t = self.X_train_resampled if use_smote and self.X_train_resampled is not None else self.X_train
        y_t = self.y_train_resampled if use_smote and self.y_train_resampled is not None else self.y_train
        
        try:
            model.fit(X_t, y_t)
            self.models[name] = model
            results = evaluate_model(model, self.X_test, self.y_test, name)
            self.results.append(results)
            self.logger.info(f"{name} - F1: {results['f1_score']:.4f}, Acc: {results['accuracy']:.4f}")
        except Exception as e:
            self.logger.error(f"Failed to train {name}: {e}")

    def train_baselines(self):
        """Trains a comprehensive set of baseline models."""
        self.logger.info("Starting baseline training...")
        
        # 1. Standard Models
        standard_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Extra Trees": ExtraTreesClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "LightGBM": LGBMClassifier(verbose=-1, random_state=42)
        }
        
        for name, model in standard_models.items():
            self.train_model(name, model, use_smote=False)

        # 2. Class Weighted Models (if imbalance exists)
        if self.check_class_imbalance() < 0.8:
            self.logger.info("Training class-weighted models...")
            weighted_models = {
                "Logistic Regression Weighted": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
                "Random Forest Weighted": RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42),
                "SVM Weighted": SVC(class_weight='balanced', probability=True, random_state=42)
            }
            for name, model in weighted_models.items():
                self.train_model(name, model, use_smote=False)

        # 3. SMOTE Models (if imbalance exists)
        if self.X_train_resampled is not None:
            self.logger.info("Training SMOTE-based models...")
            smote_models = {
                "Logistic Regression SMOTE": LogisticRegression(max_iter=1000, random_state=42),
                "Random Forest SMOTE": RandomForestClassifier(n_estimators=100, random_state=42),
                "XGBoost SMOTE": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
                "LightGBM SMOTE": LGBMClassifier(verbose=-1, random_state=42)
            }
            for name, model in smote_models.items():
                self.train_model(name, model, use_smote=True)

    def hyperparameter_tuning(self, top_n=3):
        """Tunes the top N performing models."""
        self.logger.info(f"Tuning top {top_n} models...")
        
        if not self.results:
            self.logger.warning("No results to tune from.")
            return

        sorted_results = sorted(self.results, key=lambda x: x['f1_score'], reverse=True)
        top_models = [r['model_name'] for r in sorted_results[:top_n]]
        
        # Define parameter grids (simplified for performance)
        param_grids = {
            "Random Forest": {'n_estimators': [100, 200], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]},
            "XGBoost": {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]},
            "LightGBM": {'n_estimators': [100, 200], 'max_depth': [5, 10], 'learning_rate': [0.01, 0.1]},
            "Logistic Regression": {'C': [0.1, 1, 10], 'solver': ['liblinear']},
            "SVM": {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
            "Decision Tree": {'max_depth': [5, 10, None], 'min_samples_split': [2, 5]},
            "KNN": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
            "AdaBoost": {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            "Extra Trees": {'n_estimators': [100, 200], 'max_depth': [10, 20, None]},
            "Gradient Boosting": {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}
        }
        
        # Add variations for Weighted/SMOTE
        full_param_grids = {}
        for base, grid in param_grids.items():
            full_param_grids[base] = grid
            full_param_grids[f"{base} Weighted"] = grid
            full_param_grids[f"{base} SMOTE"] = grid

        for name in top_models:
            if name not in full_param_grids:
                self.logger.info(f"Skipping tuning for {name} (no grid defined)")
                continue
                
            self.logger.info(f"Tuning {name}...")
            base_model = self.models[name]
            
            # Determine data to use
            use_smote = "SMOTE" in name
            X_t = self.X_train_resampled if use_smote and self.X_train_resampled is not None else self.X_train
            y_t = self.y_train_resampled if use_smote and self.y_train_resampled is not None else self.y_train
            
            try:
                grid = GridSearchCV(base_model, full_param_grids[name], cv=3, scoring='f1', n_jobs=-1, verbose=0)
                grid.fit(X_t, y_t)
                
                best_model = grid.best_estimator_
                tuned_name = f"{name} Tuned"
                self.models[tuned_name] = best_model
                
                results = evaluate_model(best_model, self.X_test, self.y_test, tuned_name)
                self.results.append(results)
                self.logger.info(f"Tuned {name} -> F1: {results['f1_score']:.4f}")
            except Exception as e:
                self.logger.error(f"Tuning failed for {name}: {e}")

    def select_and_save_best_model(self):
        """Selects the best model, saves it, and generates plots."""
        if not self.results:
            self.logger.error("No models trained!")
            return None

        # Select best
        best_result = max(self.results, key=lambda x: x['f1_score'])
        self.best_model_name = best_result['model_name']
        self.best_model = self.models[self.best_model_name]
        
        self.logger.info(f"Best Model: {self.best_model_name} (F1: {best_result['f1_score']:.4f})")
        
        # Save Model
        save_model(self.best_model, self.best_model_name, self.dataset_name)
        
        # Save Results
        clean_results = {k: v for k, v in best_result.items() if k not in ['predictions', 'probabilities', 'confusion_matrix']}
        save_results(clean_results, self.dataset_name)
        
        # Visualizations
        cm = np.array(best_result['confusion_matrix'])
        plot_confusion_matrix(cm, self.best_model_name, 
                              save_path=self.results_dir / "confusion_matrix.png")
        
        if best_result.get('probabilities') is not None:
            plot_roc_curve(self.y_test, best_result['probabilities'], self.best_model_name, 
                           save_path=self.results_dir / "roc_curve.png")
                           
        # Comparison CSV
        compare_models(self.results).to_csv(self.results_dir / "model_comparison.csv", index=False)
        
        return self.best_model

    def run(self):
        """Executes the full training pipeline."""
        self.check_class_imbalance()
        self.apply_smote()
        self.train_baselines()
        self.hyperparameter_tuning()
        return self.select_and_save_best_model()
