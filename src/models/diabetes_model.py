""" Diabetes Disease ML Model Training Pipeline """

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Project root setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.diabetes_preprocessing import preprocess_diabetes_data
from src.models.utils import (
    evaluate_model, cross_validate_model,
    plot_confusion_matrix, plot_roc_curve,
    save_model, save_results, compare_models
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV

class DiabetesModelTrainer:
    """Comprehensive model training class for Diabetes prediction."""

    def __init__(self, Xtrain, Xtest, ytrain, ytest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None

    def train_baseline_models(self):
        print("=" * 70)
        print("TRAINING BASELINE MODELS")
        print("=" * 70)
        models_dict = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
            "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Extra Trees": ExtraTreesClassifier(random_state=42),
        }
        for model_name, model in models_dict.items():
            print("-" * 70)
            print(f"Training: {model_name}")
            print("-" * 70)
            model.fit(self.Xtrain, self.ytrain)
            self.models[model_name] = model

            results = evaluate_model(model, self.Xtest, self.ytest, model_name)
            self.results.append(results)

            cv_results = cross_validate_model(model, self.Xtrain, self.ytrain, cv=5, model_name=model_name)
            results['cv_results'] = cv_results

        print("Baseline model training completed!")

    def tune_best_models(self):
        print("=" * 70)
        print("HYPERPARAMETER TUNING")
        print("=" * 70)
        sorted_results = sorted(self.results, key=lambda x: x['f1_score'], reverse=True)
        top3models = [r['model_name'] for r in sorted_results[:3]]
        print(f"Top 3 models for tuning: {', '.join(top3models)}")

        param_grids = {
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, 30, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            },
            "XGBoost": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 0.9, 1.0]
            },
            "LightGBM": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15],
                "learning_rate": [0.01, 0.1, 0.3],
                "num_leaves": [31, 50, 70]
            },
            "Gradient Boosting": {
                "n_estimators": [100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.8, 1.0]
            },
            "Logistic Regression": {
                "C": [0.01, 0.1, 1, 10, 100],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "saga"]
            },
        }

        tuned_models = {}

        for model_name in top3models:
            if model_name in param_grids:
                print("-" * 70)
                print(f"Tuning {model_name}")
                print("-" * 70)

                base_model = self.models[model_name]
                param_grid = param_grids[model_name]

                grid_search = GridSearchCV(
                    base_model, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
                )
                grid_search.fit(self.Xtrain, self.ytrain)

                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best CV F1-score: {grid_search.best_score_:.4f}")

                tuned_model = grid_search.best_estimator_
                tuned_name = f"Tuned {model_name}"

                tuned_results = evaluate_model(tuned_model, self.Xtest, self.ytest, tuned_name)
                tuned_models[tuned_name] = tuned_model
                self.results.append(tuned_results)
                self.models[tuned_name] = tuned_model

        print("Hyperparameter tuning completed!")

    def select_best_model(self):
        print("=" * 70)
        print("MODEL SELECTION")
        print("=" * 70)
        comparison_df = compare_models(self.results)
        best_result = max(self.results, key=lambda x: x['f1_score'])
        self.best_model_name = best_result['model_name']
        self.best_model = self.models[self.best_model_name]
        print(f"*** BEST MODEL: {self.best_model_name} ***")
        print(f"F1-Score: {best_result['f1_score']:.4f}")
        print(f"Accuracy: {best_result['accuracy']:.4f}")
        print(f"ROC-AUC: {best_result['roc_auc']:.4f}")
        return self.best_model, self.best_model_name, comparison_df

    def save_visualizations(self):
        print("=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        results_dir = Path("results/diabetes")
        results_dir.mkdir(parents=True, exist_ok=True)
        best_result = next(r for r in self.results if r['model_name'] == self.best_model_name)
        cm = np.array(best_result['confusion_matrix'])
        plot_confusion_matrix(cm, self.best_model_name, save_path=results_dir / "confusion_matrix.png")
        if best_result['probabilities'] is not None:
            plot_roc_curve(self.ytest, best_result['probabilities'], self.best_model_name, save_path=results_dir / "roc_curve.png")
        print("Visualizations saved!")

    def save_best_model(self):
        print("=" * 70)
        print("SAVING BEST MODEL")
        print("=" * 70)
        model_path = save_model(self.best_model, self.best_model_name, "diabetes")
        best_result = next(r for r in self.results if r['model_name'] == self.best_model_name)
        best_result_clean = {k: v for k, v in best_result.items() if k not in ['predictions', 'probabilities']}
        results_path = save_results(best_result_clean, "diabetes")
        print("Model and results saved successfully!")
        return model_path, results_path

def train_diabetes_model(datapath="data/raw/diabetes.csv"):
    print("=" * 70)
    print("DIABETES PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 70)
    print("[1] Data Preprocessing")
    Xtrain, Xtest, ytrain, ytest, preprocessor = preprocess_diabetes_data(datapath)
    print("[2] Initializing Model Trainer")
    trainer = DiabetesModelTrainer(Xtrain, Xtest, ytrain, ytest)
    print("[3] Training Baseline Models")
    trainer.train_baseline_models()
    print("[4] Hyperparameter Tuning")
    trainer.tune_best_models()
    print("[5] Model Selection")
    best_model, best_name, comparison = trainer.select_best_model()
    print("[6] Generating Visualizations")
    trainer.save_visualizations()
    print("[7] Saving Best Model")
    trainer.save_best_model()
    print("=" * 70)
    print("DIABETES MODEL TRAINING COMPLETED!")
    print("=" * 70)
    return trainer

if __name__ == "__main__":
    trainer = train_diabetes_model()
