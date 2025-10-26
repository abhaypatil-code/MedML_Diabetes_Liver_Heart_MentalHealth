""" Heart Disease ML Model Training Pipeline """

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Project root setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.heart_preprocessing import preprocess_heart_data
from src.models.utils import (
    evaluate_model, cross_validate_model,
    plot_confusion_matrix, plot_roc_curve,
    save_model, save_results, compare_models
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

class HeartModelTrainer:
    """Trainer for heart disease models."""

    def __init__(self, Xtrain, Xtest, ytrain, ytest):
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        self.ytrain = ytrain
        self.ytest = ytest
        self.Xtrain_resampled = None
        self.ytrain_resampled = None
        self.models = {}
        self.results = []
        self.best_model = None
        self.best_model_name = None

    def check_class_imbalance(self):
        print("=" * 70)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("=" * 70)
        traindist = self.ytrain.value_counts()
        imbalanceratio = traindist[0] / (traindist[1] + 1e-6)
        print(f"Class 0 (No Heart Attack): {traindist[0]} ({traindist[0]/len(self.ytrain)*100:.2f}%)")
        print(f"Class 1 (Heart Attack): {traindist[1]} ({traindist[1]/len(self.ytrain)*100:.2f}%)")
        print(f"Imbalance Ratio: {imbalanceratio:.2f}:1")
        if imbalanceratio > 2:
            print("Significant class imbalance detected! SMOTE and class weights will be applied.")

    def apply_smote(self):
        print("=" * 70)
        print("APPLYING SMOTE")
        print("=" * 70)
        smote = SMOTE(random_state=42, k_neighbors=5)
        self.Xtrain_resampled, self.ytrain_resampled = smote.fit_resample(self.Xtrain, self.ytrain)
        print(f"After SMOTE: {len(self.ytrain_resampled)} samples")
        print(f"Class distribution: {dict(pd.Series(self.ytrain_resampled).value_counts())}")

    def train_baseline_models(self):
        print("=" * 70)
        print("TRAINING BASELINE MODELS")
        print("=" * 70)
        scaleposweight = len(self.ytrain) / (np.sum(self.ytrain == 1) + 1e-6)
        models_dict = {
            # Models trained with SMOTE-balanced data
            "Random Forest SMOTE": RandomForestClassifier(random_state=42, n_estimators=200),
            "XGBoost SMOTE": XGBClassifier(random_state=42, eval_metric="logloss", n_estimators=200),
            "LightGBM SMOTE": LGBMClassifier(random_state=42, verbose=-1, n_estimators=200),
            # Models trained with class weights
            "Logistic Regression Weighted": LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
            "Random Forest Weighted": RandomForestClassifier(random_state=42, n_estimators=200, class_weight="balanced"),
            "XGBoost Weighted": XGBClassifier(random_state=42, eval_metric="logloss", scale_pos_weight=scaleposweight, n_estimators=200),
            "Gradient Boosting Weighted": GradientBoostingClassifier(random_state=42, n_estimators=200),
            "CatBoost Weighted": CatBoostClassifier(random_state=42, verbose=0, iterations=200),
            # Other base models without special balancing
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "Naive Bayes": GaussianNB(),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "Extra Trees": ExtraTreesClassifier(random_state=42),
            "SVM Weighted": SVC(probability=True, random_state=42, class_weight="balanced"),
        }
        for model_name, model in models_dict.items():
            print("-" * 70)
            print(f"Training: {model_name}")
            if "SMOTE" in model_name:
                model.fit(self.Xtrain_resampled, self.ytrain_resampled)
            else:
                model.fit(self.Xtrain, self.ytrain)
            self.models[model_name] = model
            results = evaluate_model(model, self.Xtest, self.ytest, model_name)
            self.results.append(results)
        print("Baseline training completed!")

    def create_ensemble_models(self):
        print("=" * 70)
        print("CREATING ENSEMBLE MODELS")
        print("=" * 70)
        sorted_results = sorted(self.results, key=lambda x: x['f1_score'], reverse=True)
        top3names = [r['model_name'] for r in sorted_results[:3]]
        print(f"Top 3 models for ensembling: {', '.join(top3names)}")
        estimators = [(name, self.models[name]) for name in top3names]
        voting_clf = VotingClassifier(estimators=estimators, voting="soft")
        print("-" * 70)
        print("Training Voting Ensemble")
        voting_clf.fit(self.Xtrain, self.ytrain)
        self.models["Voting Ensemble"] = voting_clf
        results = evaluate_model(voting_clf, self.Xtest, self.ytest, "Voting Ensemble")
        self.results.append(results)
        print("Ensemble model created!")

    def tune_best_model(self):
        print("=" * 70)
        print("HYPERPARAMETER TUNING BEST MODEL")
        print("=" * 70)
        best_result = max(self.results, key=lambda x: x['f1_score'])
        best_name = best_result['model_name']
        print(f"Best model: {best_name}")

        if "Random Forest" in best_name:
            param_grid = {
                "n_estimators": [200, 300, 400],
                "max_depth": [15, 20, 25, None],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        elif "XGBoost" in best_name:
            param_grid = {
                "n_estimators": [200, 300, 400],
                "max_depth": [5, 7, 9],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.8, 0.9, 1.0]
            }
        elif "LightGBM" in best_name:
            param_grid = {
                "n_estimators": [200, 300, 400],
                "max_depth": [10, 15, 20],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [31, 50, 70]
            }
        else:
            print(f"No tuning grid defined for {best_name}")
            return

        basemodel = self.models[best_name]
        if "SMOTE" in best_name:
            Xtrain_use = self.Xtrain_resampled
            ytrain_use = self.ytrain_resampled
        else:
            Xtrain_use = self.Xtrain
            ytrain_use = self.ytrain

        random_search = RandomizedSearchCV(
            basemodel, param_grid, n_iter=15, cv=5, scoring="f1", n_jobs=-1, verbose=1, random_state=42
        )
        random_search.fit(Xtrain_use, ytrain_use)
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV F1-score: {random_search.best_score_:.4f}")
        tuned_model = random_search.best_estimator_
        tuned_name = f"{best_name} Tuned"
        results = evaluate_model(tuned_model, self.Xtest, self.ytest, tuned_name)
        self.models[tuned_name] = tuned_model
        self.results.append(results)
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
        print(f"Precision: {best_result['precision']:.4f}")
        print(f"Recall: {best_result['recall']:.4f}")
        print(f"ROC-AUC: {best_result['roc_auc']:.4f}")
        return self.best_model, self.best_model_name, comparison_df

    def save_visualizations(self):
        print("=" * 70)
        print("GENERATING VISUALIZATIONS")
        print("=" * 70)
        results_dir = Path("results/heart")
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
        model_path = save_model(self.best_model, self.best_model_name, "heart")
        best_result = next(r for r in self.results if r['model_name'] == self.best_model_name)
        best_result_clean = {k: v for k, v in best_result.items() if k not in ['predictions', 'probabilities']}
        results_path = save_results(best_result_clean, "heart")
        print("Model and results saved!")
        return model_path, results_path

def train_heart_model(datapath="data/raw/heart.csv"):
    print("=" * 70)
    print("HEART DISEASE PREDICTION - MODEL TRAINING PIPELINE")
    print("=" * 70)
    print("[1] Data Preprocessing")
    Xtrain, Xtest, ytrain, ytest, preprocessor = preprocess_heart_data(datapath)
    print("[2] Initializing Model Trainer")
    trainer = HeartModelTrainer(Xtrain, Xtest, ytrain, ytest)
    print("[3] Checking Class Imbalance")
    trainer.check_class_imbalance()
    print("[4] Applying SMOTE")
    trainer.apply_smote()
    print("[5] Training Baseline Models")
    trainer.train_baseline_models()
    print("[6] Creating Ensemble Models")
    trainer.create_ensemble_models()
    print("[7] Hyperparameter Tuning")
    trainer.tune_best_model()
    print("[8] Model Selection")
    best_model, best_name, comparison = trainer.select_best_model()
    print("[9] Generating Visualizations")
    trainer.save_visualizations()
    print("[10] Saving Best Model")
    trainer.save_best_model()
    print("=" * 70)
    print("HEART MODEL TRAINING COMPLETED!")
    print("=" * 70)
    return trainer

if __name__ == "__main__":
    trainer = train_heart_model()
