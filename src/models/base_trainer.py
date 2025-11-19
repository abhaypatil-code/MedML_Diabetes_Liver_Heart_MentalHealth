import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from src.utils.common import setup_logging, save_model
from src.utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from pathlib import Path

class BaseTrainer:
    """Base class for model training and evaluation."""
    
    def __init__(self, dataset_name, X_train, X_test, y_train, y_test):
        self.dataset_name = dataset_name
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.logger = setup_logging(f"{dataset_name}_trainer")
        self.models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "SVM": SVC(probability=True),
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        self.best_model = None
        self.best_score = 0
        self.results = {}
        
        self.models_dir = Path("models")
        self.results_dir = Path(f"results/{dataset_name}")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def train_and_evaluate(self):
        """Train multiple models and evaluate them."""
        self.logger.info("Starting model training...")
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            
            y_pred = model.predict(self.X_test)
            y_prob = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else None
            
            # Metrics
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average='weighted')
            rec = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            roc = roc_auc_score(self.y_test, y_prob) if y_prob is not None and len(np.unique(self.y_test)) == 2 else 0
            
            self.results[name] = {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC AUC": roc,
                "Model": model
            }
            
            self.logger.info(f"{name} Results - Acc: {acc:.4f}, F1: {f1:.4f}")
            
            # Save visualizations for each model
            plot_confusion_matrix(self.y_test, y_pred, classes=np.unique(self.y_test), 
                                  output_path=self.results_dir / f"confusion_matrix_{name}.png",
                                  title=f"Confusion Matrix - {name}")
            
            if y_prob is not None and len(np.unique(self.y_test)) == 2:
                plot_roc_curve(self.y_test, y_prob, 
                               output_path=self.results_dir / f"roc_curve_{name}.png",
                               title=f"ROC Curve - {name}")

    def select_best_model(self, metric="F1 Score"):
        """Select the best model based on a metric."""
        best_name = max(self.results, key=lambda x: self.results[x][metric])
        self.best_model = self.results[best_name]["Model"]
        self.best_score = self.results[best_name][metric]
        self.logger.info(f"Best model selected: {best_name} with {metric} = {self.best_score:.4f}")
        
        # Save best model
        save_model(self.best_model, self.models_dir / f"{self.dataset_name}_best_model.pkl")
        
        # Feature importance for best model
        plot_feature_importance(self.best_model, self.X_train.columns, 
                                output_path=self.results_dir / "best_model_feature_importance.png",
                                title=f"Feature Importance - {best_name}")
        
        return self.best_model
