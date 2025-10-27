# src/models/heart_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import joblib
import warnings
warnings.filterwarnings("ignore")

# FIXED IMPORT: Correct path for src structure
from src.preprocessing.heart_preprocessing import preprocess_heart_data

class HeartDiseaseModelTrainer:
    def __init__(self, X_train, X_test, y_train, y_test, preprocessor):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.preprocessor = preprocessor
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        self.results_dir = Path("results/heart")
        self.models_dir = Path("models")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def train_models(self):
        print("\n[Training Models]")
        
        models_to_train = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
            "SVM": SVC(random_state=42, probability=True)
        }
        
        for name, model in models_to_train.items():
            print(f"  Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            self.results[name] = {
                "accuracy": accuracy_score(self.y_test, y_pred),
                "precision": precision_score(self.y_test, y_pred, average='weighted'),
                "recall": recall_score(self.y_test, y_pred, average='weighted'),
                "f1": f1_score(self.y_test, y_pred, average='weighted'),
                "roc_auc": roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else None,
                "y_pred": y_pred,
                "y_pred_proba": y_pred_proba
            }
            
        return self

    def select_best_model(self):
        print("\n[Selecting Best Model]")
        best_score = 0
        for name, metrics in self.results.items():
            if metrics["f1"] > best_score:
                best_score = metrics["f1"]
                self.best_model_name = name
                self.best_model = self.models[name]
        
        print(f"  Best Model: {self.best_model_name} (F1 Score: {best_score:.4f})")
        return self

    def save_best_model(self):
        joblib.dump(self.best_model, self.models_dir / "heart_best_model.pkl")
        joblib.dump(self.preprocessor, self.models_dir / "heart_preprocessor.pkl")
        
        model_info = {
            "model_name": self.best_model_name,
            "metrics": self.results[self.best_model_name]
        }
        joblib.dump(model_info, self.models_dir / "heart_model_info.pkl")
        return self

    def generate_evaluation_report(self):
        print("\n[Generating Evaluation Report]")
        
        # Comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics_df = pd.DataFrame({
            name: [res["accuracy"], res["precision"], res["recall"], res["f1"]]
            for name, res in self.results.items()
        }, index=["Accuracy", "Precision", "Recall", "F1 Score"])
        
        metrics_df.T.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title("Model Comparison")
        axes[0, 0].set_ylabel("Score")
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Confusion Matrix for best model
        best_y_pred = self.results[self.best_model_name]["y_pred"]
        cm = confusion_matrix(self.y_test, best_y_pred)

        # Create proper labels
        labels = ['No Risk (0)', 'Risk (1)']

        # Plot with better formatting
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],xticklabels=labels, yticklabels=labels,cbar_kws={'label': 'Count'},linewidths=0.5, linecolor='gray')

        axes[0, 1].set_title(f"Confusion Matrix - {self.best_model_name}", fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel("Actual Label", fontsize=10, fontweight='bold')
        axes[0, 1].set_xlabel("Predicted Label", fontsize=10, fontweight='bold')

        # Add accuracy annotation
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        axes[0, 1].text(0.5, -0.15, f'Accuracy: {accuracy:.2%}', 
                    ha='center', va='top', transform=axes[0, 1].transAxes,
                    fontsize=10, fontweight='bold')
        
        # ROC Curve
        for name, res in self.results.items():
            if res["y_pred_proba"] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, res["y_pred_proba"])
                axes[1, 0].plot(fpr, tpr, label=f'{name} (AUC={res["roc_auc"]:.3f})')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1, 0].set_xlabel("False Positive Rate")
        axes[1, 0].set_ylabel("True Positive Rate")
        axes[1, 0].set_title("ROC Curves")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Feature Importance (if available)
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-10:]
            feature_names = self.X_train.columns
            axes[1, 1].barh(range(len(indices)), importances[indices], color='teal')
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([feature_names[i] for i in indices])
            axes[1, 1].set_xlabel("Importance")
            axes[1, 1].set_title(f"Top 10 Features - {self.best_model_name}")
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "model_evaluation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save metrics to CSV
        metrics_df.to_csv(self.results_dir / "model_metrics.csv")
        
        return self

def train_heart_model():
    """Train heart disease prediction model"""
    print("\n" + "="*80)
    print("HEART DISEASE PREDICTION - MODEL TRAINING")
    print("="*80)
    
    # Preprocessing
    print("\n[Step 1: Data Preprocessing]")
    X_train, X_test, y_train, y_test, preprocessor = preprocess_heart_data("data/raw/heart.csv")
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Features: {X_train.shape[1]}")
    
    # Model Training
    trainer = HeartDiseaseModelTrainer(X_train, X_test, y_train, y_test, preprocessor)
    trainer.train_models()\
           .select_best_model()\
           .save_best_model()\
           .generate_evaluation_report()
    
    print("\n" + "="*80)
    print("HEART DISEASE MODEL TRAINING COMPLETED")
    print("="*80)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Test Accuracy: {trainer.results[trainer.best_model_name]['accuracy']:.4f}")
    print(f"F1 Score: {trainer.results[trainer.best_model_name]['f1']:.4f}")
    print(f"\nModel saved to: models/heart_best_model.pkl")
    print(f"Results saved to: results/heart/")
    
    return trainer

if __name__ == "__main__":
    trainer = train_heart_model()
