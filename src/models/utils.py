import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    log_loss
)
from sklearn.model_selection import cross_val_score, cross_val_predict
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a trained model on test data, return metrics and confusion matrix."""
    y_pred = model.predict(X_test)
    results = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="binary" if len(np.unique(y_test))==2 else "weighted"),
        "recall": recall_score(y_test, y_pred, average="binary" if len(np.unique(y_test))==2 else "weighted"),
        "f1_score": f1_score(y_test, y_pred, average="binary" if len(np.unique(y_test))==2 else "weighted"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "predictions": y_pred.tolist(),
    }
    # ROC-AUC and probabilities
    try:
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:,1] if X_test.shape[1] > 1 else model.predict_proba(X_test)
            results["roc_auc"] = roc_auc_score(y_test, y_proba)
            results["probabilities"] = y_proba.tolist()
            results["log_loss"] = log_loss(y_test, y_proba)
        else:
            results["roc_auc"] = None
            results["probabilities"] = None
            results["log_loss"] = None
    except Exception:
        results["roc_auc"] = None
        results["probabilities"] = None
        results["log_loss"] = None
    return results

def cross_validate_model(model, X_train, y_train, cv=5, model_name="Model"):
    """Perform cross-validation and return mean scores."""
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    scores = {}
    for metric in scoring:
        try:
            cv_score = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
            scores[metric] = np.mean(cv_score)
        except Exception:
            scores[metric] = None
    return {
        "model_name": model_name,
        "cv_scores": scores
    }

def plot_confusion_matrix(cm, model_name, save_path=None):
    """Plot confusion matrix as heatmap."""
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_roc_curve(y_true, y_proba, model_name, save_path=None):
    """Plot ROC curve given test labels and predicted probabilities."""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC: {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
        plt.close()

def save_model(model, model_name, disease_type):
    """Save model object as pickle to models directory."""
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{disease_type}_{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    return model_path

def save_results(results_dict, disease_type):
    """Save evaluation results to results directory in CSV format."""
    results_dir = Path("results") / disease_type
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "model_results.csv"
    df = pd.DataFrame([results_dict])
    df.to_csv(results_path, index=False)
    return results_path

def compare_models(results_list):
    """Compare all models - return DataFrame of key metrics."""
    df = pd.DataFrame(results_list)
    if "predictions" in df.columns:
        df = df.drop(columns=["predictions"])
    if "probabilities" in df.columns:
        df = df.drop(columns=["probabilities"])
    if "classification_report" in df.columns:
        df = df.drop(columns=["classification_report"])
    return df.sort_values("f1_score", ascending=False)

