# Disease Prediction ML Project

## Overview
This project implements a comprehensive Machine Learning pipeline to predict four different diseases:
1.  **Liver Disease** (ILPD)
2.  **Heart Disease**
3.  **Diabetes**
4.  **Mental Health** (Depression, Anxiety, Sleepiness)

The project uses a modular architecture with robust preprocessing, advanced model training (including SMOTE and hyperparameter tuning), and detailed evaluation.

## Project Structure
```
ML_Models/
├── data/
│   ├── raw/            # Original datasets
│   └── splits/         # Processed train/test splits
├── models/             # Trained .pkl models
├── results/            # Evaluation metrics and plots
├── scalers/            # Saved scalers and encoders
├── src/
│   ├── models/         # Training logic (AdvancedTrainer)
│   ├── preprocessing/  # Data cleaning and feature engineering
│   └── utils/          # Helper functions
├── main.py             # Main execution script
├── EXECUTION_FLOW.md   # Detailed pipeline documentation
└── requirements.txt    # Python dependencies
```

## Key Features
- **Modular Design**: Separate modules for preprocessing and training.
- **Advanced Preprocessing**: Missing value imputation, outlier handling, and feature engineering.
- **Robust Training**:
    - **Class Imbalance Handling**: Automatic detection and SMOTE application.
    - **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost, SVM, LightGBM, etc.
    - **Hyperparameter Tuning**: GridSearchCV for top-performing models.
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Visualizations**: Confusion Matrices, ROC Curves, Feature Importance.

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Run All Pipelines
To train models for all datasets and generate results:
```bash
python main.py
```

### Run Specific Pipeline
To train a model for a specific disease (e.g., Diabetes):
```bash
python src/models/diabetes_model.py
```

## Results
After running the pipeline, check the `results/` directory for:
- **Confusion Matrices**: `confusion_matrix.png`
- **ROC Curves**: `roc_curve.png`
- **Model Comparison**: `model_comparison.csv`

The best trained models will be saved in the `models/` directory.

## Customization
- **Preprocessing**: Modify `src/preprocessing/<disease>_preprocessing.py` to add new features.
- **Models**: Update `src/models/advanced_trainer.py` to add new algorithms or change grid search parameters.