# Execution Flow

This document outlines the step-by-step execution flow of the Disease Prediction ML Pipeline.

## 1. Data Ingestion
- **Source**: `data/raw/`
- **Files**:
    - `liver.csv`: Liver disease dataset.
    - `heart.csv`: Heart disease dataset.
    - `diabetes.csv`: Diabetes dataset.
    - `mental_health.csv`: Mental health dataset.

## 2. Preprocessing Pipeline
For each dataset, the following steps are executed via `src/preprocessing/`:

1.  **Loading**: Data is loaded into a Pandas DataFrame.
2.  **Cleaning**:
    - Missing values are imputed (Median/Mean/Mode).
    - Inconsistencies are fixed (e.g., mapping "1/2" to "1/0").
    - Outliers are handled using IQR capping.
3.  **Feature Engineering**: New features are created based on domain knowledge (e.g., BMI, Ratios).
4.  **Splitting**: Data is split into Train (80%) and Test (20%) sets, stratified by target.
5.  **Normalization**: Features are scaled using `StandardScaler`.
6.  **Saving**:
    - Processed splits -> `data/splits/`
    - Scalers/Encoders -> `scalers/`

## 3. Model Training Pipeline
The `AdvancedTrainer` class (`src/models/advanced_trainer.py`) orchestrates the training:

1.  **Class Imbalance Check**: Checks if the target class is imbalanced.
2.  **SMOTE Application**: If imbalanced, Synthetic Minority Over-sampling Technique (SMOTE) is applied to the training set.
3.  **Baseline Training**:
    - Trains multiple algorithms: Logistic Regression, Random Forest, XGBoost, SVM, etc.
    - Trains variants: Standard, Class-Weighted, and SMOTE-based.
4.  **Hyperparameter Tuning**:
    - Selects the top 3 performing models.
    - Tunes hyperparameters using `GridSearchCV` (optimizing for F1-score).
5.  **Evaluation**:
    - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
    - Visualizations: Confusion Matrix, ROC Curve.

## 4. Model Selection & Export
1.  **Selection**: The model with the highest F1-score is selected as the Best Model.
2.  **Export**:
    - **Model**: Saved as `.pkl` in `models/` (e.g., `liver_best_model.pkl`).
    - **Results**: Metrics saved as JSON/CSV in `results/<dataset>/`.
    - **Plots**: Saved in `results/<dataset>/`.

## 5. Running the Pipeline
To execute the entire pipeline for all datasets:

```bash
python main.py
```

To run a specific pipeline (e.g., Liver):

```bash
python src/models/liver_model.py
```

## Output Directory Structure
```
ML_Models/
├── data/
│   ├── raw/            # Original CSVs
│   ├── processed/      # (Optional) Intermediate files
│   └── splits/         # Train/Test CSVs
├── models/             # Final .pkl models
├── results/            # Metrics and Plots
│   ├── liver/
│   ├── heart/
│   ├── diabetes/
│   └── mental_health/
├── scalers/            # Saved Scalers and Encoders
└── src/                # Source code
```
