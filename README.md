# Disease Prediction ML Pipeline - Complete MVP

A comprehensive machine learning pipeline for predicting **4 disease types**: Liver Disease, Heart Disease, Diabetes, and Mental Health conditions.

## 📋 Project Overview

This project implements end-to-end ML pipelines for disease prediction, including:
- **Data Visualization & EDA**
- **Preprocessing & Feature Engineering**
- **Multiple Model Training & Evaluation**
- **Automated Model Selection**
- **Production-Ready Model Export (.pkl)**

## 🗂️ Project Structure

```
.
├── data/
│   ├── raw/                      # Raw datasets
│   │   ├── diabetes.csv          ✓ Included
│   │   ├── liver.csv             ✓ Included
│   │   ├── mental_health.csv     ✓ Included
│   │   └── heart.csv             ⚠️ Not included (see below)
│   ├── processed/                # Processed datasets
│   └── splits/                   # Train/test splits
├── src/
│   ├── preprocessing/            # Data preprocessing modules
│   │   ├── diabetes_preprocessing.py
│   │   ├── liver_preprocessing.py
│   │   ├── heart_preprocessing.py
│   │   └── mental_health_preprocessing.py
│   ├── models/                   # Model training modules
│   │   ├── diabetes_model.py
│   │   ├── liver_model.py
│   │   ├── heart_model.py
│   │   ├── mental_health_model.py
│   │   └── utils.py              # Shared utilities
│   └── api/
│       └── app.py                # Flask API for predictions
├── models/                       # Saved trained models (.pkl)
├── scalers/                      # Saved scalers & encoders
├── results/                      # Evaluation results & visualizations
├── main.py                       # Main orchestration script
├── requirements.txt              # Python dependencies
└── generate_heart_data.py        # Optional: synthetic heart data
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Handle Missing Heart Dataset

**Option A: If you have the real heart.csv dataset**
```bash
# Place your heart.csv file in data/raw/
cp /path/to/your/heart.csv data/raw/
```

**Option B: Generate synthetic data (for testing only)**
```bash
python generate_heart_data.py
```

**Option C: Skip heart disease pipeline**
```bash
# The main.py script will automatically skip if heart.csv is missing
```

### 3. Run Complete Pipeline

```bash
python main.py
```

This will:
1. ✓ Train Liver Disease model
2. ✓ Train Heart Disease model (if data available)
3. ✓ Train Diabetes model
4. ✓ Train Mental Health models (3 targets)

## 📊 Individual Pipeline Execution

You can also run each pipeline independently:

```bash
# Diabetes
python -m src.models.diabetes_model

# Liver
python -m src.models.liver_model

# Heart
python -m src.models.heart_model

# Mental Health
python -m src.models.mental_health_model
```

## 🔧 Pipeline Details

### 1. Diabetes Prediction
- **Target**: Diabetes diagnosis (0/1)
- **Features**: 8 clinical measurements + engineered features
- **Models**: LogReg, RF, XGBoost, LightGBM, SVM, KNN, etc.
- **Techniques**: Median imputation, outlier capping, feature engineering

### 2. Liver Disease Prediction
- **Target**: Liver disease (0/1)
- **Features**: 10 liver function tests + engineered features
- **Models**: RF, XGBoost, LightGBM with SMOTE & class weights
- **Techniques**: SMOTE, IQR outlier handling, ratio features

### 3. Heart Disease Prediction
- **Target**: Heart attack history (0/1)
- **Features**: 20+ cardiovascular risk factors
- **Models**: RF, XGBoost, CatBoost, ensemble methods
- **Techniques**: SMOTE, medical feature engineering, ensemble voting

### 4. Mental Health Prediction
- **Targets**: Depression, Anxiety, Sleepiness (3 separate models)
- **Features**: PHQ-9, GAD-7, Epworth scores + demographics
- **Models**: Multi-label compatible classifiers
- **Techniques**: Clinical cutoffs, comorbidity features, treatment gaps

## 📈 Output Files

After running the pipeline, you'll find:

```
models/
├── diabetes_model.pkl
├── liver_model.pkl
├── heart_model.pkl
├── mental_health_depressiveness_model.pkl
├── mental_health_anxiousness_model.pkl
└── mental_health_sleepiness_model.pkl

scalers/
├── diabetes_scaler.pkl
├── liver_scaler.pkl
├── heart_scaler.pkl
└── mental_health_*_scaler.pkl

results/
├── diabetes/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── model_results.csv
├── liver/
├── heart/
└── mental_health/
    ├── depressiveness/
    ├── anxiousness/
    └── sleepiness/
```

## 🔌 API Usage (Optional)

Start the Flask API:

```bash
python src/api/app.py
```

Make predictions:

```python
import requests

# Diabetes prediction
response = requests.post('http://localhost:5000/predict/diabetes', 
                        json={'features': [6, 148, 72, 35, 0, 33.6, 0.627, 50]})
print(response.json())

# Mental health prediction
response = requests.post('http://localhost:5000/predict/mental_health/depressiveness',
                        json={'features': [19, 1, 33.3, 9, 11, 7, ...]})
print(response.json())
```

## 🛠️ Key Fixes Applied

### 1. Mental Health Preprocessing
- ✓ Fixed boolean column handling during normalization
- ✓ Ensured only numeric columns are scaled
- ✓ Added proper type conversion for target variables

### 2. Model Training
- ✓ Added error handling for SMOTE failures
- ✓ Fixed scaler saving paths per target
- ✓ Ensured proper label encoder usage

### 3. Main Pipeline
- ✓ Added comprehensive error handling
- ✓ Graceful degradation if datasets missing
- ✓ Clear success/failure reporting

## 📝 Notes

### Heart Dataset
The heart dataset (`heart.csv`) is **not included** in this repository. The pipeline expects these columns:

```
PatientID, State_Name, Age, Gender, Diabetes, Hypertension, Obesity, 
Smoking, AlcoholConsumption, PhysicalActivity, DietScore, 
CholesterolLevel, TriglycerideLevel, LDLLevel, HDLLevel, 
SystolicBP, DiastolicBP, AirPollutionExposure, FamilyHistory, 
StressLevel, HeartAttackHistory
```

Use `generate_heart_data.py` to create synthetic data for testing.

### Model Selection
Each pipeline automatically:
1. Trains 10+ baseline models
2. Applies SMOTE for imbalanced classes
3. Tunes hyperparameters for top 3 models
4. Selects best model by F1-score
5. Saves best model as `.pkl`

### Performance
- Training time: ~5-15 minutes per disease (depending on hardware)
- Models are production-ready after training
- All visualizations saved automatically

## 🐛 Troubleshooting

### Issue: ImportError or ModuleNotFoundError
```bash
# Ensure you're in project root and run:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python main.py
```

### Issue: Heart dataset missing
```bash
# Either generate synthetic data:
python generate_heart_data.py

# Or the pipeline will skip automatically
```

### Issue: Memory errors during SMOTE
- Reduce dataset size in preprocessing
- Or skip SMOTE models (baseline models still train)

### Issue: Column name mismatches
- Check your CSV files match the expected column names
- The preprocessing scripts handle most variations automatically

## 📊 Expected Results

Typical model performance:
- **Diabetes**: F1 ~0.75-0.85, Accuracy ~0.80-0.88
- **Liver**: F1 ~0.70-0.80, Accuracy ~0.75-0.85
- **Heart**: F1 ~0.75-0.88, Accuracy ~0.80-0.90
- **Mental Health**: F1 ~0.65-0.80 per target

## 🎯 Next Steps

1. ✅ All pipelines run end-to-end
2. ✅ Models saved as `.pkl` files
3. 🔄 Optional: Integrate with web UI
4. 🔄 Optional: Deploy API to cloud
5. 🔄 Optional: Add real-time prediction dashboard

## 📄 License

This is an educational/research project. Ensure compliance with medical data regulations (HIPAA, GDPR) for production use.

## ⚠️ Disclaimer

These models are for educational purposes only and should not be used for actual medical diagnosis without proper validation and regulatory approval.