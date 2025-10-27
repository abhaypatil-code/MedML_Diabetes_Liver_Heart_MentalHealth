import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from pathlib import Path
import numpy as np

class HeartPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.dforiginal = None
        self.scaler = None
        self.labelencoder = None
        self.featurenames = None
        self.visdir = Path("results/heart/visualizations")  # FIXED: Added slash
        self.visdir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        self.dforiginal = self.df.copy()
        if "PatientID" in self.df.columns:
            self.df = self.df.drop(columns=["PatientID"])
        return self

    def exploratory_analysis(self):
        plt.figure(figsize=(8, 5))
        sns.countplot(x="Heart_Attack_Risk", data=self.df)
        plt.title("Heart Attack Risk Distribution Target")
        plt.savefig(self.visdir / "heartattackriskdistribution.png")
        plt.close()

        plt.figure(figsize=(12, 10))
        sns.heatmap(self.df.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Numerical Feature Correlation Heatmap")
        plt.savefig(self.visdir / "heartnumericalcorrelation.png")
        plt.close()
        return self

    def handle_missing_values(self):
        imputer = SimpleImputer(strategy='mean')
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if self.df[col].isnull().any():
                self.df[col] = imputer.fit_transform(self.df[[col]])

        for col in self.df.select_dtypes(include=[object]).columns:
            if self.df[col].isnull().any():
                imputer = SimpleImputer(strategy='most_frequent')
                self.df[col] = imputer.fit_transform(self.df[[col]])
        return self

    def encode_categorical_variables(self):
        cat_cols = self.df.select_dtypes(include=[object]).columns
        self.labelencoder = {}  # FIXED: Dictionary instead of single encoder
        for col in cat_cols:
            le = LabelEncoder()  # FIXED: Create new encoder for each column
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.labelencoder[col] = le  # FIXED: Store each encoder
        return self

    def detect_and_handle_outliers(self):
        # FIXED: Changed 'HeartAttackRisk' to 'Heart_Attack_Risk'
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.difference(['Heart_Attack_Risk'])
        for col in numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)
        return self

    def feature_engineering(self):
        if ('Systolic_BP' in self.df.columns and 'Diastolic_BP' in self.df.columns):
            self.df["SystolicDiastolicRatio"] = self.df["Systolic_BP"] / (self.df["Diastolic_BP"] + 1e-6)
            self.df['BP_Difference'] = self.df['Systolic_BP'] - self.df['Diastolic_BP']
        return self

    def check_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self

    def train_test_split_data(self, test_size=0.2, random_state=42):
        X = self.df.drop(columns=["Heart_Attack_Risk"])
        y = self.df["Heart_Attack_Risk"]
        self.featurenames = X.columns.tolist()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        return self

    def scale_features(self):
        self.scaler = StandardScaler()
        self.X_train = pd.DataFrame(self.scaler.fit_transform(self.X_train), columns=self.featurenames, index=self.X_train.index)
        self.X_test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.featurenames, index=self.X_test.index)
        return self

    def save_processed_data(self):
        output_dir = Path("data/processed")
        splits_dir = Path("data/splits")
        scalers_dir = Path("scalers")
        output_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)
        scalers_dir.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(output_dir / "heart_processed_unscaled.csv", index=False)

        train_df = pd.concat([self.X_train.reset_index(drop=True), self.y_train.reset_index(drop=True)], axis=1)
        test_df = pd.concat([self.X_test.reset_index(drop=True), self.y_test.reset_index(drop=True)], axis=1)

        train_df.to_csv(splits_dir / "heart_train_scaled.csv", index=False)
        test_df.to_csv(splits_dir / "heart_test_scaled.csv", index=False)

        joblib.dump(self.scaler, scalers_dir / "heart_scaler.pkl")
        # FIXED: Save label encoders dictionary
        if self.labelencoder:
            joblib.dump(self.labelencoder, scalers_dir / "heart_label_encoders.pkl")
        return self

def preprocess_heart_data(filepath):
    preprocessor = HeartPreprocessor(filepath)
    (preprocessor.load_data()
     .exploratory_analysis()
     .handle_missing_values()
     .encode_categorical_variables()
     .detect_and_handle_outliers()
     .feature_engineering()
     .check_duplicates()
     .train_test_split_data()
     .scale_features()
     .save_processed_data())
    return preprocessor.X_train, preprocessor.X_test, preprocessor.y_train, preprocessor.y_test, preprocessor
