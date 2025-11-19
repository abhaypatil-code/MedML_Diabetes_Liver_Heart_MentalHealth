import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.preprocessing.base_preprocessor import BasePreprocessor
import pickle

class DiabetesPreprocessor(BasePreprocessor):
    """
    Comprehensive preprocessing class for Diabetes prediction dataset
    """
    def __init__(self, filepath):
        super().__init__(filepath, "diabetes")
        
    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        self.logger.info(f"Loaded data from {self.filepath} with shape {self.df.shape}")
        return self

    def clean_data(self):
        # Handle zero values
        zeroinvalidcols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        for col in zeroinvalidcols:
            self.df[col] = self.df[col].replace(0, np.nan)
        self.logger.info("Replaced 0 with NaN in medically invalid columns")

        # Impute missing values
        imputer = SimpleImputer(strategy="median")
        X = self.df.drop("Outcome", axis=1)
        y = self.df["Outcome"]
        X_imputed = imputer.fit_transform(X)
        X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
        self.df = pd.concat([X_imputed_df, y.reset_index(drop=True)], axis=1)
        self.logger.info("Imputed missing values with median")

        # Outlier handling
        numericcols = self.df.drop("Outcome", axis=1).columns
        for col in numericcols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            self.df[col] = np.clip(self.df[col], lower, upper)
        self.logger.info("Handled outliers using IQR capping")
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        return self

    def feature_engineering(self):
        self.df["AgeGroup"] = pd.cut(self.df["Age"], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
        self.df["BMICategory"] = pd.cut(self.df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
        self.df["GlucoseCategory"] = pd.cut(self.df["Glucose"], bins=[0, 99, 125, 200], labels=[0, 1, 2]).astype(int)
        self.df["BMIAgeInteraction"] = self.df["BMI"] * self.df["Age"]
        self.df["GlucoseBMIInteraction"] = self.df["Glucose"] * self.df["BMI"]
        self.logger.info("Created new features: AgeGroup, BMICategory, GlucoseCategory, Interactions")
        return self

    def split_data(self):
        X = self.df.drop("Outcome", axis=1)
        y = self.df["Outcome"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.logger.info(f"Split data into Train ({self.X_train.shape}) and Test ({self.X_test.shape})")
        return self

    def normalize_data(self):
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)
        
        # Save scaler
        with open(self.scalers_dir / "diabetes_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        self.logger.info("Normalized data and saved scaler")
        return self

def preprocess_diabetes_data(filepath="data/raw/diabetes.csv"):
    preprocessor = DiabetesPreprocessor(filepath)
    return preprocessor.run_pipeline()

if __name__ == "__main__":
    preprocess_diabetes_data()