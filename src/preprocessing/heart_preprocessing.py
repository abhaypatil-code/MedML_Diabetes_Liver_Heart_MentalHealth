import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from src.preprocessing.base_preprocessor import BasePreprocessor
import pickle

class HeartPreprocessor(BasePreprocessor):
    def __init__(self, filepath):
        super().__init__(filepath, "heart")
        self.label_encoders = {}
        self.imputers = {}

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        self.logger.info(f"Loaded data from {self.filepath} with shape {self.df.shape}")
        if "PatientID" in self.df.columns:
            self.df = self.df.drop(columns=["PatientID"])
            self.logger.info("Dropped PatientID column")
        return self

    def clean_data(self):
        # Handle missing values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=[object]).columns

        if not numeric_cols.empty:
            imputer_num = SimpleImputer(strategy='mean')
            self.df[numeric_cols] = imputer_num.fit_transform(self.df[numeric_cols])
            self.imputers['numeric'] = imputer_num
        
        if not categorical_cols.empty:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            self.df[categorical_cols] = imputer_cat.fit_transform(self.df[categorical_cols])
            self.imputers['categorical'] = imputer_cat
        
        self.logger.info("Imputed missing values (Mean for numeric, Most Frequent for categorical)")

        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            self.label_encoders[col] = le
        
        # Save all artifacts
        artifacts = {
            'label_encoders': self.label_encoders,
            'imputers': self.imputers
        }
        with open(self.scalers_dir / "heart_artifacts.pkl", "wb") as f:
            pickle.dump(artifacts, f)
        self.logger.info("Encoded categorical variables and saved artifacts")

        # Outlier handling
        target_col = "Heart_Attack_Risk"
        if target_col in self.df.columns:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.difference([target_col])
            for col in numeric_cols:
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
        # BP features
        if 'Systolic_BP' in self.df.columns and 'Diastolic_BP' in self.df.columns:
            self.df["SystolicDiastolicRatio"] = self.df["Systolic_BP"] / (self.df["Diastolic_BP"] + 1e-6)
            self.df['BP_Difference'] = self.df['Systolic_BP'] - self.df['Diastolic_BP']
            self.logger.info("Created BP features: SystolicDiastolicRatio, BP_Difference")
        return self

    def split_data(self):
        target_col = "Heart_Attack_Risk"
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.logger.info(f"Split data into Train ({self.X_train.shape}) and Test ({self.X_test.shape})")
        return self

    def normalize_data(self):
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)
        
        with open(self.scalers_dir / "heart_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        self.logger.info("Normalized data and saved scaler")
        return self

def preprocess_heart_data(filepath="data/raw/heart.csv"):
    preprocessor = HeartPreprocessor(filepath)
    return preprocessor.run_pipeline()

if __name__ == "__main__":
    preprocess_heart_data()
