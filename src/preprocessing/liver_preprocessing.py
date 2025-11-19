import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from src.preprocessing.base_preprocessor import BasePreprocessor
import pickle

class LiverPreprocessor(BasePreprocessor):
    def __init__(self, filepath):
        super().__init__(filepath, "liver")
        self.labelencoder = None

    def load_data(self):
        self.df = pd.read_csv(self.filepath, header=None)
        self.df.columns = [
            "Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot",
            "TP", "ALB", "AGRatio", "Selector"
        ]
        self.logger.info(f"Loaded data from {self.filepath} with shape {self.df.shape}")
        return self

    def clean_data(self):
        # Fix target variable: 1 (Disease) -> 1, 2 (No Disease) -> 0
        self.df["Selector"] = self.df["Selector"].map({1: 1, 2: 0})
        self.logger.info("Mapped target variable: 1->1, 2->0")

        # Impute missing values (AGRatio)
        imputer = SimpleImputer(strategy='median')
        self.df["AGRatio"] = imputer.fit_transform(self.df[["AGRatio"]])
        self.logger.info("Imputed missing AGRatio with median")

        # Encode Gender
        self.labelencoder = LabelEncoder()
        self.df["Gender"] = self.labelencoder.fit_transform(self.df["Gender"])
        
        # Save label encoder
        with open(self.scalers_dir / "liver_label_encoder.pkl", "wb") as f:
            pickle.dump(self.labelencoder, f)
        self.logger.info("Encoded Gender and saved label encoder")

        # Outlier handling
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ["Selector", "Gender"]]
        
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            multiplier = 3 if col in ["TB", "DB", "Alkphos", "Sgpt", "Sgot"] else 1.5
            lower = Q1 - multiplier * IQR
            upper = Q3 + multiplier * IQR
            self.df[col] = np.clip(self.df[col], lower, upper)
        self.logger.info("Handled outliers using IQR capping")

        # Remove duplicates
        self.df = self.df.drop_duplicates()
        return self

    def feature_engineering(self):
        self.df["BilirubinRatio"] = self.df["DB"] / (self.df["TB"] + 1e-6)
        self.df["SGPTSGOTRatio"] = self.df["Sgpt"] / (self.df["Sgot"] + 1e-6)
        self.df["TotalEnzymes"] = self.df["Sgpt"] + self.df["Sgot"]
        self.df["AgeGroup"] = pd.cut(self.df["Age"], bins=[0, 30, 45, 60, 100], labels=[0,1,2,3]).astype(int)
        self.df["LowProtein"] = (self.df["TP"] < 6.0).astype(int)
        self.df["HighEnzymes"] = ((self.df["Sgpt"] > 40) | (self.df["Sgot"] > 40)).astype(int)
        self.df["AgeGenderInteraction"] = self.df["Age"] * self.df["Gender"]
        
        # Handle any infs created
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.fillna(0, inplace=True)
        self.logger.info("Created new features: BilirubinRatio, SGPTSGOTRatio, TotalEnzymes, etc.")
        return self

    def split_data(self):
        X = self.df.drop("Selector", axis=1)
        y = self.df["Selector"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.logger.info(f"Split data into Train ({self.X_train.shape}) and Test ({self.X_test.shape})")
        return self

    def normalize_data(self):
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(self.X_train), columns=self.X_train.columns)
        self.X_test = pd.DataFrame(scaler.transform(self.X_test), columns=self.X_test.columns)
        
        with open(self.scalers_dir / "liver_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        self.logger.info("Normalized data and saved scaler")
        return self

def preprocess_liver_data(filepath="data/raw/liver.csv"):
    preprocessor = LiverPreprocessor(filepath)
    return preprocessor.run_pipeline()

if __name__ == "__main__":
    preprocess_liver_data()