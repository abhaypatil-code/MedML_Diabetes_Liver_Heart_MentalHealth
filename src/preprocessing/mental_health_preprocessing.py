import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from src.preprocessing.base_preprocessor import BasePreprocessor
import pickle

class MentalHealthPreprocessor(BasePreprocessor):
    def __init__(self, filepath, target="depressiveness"):
        super().__init__(filepath, "mental_health")
        self.target = target
        self.labelencoders = {}
        self.imputers = {}
        self.outlierbounds = {}

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        self.logger.info(f"Loaded data from {self.filepath} with shape {self.df.shape}")
        return self

    def clean_data(self):
        # Clean inconsistencies
        repmap = {
            "depression_severity": {"none": "None-minimal"},
            "anxiety_severity": {"0": "None-minimal"},
            "who_bmi": {"Not Availble": np.nan},
        }
        for col, mapping in repmap.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(mapping)
        
        # Convert boolean columns
        boolcols = ["depressiveness", "suicidal", "depression_diagnosis", "depression_treatment",
                    "anxiousness", "anxiety_diagnosis", "anxiety_treatment", "sleepiness"]
        for col in boolcols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0, "NAN": 0})
                self.df[col] = self.df[col].fillna(0).astype(int)
        
        # Drop unnecessary columns
        if "id" in self.df.columns:
            self.df = self.df.drop("id", axis=1)
            
        self.logger.info("Cleaned inconsistencies and converted boolean columns")
        return self

    def feature_engineering(self):
        targetcols = ["depressiveness", "anxiousness", "sleepiness"]
        for col in targetcols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0).astype(int)
        
        self.df["MentalHealthRisk"] = self.df.get("depressiveness", 0) + self.df.get("anxiousness", 0) + self.df.get("sleepiness", 0)
        
        if "phq_score" in self.df.columns and "gad_score" in self.df.columns:
            self.df["PHQGADCombined"] = self.df["phq_score"] + self.df["gad_score"]
        if "phq_score" in self.df.columns:
            self.df["ClinicalDepression"] = (self.df["phq_score"] >= 10).astype(int)
        if "age" in self.df.columns and "gad_score" in self.df.columns:
            self.df["AgeGAD"] = self.df["age"] * self.df["gad_score"]
        if "suicidal" in self.df.columns and "depressiveness" in self.df.columns:
            self.df["HighRiskProfile"] = ((self.df["suicidal"] == 1) & (self.df["depressiveness"] == 1)).astype(int)
            
        self.logger.info("Performed feature engineering")
        return self

    def split_data(self):
        targetcols = ["depressiveness", "anxiousness", "sleepiness"]
        
        # Select only numeric and non-target features for X
        X = self.df.drop(targetcols, axis=1, errors='ignore')
        # Keep object type for now as we encode later
        # X = X.select_dtypes(include=[np.number, 'object']) 
        
        y = self.df[self.target]
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.logger.info(f"Split data for target '{self.target}' into Train ({self.X_train.shape}) and Test ({self.X_test.shape})")
        return self

    def normalize_data(self):
        # Fit on train, transform both
        Xtrain = self.X_train.copy()
        Xtest = self.X_test.copy()
        
        num_cols = Xtrain.select_dtypes(include=[np.number]).columns
        cat_cols = Xtrain.select_dtypes(include=['object']).columns

        # Numeric Imputation (Median)
        if not num_cols.empty:
            num_imputer = SimpleImputer(strategy="median")
            Xtrain[num_cols] = num_imputer.fit_transform(Xtrain[num_cols])
            Xtest[num_cols] = num_imputer.transform(Xtest[num_cols])
            self.imputers['numeric'] = num_imputer

        # Categorical Imputation (Mode)
        if not cat_cols.empty:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            Xtrain[cat_cols] = cat_imputer.fit_transform(Xtrain[cat_cols])
            Xtest[cat_cols] = cat_imputer.transform(Xtest[cat_cols])
            self.imputers['categorical'] = cat_imputer

        # Ordinal Mappings
        bmimapping = {"Underweight": 0, "Normal": 1, "Overweight": 2,
                      "Class I Obesity": 3, "Class II Obesity": 4, "Class III Obesity": 5}
        depmapping = {"None-minimal": 0, "Mild": 1, "Moderate": 2,
                      "Moderately severe": 3, "Severe": 4}
        anxmapping = {"None-minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
        
        map_dict = {
            "who_bmi": (bmimapping, 1),
            "depression_severity": (depmapping, 0),
            "anxiety_severity": (anxmapping, 0)
        }

        for col, (mapping, default) in map_dict.items():
            if col in Xtrain.columns:
                Xtrain[col] = Xtrain[col].map(mapping).fillna(default)
                Xtest[col] = Xtest[col].map(mapping).fillna(default)

        # Nominal Encoding
        if "gender" in Xtrain.columns:
            le = LabelEncoder()
            Xtrain["gender"] = le.fit_transform(Xtrain["gender"].astype(str))
            Xtest["gender"] = le.transform(Xtest["gender"].astype(str))
            self.labelencoders["gender"] = le

        # Outlier Handling
        continuouscols = ["age", "bmi", "phq_score", "gad_score", "epworth_score"]
        for col in continuouscols:
            if col in Xtrain.columns:
                Q1 = Xtrain[col].quantile(0.25)
                Q3 = Xtrain[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                self.outlierbounds[col] = (lower, upper)
                Xtrain[col] = np.clip(Xtrain[col], lower, upper)
                Xtest[col] = np.clip(Xtest[col], lower, upper)

        # Final Normalization
        Xtrain = Xtrain.select_dtypes(include=[np.number])
        Xtest = Xtest.select_dtypes(include=[np.number])
        
        scaler = StandardScaler()
        self.X_train = pd.DataFrame(scaler.fit_transform(Xtrain), columns=Xtrain.columns, index=Xtrain.index)
        self.X_test = pd.DataFrame(scaler.transform(Xtest), columns=Xtest.columns, index=Xtest.index)
        
        # Save artifacts
        preprocessor_artifacts = {
            'scaler': scaler,
            'labelencoders': self.labelencoders,
            'imputers': self.imputers,
            'outlierbounds': self.outlierbounds
        }
        with open(self.scalers_dir / f"mental_health_{self.target}_preprocessors.pkl", "wb") as f:
            pickle.dump(preprocessor_artifacts, f)
            
        self.logger.info("Normalized data and saved artifacts")
        return self
        
    def save_splits(self):
        """Override to include target in filename"""
        if self.X_train is not None:
            train_df = pd.concat([self.X_train, self.y_train], axis=1)
            test_df = pd.concat([self.X_test, self.y_test], axis=1)
            
            train_path = self.splits_dir / f"mental_health_{self.target}_train.csv"
            test_path = self.splits_dir / f"mental_health_{self.target}_test.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            self.logger.info(f"Saved splits to {self.splits_dir}")

def preprocess_mental_health_data(filepath="data/raw/mental_health.csv", target="depressiveness"):
    preprocessor = MentalHealthPreprocessor(filepath, target)
    return preprocessor.run_pipeline()

if __name__ == "__main__":
    preprocess_mental_health_data()