import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings("ignore")

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

class HeartPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.dforiginal = None
        self.scaler = None
        self.labelencoders = {}
        self.imputers = {}
        self.outlierbounds = {}
        self.featurenames = None
        self.visdir = Path("results/heart/visualizations")
        self.visdir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        self.dforiginal = self.df.copy()
        return self

    def exploratory_analysis(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Missing values
        missing_data = self.df.isnull().sum()
        axes[0, 0].barh(missing_data.index, missing_data.values, color="coral")
        axes[0, 0].set_xlabel("Count")
        axes[0, 0].set_title("Missing Values per Column", fontweight="bold")
        axes[0, 0].grid(axis='x', alpha=0.3)

        # Gender distribution
        if "Gender" in self.df.columns:
            gender_counts = self.df["Gender"].value_counts()
            axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                           colors=["skyblue" if label == "Male" else "lightcoral" for label in gender_counts.index])
            axes[0, 1].set_title("Gender Distribution", fontweight="bold")
        else:
            axes[0, 1].axis('off')

        # Target variable
        if "HeartAttackHistory" in self.df.columns:
            target_counts = self.df["HeartAttackHistory"].value_counts()
            bars = axes[1, 0].bar(["No Heart Attack (0)", "Heart Attack (1)"], target_counts.values,
                                  color=['lightgreen', 'salmon'])
            axes[1, 0].set_ylabel("Count")
            axes[1, 0].set_title("Target Variable Distribution", fontweight="bold")
            axes[1, 0].grid(axis='y', alpha=0.3)
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height, 
                                f"{height/self.df.shape[0]*100:.1f}%", ha='center', va='bottom')
        else:
            axes[1, 0].axis('off')

        # Stats box
        statstext = f"""Dataset Statistics
Total Samples: {self.df.shape[0]}
Total Features: {self.df.shape[1] - 1}
Target Variable: HeartAttackHistory
"""
        if "Gender" in self.df.columns:
            gender_counts = self.df["Gender"].value_counts()
            statstext += f"Gender Distribution:\n  Male: {gender_counts.get('Male', 0)} ({gender_counts.get('Male', 0)/len(self.df)*100:.1f}%)\n"
            statstext += f"  Female: {gender_counts.get('Female', 0)} ({gender_counts.get('Female', 0)/len(self.df)*100:.1f}%)\n"
        if "HeartAttackHistory" in self.df.columns:
            target_counts = self.df["HeartAttackHistory"].value_counts()
            statstext += f"Class Distribution:\n  No Attack (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(self.df)*100:.1f}%)\n"
            statstext += f"  Heart Attack (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(self.df)*100:.1f}%)\n"

        axes[1, 1].text(0.1, 0.5, statstext, fontsize=11, verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
        axes[1, 1].axis("off")
        plt.tight_layout()
        plt.savefig(self.visdir / "01_dataset_overview.png", dpi=300, bbox_inches="tight")
        plt.close()
        return self

    def drop_unnecessary_columns(self):
        if "PatientID" in self.df.columns:
            self.df = self.df.drop("PatientID", axis=1)
        return self

    def feature_engineering(self):
        # Example medical features, actual code should ensure all divisions handle zero denominators with +1e-6
        # Cardiovascular risk score
        risk_cols = ["Diabetes", "Hypertension", "Obesity", "Smoking", "AlcoholConsumption", "FamilyHistory"]
        if all(col in self.df.columns for col in risk_cols):
            self.df["CVDRiskScore"] = self.df[risk_cols].sum(axis=1)

        # Cholesterol/HDL/LDL/Pulse/MeanArterialPressure, AgeGroup, High/Low indicators
        if "CholesterolLevel" in self.df.columns and "HDLLevel" in self.df.columns:
            self.df["CholesterolHDLRatio"] = self.df["CholesterolLevel"] / (self.df["HDLLevel"] + 1e-6)
        if "LDLLevel" in self.df.columns and "HDLLevel" in self.df.columns:
            self.df["LDLHDLRatio"] = self.df["LDLLevel"] / (self.df["HDLLevel"] + 1e-6)
        if "SystolicBP" in self.df.columns and "DiastolicBP" in self.df.columns:
            self.df["PulsePressure"] = self.df["SystolicBP"] - self.df["DiastolicBP"]
            self.df["MeanArterialPressure"] = self.df["DiastolicBP"] + self.df["PulsePressure"]/3
        if "Age" in self.df.columns:
            self.df["AgeGroup"] = pd.cut(self.df["Age"], bins=[0, 35, 50, 65, 100], labels=[0,1,2,3]).astype(int)
        if "CholesterolLevel" in self.df.columns:
            self.df["HighCholesterol"] = (self.df["CholesterolLevel"] > 200).astype(int)
        if "LDLLevel" in self.df.columns:
            self.df["HighLDL"] = (self.df["LDLLevel"] > 100).astype(int)
        if "HDLLevel" in self.df.columns:
            self.df["LowHDL"] = (self.df["HDLLevel"] < 45).astype(int)
        if "SystolicBP" in self.df.columns and "DiastolicBP" in self.df.columns:
            self.df["HypertensionStage"] = 0
            self.df.loc[
                (self.df["SystolicBP"] >= 130) | (self.df["DiastolicBP"] >= 80),
                "HypertensionStage"
            ] = 1
            self.df.loc[
                (self.df["SystolicBP"] >= 140) | (self.df["DiastolicBP"] >= 90),
                "HypertensionStage"
            ] = 2
        if all(col in self.df.columns for col in ["PhysicalActivity", "Smoking", "AlcoholConsumption", "DietScore"]):
            self.df["PoorLifestyle"] = (
                (1 - self.df["PhysicalActivity"]) +
                self.df["Smoking"] + self.df["AlcoholConsumption"] + (self.df["DietScore"] < 5).astype(int)
            )
        if all(col in self.df.columns for col in ["Age", "CholesterolLevel", "SystolicBP"]):
            self.df["AgeCholesterol"] = self.df["Age"] * self.df["CholesterolLevel"]
            self.df["AgeBP"] = self.df["Age"] * self.df["SystolicBP"]
        if all(col in self.df.columns for col in ["Obesity", "Diabetes", "Hypertension", "HighCholesterol"]):
            self.df["MetabolicSyndrome"] = (
                (self.df["Obesity"]==1) & (self.df["Diabetes"]==1) & 
                (self.df["Hypertension"]==1) & (self.df["HighCholesterol"]==1)
            ).astype(int)
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        return self

    def check_duplicates(self):
        if self.df.duplicated().sum() > 0:
            self.df = self.df.drop_duplicates()
        return self

    def train_test_split_data(self, test_size=0.2, random_state=42):
        X = self.df.drop("HeartAttackHistory", axis=1)
        y = self.df["HeartAttackHistory"]
        self.featurenames = X.columns.tolist()
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Visualization of train-test split
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        split_sizes = [len(self.Xtrain), len(self.Xtest)]
        axes[0].bar(["Train", "Test"], split_sizes, color=["steelblue", "coral"])
        axes[0].set_ylabel("Number of Samples")
        axes[0].set_title("Train-Test Split Distribution")
        axes[0].grid(axis='y', alpha=0.3)
        for i, count in enumerate(split_sizes):
            axes[0].text(i, count, f"{count} ({count/sum(split_sizes)*100:.1f}%)", ha='center', va='bottom')
        train_dist = self.ytrain.value_counts()
        axes[1].pie(train_dist.values, labels=["No Attack (0)", "Heart Attack (1)"],
                    autopct="%1.1f%%", colors=["lightgreen", "salmon"])
        axes[1].set_title("Train Set Class Distribution")
        test_dist = self.ytest.value_counts()
        axes[2].pie(test_dist.values, labels=["No Attack (0)", "Heart Attack (1)"],
                    autopct="%1.1f%%", colors=["lightgreen", "salmon"])
        axes[2].set_title("Test Set Class Distribution")
        plt.tight_layout()
        plt.savefig(self.visdir / "10_train_test_split.png", dpi=300, bbox_inches="tight")
        plt.close()
        return self

    def handle_missing_values(self):
        # Fit on train, transform train/test for each column type
        Xtrain = self.Xtrain.copy()
        Xtest = self.Xtest.copy()
        for col in Xtrain.columns:
            if Xtrain[col].dtype == "object":
                imputer = SimpleImputer(strategy="most_frequent")
            else:
                imputer = SimpleImputer(strategy="median")
            Xtrain[[col]] = imputer.fit_transform(Xtrain[[col]])
            Xtest[[col]] = imputer.transform(Xtest[[col]])
            self.imputers[col] = imputer
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        return self

    def encode_categorical_variables(self):
        # Fit on train, transform both splits
        Xtrain = self.Xtrain.copy()
        Xtest = self.Xtest.copy()
        for col in Xtrain.columns:
            if Xtrain[col].dtype == "object":
                encoder = LabelEncoder()
                Xtrain[col] = encoder.fit_transform(Xtrain[col])
                Xtest[col] = encoder.transform(Xtest[col])
                self.labelencoders[col] = encoder
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        return self

    def detect_and_handle_outliers(self):
        # IQR (fit on train, transform both)
        Xtrain = self.Xtrain.copy()
        Xtest = self.Xtest.copy()
        for col in Xtrain.select_dtypes(include=[np.number]).columns:
            Q1 = Xtrain[col].quantile(0.25)
            Q3 = Xtrain[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            Xtrain[col] = np.clip(Xtrain[col], lower, upper)
            Xtest[col] = np.clip(Xtest[col], lower, upper)
            self.outlierbounds[col] = (lower, upper)
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        return self

    def normalize_features(self):
        Xtrain = self.Xtrain.copy()
        Xtest = self.Xtest.copy()
        self.scaler = StandardScaler()
        Xtrain[:] = self.scaler.fit_transform(Xtrain)
        Xtest[:] = self.scaler.transform(Xtest)
        # Visual check
        samplecols = Xtrain.columns[:5]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].boxplot([self.Xtrain[col] for col in samplecols], labels=samplecols)
        axes[0].set_title("Before Normalization (Train Set)", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Value")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        axes[1].boxplot([Xtrain[col] for col in samplecols], labels=samplecols)
        axes[1].set_title("After Normalization (Train Set)", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Scaled Value")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.visdir / "09_feature_normalization.png", dpi=300, bbox_inches="tight")
        plt.close()
        self.Xtrain = Xtrain
        self.Xtest = Xtest
        return self

    def save_processed_data(self, output_dir="data/processed"):
        output_dir = Path(output_dir)
        splits_dir = Path("data/splits")
        scalers_dir = Path("scalers")
        output_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)
        scalers_dir.mkdir(parents=True, exist_ok=True)
        # Save full processed data (combine splits back for overall .csv)
        full_processed = pd.concat([
            pd.concat([self.Xtrain, self.ytrain.reset_index(drop=True)], axis=1),
            pd.concat([self.Xtest, self.ytest.reset_index(drop=True)], axis=1)
        ])
        full_processed.to_csv(output_dir / "heart_processed.csv", index=False)
        # Save splits
        pd.concat([self.Xtrain, self.ytrain.reset_index(drop=True)], axis=1).to_csv(splits_dir / "heart_train.csv", index=False)
        pd.concat([self.Xtest, self.ytest.reset_index(drop=True)], axis=1).to_csv(splits_dir / "heart_test.csv", index=False)
        # Save scaler, encoders, imputers, outlier bounds
        if self.scaler is not None:
            with open(scalers_dir / "heart_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
        if self.labelencoders:
            with open(scalers_dir / "heart_labelencoders.pkl", "wb") as f:
                pickle.dump(self.labelencoders, f)
        if self.imputers:
            with open(scalers_dir / "heart_imputers.pkl", "wb") as f:
                pickle.dump(self.imputers, f)
        if self.outlierbounds:
            with open(scalers_dir / "heart_outlierbounds.pkl", "wb") as f:
                pickle.dump(self.outlierbounds, f)
        return self

    def get_preprocessing_summary(self):
        summary = (
            f"Heart Disease Prediction - Preprocessing Pipeline Summary\n\n"
            f"Dataset Information:\n"
            f"  Original shape: {self.dforiginal.shape}\n"
            f"  Final shape: {self.df.shape}\n"
            f"  Total Features after engineering: {self.df.shape[1] - 1}\n"
            f"  Target variable: HeartAttackHistory\n\n"
            f"Preprocessing Steps Completed (Barriers Prevent Data Leakage):\n"
            f"  1. Data Loading\n"
            f"  2. Exploratory Data Analysis (visualizations saved)\n"
            f"  3. Unnecessary Column Removal (PatientID)\n"
            f"  4. Feature Engineering (medical features)\n"
            f"  5. Duplicate Removal\n"
            f"  6. Train-Test Split (80-20, stratified)\n"
            f"--- Data Leakage Prevention Done ---\n"
            f"  7. Missing Value Imputation (fit on train only)\n"
            f"  8. Categorical Encoding (fit on train only)\n"
            f"  9. Outlier Detection & Capping (fit on train only)\n"
            f"  10. Feature Normalization (StandardScaler, fit on train only)\n"
            f"Artifacts Saved:\n"
            f"  - Processed dataset (data/processed/heart_processed.csv)\n"
            f"  - Train split (data/splits/heart_train.csv)\n"
            f"  - Test split (data/splits/heart_test.csv)\n"
            f"  - Fitted scaler (scalers/heart_scaler.pkl)\n"
            f"  - Fitted encoders/imputers/outlier_bounds (scalers/)\n"
            f"  - Visualizations (results/heart/visualizations/)\n"
            f"Ready for Model Training!\n"
        )
        summary_path = self.visdir.parent / "preprocessing_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        return self

def preprocess_heart_data(filepath="data/raw/heart.csv"):
    preprocessor = HeartPreprocessor(filepath)
    preprocessor.load_data()\
                .exploratory_analysis()\
                .drop_unnecessary_columns()\
                .feature_engineering()\
                .check_duplicates()\
                .train_test_split_data()\
                .handle_missing_values()\
                .encode_categorical_variables()\
                .detect_and_handle_outliers()\
                .normalize_features()\
                .save_processed_data()\
                .get_preprocessing_summary()
    return preprocessor.Xtrain, preprocessor.Xtest, preprocessor.ytrain, preprocessor.ytest, preprocessor

if __name__ == "__main__":
    Xtrain, Xtest, ytrain, ytest, preprocessor = preprocess_heart_data("data/raw/heart.csv")
    print("Preprocessing completed. Processed data and artifacts saved.")