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

class MentalHealthPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.dforiginal = None
        self.scaler = None
        self.labelencoders = {}
        self.featurenames = None
        self.visdir = Path("results/mental_health/visualizations")
        self.visdir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        self.dforiginal = self.df.copy()
        return self

    def exploratory_analysis(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        missing_data = self.df.isnull().sum()
        axes[0, 0].barh(missing_data.index, missing_data.values, color="coral")
        axes[0, 0].set_xlabel("Count")
        axes[0, 0].set_title("Missing Values per Column", fontweight="bold")
        axes[0, 0].grid(axis='x', alpha=0.3)

        if "gender" in self.df.columns:
            gender_counts = self.df["gender"].value_counts()
            axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                            colors=["skyblue", "lightcoral"])
            axes[0, 1].set_title("Gender Distribution", fontweight="bold")
        else:
            axes[0, 1].axis('off')

        targetcols = ["depressiveness", "anxiousness", "sleepiness"]
        prevalence = [self.df[col].astype(str).str.upper().eq('TRUE').sum() / len(self.df) * 100 if col in self.df else 0 for col in targetcols]
        if prevalence:
            bars = axes[1, 0].barh(targetcols, prevalence, color='salmon')
            axes[1, 0].set_xlabel("Prevalence (%)")
            axes[1, 0].set_title("Mental Health Condition Prevalence", fontweight="bold")
            axes[1, 0].grid(axis='x', alpha=0.3)
        else:
            axes[1, 0].axis('off')

        statstext = f"""Dataset Statistics
Total Samples: {self.df.shape[0]}
Features: {self.df.shape[1]}
Clinical Scales: PHQ-9, GAD-7, Epworth
Gender Distribution: Male = {self.df['gender'].value_counts().get('male', 0)}, Female = {self.df['gender'].value_counts().get('female', 0)}
Conditions Prevalence:
  Depression: {prevalence[0] if len(prevalence) > 0 else 'NA'}%
  Anxiety: {prevalence[1] if len(prevalence) > 1 else 'NA'}%
  Sleepiness: {prevalence[2] if len(prevalence) > 2 else 'NA'}%
"""
        axes[1, 1].text(0.1, 0.5, statstext, fontsize=11, verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
        axes[1, 1].axis("off")
        plt.tight_layout()
        plt.savefig(self.visdir / "01_dataset_overview.png", dpi=300, bbox_inches="tight")
        plt.close()

        clinicalfeatures = ["phq_score", "gad_score", "epworth_score", "bmi", "age"]
        ncols = 3
        nrows = int(np.ceil(len(clinicalfeatures) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
        axes = axes.flatten()
        for idx, feature in enumerate(clinicalfeatures):
            if feature in self.df.columns:
                axes[idx].hist(self.df[feature].dropna(), bins=30, color="steelblue", edgecolor="black", alpha=0.7)
                axes[idx].set_title(f"{feature} Distribution", fontweight="bold")
                axes[idx].set_xlabel(feature)
                axes[idx].set_ylabel("Frequency")
        for idx in range(len(clinicalfeatures), len(axes)):
            axes[idx].axis("off")
        plt.tight_layout()
        plt.savefig(self.visdir / "02_clinical_scales.png", dpi=300, bbox_inches="tight")
        plt.close()

        numericcols = self.df.select_dtypes(include=[np.number]).columns
        if not numericcols.empty:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df[numericcols].corr()
            sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap="coolwarm", center=0,
                        square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
            plt.tight_layout()
            plt.savefig(self.visdir / "03_correlation_matrix.png", dpi=300, bbox_inches="tight")
            plt.close()
        return self

    def clean_inconsistencies(self):
        repmap = {
            "depression_severity": {"none": "None-minimal"},
            "anxiety_severity": {"0": "None-minimal"},
            "who_bmi": {"Not Availble": np.nan},
        }
        for col, mapping in repmap.items():
            if col in self.df.columns:
                self.df[col] = self.df[col].replace(mapping)
        return self

    def handle_missing_values(self):
        missingbefore = self.df.isnull().sum()
        boolcols = ["depressiveness", "suicidal", "depression_diagnosis", "depression_treatment",
                    "anxiousness", "anxiety_diagnosis", "anxiety_treatment", "sleepiness"]
        for col in boolcols:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()
                if len(mode_val) > 0:
                    self.df[col].fillna(mode_val[0], inplace=True)
        
        catcols = ["depression_severity", "anxiety_severity", "who_bmi"]
        for col in catcols:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()
                if len(mode_val) > 0:
                    self.df[col].fillna(mode_val[0], inplace=True)
        
        clincols = ["epworth_score", "bmi"]
        for col in clincols:
            if col in self.df.columns and self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)

        if missingbefore.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(missingbefore.index, missingbefore.values, color="orange")
            ax.set_xlabel("Number of Missing Values Imputed")
            ax.set_title("Missing Values Handled", fontsize=14, fontweight="bold")
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.visdir / "05_missing_value_imputation.png", dpi=300, bbox_inches="tight")
            plt.close()
        return self

    def convert_boolean_columns(self):
        boolcols = ["depressiveness", "suicidal", "depression_diagnosis", "depression_treatment",
                    "anxiousness", "anxiety_diagnosis", "anxiety_treatment", "sleepiness"]
        for col in boolcols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0, "NAN": 0})
                self.df[col] = self.df[col].fillna(0).astype(int)
        return self

    def drop_unnecessary_columns(self):
        if "id" in self.df.columns:
            self.df = self.df.drop("id", axis=1)
        return self

    def encode_categorical_variables(self):
        if "gender" in self.df.columns:
            self.labelencoders["gender"] = LabelEncoder()
            self.df["gender"] = self.labelencoders["gender"].fit_transform(self.df["gender"].astype(str))
        
        if "who_bmi" in self.df.columns:
            bmimapping = {"Underweight": 0, "Normal": 1, "Overweight": 2,
                          "Class I Obesity": 3, "Class II Obesity": 4, "Class III Obesity": 5}
            self.df["who_bmi"] = self.df["who_bmi"].map(bmimapping)
            self.df["who_bmi"] = self.df["who_bmi"].fillna(1)
        
        if "depression_severity" in self.df.columns:
            depmapping = {"None-minimal": 0, "Mild": 1, "Moderate": 2,
                          "Moderately severe": 3, "Severe": 4}
            self.df["depression_severity"] = self.df["depression_severity"].map(depmapping)
            self.df["depression_severity"] = self.df["depression_severity"].fillna(0)
        
        if "anxiety_severity" in self.df.columns:
            anxmapping = {"None-minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
            self.df["anxiety_severity"] = self.df["anxiety_severity"].map(anxmapping)
            self.df["anxiety_severity"] = self.df["anxiety_severity"].fillna(0)
        return self

    def detect_and_handle_outliers(self):
        continuouscols = ["age", "bmi", "phq_score", "gad_score", "epworth_score"]
        outlier_summary = []
        for col in continuouscols:
            if col in self.df.columns:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
                if outliers > 0:
                    outlier_summary.append((col, outliers))
                self.df[col] = np.clip(self.df[col], lower, upper)
        
        if outlier_summary:
            outlierdf = pd.DataFrame(outlier_summary, columns=["Feature", "Outliers"])
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(outlierdf["Feature"], outlierdf["Outliers"], color="tomato")
            ax.set_xlabel("Number of Outliers Capped")
            ax.set_title("Outlier Detection and Handling (IQR Method)", fontsize=14, fontweight="bold")
            ax.grid(axis='x', alpha=0.3)
            for bar, outliers in zip(bars, outlierdf["Outliers"]):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2., f"{outliers} capped", ha='left', va='center')
            plt.tight_layout()
            plt.savefig(self.visdir / "07_outlier_handling.png", dpi=300, bbox_inches="tight")
            plt.close()
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
        if "gad_score" in self.df.columns:
            self.df["ClinicalAnxiety"] = (self.df["gad_score"] >= 10).astype(int)
        if "epworth_score" in self.df.columns:
            self.df["ClinicalSleepiness"] = (self.df["epworth_score"] >= 11).astype(int)
        
        self.df["ComorbidConditions"] = ((self.df.get("depressiveness", 0) + self.df.get("anxiousness", 0)) >= 2).astype(int)
        
        if "depressiveness" in self.df.columns and "depression_treatment" in self.df.columns:
            self.df["DepressionTreatmentGap"] = ((self.df["depressiveness"] == 1) & (self.df["depression_treatment"] == 0)).astype(int)
        if "anxiousness" in self.df.columns and "anxiety_treatment" in self.df.columns:
            self.df["AnxietyTreatmentGap"] = ((self.df["anxiousness"] == 1) & (self.df["anxiety_treatment"] == 0)).astype(int)
        
        if "depression_severity" in self.df.columns:
            self.df["HighSeverity"] = (self.df["depression_severity"].fillna(0) >= 3).astype(int)
        if "who_bmi" in self.df.columns:
            self.df["BMIRisk"] = (self.df["who_bmi"].fillna(0) >= 3).astype(int)
        if "age" in self.df.columns:
            self.df["AgeGroup"] = pd.cut(self.df["age"], bins=[0, 19, 21, 25, 100], labels=[0, 1, 2, 3]).astype(float)
        if "school_year" in self.df.columns:
            self.df["AcademicPressure"] = (self.df["school_year"] >= 3).astype(int)
        if "bmi" in self.df.columns and "MentalHealthRisk" in self.df.columns:
            self.df["BMIMentalHealth"] = self.df["bmi"] * self.df["MentalHealthRisk"]
        if "age" in self.df.columns and "phq_score" in self.df.columns:
            self.df["AgePHQ"] = self.df["age"] * self.df["phq_score"]
        if "age" in self.df.columns and "gad_score" in self.df.columns:
            self.df["AgeGAD"] = self.df["age"] * self.df["gad_score"]
        if "suicidal" in self.df.columns and "depressiveness" in self.df.columns:
            self.df["HighRiskProfile"] = ((self.df["suicidal"] == 1) & (self.df["depressiveness"] == 1)).astype(int)
        
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if self.df.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            numericcols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numericcols] = imputer.fit_transform(self.df[numericcols])
        return self

    def check_duplicates(self):
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.df = self.df.drop_duplicates()
        return self

    def train_test_split_data(self, target="depressiveness", test_size=0.2, random_state=42):
        targetcols = ["depressiveness", "anxiousness", "sleepiness"]
        X = self.df.drop(targetcols, axis=1, errors='ignore')
        X = X.select_dtypes(include=[np.number])
        
        y = self.df[targetcols]
        y_target = y[target] if target in y else y[targetcols[0]]
        
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(
            X, y_target, test_size=test_size, random_state=random_state, stratify=y_target
        )
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        split_sizes = [len(self.Xtrain), len(self.Xtest)]
        axes[0].bar(["Train", "Test"], split_sizes, color=["steelblue", "coral"])
        axes[0].set_ylabel("Number of Samples")
        axes[0].set_title("Train-Test Split")
        axes[0].grid(axis='y', alpha=0.3)
        
        train_dist = self.ytrain.value_counts()
        axes[1].pie(train_dist.values, labels=["No Condition", "Has Condition"],
                    autopct="%1.1f%%", colors=["lightgreen", "salmon"])
        axes[1].set_title("Train Set Class Distribution")
        
        test_dist = self.ytest.value_counts()
        axes[2].pie(test_dist.values, labels=["No Condition", "Has Condition"],
                    autopct="%1.1f%%", colors=["lightgreen", "salmon"])
        axes[2].set_title("Test Set Class Distribution")
        
        plt.tight_layout()
        plt.savefig(self.visdir / "10_train_test_split.png", dpi=300, bbox_inches="tight")
        plt.close()
        return self

    def normalize_features(self):
        self.featurenames = self.Xtrain.columns.tolist()
        self.scaler = StandardScaler()
        
        sample_cols_before = self.Xtrain.iloc[:, :5]
        
        Xtrain_scaled = self.scaler.fit_transform(self.Xtrain)
        Xtest_scaled = self.scaler.transform(self.Xtest)
        
        self.Xtrain = pd.DataFrame(Xtrain_scaled, columns=self.featurenames, index=self.Xtrain.index)
        self.Xtest = pd.DataFrame(Xtest_scaled, columns=self.featurenames, index=self.Xtest.index)
        
        samplefeaturescols = sample_cols_before.columns[:5]
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].boxplot([sample_cols_before[col] for col in samplefeaturescols], labels=samplefeaturescols)
        axes[0].set_title("Before Normalization", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Value")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].boxplot([self.Xtrain[col] for col in samplefeaturescols], labels=samplefeaturescols)
        axes[1].set_title("After Normalization (StandardScaler)", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Scaled Value")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.visdir / "09_feature_normalization.png", dpi=300, bbox_inches="tight")
        plt.close()
        return self

    def save_processed_data(self, target="depressiveness", output_dir="data/processed"):
        output_dir = Path(output_dir)
        splits_dir = Path("data/splits")
        scalers_dir = Path("scalers")
        output_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)
        scalers_dir.mkdir(parents=True, exist_ok=True)
        
        pd.concat([self.Xtrain, self.ytrain], axis=1).to_csv(
            splits_dir / f"mental_health_{target}_train.csv", index=False
        )
        pd.concat([self.Xtest, self.ytest], axis=1).to_csv(
            splits_dir / f"mental_health_{target}_test.csv", index=False
        )
        
        if self.scaler is not None:
            with open(scalers_dir / f"mental_health_{target}_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
        if self.labelencoders:
            with open(scalers_dir / f"mental_health_{target}_labelencoders.pkl", "wb") as f:
                pickle.dump(self.labelencoders, f)
        return self

    def get_preprocessing_summary(self):
        summary = (
            f"Mental Health Prediction - Preprocessing Pipeline Summary\n\n"
            f"Dataset shape: original = {self.dforiginal.shape}, final processed\n"
            f"Preprocessing Steps Completed:\n"
            f"  1. Data Loading\n"
            f"  2. Exploratory Data Analysis\n"
            f"  3. Data cleaning\n"
            f"  4. Missing value imputation\n"
            f"  5. Boolean conversion\n"
            f"  6. Categorical encoding\n"
            f"  7. Outlier handling\n"
            f"  8. Feature engineering\n"
            f"  9. Train-Test split\n"
            f" 10. Feature normalization\n"
            f"Ready for Model Training!\n"
        )
        summary_path = self.visdir.parent / "preprocessing_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        return self

def preprocess_mental_health_data(filepath="data/raw/mental_health.csv", target="depressiveness"):
    preprocessor = MentalHealthPreprocessor(filepath)
    preprocessor.load_data()\
        .exploratory_analysis()\
        .clean_inconsistencies()\
        .handle_missing_values()\
        .convert_boolean_columns()\
        .drop_unnecessary_columns()\
        .encode_categorical_variables()\
        .detect_and_handle_outliers()\
        .feature_engineering()\
        .check_duplicates()\
        .train_test_split_data(target)\
        .normalize_features()\
        .save_processed_data(target)\
        .get_preprocessing_summary()
    return preprocessor.Xtrain, preprocessor.Xtest, preprocessor.ytrain, preprocessor.ytest, preprocessor

if __name__ == "__main__":
    Xtrain, Xtest, ytrain, ytest, preprocessor = preprocess_mental_health_data("data/raw/mental_health.csv")
    print("Preprocessing complete.")
    print(f"X_train shape: {Xtrain.shape}")
    print(f"X_test shape: {Xtest.shape}")