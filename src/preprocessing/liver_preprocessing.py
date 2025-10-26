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

class LiverPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.dforiginal = None
        self.scaler = None
        self.labelencoder = None
        self.featurenames = None
        self.visdir = Path("results/liver/visualizations")
        self.visdir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        # Load liver dataset with no header and assign correct column names
        self.df = pd.read_csv(self.filepath, header=None)
        self.df.columns = [
            "Age", "Gender", "TB", "DB", "Alkphos", "Sgpt", "Sgot",
            "TP", "ALB", "AGRatio", "Selector"
        ]
        self.dforiginal = self.df.copy()
        return self

    def exploratory_analysis(self):
        # Missing values visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Missing values per column
        missing_data = self.df.isnull().sum()
        axes[0, 0].barh(missing_data.index, missing_data.values, color="coral")
        axes[0, 0].set_xlabel("Count")
        axes[0, 0].set_title("Missing Values per Column", fontweight="bold")
        axes[0, 0].grid(axis='x', alpha=0.3)

        # Gender distribution
        gender_counts = self.df["Gender"].value_counts()
        axes[0, 1].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
                       colors=["skyblue", "lightcoral"])
        axes[0, 1].set_title("Gender Distribution", fontweight="bold")

        # Target variable distribution
        target_counts = self.df["Selector"].value_counts().sort_index()
        bars = axes[1, 0].bar(["Liver Disease (1)", "No Disease (0)"], target_counts.values,
                              color=['salmon', 'lightgreen'])
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Target Variable Distribution (Original)", fontweight="bold")
        axes[1, 0].grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                            f"{height/self.df.shape[0]*100:.1f}%", ha='center', va='bottom')

        # Dataset statistics text
        statstext = f"""Dataset Statistics
Total Samples: {self.df.shape[0]}
Features: {self.df.shape[1] - 1}
Target Variable: Selector
Gender Distribution:
  Male: {gender_counts.get('Male', 0)} ({gender_counts.get('Male', 0)/len(self.df)*100:.1f}%)
  Female: {gender_counts.get('Female', 0)} ({gender_counts.get('Female', 0)/len(self.df)*100:.1f}%)
Class Distribution:
  Liver Disease (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(self.df)*100:.1f}%)
  No Disease (0): {target_counts.get(2, 0)} ({target_counts.get(2, 0)/len(self.df)*100:.1f}%)
"""
        axes[1, 1].text(0.05, 0.5, statstext, fontsize=11, family='monospace',
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
        axes[1, 1].axis("off")
        plt.tight_layout()
        plt.savefig(self.visdir / "01_dataset_overview.png", dpi=300, bbox_inches="tight")
        plt.close()
        return self

    def fix_target_variable(self):
        # Convert Selector from 1, 2 to binary 1,0 for liver disease prediction
        self.df["Selector"] = self.df["Selector"].map({1:1, 2:0})
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        before_counts = self.dforiginal["Selector"].value_counts().sort_index()
        axes[0].bar(["Liver Disease (1)", "No Disease (2)"], before_counts.values,
                    color=['salmon', 'lightgreen'])
        axes[0].set_title("Before Conversion")
        axes[0].set_ylabel("Count")
        axes[0].grid(axis='y', alpha=0.3)
        
        after_counts = self.df["Selector"].value_counts().sort_index()
        axes[1].bar(["No Disease (0)", "Liver Disease (1)"], after_counts.values,
                    color=['lightgreen', 'salmon'])
        axes[1].set_title("After Conversion to Binary")
        axes[1].set_ylabel("Count")
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.visdir / "05_target_conversion.png", dpi=300, bbox_inches="tight")
        plt.close()
        return self

    def handle_missing_values(self):
        # Impute median for missing values (AGRatio)
        missing_before = self.df.isnull().sum()
        if missing_before.sum() > 0:
            imputer = SimpleImputer(strategy='median')
            self.df["AGRatio"] = imputer.fit_transform(self.df[["AGRatio"]])
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(missing_before.index, missing_before.values, color='orange')
            ax.set_xlabel("Missing Values Imputed")
            ax.set_title("Missing Values Imputed (Median Strategy)")
            ax.grid(axis='x', alpha=0.3)
            for i, (col, count) in enumerate(missing_before.items()):
                ax.text(count, i, f"{count} Imputed", ha='left', va='center')
            plt.tight_layout()
            plt.savefig(self.visdir / "06_missing_value_imputation.png", dpi=300, bbox_inches="tight")
            plt.close()
        return self

    def encode_categorical_variables(self):
        # Encode Gender column: Female=0, Male=1
        gender_counts = self.df["Gender"].value_counts()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(gender_counts.index, gender_counts.values, color=["lightblue", "pink"])
        axes[0].set_title("Original Gender Distribution")
        axes[0].set_ylabel("Count")
        axes[0].grid(axis='y', alpha=0.3)

        self.labelencoder = LabelEncoder()
        self.df["Gender"] = self.labelencoder.fit_transform(self.df["Gender"])

        gender_encoded_counts = self.df["Gender"].value_counts().sort_index()
        axes[1].bar(["Female (0)", "Male (1)"], gender_encoded_counts.values, color=["pink", "lightblue"])
        axes[1].set_title("Encoded Gender Distribution")
        axes[1].set_ylabel("Count")
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.visdir / "07_gender_encoding.png", dpi=300, bbox_inches="tight")
        plt.close()
        return self

    def detect_and_handle_outliers(self):
        # Use IQR method with 3*IQR for liver enzyme columns, 1.5*IQR for others
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in ["Selector", "Gender"]]
        outlier_summary = []
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            multiplier = 3 if col in ["TB", "DB", "Alkphos", "Sgpt", "Sgot"] else 1.5
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_summary.append((col, outliers))
                self.df[col] = np.clip(self.df[col], lower_bound, upper_bound)
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary, columns=["Feature", "Outliers"])
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(outlier_df["Feature"], outlier_df["Outliers"], color="tomato")
            ax.set_xlabel("Outliers Capped")
            ax.set_title("Outlier Detection and Handling (IQR Method)")
            ax.grid(axis='x', alpha=0.3)
            for bar, outliers in zip(bars, outlier_df["Outliers"]):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height() / 2., f"{outliers} capped", ha='left', va='center')
            plt.tight_layout()
            plt.savefig(self.visdir / "08_outlier_handling.png", dpi=300, bbox_inches="tight")
            plt.close()
        return self

    def feature_engineering(self):
        # Create new features as recommended
        self.df["BilirubinRatio"] = self.df["DB"] / (self.df["TB"] + 1e-6)
        self.df["SGPTSGOTRatio"] = self.df["Sgpt"] / (self.df["Sgot"] + 1e-6)
        self.df["TotalEnzymes"] = self.df["Sgpt"] + self.df["Sgot"]
        self.df["AgeGroup"] = pd.cut(self.df["Age"], bins=[0, 30, 45, 60, 100], labels=[0,1,2,3]).astype(int)
        self.df["LowProtein"] = (self.df["TP"] < 6.0).astype(int)
        self.df["HighEnzymes"] = ((self.df["Sgpt"] > 40) | (self.df["Sgot"] > 40)).astype(int)
        self.df["AgeGenderInteraction"] = self.df["Age"] * self.df["Gender"]

        # Replace infinite values and impute any new NaNs from creation
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if self.df.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy='median')
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])

        # Visualization of new features
        new_features = ["BilirubinRatio", "SGPTSGOTRatio", "TotalEnzymes", "AgeGroup",
                        "LowProtein", "HighEnzymes", "AgeGenderInteraction"]
        n_plots = len(new_features)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten()
        for i, feature in enumerate(new_features):
            axes[i].hist(self.df[feature], bins=30, color="mediumseagreen", edgecolor="black", alpha=0.7)
            axes[i].set_title(feature, fontweight="bold")
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel("Frequency")
            axes[i].grid(axis='y', alpha=0.3)
        for i in range(n_plots, len(axes)):
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(self.visdir / "09_engineered_features.png", dpi=300, bbox_inches="tight")
        plt.close()
        return self

    def check_duplicates(self):
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.df = self.df.drop_duplicates()
        return self

    def train_test_split_data(self, test_size=0.2, random_state=42):
        # Split data into stratified train and test sets and scale features
        X = self.df.drop("Selector", axis=1)
        y = self.df["Selector"]
        self.featurenames = X.columns.tolist()

        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Scaling features with StandardScaler fit on train only
        self.scaler = StandardScaler()
        Xtrain_scaled = self.scaler.fit_transform(self.Xtrain)
        Xtest_scaled = self.scaler.transform(self.Xtest)
        self.Xtrain = pd.DataFrame(Xtrain_scaled, columns=self.featurenames, index=self.Xtrain.index)
        self.Xtest = pd.DataFrame(Xtest_scaled, columns=self.featurenames, index=self.Xtest.index)

        # Plot train-test split distribution and class balance
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        split_sizes = [len(self.Xtrain), len(self.Xtest)]
        axes[0].bar(["Train", "Test"], split_sizes, color=["steelblue", "coral"])
        axes[0].set_ylabel("Number of Samples")
        axes[0].set_title("Train-Test Split Distribution")
        axes[0].grid(axis='y', alpha=0.3)
        for i, count in enumerate(split_sizes):
            axes[0].text(i, count, f"{count} ({count/sum(split_sizes)*100:.1f}%)", ha='center', va='bottom')

        # Train class distribution pie
        train_dist = self.ytrain.value_counts().sort_index()
        axes[1].pie(train_dist.values, labels=["No Disease (0)", "Liver Disease (1)"],
                    autopct="%1.1f%%", colors=["lightgreen", "salmon"])
        axes[1].set_title("Train Set Class Distribution")

        # Test class distribution pie
        test_dist = self.ytest.value_counts().sort_index()
        axes[2].pie(test_dist.values, labels=["No Disease (0)", "Liver Disease (1)"],
                    autopct="%1.1f%%", colors=["lightgreen", "salmon"])
        axes[2].set_title("Test Set Class Distribution")

        plt.tight_layout()
        plt.savefig(self.visdir / "11_train_test_split.png", dpi=300, bbox_inches="tight")
        plt.close()

        return self

    def save_processed_data(self, output_dir="data/processed"):
        output_dir = Path(output_dir)
        splits_dir = Path("data/splits")
        scalers_dir = Path("scalers")
        output_dir.mkdir(parents=True, exist_ok=True)
        splits_dir.mkdir(parents=True, exist_ok=True)
        scalers_dir.mkdir(parents=True, exist_ok=True)

        # Save the unscaled processed dataset
        self.df.to_csv(output_dir / "liver_processed_unscaled.csv", index=False)

        # Save scaled train and test splits
        train_df = pd.concat([self.Xtrain.reset_index(drop=True), self.ytrain.reset_index(drop=True)], axis=1)
        test_df = pd.concat([self.Xtest.reset_index(drop=True), self.ytest.reset_index(drop=True)], axis=1)
        train_df.to_csv(splits_dir / "liver_train_scaled.csv", index=False)
        test_df.to_csv(splits_dir / "liver_test_scaled.csv", index=False)

        # Save scaler and label encoder
        if self.scaler:
            with open(scalers_dir / "liver_scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)
        if self.labelencoder:
            with open(scalers_dir / "liver_label_encoder.pkl", "wb") as f:
                pickle.dump(self.labelencoder, f)
        return self

    def get_preprocessing_summary(self):
        summary = (
            f"Liver Disease Prediction - Preprocessing Pipeline Summary\n\n"
            f"Dataset Information:\n"
            f"  Original shape: {self.dforiginal.shape}\n"
            f"  Final shape (unscaled): {self.df.shape}\n"
            f"  Total Features after engineering: {self.df.shape[1] - 1}\n"
            f"  Target variable: Selector (1=Liver Disease, 0=No Disease)\n\n"
            f"Preprocessing Steps Completed:\n"
            f"  1. Loaded Data with column names\n"
            f"  2. Exploratory Data Analysis with visualizations\n"
            f"  3. Target Variable Conversion to binary\n"
            f"  4. Missing Values Imputation (median for AGRatio)\n"
            f"  5. Categorical Encoding (Gender)\n"
            f"  6. Outlier Detection and Capping using IQR\n"
            f"  7. Feature Engineering (7 new features created)\n"
            f"  8. Duplicate Removal\n"
            f"  9. Train-Test Split (80-20 stratified)\n"
            f"  10. Feature Normalization on Train only, transform Test\n"
            f"Outputs Generated:\n"
            f"  - Unscaled processed data (data/processed/liver_processed_unscaled.csv)\n"
            f"  - Scaled train and test splits (data/splits/liver_train_scaled.csv, liver_test_scaled.csv)\n"
            f"  - Scaler and label encoder saved (scalers/)\n"
            f"  - Visualizations saved (results/liver/visualizations/)\n"
            f"Ready for Model Training!\n"
        )
        summary_path = self.visdir.parent / "preprocessing_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary)
        return self

def preprocess_liver_data(filepath="data/raw/liver.csv"):
    preprocessor = LiverPreprocessor(filepath)
    preprocessor.load_data()\
                .exploratory_analysis()\
                .fix_target_variable()\
                .handle_missing_values()\
                .encode_categorical_variables()\
                .detect_and_handle_outliers()\
                .feature_engineering()\
                .check_duplicates()\
                .train_test_split_data()\
                .save_processed_data()\
                .get_preprocessing_summary()
    return preprocessor.Xtrain, preprocessor.Xtest, preprocessor.ytrain, preprocessor.ytest, preprocessor

if __name__ == "__main__":
    Xtrain, Xtest, ytrain, ytest, preprocessor = preprocess_liver_data("data/raw/liver.csv")
    print("Preprocessing completed. Processed data and artifacts saved.")