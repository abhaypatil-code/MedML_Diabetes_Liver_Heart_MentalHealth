import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle
import warnings
warnings.filterwarnings("ignore")

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

class DiabetesPreprocessor:
    """
    Comprehensive preprocessing class for Diabetes prediction dataset
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.dforiginal = None
        self.scaler = None
        self.featurenames = None
        
        # Attributes to hold splits
        self.Xtrain = None
        self.Xtest = None
        self.ytrain = None
        self.ytest = None
        
        # Visualization output directory
        self.visdir = Path("results/diabetes/visualizations")
        self.visdir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        self.df = pd.read_csv(self.filepath)
        self.dforiginal = self.df.copy()
        return self

    def exploratory_analysis(self):
        # Data overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        # 1. Missing values
        missing_data = self.df.isnull().sum()
        axes[0, 0].barh(missing_data.index, missing_data.values, color="coral")
        axes[0, 0].set_xlabel("Count")
        axes[0, 0].set_title("Missing Values per Column", fontweight="bold")
        axes[0, 0].grid(axis='x', alpha=0.3)

        # 2. Data types
        dtype_counts = self.df.dtypes.value_counts()
        axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%',
                       colors=["skyblue", "lightcoral"])
        axes[0, 1].set_title("Data Types Distribution", fontweight="bold")

        # 3. Target variable
        target_counts = self.df["Outcome"].value_counts()
        bars = axes[1, 0].bar(["No Diabetes (0)", "Diabetes (1)"], target_counts.values,
                              color=['lightgreen', 'salmon'])
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Target Variable Distribution", fontweight="bold")
        axes[1, 0].grid(axis='y', alpha=0.3)
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height, 
                            f"{height/self.df.shape[0]*100:.1f}%", ha='center', va='bottom')
        # 4. Dataset Stats
        statstext = f"""Dataset Statistics
Total Samples: {self.df.shape[0]}
Total Features: {self.df.shape[1] - 1}
Target Variable: Outcome
Class Balance:
  No Diabetes (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/len(self.df)*100:.1f}%)
  Diabetes (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/len(self.df)*100:.1f}%)
Data Types: Numerical ({self.df.shape[1]-1}), Categorical (1 Target)"""
        axes[1, 1].text(0.1, 0.5, statstext, fontsize=11, verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3))
        axes[1, 1].axis("off")
        plt.tight_layout()
        plt.savefig(self.visdir / "01_dataset_overview.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 5. Feature distributions
        numericcols = self.df.drop("Outcome", axis=1).columns
        ncols = 3
        nrows = int(np.ceil(len(numericcols) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
        axes = axes.flatten()
        for idx, col in enumerate(numericcols):
            axes[idx].hist(self.df[col], bins=30, color="steelblue", edgecolor="black", alpha=0.7)
            axes[idx].set_title(f"{col} Distribution", fontweight="bold")
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Frequency")
            axes[idx].grid(axis="y", alpha=0.3)
            meanval = self.df[col].mean()
            medianval = self.df[col].median()
            axes[idx].axvline(meanval, color="red", linestyle="--", linewidth=2, label=f"Mean {meanval:.1f}")
            axes[idx].axvline(medianval, color="green", linestyle="--", linewidth=2, label=f"Median {medianval:.1f}")
            axes[idx].legend()
        for idx in range(len(numericcols), len(axes)):
            axes[idx].axis("off")
        plt.tight_layout()
        plt.savefig(self.visdir / "02_feature_distributions.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 6. Correlation matrix
        plt.figure(figsize=(12, 10))
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        plt.savefig(self.visdir / "03_correlation_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 7. Boxplots for outlier visualization
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
        axes = axes.flatten()
        for idx, col in enumerate(numericcols):
            axes[idx].boxplot(self.df[col].dropna(), vert=True, patch_artist=True,
                              boxprops=dict(facecolor="lightblue", color="blue"),
                              medianprops=dict(color="red", linewidth=2))
            axes[idx].set_title(f"{col} - Outlier Detection (Raw)", fontweight="bold")
            axes[idx].set_ylabel(col)
            axes[idx].grid(axis="y", alpha=0.3)
        for idx in range(len(numericcols), len(axes)):
            axes[idx].axis("off")
        plt.tight_layout()
        plt.savefig(self.visdir / "04_outlier_detection.png", dpi=300, bbox_inches="tight")
        plt.close()

        return self

    def handle_zero_values(self):
        """
        Replace zeros in medically invalid columns with NaN
        """
        zeroinvalidcols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        zerocounts = [self.df[col].eq(0).sum() for col in zeroinvalidcols]
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(zeroinvalidcols, zerocounts, color="indianred")
        ax.set_xlabel("Number of Zero Values")
        ax.set_title("Zero Values in Features (Medically Impossible)", fontsize=14, fontweight="bold")
        ax.grid(axis='x', alpha=0.3)
        for bar, count in zip(bars, zerocounts):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2., f"{count}", ha='left', va='center')
        plt.tight_layout()
        plt.savefig(self.visdir / "05_zero_values_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
        for col in zeroinvalidcols:
            self.df[col] = self.df[col].replace(0, np.nan)
        return self

    def handle_missing_values(self):
        """
        Impute missing values using median strategy and visualize
        """
        missingbefore = self.df.isnull().sum()
        if missingbefore.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(missingbefore.index, missingbefore.values, color="orange")
            ax.set_xlabel("Number of Missing Values Imputed")
            ax.set_title("Missing Values Handled via Median Imputation", fontsize=14, fontweight="bold")
            ax.grid(axis='x', alpha=0.3)
            for i, (col, count) in enumerate(missingbefore.items()):
                ax.text(count, i, f"{count} Imputed", ha='left', va='center')
            plt.tight_layout()
            plt.savefig(self.visdir / "06_missing_value_imputation.png", dpi=300, bbox_inches="tight")
            plt.close()
        # Median impute
        X = self.df.drop("Outcome", axis=1)
        y = self.df["Outcome"]
        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X)
        X_imputed_df = pd.DataFrame(X_imputed, columns=X.columns)
        self.df = pd.concat([X_imputed_df, y.reset_index(drop=True)], axis=1)
        return self

    def detect_and_handle_outliers(self):
        """
        Detect and cap/floor outliers using the IQR method with visualization
        """
        numericcols = self.df.drop("Outcome", axis=1).columns
        outlier_summary = []
        for col in numericcols:
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
        """
        Create new features for improved prediction
        """
        # Age bins
        self.df["AgeGroup"] = pd.cut(self.df["Age"], bins=[0, 30, 45, 60, 100], labels=[0, 1, 2, 3]).astype(int)
        self.df["BMICategory"] = pd.cut(self.df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=[0, 1, 2, 3]).astype(int)
        self.df["GlucoseCategory"] = pd.cut(self.df["Glucose"], bins=[0, 99, 125, 200], labels=[0, 1, 2]).astype(int)
        self.df["BMIAgeInteraction"] = self.df["BMI"] * self.df["Age"]
        self.df["GlucoseBMIInteraction"] = self.df["Glucose"] * self.df["BMI"]
        return self

    def check_duplicates(self):
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.df = self.df.drop_duplicates()
        return self

    def train_test_split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test with stratification, visualize class balance
        """
        X = self.df.drop("Outcome", axis=1)
        y = self.df["Outcome"]
        self.Xtrain, self.Xtest, self.ytrain, self.ytest = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        # Train/Test Split distribution visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        splitdata = ["Train", "Test"]
        splitcounts = [len(self.Xtrain), len(self.Xtest)]
        colors = ["steelblue", "coral"]
        bars = axes[0].bar(splitdata, splitcounts, color=colors, edgecolor="black")
        axes[0].set_ylabel("Number of Samples")
        axes[0].set_title("Train-Test Split Distribution", fontsize=12, fontweight="bold")
        axes[0].grid(axis='y', alpha=0.3)
        for bar, count in zip(bars, splitcounts):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height, f"{count} ({height/(len(self.Xtrain)+len(self.Xtest))*100:.1f}%)", ha='center', va='bottom')
        # Class balance in train
        traindist = self.ytrain.value_counts()
        axes[1].pie(traindist.values, labels=["No Diabetes", "Diabetes"], autopct='%1.1f%%',
                    colors=['lightgreen', 'salmon'])
        axes[1].set_title("Train Set Class Distribution", fontsize=12, fontweight="bold")

        # Class balance in test
        testdist = self.ytest.value_counts()
        axes[2].pie(testdist.values, labels=["No Diabetes", "Diabetes"], autopct='%1.1f%%',
                    colors=['lightgreen', 'salmon'])
        axes[2].set_title("Test Set Class Distribution", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.visdir / "09_train_test_split.png", dpi=300, bbox_inches="tight")
        plt.close()
        return self

    def normalize_features(self):
        """
        Normalize features using StandardScaler (fit only on train)
        """
        if any(getattr(self, attr) is None for attr in ["Xtrain", "Xtest"]):
            raise AttributeError("train_test_split_data must be called before normalize_features.")
        self.featurenames = self.Xtrain.columns.tolist()

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        samplefeatures = self.Xtrain.iloc[:, :5]  # Visualize first 5 features
        axes[0].boxplot([samplefeatures[col].values for col in samplefeatures.columns], labels=samplefeatures.columns)
        axes[0].set_title("Before Normalization (Train Set)", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("Value")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        self.scaler = StandardScaler()
        Xtrain_scaled = self.scaler.fit_transform(self.Xtrain)
        Xtest_scaled = self.scaler.transform(self.Xtest)
        X_scaled_sample = Xtrain_scaled[:, :5]
        axes[1].boxplot([X_scaled_sample[:, i] for i in range(5)], labels=samplefeatures.columns)
        axes[1].set_title("After Normalization (Train Set)", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Scaled Value")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.visdir / "08_feature_normalization.png", dpi=300, bbox_inches="tight")
        plt.close()
        # Overwrite train/test splits with normalized values (still as DataFrame for clarity)
        self.Xtrain = pd.DataFrame(Xtrain_scaled, columns=self.featurenames, index=self.Xtrain.index)
        self.Xtest = pd.DataFrame(Xtest_scaled, columns=self.featurenames, index=self.Xtest.index)
        return self

    def save_data(self, output_dir="data/processed"):
        """
        Save processed unscaled data, scaled splits, and fitted scaler pkl
        """
        processeddir = Path(output_dir)
        splitsdir = Path("data/splits")
        scalersdir = Path("scalers")
        processeddir.mkdir(parents=True, exist_ok=True)
        splitsdir.mkdir(parents=True, exist_ok=True)
        scalersdir.mkdir(parents=True, exist_ok=True)
        # Save full unscaled processed data
        self.df.to_csv(processeddir / "diabetes_processed.csv", index=False)
        # Save scaled splits
        traindf = pd.concat([self.Xtrain, self.ytrain.reset_index(drop=True)], axis=1)
        testdf = pd.concat([self.Xtest, self.ytest.reset_index(drop=True)], axis=1)
        traindf.to_csv(splitsdir / "diabetes_train.csv", index=False)
        testdf.to_csv(splitsdir / "diabetes_test.csv", index=False)
        # Save scaler
        if self.scaler is not None:
            scalerpath = scalersdir / "diabetes_scaler.pkl"
            with open(scalerpath, "wb") as f:
                pickle.dump(self.scaler, f)
        return self

    def get_preprocessing_summary(self):
        summary = (
            f"Diabetes Prediction - Preprocessing Pipeline Summary\n\n"
            f"Dataset Information:\n"
            f"  Original shape: {self.dforiginal.shape}\n"
            f"  Final shape (cleaned, unscaled): {self.df.shape}\n"
            f"  Total features after engineering: {self.df.shape[1] - 1}\n"
            f"  Target variable: Outcome\n\n"
            f"Preprocessing Steps Completed (Correct Order):\n"
            f"  1. Data Loading\n"
            f"  2. Exploratory Data Analysis (visualizations saved)\n"
            f"  3. Zero Values Handling (medically impossible NaN)\n"
            f"  4. Missing Values Imputation (median strategy)\n"
            f"  5. Outlier Detection and Capping (IQR method)\n"
            f"  6. Feature Engineering (new features created)\n"
            f"  7. Duplicate Removal\n"
            f"  8. Train-Test Split (80-20, stratified)\n"
            f"  9. Feature Normalization (StandardScaler, fitted on TRAIN only)\n"
            f"Outputs Generated:\n"
            f"  - Processed unscaled dataset (data/processed/diabetes_processed.csv)\n"
            f"  - Fitted scaler from train data (scalers/diabetes_scaler.pkl)\n"
            f"  - SCALED Train split (data/splits/diabetes_train.csv)\n"
            f"  - SCALED Test split (data/splits/diabetes_test.csv)\n"
            f"  - Visualizations (results/diabetes/visualizations/, 9+ images)\n"
            f"Ready for Model Training!\n"
        )
        summarypath = self.visdir.parent / "preprocessing_summary.txt"
        with open(summarypath, "w", encoding="utf-8") as f:
            f.write(summary)
        return self

def preprocess_diabetes_data(filepath="data/raw/diabetes.csv"):
    preprocessor = DiabetesPreprocessor(filepath)
    preprocessor.load_data()\
                .exploratory_analysis()\
                .handle_zero_values()\
                .handle_missing_values()\
                .detect_and_handle_outliers()\
                .feature_engineering()\
                .check_duplicates()\
                .train_test_split_data()\
                .normalize_features()\
                .save_data()\
                .get_preprocessing_summary()
    return preprocessor.Xtrain, preprocessor.Xtest, preprocessor.ytrain, preprocessor.ytest, preprocessor

if __name__ == "__main__":
    Xtrain, Xtest, ytrain, ytest, preprocessor = preprocess_diabetes_data("data/raw/diabetes.csv")
    print("Preprocessing completed. Processed data and artifacts saved.")