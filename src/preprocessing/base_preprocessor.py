from abc import ABC, abstractmethod
import pandas as pd
from pathlib import Path
from src.utils.common import setup_logging

class BasePreprocessor(ABC):
    """Abstract base class for data preprocessing."""
    
    def __init__(self, filepath, dataset_name):
        self.filepath = Path(filepath)
        self.dataset_name = dataset_name
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.logger = setup_logging(f"{dataset_name}_preprocessor")
        
        # Output directories
        self.processed_dir = Path("data/processed")
        self.splits_dir = Path("data/splits")
        self.scalers_dir = Path("scalers")
        self.vis_dir = Path(f"results/{dataset_name}/visualizations")
        
        for d in [self.processed_dir, self.splits_dir, self.scalers_dir, self.vis_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_data(self):
        """Load data from source."""
        pass

    @abstractmethod
    def clean_data(self):
        """Handle missing values, duplicates, etc."""
        pass

    @abstractmethod
    def feature_engineering(self):
        """Create new features."""
        pass
        
    @abstractmethod
    def split_data(self):
        """Split into train/test."""
        pass

    @abstractmethod
    def normalize_data(self):
        """Scale features."""
        pass

    def save_splits(self):
        """Save train/test splits to CSV."""
        if self.X_train is not None:
            train_df = pd.concat([self.X_train, self.y_train], axis=1)
            test_df = pd.concat([self.X_test, self.y_test], axis=1)
            
            train_path = self.splits_dir / f"{self.dataset_name}_train.csv"
            test_path = self.splits_dir / f"{self.dataset_name}_test.csv"
            
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            self.logger.info(f"Saved splits to {self.splits_dir}")
        else:
            self.logger.warning("No data to save!")

    def run_pipeline(self):
        """Execute the full preprocessing pipeline."""
        self.logger.info("Starting preprocessing pipeline...")
        self.load_data()
        self.clean_data()
        self.feature_engineering()
        self.split_data()
        self.normalize_data()
        self.save_splits()
        self.logger.info("Preprocessing pipeline completed.")
        return self.X_train, self.X_test, self.y_train, self.y_test
