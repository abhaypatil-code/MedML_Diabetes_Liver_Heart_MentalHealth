import logging
import pickle
from pathlib import Path
import sys

def setup_logging(name, log_file=None, level=logging.INFO):
    """Function to setup a logger; can write to file and stdout."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def save_model(model, filepath):
    """Save a model to a pickle file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load a model from a pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
