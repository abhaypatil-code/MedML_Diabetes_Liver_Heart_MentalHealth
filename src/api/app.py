from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Set base directories for models and scalers
# Assuming 'models' and 'scalers' directories are in the same directory as app.py
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
SCALERS_DIR = BASE_DIR / "scalers"

# --- Load Models and Scalers ---

def load_pickle(file_path):
    """Helper function to load pickle files."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        # Log this error in a real application
        print(f"Error: Could not find file at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Load Liver model and scaler
liver_model = load_pickle(MODELS_DIR / "liver_model.pkl")
liver_scaler = load_pickle(SCALERS_DIR / "liver_scaler.pkl")

# Load Heart model and scaler
heart_model = load_pickle(MODELS_DIR / "heart_model.pkl")
heart_scaler = load_pickle(SCALERS_DIR / "heart_scaler.pkl")

# Load Diabetes model and scaler
diabetes_model = load_pickle(MODELS_DIR / "diabetes_model.pkl")
diabetes_scaler = load_pickle(SCALERS_DIR / "diabetes_scaler.pkl")

# Load Mental Health models and scalers for each target
mental_health_models = {
    "depressiveness": load_pickle(MODELS_DIR / "mental_health_depressiveness_model.pkl"),
    "anxiousness": load_pickle(MODELS_DIR / "mental_health_anxiousness_model.pkl"),
    "sleepiness": load_pickle(MODELS_DIR / "mental_health_sleepiness_model.pkl"),
}
mental_health_scalers = {
    "depressiveness": load_pickle(SCALERS_DIR / "mental_health_depressiveness_scaler.pkl"),
    "anxiousness": load_pickle(SCALERS_DIR / "mental_health_anxiousness_scaler.pkl"),
    "sleepiness": load_pickle(SCALERS_DIR / "mental_health_sleepiness_scaler.pkl"),
}

# --- API Routes ---

@app.route("/")
def home():
    """Welcome endpoint."""
    return "Welcome to the Disease Prediction API"

@app.route("/predict/liver", methods=["POST"])
def predict_liver():
    """Predicts liver disease risk."""
    if not liver_model or not liver_scaler:
        return jsonify({"error": "Liver model or scaler not loaded"}), 500
        
    try:
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400
            
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = liver_scaler.transform(features)
        prediction = liver_model.predict(features_scaled)[0]
        return jsonify({"prediction": int(prediction)})
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    """Predicts heart attack risk."""
    if not heart_model or not heart_scaler:
        return jsonify({"error": "Heart model or scaler not loaded"}), 500

    try:
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400
            
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = heart_scaler.transform(features)
        prediction = heart_model.predict(features_scaled)[0]
        return jsonify({"prediction": int(prediction)})
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    """Predicts diabetes risk."""
    if not diabetes_model or not diabetes_scaler:
        return jsonify({"error": "Diabetes model or scaler not loaded"}), 500
        
    try:
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400
            
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = diabetes_scaler.transform(features)
        prediction = diabetes_model.predict(features_scaled)[0]
        return jsonify({"prediction": int(prediction)})
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

@app.route("/predict/mental_health/<target>", methods=["POST"])
def predict_mental_health(target):
    """Predicts mental health risk for a specific target."""
    if target not in mental_health_models or not mental_health_scalers:
        return jsonify({"error": "Invalid mental health target or model/scaler not loaded"}), 400
    
    model = mental_health_models.get(target)
    scaler = mental_health_scalers.get(target)

    if not model or not scaler:
        return jsonify({"error": f"Model or scaler for '{target}' not loaded"}), 500

    try:
        data = request.get_json()
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key"}), 400
            
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        return jsonify({f"{target}_prediction": int(prediction)})
        
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 400

if __name__ == "__main__":
    # Note: 'debug=True' is for development only. 
    # Use a production server like Gunicorn or uWSGI for deployment.
    app.run(debug=True, host='0.0.0.0', port=5000)