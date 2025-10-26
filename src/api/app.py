from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# Load models and scalers
MODELS_DIR = Path("models")
SCALERS_DIR = Path("scalers")

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Load Liver model
liver_model = load_pickle(MODELS_DIR / "liver_model.pkl")
liver_scaler = load_pickle(SCALERS_DIR / "liver_scaler.pkl")

# Load Heart model
heart_model = load_pickle(MODELS_DIR / "heart_model.pkl")
heart_scaler = load_pickle(SCALERS_DIR / "heart_scaler.pkl")

# Load Diabetes model
diabetes_model = load_pickle(MODELS_DIR / "diabetes_model.pkl")
diabetes_scaler = load_pickle(SCALERS_DIR / "diabetes_scaler.pkl")

# Load Mental Health models for each target
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

@app.route("/")
def home():
    return "Welcome to the Disease Prediction API"

@app.route("/predict/liver", methods=["POST"])
def predict_liver():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = liver_scaler.transform(features)
    prediction = liver_model.predict(features_scaled)[0]
    return jsonify({"prediction": int(prediction)})

@app.route("/predict/heart", methods=["POST"])
def predict_heart():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = heart_scaler.transform(features)
    prediction = heart_model.predict(features_scaled)[0]
    return jsonify({"prediction": int(prediction)})

@app.route("/predict/diabetes", methods=["POST"])
def predict_diabetes():
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = diabetes_scaler.transform(features)
    prediction = diabetes_model.predict(features_scaled)[0]
    return jsonify({"prediction": int(prediction)})

@app.route("/predict/mental_health/<target>", methods=["POST"])
def predict_mental_health(target):
    if target not in mental_health_models:
        return jsonify({"error": "Invalid mental health target"}), 400
    data = request.get_json()
    features = np.array(data["features"]).reshape(1, -1)
    features_scaled = mental_health_scalers[target].transform(features)
    model = mental_health_models[target]
    prediction = model.predict(features_scaled)[0]
    return jsonify({f"{target}_prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)
