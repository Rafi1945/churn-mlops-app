import os
import joblib
import json


MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model.joblib")
METRICS_PATH = os.getenv("METRICS_PATH", "models/metrics.json")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)
    return model


def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return {}

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    return metrics