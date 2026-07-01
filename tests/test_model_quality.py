import json
import os
import joblib
import pandas as pd


MODEL_PATH = "models/churn_model.joblib"
METRICS_PATH = "models/metrics.json"
TEST_PATH = "data/test.csv"


def test_model_file_exists():
    assert os.path.exists(MODEL_PATH)


def test_metrics_file_exists():
    assert os.path.exists(METRICS_PATH)


def test_model_metrics_are_acceptable():
    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    assert metrics["accuracy"] >= 0.60
    assert metrics["precision"] >= 0.55
    assert metrics["recall"] >= 0.55
    assert metrics["f1"] >= 0.55
    assert metrics["roc_auc"] >= 0.70


def test_model_can_make_prediction():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(TEST_PATH).dropna()

    X = df.drop(columns=["CustomerID", "Churn"])

    sample = X.head(1)

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1]


def test_model_can_return_probability():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(TEST_PATH).dropna()

    X = df.drop(columns=["CustomerID", "Churn"])

    sample = X.head(1)

    probability = model.predict_proba(sample)[0][1]

    assert 0.0 <= probability <= 1.0