import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

from src.validate_data import validate_churn_data


TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

TARGET_COLUMN = "Churn"
DROP_COLUMNS = ["CustomerID"]

NUMERIC_FEATURES = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Subscription Type",
    "Contract Length"
]

DECISION_THRESHOLD = 0.5


def load_and_prepare_data():
    validate_churn_data(TRAIN_PATH)
    validate_churn_data(TEST_PATH)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    full_df = pd.concat([train_df, test_df], ignore_index=True)

    full_df = full_df.dropna()

    full_df[TARGET_COLUMN] = full_df[TARGET_COLUMN].astype(int)

    X = full_df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN])
    y = full_df[TARGET_COLUMN]

    return X, y, full_df


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_model():
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=1,
        eval_metric="logloss",
        random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier)
        ]
    )

    return model


def evaluate_model(model, X_test, y_test):
    prediction_probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (prediction_probabilities >= DECISION_THRESHOLD).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, prediction_probabilities),
        "decision_threshold": DECISION_THRESHOLD
    }

    return metrics, predictions, prediction_probabilities


def save_outputs(model, metrics):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, MODEL_PATH)

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model saved to:", MODEL_PATH)
    print("Metrics saved to:", METRICS_PATH)


def train():
    X, y, full_df = load_and_prepare_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTrain Target Distribution:")
    print(y_train.value_counts(normalize=True))

    print("\nTest Target Distribution:")
    print(y_test.value_counts(normalize=True))

    print("\nAverage numeric values by churn:")
    print(full_df.groupby(TARGET_COLUMN).mean(numeric_only=True))

    model = build_model()

    mlflow.set_experiment("churn-classification-production-demo")

    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("scale_pos_weight", 1)
        mlflow.log_param("eval_metric", "logloss")
        mlflow.log_param("decision_threshold", DECISION_THRESHOLD)
        mlflow.log_param("total_rows", len(full_df))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        model.fit(X_train, y_train)

        train_probabilities = model.predict_proba(X_train)[:, 1]
        train_predictions = (train_probabilities >= DECISION_THRESHOLD).astype(int)

        print("\nTRAIN PERFORMANCE:")
        print("Accuracy:", accuracy_score(y_train, train_predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(y_train, train_predictions))

        metrics, predictions, prediction_probabilities = evaluate_model(
            model,
            X_test,
            y_test
        )

        print("\nTEST PERFORMANCE:")
        print("Accuracy:", metrics["accuracy"])
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, predictions))

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        save_outputs(model, metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="ChurnClassifier",
            skops_trusted_types=[
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier"
            ]
        )

        print("\nTraining completed successfully")
        print("\nModel Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    train()