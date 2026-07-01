import pandas as pd
from src.validate_data import validate_churn_data


TRAIN_PATH = "data/train.csv"
TEST_PATH = "data/test.csv"


def test_train_dataset_is_valid():
    assert validate_churn_data(TRAIN_PATH) is True


def test_test_dataset_is_valid():
    assert validate_churn_data(TEST_PATH) is True


def test_required_columns_exist():
    df = pd.read_csv(TRAIN_PATH)

    required_columns = [
        "CustomerID",
        "Age",
        "Gender",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Subscription Type",
        "Contract Length",
        "Total Spend",
        "Last Interaction",
        "Churn"
    ]

    for column in required_columns:
        assert column in df.columns


def test_churn_target_is_binary():
    df = pd.read_csv(TRAIN_PATH).dropna()

    assert set(df["Churn"].unique()).issubset({0, 1, 0.0, 1.0})


def test_customer_id_not_used_as_feature():
    feature_columns = [
        "Age",
        "Gender",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Subscription Type",
        "Contract Length",
        "Total Spend",
        "Last Interaction"
    ]

    assert "CustomerID" not in feature_columns