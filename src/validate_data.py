import pandas as pd


REQUIRED_COLUMNS = [
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

VALID_GENDERS = ["Male", "Female"]
VALID_SUBSCRIPTION_TYPES = ["Basic", "Standard", "Premium"]
VALID_CONTRACT_LENGTHS = ["Monthly", "Quarterly", "Annual"]


def validate_churn_data(file_path):
    df = pd.read_csv(file_path)

    errors = []

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            errors.append(f"Missing required column: {column}")

    if len(df) < 100:
        errors.append("Dataset must contain at least 100 rows")

    df = df.dropna()

    numeric_columns = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction"
    ]

    for column in numeric_columns:
        if column in df.columns:
            if (df[column] < 0).any():
                errors.append(f"{column} contains negative values")

    if "Gender" in df.columns:
        invalid_values = set(df["Gender"].unique()) - set(VALID_GENDERS)
        if invalid_values:
            errors.append(f"Invalid Gender values found: {invalid_values}")

    if "Subscription Type" in df.columns:
        invalid_values = set(df["Subscription Type"].unique()) - set(VALID_SUBSCRIPTION_TYPES)
        if invalid_values:
            errors.append(f"Invalid Subscription Type values found: {invalid_values}")

    if "Contract Length" in df.columns:
        invalid_values = set(df["Contract Length"].unique()) - set(VALID_CONTRACT_LENGTHS)
        if invalid_values:
            errors.append(f"Invalid Contract Length values found: {invalid_values}")

    if "Churn" in df.columns:
        invalid_targets = set(df["Churn"].unique()) - {0, 1, 0.0, 1.0}
        if invalid_targets:
            errors.append(f"Invalid Churn values found: {invalid_targets}")

    if errors:
        raise ValueError("Data validation failed: " + " | ".join(errors))

    return True


if __name__ == "__main__":
    validate_churn_data("data/train.csv")
    validate_churn_data("data/test.csv")
    print("Data validation passed successfully")