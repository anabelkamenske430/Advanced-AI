"""
Commit D - Adult Income Dataset Identification, Description, and Preparation
Rubric Sections: D, D1, D2, D3
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -----------------------------
# D & D1: Dataset Identification & Description
# -----------------------------

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

ADULT_FEATURE_DESCRIPTION = """
The UCI Adult Income dataset contains 48,842 rows and 15 columns.
It includes demographic and employment-related attributes such as:
- age (numeric)
- workclass (categorical)
- fnlwgt (numeric)
- education (categorical)
- education_num (numeric)
- marital_status (categorical)
- occupation (categorical)
- relationship (categorical)
- race (categorical)
- sex (categorical)
- capital_gain (numeric)
- capital_loss (numeric)
- hours_per_week (numeric)
- native_country (categorical)
The target variable is 'income', labeled as <=50K or >50K.
"""

# -----------------------------
# D2: Reason for Choosing Dataset
# -----------------------------

ADULT_REASON_FOR_CHOICE = """
The Adult Income dataset was chosen because it is a widely used benchmark
for classification tasks involving structured tabular data. It contains a mix
of categorical and numerical features, making it ideal for demonstrating
advanced AI methods such as deep neural networks. Its real-world relevance
and moderate size make it suitable for training, evaluation, and optimization
within the Cloud Academy environment.
"""

# -----------------------------
# D3: Data Preparation
# -----------------------------

def load_raw_adult_dataset(path="data/adult.data"):
    """Load raw Adult Income dataset with column names applied."""
    df = pd.read_csv(
        path,
        header=None,
        names=ADULT_COLUMNS,
        na_values=" ?",
        skipinitialspace=True,
    )
    return df


def prepare_adult_dataset(df: pd.DataFrame):
    """
    Clean, encode, and split the Adult Income dataset.
    This function performs:
    - Missing value handling
    - One-hot encoding for categorical features
    - Standardization for numeric features
    - Train/test split
    """
    # Remove rows with missing values
    df = df.dropna()

    # Separate features and target
    X = df.drop("income", axis=1)
    y = df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

    # Identify categorical and numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Ensure all columns are numeric
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")

    # Drop any columns that became NaN due to coercion
    X_encoded = X_encoded.dropna(axis=1)

    # Standardize numeric features
    scaler = StandardScaler()
    X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

    # Convert everything to float32 (critical for TensorFlow)
    X_encoded = X_encoded.astype("float32")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler


# -----------------------------
# Quick test function
# -----------------------------

def test_preparation():
    df = load_raw_adult_dataset()
    print("Raw dataset shape:", df.shape)
    X_train, X_test, y_train, y_test, scaler = prepare_adult_dataset(df)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)


if __name__ == "__main__":
    test_preparation()