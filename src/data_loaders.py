"""
Commit B - Dataset Loading Utilities
Rubric Section: B
"""

import os
import pandas as pd

# -------------------------
# Adult Income Dataset Loader
# -------------------------

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

def load_adult_dataset(path="data/adult.data"):
    """
    Loads the UCI Adult Income dataset.
    Only loads raw data — no cleaning or preprocessing yet.
    """
    df = pd.read_csv(
        path,
        header=None,
        names=ADULT_COLUMNS,
        na_values=" ?",
        skipinitialspace=True,
    )
    return df


# -------------------------
# SMS Spam Dataset Loader
# -------------------------

def load_sms_dataset(path="data/spam.csv"):
    """
    Loads the SMS Spam dataset.
    Only loads raw data — no cleaning or preprocessing yet.
    """
    df = pd.read_csv(path, encoding="latin-1")

    # Normalize column names if needed
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})

    # Keep only the relevant columns
    df = df[["label", "message"]]

    return df