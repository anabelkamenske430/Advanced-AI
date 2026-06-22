"""
Commit G - SMS Spam Dataset Identification, Description, and Preparation
Rubric Sections: G, G1, G2, G3
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# -----------------------------
# G & G1: Dataset Identification & Description
# -----------------------------

SMS_FEATURE_DESCRIPTION = """
The SMS Spam Collection dataset contains 5,574 SMS messages labeled as either
'spam' or 'ham'. It includes the following features:

- label: the classification label ('ham' for legitimate messages, 'spam' for unsolicited messages)
- message: the raw text content of the SMS message

This dataset is widely used for benchmarking text classification models.
"""

# -----------------------------
# G2: Reason for Choosing Dataset
# -----------------------------

SMS_REASON_FOR_CHOICE = """
The SMS Spam dataset was chosen because it is ideal for demonstrating
probabilistic machine learning methods such as Naive Bayes. The dataset is
clean, well-structured, and commonly used in academic and industry benchmarks
for spam detection. Its text-based nature makes it suitable for TF-IDF
vectorization and probabilistic classification.
"""

# -----------------------------
# G3: Data Preparation
# -----------------------------

def load_raw_sms_dataset(path="data/spam.csv"):
    """
    Load the SMS Spam dataset.
    Only loads raw data — no modeling yet.
    """
    df = pd.read_csv(path, encoding="latin-1")

    # Normalize column names if needed
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})

    df = df[["label", "message"]]
    df = df.dropna()

    return df


def prepare_sms_dataset(df: pd.DataFrame):
    """
    Prepare the SMS dataset for modeling.
    This includes:
    - Label encoding (ham=0, spam=1)
    - Train/test split
    - TF-IDF vectorization (fit only on training data)
    """

    # Encode labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X = df["message"]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF vectorizer (fit on training only)
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


# -----------------------------
# Optional quick test
# -----------------------------

def test_preparation():
    df = load_raw_sms_dataset()
    print("Raw SMS dataset shape:", df.shape)

    X_train, X_test, y_train, y_test, vectorizer = prepare_sms_dataset(df)
    print("Train TF-IDF shape:", X_train.shape)
    print("Test TF-IDF shape:", X_test.shape)


if __name__ == "__main__":
    test_preparation()