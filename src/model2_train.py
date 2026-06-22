"""
Commit H - Model 2 Development, Training, Validation, and Documentation
Rubric Sections: H1, H2, H3, H4, H5

Model: D804_PA_Model_SpamDetector
Probabilistic Method: Multinomial Naive Bayes
"""

import os
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from model2_data_prep import load_raw_sms_dataset, prepare_sms_dataset
from model2_config import MODEL_2_NAME, MODEL_2_GOAL


# ---------------------------------------------------------
# H1: Model Development Method (Naive Bayes)
# ---------------------------------------------------------

def build_naive_bayes_model():
    """
    Build a Multinomial Naive Bayes classifier.
    This satisfies H1: explaining the model development method.
    """
    return MultinomialNB()


# ---------------------------------------------------------
# H2: Model Training
# ---------------------------------------------------------

def train_model(X_train_tfidf, y_train):
    """
    Train the Naive Bayes classifier on TF-IDF features.
    This satisfies H2: explaining how the model was trained.
    """
    model = build_naive_bayes_model()
    model.fit(X_train_tfidf, y_train)
    return model


# ---------------------------------------------------------
# H3: Results Validation
# ---------------------------------------------------------

def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate the model using accuracy, precision, recall, F1, ROC-AUC,
    and confusion matrix. This satisfies H3.
    """
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
    }

    return metrics


# ---------------------------------------------------------
# H4: Deployment Preparation
# ---------------------------------------------------------

def save_model(model, vectorizer, output_dir="models"):
    """
    Save the trained model and TF-IDF vectorizer for deployment.
    This satisfies H4: explaining how the model could be deployed.
    """
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "sms_spam_nb_model.joblib")
    vectorizer_path = os.path.join(output_dir, "sms_spam_tfidf_vectorizer.joblib")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Model saved to {model_path}")
    print(f"Vectorizer saved to {vectorizer_path}")


# ---------------------------------------------------------
# H5: Main Execution + Documentation
# ---------------------------------------------------------

def main():
    print(f"Training Model 2: {MODEL_2_NAME}")
    print(f"Goal: {MODEL_2_GOAL}")

    # Load and prepare data
    df = load_raw_sms_dataset()
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = prepare_sms_dataset(df)

    # Train model
    model = train_model(X_train_tfidf, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_test_tfidf, y_test)

    # Print metrics to console
    print("\nModel 2 Evaluation Metrics:")
    for k, v in metrics.items():
        if k in ["confusion_matrix", "classification_report"]:
            print(f"\n{k}:\n{v}")
        else:
            print(f"{k}: {v:.4f}")

    # Save metrics to file
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics_model2.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}:\n{v}\n\n")

    # Save model + vectorizer for deployment
    save_model(model, vectorizer)


if __name__ == "__main__":
    main()