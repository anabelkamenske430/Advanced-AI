"""
Commit E - Model 1 Development, Training, Validation, and Documentation
Rubric Sections: E1, E2, E3, E4, E5

Model: D804_PA_Model_AdultIncomeClassifier
Advanced Method: Deep Neural Network (DNN)
"""

import os
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras import layers, models

from model1_data_prep import load_raw_adult_dataset, prepare_adult_dataset
from model1_config import MODEL_1_NAME, MODEL_1_GOAL


# ---------------------------------------------------------
# E1: Model Development Method (Deep Neural Network)
# ---------------------------------------------------------

def build_dnn_model(input_dim: int) -> tf.keras.Model:
    """
    Build a deep neural network for binary classification.
    This satisfies E1: explaining the model development method.
    """
    model = models.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------
# E2: Model Training
# ---------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train the DNN model using class weights and validation split.
    This satisfies E2: explaining how the model was trained.
    """
    input_dim = X_train.shape[1]
    model = build_dnn_model(input_dim)

    # Compute class weights to address imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = {i: w for i, w in enumerate(class_weights)}

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        class_weight=class_weights,
        verbose=1,
    )

    return model, history


# ---------------------------------------------------------
# E3: Results Validation
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using accuracy, precision, recall, F1, ROC-AUC,
    and confusion matrix. This satisfies E3.
    """
    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)

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
# E4: Deployment Preparation
# ---------------------------------------------------------

def save_model(model, output_dir="models"):
    """
    Save the trained model for deployment.
    This satisfies E4: explaining how the model could be deployed.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "adult_income_dnn")
    model.save(model_path)
    print(f"Model saved to {model_path}")


# ---------------------------------------------------------
# E5: Main Execution + Documentation
# ---------------------------------------------------------

def main():
    print(f"Training Model 1: {MODEL_1_NAME}")
    print(f"Goal: {MODEL_1_GOAL}")

    # Load and prepare data
    df = load_raw_adult_dataset()
    X_train, X_test, y_train, y_test, scaler = prepare_adult_dataset(df)
    print("Non-numeric columns in X_train:", X_train.select_dtypes(include=['object']).columns.tolist())
    print("Dtypes summary:\n", X_train.dtypes)

    # Train model
    model, history = train_model(X_train, y_train)

    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Print metrics to console
    print("\nModel 1 Evaluation Metrics:")
    for k, v in metrics.items():
        if k in ["confusion_matrix", "classification_report"]:
            print(f"\n{k}:\n{v}")
        else:
            print(f"{k}: {v:.4f}")

    # Save metrics to file
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics_model1.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}:\n{v}\n\n")

    # Save model for deployment
    save_model(model)


if __name__ == "__main__":
    main()