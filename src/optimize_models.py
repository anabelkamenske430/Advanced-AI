"""
Commit J - Optimization for Both Models
Rubric Section: J

This script performs:
1. Optimization of both models
2. Benchmark identification
3. Evaluation of improvements

Outputs are written to:
    reports/optimization_summary.txt
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow.keras import layers, models

from model1_data_prep import load_raw_adult_dataset, prepare_adult_dataset
from model2_data_prep import load_raw_sms_dataset


# ---------------------------------------------------------
# Model 1 Optimization (Adult Income)
# ---------------------------------------------------------

def optimize_model1():
    df = load_raw_adult_dataset()
    X_train, X_test, y_train, y_test, scaler = prepare_adult_dataset(df)

    # Baseline model
    baseline_model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    baseline_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    baseline_model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    y_pred_base = (baseline_model.predict(X_test).ravel() >= 0.5).astype(int)
    baseline_acc = accuracy_score(y_test, y_pred_base)
    baseline_f1 = f1_score(y_test, y_pred_base)

    # Optimized model
    optimized_model = models.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])
    optimized_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    optimized_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred_opt = (optimized_model.predict(X_test).ravel() >= 0.5).astype(int)
    opt_acc = accuracy_score(y_test, y_pred_opt)
    opt_f1 = f1_score(y_test, y_pred_opt)

    return {
        "baseline_accuracy": baseline_acc,
        "baseline_f1": baseline_f1,
        "optimized_accuracy": opt_acc,
        "optimized_f1": opt_f1,
    }


# ---------------------------------------------------------
# Model 2 Optimization (SMS Spam)
# ---------------------------------------------------------

def optimize_model2():
    df = load_raw_sms_dataset()
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline pipeline
    baseline_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB())
    ])
    baseline_pipeline.fit(X_train, y_train)
    y_pred_base = baseline_pipeline.predict(X_test)
    baseline_acc = accuracy_score(y_test, y_pred_base)
    baseline_f1 = f1_score(y_test, y_pred_base)

    # Optimized pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", MultinomialNB())
    ])

    param_grid = {
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [1, 2, 5],
        "clf__alpha": [0.5, 1.0, 1.5],
    }

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="f1",
        cv=3,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred_opt = best_model.predict(X_test)
    opt_acc = accuracy_score(y_test, y_pred_opt)
    opt_f1 = f1_score(y_test, y_pred_opt)

    return {
        "baseline_accuracy": baseline_acc,
        "baseline_f1": baseline_f1,
        "optimized_accuracy": opt_acc,
        "optimized_f1": opt_f1,
        "best_params": grid.best_params_,
    }


# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------

def main():
    os.makedirs("reports", exist_ok=True)

    model1_results = optimize_model1()
    model2_results = optimize_model2()

    with open("reports/optimization_summary.txt", "w") as f:
        f.write("Model 1 Optimization Results:\n")
        for k, v in model1_results.items():
            f.write(f"{k}: {v}\n")

        f.write("\nModel 2 Optimization Results:\n")
        for k, v in model2_results.items():
            f.write(f"{k}: {v}\n")

    print("Optimization complete. Results saved to reports/optimization_summary.txt")


if __name__ == "__main__":
    main()