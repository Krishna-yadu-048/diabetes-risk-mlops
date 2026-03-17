"""
evaluate.py

Loads the Production model from the MLflow Model Registry and
evaluates it on the held-out test set. Saves a metrics.json file
so the result is trackable in Git (alongside the DVC-tracked data).

Run this after promoting a model to Production in the MLflow UI.
"""

import json
import os
from pathlib import Path

import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_PATH = Path("data/processed/diabetes_clean.csv")
METRICS_PATH = Path("metrics/metrics.json")
TARGET_COL = "Outcome"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-risk-model")
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("--- Evaluation ---")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # Load the Production model from the registry
    model_uri = f"models:/{MODEL_NAME}@Production"
    print(f"Loading model from: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"ERROR: Could not load model. Is a version promoted to Production?")
        print(f"  Detail: {e}")
        raise

    print(f"Loading data from {PROCESSED_PATH} ...")
    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Use the same split and scaling as train.py
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    _, X_test, _, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Test set: {len(X_test)} rows")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model_name": MODEL_NAME,
        "model_alias": "Production",
        "test_rows": len(X_test),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
    }

    print("\nResults:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    METRICS_PATH.parent.mkdir(exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetrics saved to {METRICS_PATH}")
    print("--- Done ---")


if __name__ == "__main__":
    main()
