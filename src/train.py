"""
train.py

Reads the processed diabetes dataset, trains three models
(Logistic Regression, Random Forest, XGBoost), logs every run
to MLflow, and registers the best-performing model in the
MLflow Model Registry under the name defined in MODEL_NAME.

The best model is chosen by ROC-AUC on the validation set.
After this script runs, go to the MLflow UI and manually
promote the best run from Staging → Production.
"""

import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Config ────────────────────────────────────────────────────────────────────
PROCESSED_PATH = Path("data/processed/diabetes_clean.csv")
METRICS_DIR = Path("metrics")
TARGET_COL = "Outcome"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_NAME = os.getenv("MODEL_NAME", "diabetes-risk-model")

# Hyperparameters for each model — all in one place, easy to tweak
MODEL_CONFIGS = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "params": {
            "model_type": "logistic_regression",
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
        },
    },
    "random_forest": {
        "model": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=RANDOM_STATE,
        ),
        "params": {
            "model_type": "random_forest",
            "n_estimators": 100,
            "max_depth": 6,
            "random_state": RANDOM_STATE,
        },
    },
    "xgboost": {
        "model": XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
        "params": {
            "model_type": "xgboost",
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": RANDOM_STATE,
        },
    },
}
# ──────────────────────────────────────────────────────────────────────────────


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def train_and_log(name: str, config: dict, X_train, X_val, y_train, y_val):
    """Trains a single model, logs everything to MLflow, returns the run's AUC."""

    model = config["model"]
    params = config["params"]

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_val, y_pred), 4),
            "f1_score": round(f1_score(y_val, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_val, y_prob), 4),
        }

        mlflow.log_metrics(metrics)

        # Log the model artifact and register it
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )

        print(
            f"  [{name}] AUC: {metrics['roc_auc']} | "
            f"F1: {metrics['f1_score']} | "
            f"Acc: {metrics['accuracy']} | "
            f"Run ID: {run.info.run_id}"
        )

        return metrics["roc_auc"], run.info.run_id


def main():
    print("--- Training ---")

    # Set up MLflow tracking URI
    # If MLFLOW_TRACKING_URI is set (e.g. DagsHub), it will use that.
    # Otherwise defaults to local ./mlruns folder.
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI: {tracking_uri}")

    # Explicitly create the experiment if it does not exist yet,
    # then set it as active. This avoids runs silently landing in Default.
    experiment = mlflow.get_experiment_by_name("diabetes-risk")
    if experiment is None:
        mlflow.create_experiment("diabetes-risk")
    mlflow.set_experiment("diabetes-risk")

    print(f"Loading data from {PROCESSED_PATH} ...")
    X, y = load_data()

    # Scale features — important for Logistic Regression
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)} rows | Val: {len(X_val)} rows")

    print("\nTraining all models ...")
    results = {}

    for name, config in MODEL_CONFIGS.items():
        auc, run_id = train_and_log(name, config, X_train, X_val, y_train, y_val)
        results[name] = {"auc": auc, "run_id": run_id}

    # Find the best model by AUC
    best_name = max(results, key=lambda k: results[k]["auc"])
    best_auc = results[best_name]["auc"]
    print(f"\nBest model: {best_name} (AUC: {best_auc})")

    # Save a summary so evaluate.py and the README can reference it
    METRICS_DIR.mkdir(exist_ok=True)
    summary = {
        "best_model": best_name,
        "best_auc": best_auc,
        "all_results": results,
    }
    with open(METRICS_DIR / "train_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to {METRICS_DIR}/train_summary.json")

    # Automatically set the 'Production' alias on the best model version.
    # In MLflow 2.9+ model stages (Staging/Production) were deprecated
    # in favour of aliases. This replaces the manual UI step.
    print(f"\nSetting 'Production' alias on best model ({best_name}) ...")
    client = MlflowClient(tracking_uri=tracking_uri)

    # Find the latest version registered under MODEL_NAME
    registered = client.search_model_versions(f"name='{MODEL_NAME}'")
    # Match by run_id to find exactly the best run's version
    best_run_id = results[best_name]["run_id"]
    best_version = None
    for mv in registered:
        if mv.run_id == best_run_id:
            best_version = mv.version
            break

    if best_version:
        client.set_registered_model_alias(MODEL_NAME, "Production", best_version)
        print(f"  ✅ Alias 'Production' set on version {best_version} ({best_name})")
    else:
        print("  ⚠️  Could not find matching model version — set the alias manually via Python:")
        print("     from mlflow.tracking import MlflowClient")
        print("     client = MlflowClient(tracking_uri='$(pwd)/mlruns')")
        print(f"     client.set_registered_model_alias('{MODEL_NAME}', 'Production', '<version_number>')")

    print("--- Done ---")


if __name__ == "__main__":
    main()
