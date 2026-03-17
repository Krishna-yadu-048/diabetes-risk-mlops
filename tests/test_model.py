"""
test_model.py

Unit tests for core ML logic:
  - The hand-written validate() function in src/validate.py
  - Training output shape
  - predict_proba correctness (output is a valid probability)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.validate import clean, validate

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_valid_df(n=20) -> pd.DataFrame:
    """Creates a small valid diabetes dataframe for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "Pregnancies": rng.integers(0, 10, n),
        "Glucose": rng.integers(80, 200, n).astype(float),
        "BloodPressure": rng.integers(60, 120, n).astype(float),
        "SkinThickness": rng.integers(10, 50, n).astype(float),
        "Insulin": rng.integers(30, 200, n).astype(float),
        "BMI": rng.uniform(18, 45, n),
        "DiabetesPedigreeFunction": rng.uniform(0.1, 1.5, n),
        "Age": rng.integers(21, 70, n),
        "Outcome": rng.integers(0, 2, n),
    })


# ── Validation tests ──────────────────────────────────────────────────────────

def test_validate_passes_on_clean_data():
    df = make_valid_df()
    errors = validate(df)
    assert errors == [], f"Expected no errors, got: {errors}"


def test_validate_catches_missing_column():
    df = make_valid_df().drop(columns=["Glucose"])
    errors = validate(df)
    assert any("Glucose" in e for e in errors)


def test_validate_catches_extra_column():
    df = make_valid_df()
    df["extra_col"] = 1
    errors = validate(df)
    assert any("extra_col" in e for e in errors)


def test_validate_catches_out_of_range_age():
    df = make_valid_df()
    df.loc[0, "Age"] = 999  # impossible age
    errors = validate(df)
    assert any("Age" in e for e in errors)


def test_validate_catches_invalid_outcome():
    df = make_valid_df()
    df.loc[0, "Outcome"] = 5  # should be 0 or 1
    errors = validate(df)
    assert any("Outcome" in e for e in errors)


# ── Cleaning tests ────────────────────────────────────────────────────────────

def test_clean_removes_rows_with_zero_glucose():
    df = make_valid_df(30)
    # Introduce a few impossible zeros
    df.loc[0, "Glucose"] = 0
    df.loc[1, "BMI"] = 0
    cleaned = clean(df)
    # Those rows should be gone
    assert len(cleaned) == 28
    assert (cleaned["Glucose"] > 0).all()
    assert (cleaned["BMI"] > 0).all()


def test_clean_preserves_valid_rows():
    df = make_valid_df(20)
    cleaned = clean(df)
    # All rows are valid so nothing should be dropped
    assert len(cleaned) == 20


# ── Model training tests ──────────────────────────────────────────────────────

def test_logistic_regression_trains_and_predicts():
    df = make_valid_df(50)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(y)
    assert set(preds).issubset({0, 1})


def test_random_forest_predict_proba_is_valid():
    df = make_valid_df(50)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    proba = model.predict_proba(X)
    # Shape should be (n_samples, 2)
    assert proba.shape == (len(X), 2)
    # Each row should sum to roughly 1
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # All probabilities should be between 0 and 1
    assert (proba >= 0).all()
    assert (proba <= 1).all()


def test_training_output_has_correct_shape():
    df = make_valid_df(60)
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    model = RandomForestClassifier(n_estimators=5, random_state=0)
    model.fit(X, y)

    preds = model.predict(X)
    assert preds.shape == (60,)
