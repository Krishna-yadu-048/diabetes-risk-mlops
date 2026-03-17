"""
validate.py

Loads the raw Pima Indians Diabetes dataset, runs a hand-written
validation function, cleans the data, and writes the result to
data/processed/diabetes_clean.csv.

The Pima dataset encodes missing values as 0 for physiological
measurements that cannot be zero (glucose, blood pressure, etc.).
This script replaces those zeros with NaN and drops the rows.
"""

import sys
from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/raw/diabetes.csv")
PROCESSED_PATH = Path("data/processed/diabetes_clean.csv")

# Columns where 0 is biologically impossible — treated as missing
ZERO_IS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

EXPECTED_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

# Sensible value ranges for a sanity check
VALUE_RANGES = {
    "Pregnancies": (0, 20),
    "Glucose": (0, 300),
    "BloodPressure": (0, 200),
    "SkinThickness": (0, 100),
    "Insulin": (0, 900),
    "BMI": (0, 70),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (0, 120),
    "Outcome": (0, 1),
}


def validate(df: pd.DataFrame) -> list[str]:
    """
    Runs a series of checks on the raw dataframe.
    Returns a list of error messages. Empty list means all checks passed.
    """
    errors = []

    # Check all expected columns are present
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # Check there are no unexpected extra columns
    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")

    # Check for nulls in raw data (before our zero replacement)
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        errors.append(f"Null values found: {null_cols.to_dict()}")

    # Check value ranges
    for col, (lo, hi) in VALUE_RANGES.items():
        if col not in df.columns:
            continue
        out_of_range = df[(df[col] < lo) | (df[col] > hi)]
        if not out_of_range.empty:
            errors.append(
                f"Column '{col}' has {len(out_of_range)} rows outside range [{lo}, {hi}]"
            )

    # Check Outcome is binary
    if "Outcome" in df.columns:
        unique_outcomes = set(df["Outcome"].unique())
        if not unique_outcomes.issubset({0, 1}):
            errors.append(f"Outcome column has unexpected values: {unique_outcomes}")

    return errors


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces impossible zeros with NaN for physiological columns,
    then drops rows with any NaN. Returns the cleaned dataframe.
    """
    df = df.copy()

    # Replace 0 with NaN for columns where 0 means missing
    df[ZERO_IS_MISSING] = df[ZERO_IS_MISSING].replace(0, float("nan"))

    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)

    print(f"  Dropped {rows_before - rows_after} rows with missing values.")
    print(f"  Remaining rows: {rows_after}")

    return df


def main():
    print("--- Data Validation ---")

    if not RAW_PATH.exists():
        print(f"ERROR: Raw data not found at {RAW_PATH}")
        print("Download the Pima Indians Diabetes dataset from Kaggle and place it at data/raw/diabetes.csv")
        sys.exit(1)

    print(f"Loading raw data from {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    print(f"  Raw shape: {df.shape}")

    print("Running validation checks ...")
    errors = validate(df)

    if errors:
        print("\nValidation FAILED. Issues found:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("  All checks passed.")

    print("Cleaning data ...")
    df_clean = clean(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(PROCESSED_PATH, index=False)
    print(f"Saved cleaned data to {PROCESSED_PATH}")
    print("--- Done ---")


if __name__ == "__main__":
    main()
