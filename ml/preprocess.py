"""
Feature engineering and preprocessing pipeline.

The same pipeline is used at training time AND at prediction time so that
feature transformations are always identical.  The fitted pipeline object
is saved to disk together with the trained model.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ── feature definitions 

NUMERIC_FEATURES = ["age", "billing_amount", "length_of_stay",
                    "admission_month", "admission_year"]

CATEGORICAL_FEATURES = [
    "gender", "blood_type", "medical_condition",
    "insurance_provider", "admission_type", "medication",
]

TARGET_COLUMN = "test_results"

# Canonical target class order (used by LabelEncoder)
TARGET_CLASSES = ["Abnormal", "Inconclusive", "Normal"]


# ── column transformer 

def build_feature_pipeline() -> ColumnTransformer:
    """
    Return an unfitted ColumnTransformer that:
    - Scales numeric features with StandardScaler
    - One-hot-encodes categorical features (unknown values at predict time
      are silently ignored)
    """
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",   # drop any extra columns silently
    )

    return preprocessor


# ── target encoder 

def build_label_encoder() -> LabelEncoder:
    """
    Return a LabelEncoder pre-fitted on TARGET_CLASSES so that class
    indices are deterministic across all training runs.
    """
    le = LabelEncoder()
    le.fit(TARGET_CLASSES)
    return le


# ── dataframe helpers 

def load_training_data(session) -> pd.DataFrame:
    """Read all valid cleaned records from the database."""
    query = (
        "SELECT age, gender, blood_type, medical_condition, "
        "insurance_provider, billing_amount, admission_type, "
        "medication, length_of_stay, admission_month, admission_year, "
        "test_results "
        "FROM cleaned_patients "
        "WHERE is_valid = TRUE"
    )
    df = pd.read_sql(query, session.bind)
    print(f"[preprocess] Loaded {len(df):,} training rows from DB")
    return df


def prepare_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into feature matrix X and target series y.
    Drops rows with nulls in required columns.
    """
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    df = df.dropna(subset=required)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    # Standardise target strings (just in case)
    y = y.str.strip().str.title()

    return X, y


def prepare_single_record(data: dict) -> pd.DataFrame:
    """
    Convert a single API payload dict to a one-row DataFrame with the
    exact columns expected by the feature pipeline.

    The API uses original column names with spaces; we map them here.
    """
    mapping = {
        "Age": "age",
        "Gender": "gender",
        "Blood Type": "blood_type",
        "Medical Condition": "medical_condition",
        "Billing Amount": "billing_amount",
        "Admission Type": "admission_type",
        "Insurance Provider": "insurance_provider",
        "Medication": "medication",
    }

    row = {}
    for api_key, col in mapping.items():
        value = data.get(api_key) or data.get(col)
        row[col] = value

    # Derived columns not in the API payload — use sensible defaults
    row.setdefault("length_of_stay", 0)
    row.setdefault("admission_month", 1)
    row.setdefault("admission_year", 2024)

    # Standardise string fields
    for col in CATEGORICAL_FEATURES:
        if col in row and isinstance(row[col], str):
            row[col] = row[col].strip().title()

    df = pd.DataFrame([row])
    return df
