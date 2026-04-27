"""
tests/test_ml.py
────────────────
Unit tests for the ML preprocessing and prediction utilities.

Run with:
    uv run pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.preprocess import (
    TARGET_CLASSES,
    build_feature_pipeline,
    build_label_encoder,
    prepare_features_and_target,
    prepare_single_record,
)


# ── label encoder ──────────────────────────────────────────────────────────

def test_label_encoder_classes():
    le = build_label_encoder()
    assert list(le.classes_) == sorted(TARGET_CLASSES)


def test_label_encoder_roundtrip():
    le = build_label_encoder()
    for cls in TARGET_CLASSES:
        encoded = le.transform([cls])
        decoded = le.inverse_transform(encoded)
        assert decoded[0] == cls


# ── feature pipeline ───────────────────────────────────────────────────────

def _make_sample_df(n=10) -> pd.DataFrame:
    """Create a minimal sample DataFrame for pipeline tests."""
    return pd.DataFrame({
        "age":               [30 + i for i in range(n)],
        "billing_amount":    [1000.0 + i * 50 for i in range(n)],
        "length_of_stay":    [3 + i % 7 for i in range(n)],
        "admission_month":   [1 + i % 12 for i in range(n)],
        "admission_year":    [2023] * n,
        "gender":            ["Male", "Female"] * (n // 2),
        "blood_type":        ["O+"] * n,
        "medical_condition": ["Diabetes"] * n,
        "insurance_provider":["Cigna"] * n,
        "admission_type":    ["Emergency"] * n,
        "medication":        ["Aspirin"] * n,
        "test_results":      ["Normal", "Abnormal", "Inconclusive",
                               "Normal", "Abnormal",
                               "Normal", "Abnormal", "Inconclusive",
                               "Normal", "Abnormal"],
    })


def test_pipeline_fit_transform():
    df = _make_sample_df()
    X, _ = prepare_features_and_target(df)
    pipeline = build_feature_pipeline()
    X_t = pipeline.fit_transform(X)
    assert X_t.shape[0] == len(df)
    assert X_t.shape[1] > 5          # numeric + one-hot expanded


def test_pipeline_handles_unknown_categories():
    df = _make_sample_df()
    X_train, _ = prepare_features_and_target(df)
    pipeline = build_feature_pipeline()
    pipeline.fit(X_train)

    # Unseen category should not raise
    unseen = pd.DataFrame([{
        "age": 50,
        "billing_amount": 3000.0,
        "length_of_stay": 5,
        "admission_month": 6,
        "admission_year": 2024,
        "gender": "Male",
        "blood_type": "X+",            # unknown
        "medical_condition": "Unknown",  # unknown
        "insurance_provider": "Cigna",
        "admission_type": "Emergency",
        "medication": "Aspirin",
    }])

    result = pipeline.transform(unseen)
    assert result.shape[0] == 1


# ── prepare_features_and_target 
def test_prepare_features_drops_nulls():
    df = _make_sample_df()
    df.loc[0, "age"] = None
    X, y = prepare_features_and_target(df)
    assert len(X) == len(df) - 1


def test_prepare_features_target_title_case():
    df = _make_sample_df()
    df["test_results"] = df["test_results"].str.lower()   # force lowercase
    _, y = prepare_features_and_target(df)
    for label in y:
        assert label == label.title()


# ── prepare_single_record ─

def test_prepare_single_record_returns_dataframe():
    payload = {
        "Age": 45,
        "Gender": "male",
        "Blood Type": "o+",
        "Medical Condition": "diabetes",
        "Billing Amount": 2000.5,
        "Admission Type": "emergency",
        "Insurance Provider": "Cigna",
        "Medication": "aspirin",
    }
    df = prepare_single_record(payload)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    # Check title-casing was applied
    assert df["gender"].iloc[0] == "Male"
    assert df["medical_condition"].iloc[0] == "Diabetes"


def test_prepare_single_record_accepts_snake_case():
    payload = {
        "age": 30,
        "gender": "Female",
        "blood_type": "A+",
        "medical_condition": "Arthritis",
        "billing_amount": 1500.0,
        "admission_type": "Elective",
        "insurance_provider": "Aetna",
        "medication": "Ibuprofen",
    }
    df = prepare_single_record(payload)
    assert df["age"].iloc[0] == 30
