"""
Step 2 — Data Cleaning & Transformation

Reads from raw_patients, applies all cleaning rules, and writes the
results to cleaned_patients.  Safe to re-run (duplicates are skipped).

Cleaning steps
--------------
1. Drop records with nulls in critical columns
2. Remove exact duplicates
3. Parse + validate date columns → derive length_of_stay
4. Standardise categorical string values (title-case, strip whitespace)
5. Remove clearly invalid values (age ≤ 0, billing ≤ 0, LOS < 0)
6. Verify target column contains only allowed classes

Usage:
    uv run python scripts/clean.py
"""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import CleanedPatient, create_all_tables, get_engine, get_session

#  constants 

ALLOWED_TEST_RESULTS = {"Normal", "Abnormal", "Inconclusive"}

CRITICAL_COLUMNS = [
    "age", "gender", "blood_type", "medical_condition",
    "billing_amount", "admission_type", "medication", "test_results",
    "date_of_admission", "discharge_date",
]


# ── cleaning helpers ─

def load_raw(session) -> pd.DataFrame:
    query = "SELECT * FROM raw_patients"
    df = pd.read_sql(query, session.bind)
    print(f"[clean] Loaded {len(df):,} rows from raw_patients")
    return df


def drop_critical_nulls(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=CRITICAL_COLUMNS)
    dropped = before - len(df)
    if dropped:
        print(f"[clean] Dropped {dropped:,} rows with null critical fields")
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(
        subset=["name", "date_of_admission", "hospital"], keep="first"
    )
    dropped = before - len(df)
    if dropped:
        print(f"[clean] Dropped {dropped:,} exact duplicate rows")
    return df


def parse_dates_and_los(df: pd.DataFrame) -> pd.DataFrame:
    """Parse date strings and compute length_of_stay in days."""
    df["date_of_admission"] = pd.to_datetime(
        df["date_of_admission"], errors="coerce", dayfirst=False
    )
    df["discharge_date"] = pd.to_datetime(
        df["discharge_date"], errors="coerce", dayfirst=False
    )

    # Drop rows where dates couldn't be parsed
    before = len(df)
    df = df.dropna(subset=["date_of_admission", "discharge_date"])
    if len(df) < before:
        print(f"[clean] Dropped {before - len(df):,} rows with unparseable dates")

    df["length_of_stay"] = (
        df["discharge_date"] - df["date_of_admission"]
    ).dt.days

    df["admission_month"] = df["date_of_admission"].dt.month
    df["admission_year"] = df["date_of_admission"].dt.year

    # Remove negative or zero LOS
    before = len(df)
    df = df[df["length_of_stay"] > 0]
    if len(df) < before:
        print(f"[clean] Dropped {before - len(df):,} rows with invalid LOS")

    return df


def standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Title-case + strip all string columns."""
    string_cols = [
        "gender", "blood_type", "medical_condition", "admission_type",
        "medication", "insurance_provider", "test_results",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    return df


def fix_test_results(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only records with recognised target class labels."""
    before = len(df)
    df = df[df["test_results"].isin(ALLOWED_TEST_RESULTS)]
    dropped = before - len(df)
    if dropped:
        print(
            f"[clean] Dropped {dropped:,} rows with unknown test_results values"
        )
    return df


def remove_invalid_numerics(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[(df["age"] > 0) & (df["age"] <= 120)]
    df = df[df["billing_amount"] > 0]
    dropped = before - len(df)
    if dropped:
        print(f"[clean] Dropped {dropped:,} rows with invalid numeric values")
    return df


def build_cleaned_records(df: pd.DataFrame) -> list[dict]:
    """Select and rename columns for cleaned_patients table."""
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "age": int(row["age"]),
                "gender": row["gender"],
                "blood_type": row["blood_type"],
                "medical_condition": row["medical_condition"],
                "insurance_provider": row.get("insurance_provider", "Unknown"),
                "billing_amount": float(row["billing_amount"]),
                "admission_type": row["admission_type"],
                "medication": row["medication"],
                "length_of_stay": int(row["length_of_stay"]),
                "admission_month": int(row["admission_month"]),
                "admission_year": int(row["admission_year"]),
                "test_results": row["test_results"],
                "is_valid": True,
            }
        )
    return records


def upsert_cleaned(records: list[dict]) -> int:
    """Insert cleaned records, skip on unique-constraint conflicts."""
    inserted = 0
    batch_size = 500

    with get_session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            stmt = (
                pg_insert(CleanedPatient)
                .values(batch)
                .on_conflict_do_nothing(constraint="uq_cleaned_patient")
            )
            result = session.execute(stmt)
            inserted += result.rowcount
            print(
                f"[clean] Batch {i // batch_size + 1}: {result.rowcount} new rows"
            )

    return inserted


# ── entry point ────────────────────────────────────────────────────────────

def run() -> None:
    create_all_tables(get_engine())

    with get_session() as session:
        df = load_raw(session)

    df = drop_critical_nulls(df)
    df = drop_duplicates(df)
    df = parse_dates_and_los(df)
    df = standardise_categoricals(df)
    df = fix_test_results(df)
    df = remove_invalid_numerics(df)

    print(f"\n[clean] Clean dataset: {len(df):,} rows remain after all checks")

    class_dist = df["test_results"].value_counts()
    print(f"\n[clean] Class distribution:\n{class_dist.to_string()}\n")

    records = build_cleaned_records(df)
    inserted = upsert_cleaned(records)

    print(
        f"\n[clean] Done — {inserted:,} new rows inserted into cleaned_patients."
    )


if __name__ == "__main__":
    run()
