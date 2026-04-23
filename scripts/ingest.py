"""
Step 1 — Data Ingestion

Loads the raw healthcare CSV into PostgreSQL table `raw_patients`.
Duplicate rows are silently skipped so the script is safe to re-run.

Usage:
    uv run python scripts/ingest.py
    uv run python scripts/ingest.py --csv data/raw/healthcare_dataset.csv
"""

import argparse
import shutil
import sys
from pathlib import Path

import kagglehub
import pandas as pd
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import RawPatient, create_all_tables, get_engine, get_session

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "raw" / "healthcare_dataset.csv"


# ── helpers 
def download_dataset(dest: Path) -> None:
    """Download dataset from Kaggle Hub if not already present."""
    if dest.exists():
        print(f"[ingest] Dataset already exists at {dest}, skipping download.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print("[ingest] Downloading dataset from Kaggle Hub...")

    path = kagglehub.dataset_download("prasad22/healthcare-dataset")
    csv_file = next(Path(path).glob("*.csv"))
    shutil.copy(csv_file, dest)

    print(f"[ingest] Dataset saved to {dest}")


def load_csv(path: Path) -> pd.DataFrame:
    """Read CSV and normalise column names to snake_case."""
    df = pd.read_csv(path)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s/]", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    print(f"[ingest] Loaded {len(df):,} rows from {path.name}")
    print(f"[ingest] Columns: {list(df.columns)}")
    return df


def map_to_raw_records(df: pd.DataFrame) -> list[dict]:
    """Map DataFrame rows to dicts matching the RawPatient schema."""
    column_map = {
        "name": "name",
        "age": "age",
        "gender": "gender",
        "blood_type": "blood_type",
        "medical_condition": "medical_condition",
        "date_of_admission": "date_of_admission",
        "doctor": "doctor",
        "hospital": "hospital",
        "insurance_provider": "insurance_provider",
        "billing_amount": "billing_amount",
        "room_number": "room_number",
        "admission_type": "admission_type",
        "discharge_date": "discharge_date",
        "medication": "medication",
        "test_results": "test_results",
    }
    records = []
    for _, row in df.iterrows():
        record = {db_col: row.get(csv_col) for csv_col, db_col in column_map.items()}
        records.append(record)
    return records


def upsert_raw_records(records: list[dict]) -> int:
    """Insert records, skipping duplicates. Returns rows inserted."""
    inserted = 0
    batch_size = 500

    with get_session() as session:
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            stmt = (
                pg_insert(RawPatient)
                .values(batch)
                .on_conflict_do_nothing(constraint="uq_raw_patient_admission")
            )
            result = session.execute(stmt)
            inserted += result.rowcount
            print(f"[ingest] Batch {i // batch_size + 1}: {result.rowcount} new rows")

    return inserted


# ── entry point 

def run(csv_path: Path) -> None:
    # Auto-download if file is missing
    if not csv_path.exists():
        download_dataset(csv_path)

    create_all_tables(get_engine())

    df = load_csv(csv_path)
    records = map_to_raw_records(df)
    total_inserted = upsert_raw_records(records)

    print(f"\n[ingest] Done — {total_inserted:,} new rows inserted into raw_patients.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest raw healthcare CSV into PostgreSQL")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    run(args.csv)