"""
Step 3 — Verify & summarise the cleaned data loaded into the database.

Prints row counts, class distribution, and a sample of records

Usage:
    uv run python scripts/load.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import get_session


def run() -> None:
    with get_session() as session:
        raw_count = session.execute(
            __import__("sqlalchemy").text("SELECT COUNT(*) FROM raw_patients")
        ).scalar()

        clean_count = session.execute(
            __import__("sqlalchemy").text("SELECT COUNT(*) FROM cleaned_patients")
        ).scalar()

        class_dist_rows = session.execute(
            __import__("sqlalchemy").text(
                "SELECT test_results, COUNT(*) AS cnt "
                "FROM cleaned_patients GROUP BY test_results ORDER BY cnt DESC"
            )
        ).fetchall()

        sample = pd.read_sql(
            "SELECT age, gender, blood_type, medical_condition, "
            "admission_type, billing_amount, medication, "
            "length_of_stay, test_results "
            "FROM cleaned_patients LIMIT 5",
            session.bind,
        )

    print("=" * 55)
    print("  Database Load Summary")
    print("=" * 55)
    print(f"  raw_patients    : {raw_count:>8,} rows")
    print(f"  cleaned_patients: {clean_count:>8,} rows")
    print()

    print("  Class distribution (cleaned_patients)")
    print("  " + "-" * 35)
    total = sum(r[1] for r in class_dist_rows)
    for label, cnt in class_dist_rows:
        pct = 100 * cnt / total if total else 0
        print(f"  {label:<20} {cnt:>6,}  ({pct:.1f} %)")
    print()

    print("  Sample records (first 5)")
    print(sample.to_string(index=False))
    print("=" * 55)


if __name__ == "__main__":
    run()
