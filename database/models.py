"""
SQLAlchemy ORM table definitions.

Tables
------
raw_patients        – Unmodified records exactly as loaded from the CSV.
cleaned_patients    – Cleaned, standardised records ready for ML training.
model_runs          – Audit log of every training run (metrics + model path).
"""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────
# Raw data table — insert-only, never modified
# ─────────────────────────────────────────────
class RawPatient(Base):
    __tablename__ = "raw_patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(120))
    age = Column(Integer)
    gender = Column(String(20))
    blood_type = Column(String(10))
    medical_condition = Column(String(100))
    date_of_admission = Column(String(30))   # stored as string at raw stage
    doctor = Column(String(120))
    hospital = Column(String(200))
    insurance_provider = Column(String(100))
    billing_amount = Column(Float)
    room_number = Column(Integer)
    admission_type = Column(String(50))
    discharge_date = Column(String(30))
    medication = Column(String(100))
    test_results = Column(String(30))
    loaded_at = Column(DateTime, default=datetime.utcnow)

    # Prevent duplicate rows from re-ingestion runs
    __table_args__ = (
        UniqueConstraint(
            "name", "date_of_admission", "hospital",
            name="uq_raw_patient_admission",
        ),
    )


# ─────────────────────────────────────────────
# Cleaned / feature-engineered table
# ─────────────────────────────────────────────
class CleanedPatient(Base):
    __tablename__ = "cleaned_patients"

    id = Column(Integer, primary_key=True, autoincrement=True)
    age = Column(Integer, nullable=False)
    gender = Column(String(20), nullable=False)
    blood_type = Column(String(10), nullable=False)
    medical_condition = Column(String(100), nullable=False)
    insurance_provider = Column(String(100), nullable=False)
    billing_amount = Column(Float, nullable=False)
    admission_type = Column(String(50), nullable=False)
    medication = Column(String(100), nullable=False)
    length_of_stay = Column(Integer)          # derived: discharge - admission
    admission_month = Column(Integer)         # derived
    admission_year = Column(Integer)          # derived
    test_results = Column(String(30), nullable=False)   # target label
    cleaned_at = Column(DateTime, default=datetime.utcnow)
    is_valid = Column(Boolean, default=True)

    __table_args__ = (
        UniqueConstraint(
            "age", "gender", "blood_type", "medical_condition",
            "admission_type", "billing_amount", "medication",
            name="uq_cleaned_patient",
        ),
    )


# ─────────────────────────────────────────────
# ML training audit log
# ─────────────────────────────────────────────
class ModelRun(Base):
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    model_name = Column(String(100), nullable=False)
    accuracy = Column(Float)
    precision_score = Column(Float)
    recall_score = Column(Float)
    f1_score = Column(Float)
    training_samples = Column(Integer)
    test_samples = Column(Integer)
    model_path = Column(String(300))
    notes = Column(Text)


def create_all_tables(engine) -> None:
    """Create all tables if they do not already exist."""
    Base.metadata.create_all(engine)
    print("[DB] All tables created (or already exist).")