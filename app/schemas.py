""""
Pydantic v2 request / response models for the FastAPI endpoints.
Field aliases allow the API to accept original column names with spaces.
"""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """
    Input payload for POST /predict.
    Accepts both original casing (e.g. "Blood Type") via alias and
    snake_case field name.
    """

    model_config = {"populate_by_name": True}

    age: int = Field(..., alias="Age", ge=1, le=120, description="Patient age in years")
    gender: str = Field(..., alias="Gender", description="Male or Female")
    blood_type: str = Field(..., alias="Blood Type", description="e.g. O+, A-, B+")
    medical_condition: str = Field(..., alias="Medical Condition", description="e.g. Diabetes")
    billing_amount: float = Field(..., alias="Billing Amount", gt=0)
    admission_type: str = Field(
        ..., alias="Admission Type", description="Emergency / Elective / Urgent"
    )
    insurance_provider: str = Field(..., alias="Insurance Provider", description="e.g. Cigna")
    medication: str = Field(..., alias="Medication", description="e.g. Aspirin")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        allowed = {"male", "female"}
        if v.lower() not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v.strip().title()

    @field_validator("admission_type")
    @classmethod
    def validate_admission_type(cls, v: str) -> str:
        allowed = {"emergency", "elective", "urgent"}
        if v.lower() not in allowed:
            raise ValueError(f"admission_type must be one of {allowed}")
        return v.strip().title()

    @field_validator("blood_type")
    @classmethod
    def validate_blood_type(cls, v: str) -> str:
        allowed = {"a+", "a-", "b+", "b-", "ab+", "ab-", "o+", "o-"}
        if v.lower().replace(" ", "") not in allowed:
            raise ValueError(f"blood_type must be one of {allowed}")
        return v.strip().upper()

    def to_payload_dict(self) -> dict:
        """Return a plain dict keyed by original API names (with spaces)."""
        return {
            "Age": self.age,
            "Gender": self.gender,
            "Blood Type": self.blood_type,
            "Medical Condition": self.medical_condition,
            "Billing Amount": self.billing_amount,
            "Admission Type": self.admission_type,
            "Insurance Provider": self.insurance_provider,
            "Medication": self.medication,
        }


class PredictResponse(BaseModel):
    predicted_test_result: str
    model_name: str | None = None


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    model_loaded: bool
    version: str = "1.0.0"


class TrainResponse(BaseModel):
    status: str
    message: str
