"""
All FastAPI route handlers.

Endpoints
---------
GET  /              → redirect to /docs
GET  /health        → liveness + readiness check
POST /predict       → run inference on a single patient record
POST /train         → manually trigger a retraining run (admin)
GET  /model-info    → return metadata about the currently loaded model
"""

import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import RedirectResponse

from app.model_loader import get_artefact, reload_model
from app.schemas import HealthResponse, PredictRequest, PredictResponse, TrainResponse
from database import test_connection
from ml.predict import predict_single

logger = logging.getLogger("healthcare_api")
router = APIRouter()


# ── root 

@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


# ── health 

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Liveness + readiness probe.
    Returns DB connectivity status and whether a model is loaded.
    """
    db_ok = test_connection()
    try:
        get_artefact()
        model_ok = True
    except FileNotFoundError:
        model_ok = False

    return HealthResponse(
        status="ok" if (db_ok and model_ok) else "degraded",
        db_connected=db_ok,
        model_loaded=model_ok,
    )


# ── predict 

@router.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Predict patient test result",
)
async def predict(request: PredictRequest):
    """
    Predict whether a patient's test result will be
    **Normal**, **Abnormal**, or **Inconclusive**.

    ### Example request body
    ```json
    {
      "Age": 45,
      "Gender": "Male",
      "Blood Type": "O+",
      "Medical Condition": "Diabetes",
      "Billing Amount": 2000.50,
      "Admission Type": "Emergency",
      "Insurance Provider": "Cigna",
      "Medication": "Aspirin"
    }
    ```
    """
    try:
        payload = request.to_payload_dict()
        result = predict_single(payload)

        artefact = get_artefact()
        model_name = artefact.get("model_name", "unknown")

        logger.info(
            "Prediction → %s | age=%s gender=%s condition=%s",
            result,
            request.age,
            request.gender,
            request.medical_condition,
        )

        return PredictResponse(
            predicted_test_result=result,
            model_name=model_name,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model not found. "
                "Please run a training cycle first via POST /train."
            ),
        ) from exc

    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(exc)}",
        ) from exc


# ── manual retrain 

def _background_train():
    """Run training in a background thread and reload model on completion."""
    try:
        from ml.train import run_training
        run_training()
        reload_model()
        logger.info("Background training completed and model reloaded.")
    except Exception as exc:
        logger.exception("Background training failed: %s", exc)


@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Admin"],
    summary="Trigger manual model retraining",
)
async def trigger_training(background_tasks: BackgroundTasks):
    """
    Manually trigger a model retraining run.
    The training happens asynchronously in the background.
    The response is returned immediately with status 202.
    """
    background_tasks.add_task(_background_train)
    logger.info("Manual training triggered via POST /train")
    return TrainResponse(
        status="accepted",
        message=(
            "Training started in the background. "
            "Check /health to confirm the model reloads after completion."
        ),
    )


# ── model info ─

@router.get("/model-info", tags=["System"])
async def model_info():
    """Return metadata about the currently loaded model."""
    try:
        artefact = get_artefact()
        return {
            "model_name": artefact.get("model_name"),
            "trained_at": artefact.get("trained_at"),
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No model loaded yet.",
        )
