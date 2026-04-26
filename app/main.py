"""
FastAPI application entry point.

Responsibilities
----------------
1. Create the FastAPI app with metadata and CORS settings.
2. Register all routes from app/routes.py.
3. On startup:
   - Test the database connection.
   - Create all tables if they don't exist.
   - Load the saved model (if it exists).
   - Start the APScheduler job that retrains every Saturday at 12:00 noon UTC.
4. On shutdown:
   - Gracefully stop the scheduler.

Run locally:
    uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Allow project root imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.routes import router
from app.utils import configure_logging, ensure_model_exists, get_version
from database import create_all_tables, get_engine, test_connection

logger = configure_logging()

# ── scheduler

scheduler = BackgroundScheduler(timezone="UTC")


def scheduled_retrain():
    """Weekly retraining job — runs every Saturday at 12:00 UTC."""
    logger.info("Scheduled weekly retraining job triggered.")
    try:
        from ml.train import run_training
        from ml.predict import reload_model

        run_training()
        reload_model()
        logger.info(" Weekly retraining completed and model reloaded.")
    except Exception as exc:
        logger.exception(" Weekly retraining failed: %s", exc)


# ── lifespan 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.
    Code before `yield` runs on startup; code after runs on shutdown.
    """
    # ── Startup 
    logger.info(" Starting Healthcare ML API v%s …", get_version())

    # 1. Database
    if test_connection():
        logger.info(" Database connection OK")
        create_all_tables(get_engine())
    else:
        logger.warning(
            " Database unreachable. Predictions will still work if a "
            "model is already saved; DB-dependent endpoints will fail."
        )

    # 2. Model
    if ensure_model_exists():
        try:
            from app.model_loader import load_model
            load_model()
            logger.info(" Model loaded from disk")
        except Exception as exc:
            logger.warning("Could not load model: %s", exc)
    else:
        logger.warning(
            " No saved model found. "
            "POST /train or wait for Saturday's scheduled run."
        )

    # 3. Scheduler — every Saturday at noon UTC
    scheduler.add_job(
        scheduled_retrain,
        trigger=CronTrigger(day_of_week="sat", hour=12, minute=0),
        id="weekly_retrain",
        replace_existing=True,
        misfire_grace_time=3600,   # allow up to 1 hour of misfire
    )
    scheduler.start()
    logger.info(
        " Scheduler started. Next retraining: Saturday 12:00 UTC"
    )

    yield  # ── application runs here 

    # ── Shutdown 
    scheduler.shutdown(wait=False)
    logger.info(" Scheduler stopped. API shutting down.")


# ── app factory 

def create_app() -> FastAPI:
    app = FastAPI(
        title="Healthcare ML API",
        description=(
            "End-to-end healthcare analytics system.\n\n"
            "Predicts patient test results (Normal / Abnormal / Inconclusive) "
            "using a machine learning model retrained every Saturday at 12:00 UTC."
        ),
        version=get_version(),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — allow all origins; restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(router)

    # Serve the HTML frontend at /ui
    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
    if frontend_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app


# Module-level app instance used by uvicorn
app = create_app()


# ── dev server 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
