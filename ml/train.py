"""
Weekly training pipeline — runs every Saturday at 12:00 noon (scheduled
by the APScheduler job in app/main.py).

To run:
    uv run python ml/train.py

Models trained
--------------
1. XGBoost          
2. Random Forest    (ensemble baseline)
3. Logistic Regression (linear baseline)

The best model by weighted F1 is saved to  models/model.joblib .
Evaluation metrics for every run are stored in the model_runs table.
"""

import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from database import ModelRun, get_session
from ml.evaluate import evaluate_model, print_evaluation_report
from ml.preprocess import (
    build_feature_pipeline,
    build_label_encoder,
    load_training_data,
    prepare_features_and_target,
)

# ── constants 

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ── model definitions ─

def get_candidate_models(n_classes: int) -> dict:
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        # "LogisticRegression": LogisticRegression(
        #     multi_class="multinomial",
        #     solver="lbfgs",
        #     max_iter=1000,
        #     C=1.0,
        #     class_weight="balanced",
        #     random_state=RANDOM_STATE,
        #     n_jobs=-1,
        # ),
        "LogisticRegression": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


# ── training helpers ─

def build_full_pipeline(preprocessor, classifier) -> Pipeline:
    """Wrap preprocessor + classifier into a single sklearn Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def save_model_artefact(pipeline: Pipeline, label_encoder, model_name: str) -> Path:
    """Save the fitted pipeline + label encoder to disk."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    versioned_path = MODEL_DIR / f"{model_name}_{timestamp}.joblib"

    artefact = {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "model_name": model_name,
        "trained_at": datetime.utcnow().isoformat(),
    }

    joblib.dump(artefact, versioned_path)
    joblib.dump(artefact, BEST_MODEL_PATH)   # overwrite the "latest" link

    print(f"[train] Model saved → {versioned_path}")
    print(f"[train] Latest symlink → {BEST_MODEL_PATH}")

    return versioned_path


def log_model_run(
    model_name: str,
    metrics: dict,
    model_path: Path,
    n_train: int,
    n_test: int,
) -> None:
    """Persist evaluation metrics to the model_runs audit table."""
    with get_session() as session:
        run = ModelRun(
            run_at=datetime.utcnow(),
            model_name=model_name,
            accuracy=metrics["accuracy"],
            precision_score=metrics["precision"],
            recall_score=metrics["recall"],
            f1_score=metrics["f1"],
            training_samples=n_train,
            test_samples=n_test,
            model_path=str(model_path),
        )
        session.add(run)
    print(f"[train] Metrics logged to model_runs table.")


# ── main training loop

def run_training() -> None:
    print(f"\n{'='*60}")
    print(f"  Healthcare ML Training Run — {datetime.utcnow().isoformat()}")
    print(f"{'='*60}\n")

    # 1. Load data 
    with get_session() as session:
        df = load_training_data(session)

    if len(df) < 100:
        print(
            "[train] ERROR: Not enough data to train "
            f"(found {len(df)} rows, need ≥ 100). Aborting."
        )
        return

    # 2. Prepare features 
    X, y = prepare_features_and_target(df)
    label_encoder = build_label_encoder()
    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )

    print(
        f"[train] Train: {len(X_train):,} rows | "
        f"Test: {len(X_test):,} rows | "
        f"Classes: {list(label_encoder.classes_)}\n"
    )

    # 3. Train each candidate model 
    n_classes = len(label_encoder.classes_)
    candidates = get_candidate_models(n_classes)
    results: list[dict] = []

    for model_name, classifier in candidates.items():
        print(f"[train] Training {model_name} …")
        preprocessor = build_feature_pipeline()
        pipeline = build_full_pipeline(preprocessor, classifier)

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test, label_encoder)
        print_evaluation_report(model_name, metrics)

        results.append(
            {
                "name": model_name,
                "metrics": metrics,
                "pipeline": pipeline,
            }
        )

    # 4. Select the best model (by weighted F1) 
    best = max(results, key=lambda r: r["metrics"]["f1"])
    print(
        f"\n[train] ✓ Best model: {best['name']} "
        f"(F1 = {best['metrics']['f1']:.4f})\n"
    )

    # 5. Save artefact 
    saved_path = save_model_artefact(best["pipeline"], label_encoder, best["name"])

    # 6. Log to DB 
    log_model_run(
        model_name=best["name"],
        metrics=best["metrics"],
        model_path=saved_path,
        n_train=len(X_train),
        n_test=len(X_test),
    )

    print(f"\n[train] Weekly training run complete.\n")


if __name__ == "__main__":
    run_training()
