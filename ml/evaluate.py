"""
Model evaluation utilities.

Computes accuracy, precision, recall, F1, and a confusion matrix.
All functions accept the fitted sklearn Pipeline directly so that
features are always transformed through the same preprocessor.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(pipeline, X_test, y_test, label_encoder) -> dict:
    """
    Run inference on the test set and return a metrics dictionary.

    Parameters
    ----------
    pipeline       : fitted sklearn Pipeline (preprocessor + classifier)
    X_test         : feature DataFrame (not yet transformed)
    y_test         : encoded integer labels (numpy array)
    label_encoder  : fitted LabelEncoder (for class names in report)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, confusion_matrix,
                    classification_report (string), y_pred
    """
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "y_pred": y_pred,
    }


def print_evaluation_report(model_name: str, metrics: dict) -> None:
    """Pretty-print evaluation metrics to stdout."""
    sep = "─" * 50
    print(sep)
    print(f"  Model  : {model_name}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print()
    print("  Classification Report:")
    for line in metrics["classification_report"].splitlines():
        print(f"    {line}")
    print()
    print("  Confusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    for row in cm:
        print("    " + "  ".join(f"{v:4d}" for v in row))
    print(sep)
