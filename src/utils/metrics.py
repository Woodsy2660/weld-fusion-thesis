"""Shared evaluation metrics for all modalities."""
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred, class_names=None):
    """Compute all evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "report": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True
        ),
    }
