from typing import Any
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    precision = float(precision_score(y_true, y_pred))
    recall = float(recall_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred))
    roc_auc = float(roc_auc_score(y_true, y_pred))

    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        y_true, y_pred
    ).ravel()

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "cm_true_negative": float(true_negative),
        "cm_false_positive": float(false_positive),
        "cm_false_negative": float(false_negative),
        "cm_true_positive": float(true_positive),
    }
