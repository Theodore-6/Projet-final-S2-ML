"""Student-owned metrics contract.

Students must implement ``compute_metrics`` to return the evaluation metrics
that matter for their project.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return the metrics used to compare model performance.

    Expected return value:
        A dictionary mapping metric names to numeric values, for example:
        ``{"accuracy": 0.91, "f1": 0.88}``.

    Constraints:
    - Every value must be numeric and convertible to ``float``.
    - Use the same metric set for every model so results remain comparable.
    - Keep metric names stable because they are written to
      ``results/model_metrics.csv``.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def compute_class_metrics(y_true: Any, y_pred: Any) -> pd.DataFrame:
    labels = sorted(pd.Series(y_true).astype(str).unique().tolist())
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    return pd.DataFrame(
        {
            "case_category": labels,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": support,
        }
    )


def compute_confusion_df(y_true: Any, y_pred: Any) -> pd.DataFrame:
    labels = sorted(pd.Series(y_true).astype(str).unique().tolist())
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    confusion_df = pd.DataFrame(matrix, index=labels, columns=labels)
    confusion_df.index.name = "actual_category"
    return confusion_df.reset_index()
