"""Helpers for saving evaluation results."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from config import CLASS_METRICS_FILE, CONFUSION_MATRIX_FILE, MODEL_METRICS_FILE


def write_metrics(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Write model metrics to ``results/model_metrics.csv`` and return a DataFrame."""

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(MODEL_METRICS_FILE, index=False)
    return metrics_df


def write_class_metrics(class_metrics_df: pd.DataFrame) -> pd.DataFrame:
    class_metrics_df.to_csv(CLASS_METRICS_FILE, index=False)
    return class_metrics_df


def write_confusion_matrix(confusion_df: pd.DataFrame) -> pd.DataFrame:
    confusion_df.to_csv(CONFUSION_MATRIX_FILE, index=False)
    return confusion_df
