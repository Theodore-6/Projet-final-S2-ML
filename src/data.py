"""Student-owned dataset loading contract.

Students must implement ``load_dataset_split`` so that ``scripts/main.py`` can
evaluate every configured model on the same test split.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROCESSED_DATA_FILE, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE, TEXT_COLUMN

def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    """Return the dataset split used for model evaluation.

    Expected return value:
        A tuple ``(X_train, X_test, y_train, y_test)``.

    Constraints:
    - ``X_train`` and ``X_test`` must contain feature data in a format accepted
      by the trained models stored in ``config.MODELS``.
    - ``y_train`` and ``y_test`` must contain the corresponding targets.
    - ``y_test`` must align with the predictions produced by each loaded model.

    Typical choices for the return types are ``pandas.DataFrame`` /
    ``pandas.Series`` or ``numpy.ndarray``.
    """

    if not PROCESSED_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {PROCESSED_DATA_FILE}. "
            "Add the dataset or generate it before running scripts/main.py."
        )

    df = pd.read_csv(PROCESSED_DATA_FILE)

    required_columns = {TEXT_COLUMN, TARGET_COLUMN}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Dataset is missing required columns: {missing}."
        )

    df = df.dropna(subset=[TEXT_COLUMN, TARGET_COLUMN]).copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).str.strip()
    df = df[df[TEXT_COLUMN] != ""]

    if df.empty:
        raise ValueError("Dataset is empty after dropping invalid text rows.")

    X = df[TEXT_COLUMN]
    y = df[TARGET_COLUMN]

    stratify = y if y.nunique() > 1 else None

    split = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    return tuple(split)
