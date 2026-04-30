from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DATA_DIR, MODELS, RESULTS_DIR  # noqa: E402
from model_io import load_model  # noqa: E402


BENCHMARK_FILE = DATA_DIR / "admin_robustness_benchmark.csv"
ROBUSTNESS_DETAILS_FILE = RESULTS_DIR / "robustness_predictions.csv"
ROBUSTNESS_SUMMARY_FILE = RESULTS_DIR / "robustness_metrics.csv"


def _evaluate_model(model_key: str, model, benchmark_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    predictions = model.predict(benchmark_df["facts_summary"])
    details_df = benchmark_df.copy()
    details_df["model_key"] = model_key
    details_df["predicted_category"] = predictions
    details_df["is_correct"] = details_df["predicted_category"] == details_df["expected_category"]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(benchmark_df["facts_summary"])
        details_df["top_probability"] = probabilities.max(axis=1)
    else:
        details_df["top_probability"] = np.nan

    consistency_scores = []
    for _, group_df in details_df.groupby("case_id"):
        modal_count = group_df["predicted_category"].value_counts().max()
        consistency_scores.append(modal_count / len(group_df))

    summary = {
        "model_key": model_key,
        "accuracy_expected_category": float(details_df["is_correct"].mean()),
        "paraphrase_consistency": float(sum(consistency_scores) / len(consistency_scores)),
        "mean_top_probability": (
            float(details_df["top_probability"].dropna().mean())
            if details_df["top_probability"].notna().any()
            else np.nan
        ),
    }
    return details_df, summary


def main() -> None:
    benchmark_df = pd.read_csv(BENCHMARK_FILE)
    all_details = []
    all_summaries = []

    for model_key, model_config in MODELS.items():
        model_path = Path(model_config["path"])
        if not model_path.exists():
            print(f"Skipping {model_key}: missing model at {model_path}")
            continue

        model = load_model(model_path)
        details_df, summary = _evaluate_model(model_key, model, benchmark_df)
        summary["model_name"] = model_config["name"]
        all_details.append(details_df)
        all_summaries.append(summary)

    if not all_summaries:
        raise FileNotFoundError("No trained model found. Run scripts/train_baselines.py first.")

    results_df = pd.DataFrame(all_summaries)
    details_df = pd.concat(
        [frame for frame in all_details if not frame.empty],
        ignore_index=True,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(ROBUSTNESS_SUMMARY_FILE, index=False)
    details_df.to_csv(ROBUSTNESS_DETAILS_FILE, index=False)

    print("Robustness evaluation completed.")
    print(results_df.to_string(index=False))
    print(f"\nSaved: {ROBUSTNESS_SUMMARY_FILE}")
    print(f"Saved: {ROBUSTNESS_DETAILS_FILE}")


if __name__ == "__main__":
    main()
