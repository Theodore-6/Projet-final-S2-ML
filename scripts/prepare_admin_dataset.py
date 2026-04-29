from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from config import DATA_DIR  # noqa: E402


DEFAULT_INPUT_FILE = DATA_DIR / "processed_conseil_etat_june_2022.csv"
DEFAULT_OUTPUT_FILE = DATA_DIR / "processed_legal_cases_admin.csv"
MIN_SUMMARY_LENGTH = 120

OUTPUT_FIELDS = [
    "facts_summary",
    "case_category",
    "likely_outcome",
    "source_file",
    "source_prefix",
    "date_lecture",
    "type_decision",
    "type_recours",
    "solution",
    "text_length",
]


def slugify_case_category(value: str) -> str:
    value = value.strip().lower()
    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "à": "a",
        "â": "a",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ù": "u",
        "û": "u",
        "'": "",
        "-": "_",
        " ": "_",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def simplify_recours(type_recours: str) -> str:
    normalized = slugify_case_category(type_recours)
    if normalized in {"plein_contentieux", "exces_de_pouvoir"}:
        return normalized
    return "autres_recours"


def simplify_outcome(solution: str) -> str:
    normalized = solution.lower()
    if "satisfaction totale" in normalized:
        return "favorable_requerant"
    if "satisfaction partielle" in normalized:
        return "partiellement_favorable"
    if "annulation" in normalized:
        return "favorable_requerant"
    if "rejet" in normalized:
        return "defavorable_requerant"
    if "desistement" in normalized or "désistement" in normalized:
        return "desistement"
    if "non-lieu" in normalized:
        return "non_lieu"
    if "renvoi" in normalized:
        return "renvoi"
    return "autre_issue"


def extract_facts_summary(full_text: str, max_chars: int = 2200) -> str:
    text = " ".join(full_text.split())
    if not text:
        return ""

    lower_text = text.lower()
    start_markers = [
        "vu la procedure suivante :",
        "vu les procedures suivantes :",
        "vu la procédure suivante :",
        "vu les procédures suivantes :",
    ]
    end_markers = [
        "vu les autres pieces du dossier",
        "vu les autres pièces du dossier",
        "vu :",
        "considerant ce qui suit",
        "considérant ce qui suit",
        "d e c i d e",
        "o r d o n n e",
    ]

    start_index = 0
    for marker in start_markers:
        marker_index = lower_text.find(marker)
        if marker_index != -1:
            start_index = marker_index + len(marker)
            break

    end_index = len(text)
    for marker in end_markers:
        marker_index = lower_text.find(marker, start_index)
        if marker_index != -1:
            end_index = min(end_index, marker_index)

    summary = text[start_index:end_index].strip()
    if not summary:
        summary = text[:max_chars].strip()

    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].strip()

    return summary


def prepare_rows(input_file: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with input_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            full_text = row.get("texte_integral", "")
            facts_summary = extract_facts_summary(full_text)
            if not facts_summary or len(facts_summary) < MIN_SUMMARY_LENGTH:
                continue

            type_recours = (row.get("type_recours") or "").strip()
            solution = (row.get("solution") or "").strip()
            if not type_recours:
                continue

            rows.append(
                {
                    "facts_summary": facts_summary,
                    "case_category": simplify_recours(type_recours),
                    "likely_outcome": simplify_outcome(solution),
                    "source_file": row.get("source_file", ""),
                    "source_prefix": row.get("source_prefix", ""),
                    "date_lecture": row.get("date_lecture", ""),
                    "type_decision": row.get("type_decision", ""),
                    "type_recours": type_recours,
                    "solution": solution,
                    "text_length": row.get("text_length", ""),
                }
            )
    return rows


def write_rows(rows: list[dict[str, str]], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare an ML-ready administrative law dataset from Conseil d'Etat decisions."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Flat CSV generated from Conseil d'Etat XML decisions.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="ML-ready CSV file to create.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = prepare_rows(args.input_file)
    write_rows(rows, args.output_file)
    print(f"rows: {len(rows)}")
    print(f"written: {args.output_file}")


if __name__ == "__main__":
    main()
