from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys
import xml.etree.ElementTree as ET


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from config import DATA_DIR  # noqa: E402


DEFAULT_INPUT_DIR = DATA_DIR / "external" / "conseil_etat" / "CE_202206"
DEFAULT_OUTPUT_FILE = DATA_DIR / "processed_conseil_etat_june_2022.csv"

FIELDS = [
    "source_file",
    "source_prefix",
    "code_juridiction",
    "nom_juridiction",
    "numero_dossier",
    "date_lecture",
    "numero_ecli",
    "type_decision",
    "type_recours",
    "code_publication",
    "solution",
    "formation_jugement",
    "texte_integral",
    "text_length",
]


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def extract_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return normalize_text(" ".join(text for text in node.itertext() if text))


def extract_field(document: ET.Element, tag: str) -> str:
    node = document.find(f".//{tag}")
    if node is None:
        return ""
    return normalize_text(node.text)


def parse_case(path: Path) -> dict[str, str | int]:
    tree = ET.parse(path)
    document = tree.getroot()
    full_text = extract_text(document.find(".//Texte_Integral"))

    return {
        "source_file": path.name,
        "source_prefix": path.name.split("_", 1)[0],
        "code_juridiction": extract_field(document, "Code_Juridiction"),
        "nom_juridiction": extract_field(document, "Nom_Juridiction"),
        "numero_dossier": extract_field(document, "Numero_Dossier"),
        "date_lecture": extract_field(document, "Date_Lecture"),
        "numero_ecli": extract_field(document, "Numero_ECLI"),
        "type_decision": extract_field(document, "Type_Decision"),
        "type_recours": extract_field(document, "Type_Recours"),
        "code_publication": extract_field(document, "Code_Publication"),
        "solution": extract_field(document, "Solution"),
        "formation_jugement": extract_field(document, "Formation_Jugement"),
        "texte_integral": full_text,
        "text_length": len(full_text),
    }


def build_dataset(input_dir: Path, output_file: Path) -> list[dict[str, str | int]]:
    rows = [parse_case(path) for path in sorted(input_dir.glob("*.xml"))]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_summary(rows: list[dict[str, str | int]]) -> None:
    prefixes = Counter(row["source_prefix"] for row in rows)
    decision_types = Counter(row["type_decision"] for row in rows if row["type_decision"])
    recours_types = Counter(row["type_recours"] for row in rows if row["type_recours"])
    solutions = Counter(row["solution"] for row in rows if row["solution"])
    lengths = [int(row["text_length"]) for row in rows if row["text_length"]]

    print(f"cases: {len(rows)}")
    print(f"prefixes: {dict(prefixes)}")
    print(f"type_decision_top: {decision_types.most_common(5)}")
    print(f"type_recours_top: {recours_types.most_common(5)}")
    print(f"solution_top: {solutions.most_common(8)}")
    if lengths:
        print(
            "text_length_stats:"
            f" min={min(lengths)} avg={round(sum(lengths) / len(lengths), 1)} max={max(lengths)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Conseil d'Etat XML decisions into a flat CSV dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing Conseil d'Etat XML files.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="CSV file to create.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_dataset(args.input_dir, args.output_file)
    print_summary(rows)
    print(f"written: {args.output_file}")


if __name__ == "__main__":
    main()
