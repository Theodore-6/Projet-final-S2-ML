# Legal Case Triage Assistant

This project adapts the upstream ML proof-of-concept template into a legal triage assistant. The current MVP is intentionally scoped to French administrative law using Conseil d'Etat decisions. The goal is to classify the probable recours category from a short summary of facts, then surface empirical outcome hints from similar cases.

The product positioning is intentionally narrow: this is an orientation tool for pre-analysis, not an automated lawyer and not legal advice.

## Business Objective

Legal teams lose time qualifying incoming cases before routing them to the right specialist. This project targets that first step:

- reduce manual time spent on initial dossier review,
- standardize case qualification across similar fact patterns,
- help prioritize dossiers that need faster human review,
- create a simple front-end that demonstrates business value to a cabinet or legaltech.

## Project Scope

The main ML task is:

- predict the legal case category from a text summary of facts.

The business-facing bonus exposed in the app is:

- show empirical outcome tendencies from similar labeled cases in the starter dataset.

Current target categories:

- `plein_contentieux`
- `exces_de_pouvoir`
- `autres_recours`

## Repository Workflow

This repository follows the upstream template workflow:

- `scripts/main.py` evaluates saved models on a shared test split,
- metrics are written to `results/model_metrics.csv`,
- `src/app.py` renders the Streamlit demo.

Additional project-specific files:

- `data/processed_conseil_etat_june_2022.csv`: flat extract of Conseil d'Etat XML decisions,
- `data/processed_legal_cases_admin.csv`: ML-ready administrative law dataset,
- `scripts/parse_conseil_etat.py`: converts raw XML files to a structured CSV,
- `scripts/prepare_admin_dataset.py`: builds the ML-ready dataset used by the app and training pipeline,
- `scripts/train_baselines.py`: trains baseline text models and saves them in `models/`.

## Dataset Schema

The current working dataset comes from the June 2022 Conseil d'Etat open-data batch and is transformed into an ML-ready administrative law dataset.

Expected columns:

- `facts_summary`: short free-text description of the facts,
- `case_category`: simplified recours category predicted by the models,
- `likely_outcome`: coarse empirical outcome label for dashboard analysis.

## Baseline Models

Configured baseline models:

- `TF-IDF + Logistic Regression`
- `TF-IDF + Linear SVM`

These are strong, interpretable baselines for short legal text classification and fit the template evaluation flow well.

## How To Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Build the real administrative law dataset:

```bash
/usr/bin/python3 scripts/parse_conseil_etat.py
/usr/bin/python3 scripts/prepare_admin_dataset.py
```

3. Train baseline models:

```bash
python scripts/train_baselines.py
```

4. Evaluate models and launch the app:

```bash
python scripts/main.py
```

To only regenerate metrics without starting Streamlit:

```bash
python scripts/main.py --no-streamlit
```

## Expected Output

After a successful run, you should have:

- serialized models in `models/`,
- evaluation results in `results/model_metrics.csv`,
- a Streamlit app showing:
  - the business objective,
  - dataset coverage,
  - model comparison,
  - an interactive text demo,
  - empirical outcome hints for similar cases.

## Important Limits

- The current MVP is based on administrative decisions only and does not cover all French law.
- Output probabilities or outcome hints must be framed as empirical signals, not certainties.
- The tool supports legal orientation only and does not replace professional legal analysis.
