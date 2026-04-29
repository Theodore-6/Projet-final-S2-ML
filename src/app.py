"""Streamlit entry point for the legal case triage demo."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from config import (
    MODEL_METRICS_FILE,
    MODELS,
    OUTCOME_COLUMN,
    PROCESSED_DATA_FILE,
    PROJECT_SUBTITLE,
    PROJECT_TITLE,
    TARGET_COLUMN,
    TEXT_COLUMN,
)
from model_io import load_model


def _load_dataset() -> Optional[pd.DataFrame]:
    if not PROCESSED_DATA_FILE.exists():
        return None

    return pd.read_csv(PROCESSED_DATA_FILE)


def _load_metrics() -> Optional[pd.DataFrame]:
    if not MODEL_METRICS_FILE.exists():
        return None

    return pd.read_csv(MODEL_METRICS_FILE)


def _load_demo_model():
    for model_key in ("log_reg_legal", "linear_svm_legal"):
        model_config = MODELS.get(model_key)
        if model_config and model_config["path"].exists():
            return model_config, load_model(model_config["path"])

    for model_config in MODELS.values():
        if model_config["path"].exists():
            return model_config, load_model(model_config["path"])

    return None, None


def _predict_case(model, facts_summary: str) -> tuple[str, Optional[pd.DataFrame]]:
    prediction = model.predict([facts_summary])[0]

    if not hasattr(model, "predict_proba"):
        return prediction, None

    probabilities = model.predict_proba([facts_summary])[0]
    labels = model.classes_
    probability_df = pd.DataFrame(
        {"case_category": labels, "probability": probabilities}
    ).sort_values("probability", ascending=False)

    return prediction, probability_df


def _empirical_outcomes(
    dataset_df: Optional[pd.DataFrame], predicted_category: str
) -> Optional[pd.DataFrame]:
    if dataset_df is None or OUTCOME_COLUMN not in dataset_df.columns:
        return None

    subset = dataset_df[dataset_df[TARGET_COLUMN] == predicted_category]
    if subset.empty:
        return None

    outcome_df = (
        subset[OUTCOME_COLUMN]
        .value_counts(normalize=True)
        .rename_axis(OUTCOME_COLUMN)
        .reset_index(name="share")
    )
    outcome_df["share"] = (outcome_df["share"] * 100).round(1)
    return outcome_df


def build_app() -> None:
    st.set_page_config(page_title=PROJECT_TITLE, layout="wide")

    dataset_df = _load_dataset()
    metrics_df = _load_metrics()
    model_config, model = _load_demo_model()

    st.title(PROJECT_TITLE)
    st.caption(PROJECT_SUBTITLE)

    st.markdown(
        """
        Cet outil aide a qualifier un dossier juridique a partir d'un resume de
        faits. Le MVP actuel est centre sur des decisions du Conseil d'Etat en
        droit administratif. Il doit etre interprete comme un outil
        d'orientation et de productivite, pas comme un conseil juridique.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Objectif metier", "Triage initial")
    with col2:
        st.metric(
            "Categories suivies",
            dataset_df[TARGET_COLUMN].nunique() if dataset_df is not None else 0,
        )
    with col3:
        st.metric(
            "Cas illustratifs",
            len(dataset_df) if dataset_df is not None else 0,
        )

    st.subheader("Interet business")
    st.markdown(
        """
        - accelerer la pre-analyse des dossiers entrants,
        - orienter plus vite un dossier vers le bon specialiste,
        - homogeniser la qualification initiale des litiges,
        - fournir un signal empirique simple sur des cas similaires.
        """
    )

    st.subheader("Vue dataset")
    if dataset_df is None:
        st.warning("Le dataset de travail est introuvable.")
    else:
        left, right = st.columns((2, 1))
        with left:
            st.dataframe(dataset_df.head(10), width="stretch")
        with right:
            category_counts = (
                dataset_df[TARGET_COLUMN]
                .value_counts()
                .rename_axis(TARGET_COLUMN)
                .reset_index(name="count")
            )
            st.dataframe(category_counts, width="stretch")

    st.subheader("Comparaison des modeles")
    if metrics_df is None:
        st.info(
            "Aucun resultat disponible pour le moment. Lance `python scripts/train_baselines.py` "
            "puis `python scripts/main.py`."
        )
    else:
        st.dataframe(metrics_df, width="stretch")

    st.subheader("Demo interactive")
    facts_summary = st.text_area(
        "Resume les faits de l'affaire",
        placeholder=(
            "Exemple: Un salarie conteste son licenciement apres plusieurs "
            "heures supplementaires non payees."
        ),
        height=160,
    )

    if st.button("Analyser le dossier", type="primary"):
        if not facts_summary.strip():
            st.warning("Saisis d'abord un resume de faits.")
        elif model is None or model_config is None:
            st.error(
                "Aucun modele entraine disponible. Lance `python scripts/train_baselines.py`."
            )
        else:
            predicted_category, probability_df = _predict_case(model, facts_summary)
            st.success(f"Categorie predite: `{predicted_category}`")
            st.caption(f"Modele utilise: {model_config['name']}")

            if probability_df is not None:
                st.dataframe(
                    probability_df.head(3),
                    width="stretch",
                    hide_index=True,
                )

            outcome_df = _empirical_outcomes(dataset_df, predicted_category)
            if outcome_df is not None:
                st.markdown(
                    "Estimation empirique issue de cas administratifs similaires du dataset:"
                )
                st.dataframe(outcome_df, width="stretch", hide_index=True)
            else:
                st.info(
                    "Pas d'estimation empirique disponible pour cette categorie."
                )

    st.subheader("Limites")
    st.markdown(
        """
        - Le dataset actuel couvre surtout le droit administratif et non l'ensemble du droit francais.
        - Les probabilites sont des signaux statistiques, pas des certitudes.
        - Toute analyse finale doit rester sous controle humain.
        """
    )


if __name__ == "__main__":
    build_app()
