"""Streamlit entry point for the legal case triage demo."""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    DATA_DIR,
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


CATEGORY_LABELS = {
    "plein_contentieux": "Plein contentieux",
    "exces_de_pouvoir": "Recours pour exces de pouvoir",
    "autres_recours": "Autres recours administratifs",
}

OUTCOME_LABELS = {
    "defavorable_requerant": "Defavorable au requerant",
    "favorable_requerant": "Favorable au requerant",
    "partiellement_favorable": "Partiellement favorable au requerant",
    "desistement": "Desistement du requerant",
    "renvoi": "Renvoi devant une autre juridiction ou formation",
    "non_lieu": "Non-lieu",
    "autre_issue": "Autre issue procedurale",
}

OUTCOME_USER_MEANINGS = {
    "defavorable_requerant": "Si vous etes le requerant : vous perdez ou vous n'obtenez pas ce que vous demandiez.",
    "favorable_requerant": "Si vous etes le requerant : vous gagnez.",
    "partiellement_favorable": "Si vous etes le requerant : vous gagnez seulement en partie.",
    "desistement": "Si vous etes le requerant : vous retirez votre requete.",
    "renvoi": "L'affaire est renvoyee devant une autre juridiction ou formation ; ce n'est pas une victoire nette immediate.",
    "non_lieu": "Le juge estime qu'il n'y a plus lieu de statuer au fond.",
    "autre_issue": "Issue procedurale diverse qui ne se lit pas comme une victoire ou une defaite simple.",
}

OUTCOME_WINNER_LABELS = {
    "defavorable_requerant": "La partie defenderesse gagne le plus souvent (souvent l'administration)",
    "favorable_requerant": "Le requerant gagne le plus souvent",
    "partiellement_favorable": "Le requerant gagne en partie",
    "desistement": "Pas de gagnant clair : le requerant retire sa requete",
    "renvoi": "Pas de gagnant clair a ce stade",
    "non_lieu": "Pas de gagnant clair a ce stade",
    "autre_issue": "Pas de gagnant clair",
}

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "f1_macro": "F1 macro",
    "precision_macro": "Precision macro",
    "recall_macro": "Recall macro",
}

SPECIALIST_GUIDANCE = {
    "plein_contentieux": {
        "specialist": "Avocat en plein contentieux administratif",
        "orientation": (
            "A orienter vers un specialiste des litiges ou l'on demande une "
            "condamnation, une indemnisation, un paiement, une responsabilite "
            "de la puissance publique ou un contentieux fiscal / contractuel."
        ),
        "examples": (
            "Exemples frequents : responsabilite hospitaliere, indemnisation, "
            "marches publics, fiscalite, sanctions administratives avec enjeu financier."
        ),
    },
    "exces_de_pouvoir": {
        "specialist": "Avocat en recours contre les decisions administratives",
        "orientation": (
            "A orienter vers un specialiste qui attaque la legalite d'un acte "
            "administratif et cherche surtout son annulation."
        ),
        "examples": (
            "Exemples frequents : refus d'autorisation, decret, arrete, "
            "decision d'administration, sanction administrative contestee."
        ),
    },
    "autres_recours": {
        "specialist": "Avocat en contentieux administratif a confirmer",
        "orientation": (
            "Categorie plus heterogene. Une revue humaine est necessaire pour "
            "identifier le sous-contentieux exact avant d'orienter le dossier."
        ),
        "examples": (
            "Exemples possibles : pension, execution, interpretation, revision "
            "ou autres recours plus techniques."
        ),
    },
}

DOMAIN_KEYWORDS = {
    "administratif": [
        "administration",
        "administratif",
        "arrete",
        "autorisation",
        "collectivite",
        "commune",
        "conseil d'etat",
        "decision administrative",
        "decret",
        "etat",
        "impot",
        "impots",
        "mairie",
        "ministere",
        "permis",
        "prefet",
        "prefecture",
        "recours",
        "refus",
        "sanction administrative",
        "service public",
        "titre de sejour",
    ],
    "travail": [
        "employe",
        "employeur",
        "entreprise",
        "harcelement",
        "licenciement",
        "patron",
        "prudhom",
        "prud'homme",
        "prudhommes",
        "prud'hommes",
        "salaire",
        "salarie",
        "stage",
        "travail",
    ],
    "penal": [
        "agression",
        "amende",
        "arme",
        "coupable",
        "delit",
        "escroquerie",
        "garde a vue",
        "homicide",
        "peine",
        "plainte",
        "police",
        "prison",
        "viol",
        "vol",
    ],
    "famille": [
        "divorce",
        "enfant",
        "famille",
        "garde",
        "mariage",
        "pension alimentaire",
        "separation",
        "succession",
        "tutelle",
    ],
}

DOMAIN_SPECIALISTS = {
    "administratif": "Avocat en droit administratif",
    "travail": "Avocat en droit du travail",
    "penal": "Avocat en droit penal",
    "famille": "Avocat en droit de la famille",
}

REFERENCE_METADATA_FILE = DATA_DIR / "processed_conseil_etat_june_2022.csv"


def _humanize_category(value: str) -> str:
    return CATEGORY_LABELS.get(value, value)


def _humanize_outcome(value: str) -> str:
    return OUTCOME_LABELS.get(value, value)


def _normalize_text(value: str) -> str:
    normalized = value.lower()
    replacements = {
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "à": "a",
        "â": "a",
        "î": "i",
        "ï": "i",
        "ô": "o",
        "ö": "o",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "'": " ",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized


def _detect_legal_domain(facts_summary: str) -> dict[str, object]:
    normalized = _normalize_text(facts_summary)
    scores = {}
    matched_terms = {}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        matches = []
        for keyword in keywords:
            keyword_normalized = _normalize_text(keyword)
            if keyword_normalized in normalized:
                matches.append(keyword)
        scores[domain] = len(matches)
        matched_terms[domain] = matches

    top_domain = max(scores, key=scores.get)
    top_score = scores[top_domain]
    second_score = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0

    return {
        "scores": scores,
        "matched_terms": matched_terms,
        "top_domain": top_domain,
        "top_score": top_score,
        "second_score": second_score,
        "is_confident": top_score >= 2 and top_score > second_score,
    }


def _specialist_guidance(predicted_category: str) -> dict[str, str]:
    return SPECIALIST_GUIDANCE.get(
        predicted_category,
        {
            "specialist": "Specialiste administratif a confirmer",
            "orientation": "Une qualification humaine complementaire est necessaire.",
            "examples": "Le dossier sort du perimetre le plus stable du MVP.",
        },
    )


def _load_dataset() -> Optional[pd.DataFrame]:
    if not PROCESSED_DATA_FILE.exists():
        return None

    return pd.read_csv(PROCESSED_DATA_FILE)


def _load_metrics() -> Optional[pd.DataFrame]:
    if not MODEL_METRICS_FILE.exists():
        return None

    return pd.read_csv(MODEL_METRICS_FILE)


def _load_reference_metadata() -> Optional[pd.DataFrame]:
    if not REFERENCE_METADATA_FILE.exists():
        return None

    use_columns = [
        "source_file",
        "numero_ecli",
        "numero_dossier",
        "date_lecture",
        "nom_juridiction",
        "formation_jugement",
        "solution",
    ]
    return pd.read_csv(REFERENCE_METADATA_FILE, usecols=use_columns)


def _load_demo_model():
    for model_key in ("log_reg_legal", "linear_svm_legal"):
        model_config = MODELS.get(model_key)
        if model_config and model_config["path"].exists():
            return model_config, load_model(model_config["path"])

    for model_config in MODELS.values():
        if model_config["path"].exists():
            return model_config, load_model(model_config["path"])

    return None, None


def _format_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    display_df = metrics_df.copy()
    display_df["model_name"] = display_df["model_name"].astype(str)

    for column in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
        if column in display_df.columns:
            display_df[column] = (display_df[column] * 100).round(1)

    rename_map = {
        "model_name": "Modele",
        "accuracy": "Accuracy (%)",
        "f1_macro": "F1 macro (%)",
        "precision_macro": "Precision macro (%)",
        "recall_macro": "Recall macro (%)",
    }
    keep_columns = [column for column in rename_map if column in display_df.columns]
    return display_df[keep_columns].rename(columns=rename_map)


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


def _format_probability_df(probability_df: pd.DataFrame) -> pd.DataFrame:
    display_df = probability_df.copy()
    display_df["category_label"] = display_df["case_category"].map(_humanize_category)
    display_df["probability"] = (display_df["probability"] * 100).round(1)
    return display_df.rename(
        columns={
            "category_label": "Categorie interpretee",
            "case_category": "Code categorie",
            "probability": "Probabilite (%)",
        }
    )[
        ["Categorie interpretee", "Code categorie", "Probabilite (%)"]
    ]


def _format_outcome_df(outcome_df: pd.DataFrame) -> pd.DataFrame:
    display_df = outcome_df.copy()
    display_df["outcome_label"] = display_df[OUTCOME_COLUMN].map(_humanize_outcome)
    display_df["winner_label"] = display_df[OUTCOME_COLUMN].map(OUTCOME_WINNER_LABELS)
    display_df["user_meaning"] = display_df[OUTCOME_COLUMN].map(
        OUTCOME_USER_MEANINGS
    )
    return display_df.rename(
        columns={
            "outcome_label": "Lecture metier",
            "winner_label": "Qui est le plus souvent avantagé",
            "user_meaning": "Si vous etes le requerant, cela signifie",
            "share": "Part observee (%)",
        }
    )[
        [
            "Lecture metier",
            "Qui est le plus souvent avantagé",
            "Si vous etes le requerant, cela signifie",
            "Part observee (%)",
        ]
    ]


def _find_similar_jurisprudence(
    model,
    dataset_df: Optional[pd.DataFrame],
    reference_df: Optional[pd.DataFrame],
    facts_summary: str,
    top_n: int = 3,
) -> Optional[pd.DataFrame]:
    if dataset_df is None or dataset_df.empty:
        return None

    pipeline_steps = getattr(model, "named_steps", {})
    vectorizer = pipeline_steps.get("tfidf")
    if vectorizer is None or not hasattr(vectorizer, "transform"):
        return None

    dataset_vectors = vectorizer.transform(dataset_df[TEXT_COLUMN].astype(str))
    query_vector = vectorizer.transform([facts_summary])
    similarities = cosine_similarity(query_vector, dataset_vectors).ravel()

    similar_df = dataset_df.copy()
    similar_df["similarity"] = similarities
    similar_df = similar_df.sort_values("similarity", ascending=False).head(top_n)

    if reference_df is not None and "source_file" in similar_df.columns:
        similar_df = similar_df.merge(reference_df, on="source_file", how="left", suffixes=("", "_ref"))

    similar_df["categorie_lisible"] = similar_df[TARGET_COLUMN].map(_humanize_category)
    similar_df["similarity"] = (similar_df["similarity"] * 100).round(1)
    similar_df["resume_court"] = (
        similar_df[TEXT_COLUMN]
        .astype(str)
        .str.slice(0, 220)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        + "..."
    )

    ecli_column = "numero_ecli" if "numero_ecli" in similar_df.columns else None
    dossier_column = "numero_dossier" if "numero_dossier" in similar_df.columns else None
    juridiction_column = "nom_juridiction" if "nom_juridiction" in similar_df.columns else None

    if ecli_column is None:
        similar_df["numero_ecli"] = ""
        ecli_column = "numero_ecli"
    if dossier_column is None:
        similar_df["numero_dossier"] = ""
        dossier_column = "numero_dossier"
    if juridiction_column is None:
        similar_df["nom_juridiction"] = ""
        juridiction_column = "nom_juridiction"

    return similar_df.rename(
        columns={
            "date_lecture": "Date",
            "categorie_lisible": "Categorie",
            "solution": "Issue observee",
            "similarity": "Proximite textuelle (%)",
            "resume_court": "Extrait de faits proches",
            ecli_column: "ECLI",
            dossier_column: "Numero dossier",
            juridiction_column: "Juridiction",
        }
    )[
        [
            "ECLI",
            "Numero dossier",
            "Date",
            "Juridiction",
            "Categorie",
            "Issue observee",
            "Proximite textuelle (%)",
            "Extrait de faits proches",
        ]
    ]


def _top_linear_evidence(
    model, facts_summary: str, predicted_category: str, top_n: int = 6
) -> Optional[pd.DataFrame]:
    pipeline_steps = getattr(model, "named_steps", {})
    vectorizer = pipeline_steps.get("tfidf")
    classifier = pipeline_steps.get("classifier")

    if vectorizer is None or classifier is None:
        return None

    if not hasattr(vectorizer, "transform") or not hasattr(vectorizer, "get_feature_names_out"):
        return None

    if not hasattr(classifier, "coef_") or not hasattr(classifier, "classes_"):
        return None

    class_labels = list(classifier.classes_)
    if predicted_category not in class_labels:
        return None

    row = vectorizer.transform([facts_summary])
    class_index = class_labels.index(predicted_category)
    contributions = row.multiply(classifier.coef_[class_index]).tocsr()
    feature_names = vectorizer.get_feature_names_out()

    evidence = []
    for feature_index, contribution in zip(contributions.indices, contributions.data):
        contribution_value = float(contribution)
        if contribution_value <= 0:
            continue
        evidence.append(
            {
                "Terme repere dans le texte": feature_names[feature_index],
                "Contribution": round(contribution_value, 4),
            }
        )

    if not evidence:
        return None

    evidence_df = pd.DataFrame(evidence).sort_values("Contribution", ascending=False)
    return evidence_df.head(top_n)


def build_app() -> None:
    st.set_page_config(page_title=PROJECT_TITLE, layout="wide")

    dataset_df = _load_dataset()
    reference_df = _load_reference_metadata()
    metrics_df = _load_metrics()
    model_config, model = _load_demo_model()

    st.title(PROJECT_TITLE)
    st.caption(PROJECT_SUBTITLE)

    st.markdown(
        """
        Cet outil aide a qualifier un dossier juridique a partir d'un resume de
        faits. Le MVP actuel est centre sur des decisions du Conseil d'Etat en
        droit administratif issues de l'open data officiel de la justice
        administrative. Il doit etre interprete comme un outil d'orientation
        et de productivite, pas comme un conseil juridique.
        """
    )

    st.error(
        "Perimetre actuel du MVP : cette application est concue uniquement pour "
        "des dossiers de droit administratif. Les cas de droit du travail, penal, "
        "famille ou contrats peuvent etre mal interpretes."
    )

    st.caption(
        "Source officielle: https://opendata.justice-administrative.fr "
        "| Extension possible: CAA et tribunaux administratifs"
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
        st.markdown(
            """
            Le jeu de donnees actuellement charge provient d'un lot Conseil d'Etat
            transforme a partir de l'open data officiel de la justice administrative.
            """
        )
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
            category_counts[TARGET_COLUMN] = category_counts[TARGET_COLUMN].map(
                _humanize_category
            )
            st.dataframe(category_counts, width="stretch")

    st.subheader("Comparaison des modeles")
    if metrics_df is None:
        st.info(
            "Aucun resultat disponible pour le moment. Lance `python scripts/train_baselines.py` "
            "puis `python scripts/main.py`."
        )
    else:
        st.dataframe(_format_metrics(metrics_df), width="stretch", hide_index=True)

        with st.expander("Comment lire les metriques"):
            st.markdown(
                """
                - `Accuracy` : part totale des predictions correctes.
                - `F1 macro` : moyenne de l'equilibre precision/rappel sur chaque classe, utile quand les classes ne sont pas parfaitement equilibrees.
                - `Precision macro` : quand le modele predit une classe, a quelle frequence cette prediction est correcte en moyenne.
                - `Recall macro` : capacite du modele a retrouver chaque classe du dataset en moyenne.
                """
            )

    st.subheader("Demo interactive")
    st.info(
        "Le modele actuel est entraine sur des decisions de droit administratif. "
        "Un cas relevant du droit du travail, du penal ou du droit de la famille "
        "peut donc etre mal oriente."
    )
    facts_summary = st.text_area(
        "Resume les faits de l'affaire (droit administratif uniquement)",
        placeholder=(
            "Exemple: Une personne conteste un refus de titre de sejour, "
            "une decision de prefecture ou un acte administratif."
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
            specialist_guidance = _specialist_guidance(predicted_category)
            domain_signal = _detect_legal_domain(facts_summary)
            top_domain = str(domain_signal["top_domain"])
            top_domain_score = int(domain_signal["top_score"])
            domain_is_confident = bool(domain_signal["is_confident"])
            st.success(
                "Categorie predite: "
                f"`{_humanize_category(predicted_category)}`"
            )
            st.caption(f"Code interne de la categorie: `{predicted_category}`")
            st.caption(f"Modele utilise: {model_config['name']}")

            st.markdown("Lecture du perimetre du cas")
            if domain_is_confident and top_domain != "administratif":
                matched_terms = ", ".join(domain_signal["matched_terms"][top_domain][:6])
                st.error(
                    f"Ce texte ressemble davantage a un dossier de **{top_domain}** "
                    f"qu'a un dossier administratif. Le modele administratif est donc "
                    f"probablement hors perimetre ici. Indices reperes : {matched_terms}."
                )
                st.info(
                    f"**Specialiste plus plausible :** {DOMAIN_SPECIALISTS[top_domain]}"
                )
            elif top_domain == "administratif" and top_domain_score >= 2:
                matched_terms = ", ".join(domain_signal["matched_terms"][top_domain][:6])
                st.info(
                    "Le texte contient plusieurs indices compatibles avec le droit "
                    f"administratif. Indices reperes : {matched_terms}."
                )
            else:
                st.warning(
                    "Le perimetre juridique du texte reste ambigu. L'orientation "
                    "automatique doit etre lue avec prudence."
                )

            st.markdown("Orientation vers le specialiste")
            if domain_is_confident and top_domain != "administratif":
                st.warning(
                    f"Je ne recommande **pas** ici un specialiste administratif comme "
                    f"orientation principale. Le dossier semble plutot relever du "
                    f"**{top_domain}**."
                )
            else:
                st.info(
                    f"**Specialiste recommande :** {specialist_guidance['specialist']}\n\n"
                    f"**Pourquoi :** {specialist_guidance['orientation']}\n\n"
                    f"**Dossiers typiques :** {specialist_guidance['examples']}"
                )

            if probability_df is not None:
                top_probability = float(probability_df.iloc[0]["probability"])
                if top_probability < 0.55:
                    st.warning(
                        "Le score de confiance reste modere. Il faut lire cette "
                        "prediction avec prudence, surtout si les faits ne relevent "
                        "pas du droit administratif."
                    )

                st.markdown("Probabilites par categorie")
                st.dataframe(
                    _format_probability_df(probability_df.head(3)),
                    width="stretch",
                    hide_index=True,
                )

            with st.expander("Pourquoi seulement 3 categories dans ce MVP ?"):
                st.markdown(
                    """
                    Le corpus actuel vient surtout d'un lot Conseil d'Etat en droit administratif.
                    Les labels du MVP derivent donc de trois grandes familles procedurales :

                    - `Recours pour exces de pouvoir` : on conteste surtout la legalite d'une decision administrative.
                    - `Plein contentieux` : on demande souvent une condamnation, une indemnisation ou une reforme plus large.
                    - `Autres recours administratifs` : categorie residuelle pour les recours moins frequents ou plus techniques.

                    Ce n'est pas encore une cartographie complete de tous les specialistes du droit francais.
                    """
                )

            st.markdown("Pourquoi cette sortie ?")
            st.warning(
                "Le modele actuel est un modele lexical. Il repere surtout des "
                "mots et groupes de mots, pas une comprehension profonde du sens "
                "juridique. Une autre formulation des memes faits peut donc produire "
                "un resultat different."
            )
            evidence_df = _top_linear_evidence(model, facts_summary, predicted_category)
            if evidence_df is not None:
                st.markdown(
                    """
                    Les termes ci-dessous sont les indices lexicaux qui ont le plus
                    pousse le modele vers la categorie predite. Ce n'est pas une
                    preuve de causalite juridique et ce n'est pas une lecture du sens
                    profond du texte.
                    """
                )
                st.dataframe(evidence_df, width="stretch", hide_index=True)
            else:
                st.info(
                    "Le modele utilise ici ne permet pas d'afficher une explication "
                    "lexicale detaillee pour cette prediction."
                )

            outcome_df = _empirical_outcomes(dataset_df, predicted_category)
            if outcome_df is not None:
                st.markdown(
                    "Estimation empirique sur des cas administratifs similaires du dataset:"
                )
                st.info(
                    "On ne parle pas ici de `culpabilite`. En contentieux "
                    "administratif, il faut plutot lire ce tableau comme une "
                    "indication de qui gagne ou perd le plus souvent. Le "
                    "`requerant` est la personne, l'entreprise ou l'entite qui a "
                    "saisi le juge. Si l'issue est `defavorable au requerant`, cela "
                    "veut dire que le requerant perd et que la partie defenderesse "
                    "(souvent l'administration) est avantagée."
                )
                st.dataframe(
                    _format_outcome_df(outcome_df),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info(
                    "Pas d'estimation empirique disponible pour cette categorie."
                )

            st.markdown("Jurisprudences administratives proches dans le corpus")
            if domain_is_confident and top_domain != "administratif":
                st.warning(
                    "Je n'affiche pas de rapprochement de jurisprudence administrative "
                    "comme reference principale ici, car le texte semble hors perimetre "
                    "du droit administratif."
                )
            else:
                similar_cases_df = _find_similar_jurisprudence(
                    model,
                    dataset_df,
                    reference_df,
                    facts_summary,
                )
                if similar_cases_df is not None:
                    st.caption(
                        "Ces references correspondent aux decisions du corpus dont le "
                        "resume de faits est le plus proche textuellement de votre saisie."
                    )
                    st.dataframe(similar_cases_df, width="stretch", hide_index=True)
                else:
                    st.info(
                        "Impossible de calculer pour l'instant un rapprochement fiable "
                        "avec la jurisprudence du corpus."
                    )

    st.subheader("Limites")
    st.markdown(
        """
        - Le dataset actuel couvre surtout le droit administratif et non l'ensemble du droit francais.
        - La source officielle permet d'etendre le projet aux CAA et aux tribunaux administratifs, mais ce MVP reste centre sur le Conseil d'Etat.
        - Le modele de base est lexical : il est sensible aux mots choisis et peut varier selon la formulation des faits.
        - Les probabilites sont des signaux statistiques, pas des certitudes.
        - Toute analyse finale doit rester sous controle humain.
        """
    )


if __name__ == "__main__":
    build_app()
