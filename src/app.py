"""Premium Streamlit dashboard for the legal case triage demo."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    ROBUSTNESS_METRICS_FILE,
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
    "defavorable_requerant": (
        "Si vous etes le requerant : vous perdez ou vous n'obtenez pas ce que "
        "vous demandiez."
    ),
    "favorable_requerant": "Si vous etes le requerant : vous gagnez.",
    "partiellement_favorable": (
        "Si vous etes le requerant : vous gagnez seulement en partie."
    ),
    "desistement": "Si vous etes le requerant : vous retirez votre requete.",
    "renvoi": (
        "L'affaire est renvoyee devant une autre juridiction ou formation ; ce "
        "n'est pas une victoire nette immediate."
    ),
    "non_lieu": "Le juge estime qu'il n'y a plus lieu de statuer au fond.",
    "autre_issue": (
        "Issue procedurale diverse qui ne se lit pas comme une victoire ou une "
        "defaite simple."
    ),
}

OUTCOME_WINNER_LABELS = {
    "defavorable_requerant": (
        "La partie defenderesse gagne le plus souvent (souvent l'administration)"
    ),
    "favorable_requerant": "Le requerant gagne le plus souvent",
    "partiellement_favorable": "Le requerant gagne en partie",
    "desistement": "Pas de gagnant clair : le requerant retire sa requete",
    "renvoi": "Pas de gagnant clair a ce stade",
    "non_lieu": "Pas de gagnant clair a ce stade",
    "autre_issue": "Pas de gagnant clair",
}

SPECIALIST_GUIDANCE = {
    "plein_contentieux": {
        "specialist": "Avocat en plein contentieux administratif",
        "orientation": (
            "A orienter vers un specialiste des litiges ou l'on demande une "
            "condamnation, une indemnisation, un paiement, une responsabilite "
            "de la puissance publique ou un contentieux fiscal ou contractuel."
        ),
        "examples": (
            "Exemples frequents : responsabilite hospitaliere, indemnisation, "
            "marches publics, fiscalite, sanctions administratives avec enjeu "
            "financier."
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

ADMIN_CATEGORY_RULES = {
    "exces_de_pouvoir": {
        "decision": [
            "refus",
            "annulation",
            "annuler",
            "arrete",
            "decret",
            "decision",
            "autorisation",
            "permis",
            "sanction",
            "oqtf",
        ],
        "autority": [
            "prefet",
            "prefecture",
            "ministre",
            "maire",
            "recteur",
            "administration",
            "office francais",
        ],
        "immigration": [
            "titre de sejour",
            "sejour",
            "visa",
            "asile",
            "etranger",
        ],
    },
    "plein_contentieux": {
        "money": [
            "indemnisation",
            "indemnitaire",
            "condamnation",
            "condamner",
            "verser",
            "somme",
            "euros",
            "interets",
            "remboursement",
        ],
        "harm": [
            "prejudice",
            "dommage",
            "responsabilite",
            "faute",
            "reparation",
        ],
        "fiscal_social": [
            "impot",
            "cotisation",
            "taxe",
            "decharge",
            "allocation",
            "pension",
            "chomage",
            "marche public",
        ],
    },
    "autres_recours": {
        "procedure": [
            "renvoi",
            "sursis a statuer",
            "question prejudicielle",
            "avis contentieux",
            "interpretation",
            "execution",
            "tribunal des pensions",
            "cour d'appel",
            "prud'hommes",
        ]
    },
}

DOMAIN_SPECIALISTS = {
    "administratif": "Avocat en droit administratif",
    "travail": "Avocat en droit du travail",
    "penal": "Avocat en droit penal",
    "famille": "Avocat en droit de la famille",
}

DOMAIN_LABELS = {
    "administratif": "Droit administratif",
    "travail": "Droit du travail",
    "penal": "Droit penal",
    "famille": "Droit de la famille",
}

REFERENCE_METADATA_FILE = DATA_DIR / "processed_conseil_etat_june_2022.csv"


def _inject_css() -> None:
    """Push Streamlit toward a premium, glass-heavy, Apple-like interface."""
    st.markdown(
        """
        <style>
            :root {
                --surface: rgba(255, 255, 255, 0.72);
                --surface-strong: rgba(255, 255, 255, 0.84);
                --surface-soft: rgba(248, 248, 250, 0.64);
                --line: rgba(15, 23, 42, 0.08);
                --line-strong: rgba(15, 23, 42, 0.12);
                --text: #0f172a;
                --muted: #667085;
                --shadow: 0 20px 60px rgba(15, 23, 42, 0.08);
                --shadow-soft: 0 8px 24px rgba(15, 23, 42, 0.05);
                --radius-xl: 24px;
                --radius-lg: 20px;
                --radius-md: 16px;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(214, 224, 255, 0.55), transparent 35%),
                    radial-gradient(circle at top right, rgba(245, 247, 250, 0.8), transparent 30%),
                    linear-gradient(180deg, #f7f8fa 0%, #eef1f5 100%);
                color: var(--text);
            }

            [data-testid="stAppViewContainer"] > .main {
                background: transparent;
            }

            .main .block-container {
                max-width: 1120px;
                padding-top: 2.5rem;
                padding-bottom: 4rem;
                padding-left: 1.4rem;
                padding-right: 1.4rem;
            }

            [data-testid="stSidebar"] {
                background: rgba(250, 250, 252, 0.74);
                border-right: 1px solid rgba(255, 255, 255, 0.55);
                backdrop-filter: blur(18px);
            }

            [data-testid="stSidebar"] > div:first-child {
                background: transparent;
            }

            [data-testid="stSidebar"] .block-container {
                padding-top: 2rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }

            header[data-testid="stHeader"] {
                background: transparent;
            }

            [data-testid="stToolbar"] {
                right: 1rem;
            }

            div.stButton > button {
                width: 100%;
                min-height: 52px;
                border-radius: 999px;
                border: 1px solid rgba(255, 255, 255, 0.72);
                background: linear-gradient(180deg, rgba(17, 24, 39, 0.92), rgba(17, 24, 39, 0.82));
                color: #ffffff;
                font-weight: 600;
                letter-spacing: -0.01em;
                box-shadow: 0 14px 32px rgba(17, 24, 39, 0.16);
                transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
            }

            div.stButton > button:hover {
                transform: translateY(-1px) scale(1.01);
                box-shadow: 0 18px 36px rgba(17, 24, 39, 0.2);
                filter: brightness(1.02);
            }

            div.stButton > button:focus:not(:active) {
                border: 1px solid rgba(15, 23, 42, 0.12);
                box-shadow: 0 0 0 4px rgba(148, 163, 184, 0.16);
            }

            div[data-baseweb="textarea"] textarea {
                border-radius: var(--radius-lg);
                background: rgba(255, 255, 255, 0.76);
                border: 1px solid rgba(255, 255, 255, 0.58);
                box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.45);
                backdrop-filter: blur(14px);
                font-size: 1rem;
                line-height: 1.6;
                padding: 1rem 1.1rem;
            }

            div[data-baseweb="select"] > div,
            div[data-baseweb="base-input"] > div {
                border-radius: 16px;
            }

            [data-testid="stDataFrame"],
            [data-testid="stMetric"],
            [data-testid="stAlert"] {
                border-radius: var(--radius-lg);
            }

            [data-testid="stDataFrame"] {
                border: 1px solid rgba(255, 255, 255, 0.45);
                box-shadow: var(--shadow-soft);
                overflow: hidden;
            }

            [data-testid="stPlotlyChart"] {
                border-radius: var(--radius-lg);
                overflow: hidden;
            }

            [data-testid="stRadio"] label {
                font-size: 0.95rem;
            }

            .hero-shell {
                padding: 1.2rem 0 1.8rem 0;
            }

            .eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.4rem 0.78rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.55);
                border: 1px solid rgba(255, 255, 255, 0.72);
                color: #475467;
                font-size: 0.8rem;
                letter-spacing: 0.02em;
                text-transform: uppercase;
                backdrop-filter: blur(16px);
            }

            .hero-title {
                margin: 1rem 0 0.55rem 0;
                font-size: clamp(2.3rem, 5vw, 4.5rem);
                line-height: 0.98;
                letter-spacing: -0.045em;
                font-weight: 700;
                color: var(--text);
            }

            .hero-subtitle {
                max-width: 760px;
                margin: 0;
                color: var(--muted);
                font-size: 1.02rem;
                line-height: 1.7;
                letter-spacing: -0.01em;
            }

            .glass-card {
                background: var(--surface);
                border: 1px solid rgba(255, 255, 255, 0.52);
                border-radius: var(--radius-xl);
                box-shadow: var(--shadow);
                backdrop-filter: blur(18px);
                -webkit-backdrop-filter: blur(18px);
                padding: 1.35rem;
                transition: transform 180ms ease, box-shadow 180ms ease;
            }

            .glass-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 24px 72px rgba(15, 23, 42, 0.1);
            }

            .kpi-card {
                min-height: 162px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
            }

            .kpi-label {
                color: var(--muted);
                font-size: 0.88rem;
                letter-spacing: -0.01em;
            }

            .kpi-value {
                margin-top: 0.75rem;
                font-size: clamp(1.9rem, 3vw, 2.7rem);
                line-height: 1;
                letter-spacing: -0.04em;
                font-weight: 700;
                color: var(--text);
            }

            .kpi-hint {
                color: #475467;
                font-size: 0.92rem;
                line-height: 1.5;
            }

            .section-heading {
                margin: 0;
                font-size: 1.45rem;
                line-height: 1.15;
                letter-spacing: -0.03em;
                font-weight: 650;
                color: var(--text);
            }

            .section-subtitle {
                margin: 0.35rem 0 0 0;
                color: var(--muted);
                font-size: 0.96rem;
                line-height: 1.6;
            }

            .micro-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.38rem 0.7rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.48);
                border: 1px solid rgba(255, 255, 255, 0.64);
                color: #344054;
                font-size: 0.84rem;
                font-weight: 500;
            }

            .signal-card {
                padding: 1rem 1.05rem;
                background: rgba(255, 255, 255, 0.44);
                border: 1px solid rgba(255, 255, 255, 0.52);
                border-radius: 18px;
                min-height: 140px;
            }

            .signal-label {
                color: var(--muted);
                font-size: 0.84rem;
                letter-spacing: -0.01em;
            }

            .signal-value {
                margin-top: 0.7rem;
                font-size: 1.08rem;
                line-height: 1.35;
                letter-spacing: -0.02em;
                font-weight: 600;
                color: var(--text);
            }

            .signal-caption {
                margin-top: 0.6rem;
                color: #475467;
                font-size: 0.9rem;
                line-height: 1.5;
            }

            .callout {
                padding: 1rem 1.05rem;
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 0.58);
                backdrop-filter: blur(12px);
                background: rgba(255, 255, 255, 0.58);
            }

            .callout-strong {
                background: rgba(255, 252, 235, 0.74);
                border-color: rgba(245, 158, 11, 0.18);
            }

            .callout-danger {
                background: rgba(255, 245, 245, 0.82);
                border-color: rgba(239, 68, 68, 0.18);
            }

            .callout-ok {
                background: rgba(239, 250, 244, 0.82);
                border-color: rgba(16, 185, 129, 0.18);
            }

            .callout-title {
                margin: 0;
                font-size: 0.95rem;
                font-weight: 650;
                color: var(--text);
            }

            .callout-copy {
                margin: 0.45rem 0 0 0;
                color: #475467;
                font-size: 0.92rem;
                line-height: 1.55;
            }

            .footer-note {
                color: var(--muted);
                font-size: 0.9rem;
                line-height: 1.6;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _glass_open(extra_class: str = "") -> None:
    st.markdown(
        f'<div class="glass-card {extra_class}">'.strip(),
        unsafe_allow_html=True,
    )


def _glass_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _metric_card(label: str, value: str, hint: str) -> str:
    return f"""
    <div class="glass-card kpi-card">
        <div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        <div class="kpi-hint">{hint}</div>
    </div>
    """


def _signal_card(label: str, value: str, caption: str) -> str:
    return f"""
    <div class="signal-card">
        <div class="signal-label">{label}</div>
        <div class="signal-value">{value}</div>
        <div class="signal-caption">{caption}</div>
    </div>
    """


def _section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <h2 class="section-heading">{title}</h2>
            <p class="section-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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


def _generic_admin_specialist() -> dict[str, str]:
    return {
        "specialist": "Qualification administrative a confirmer",
        "orientation": (
            "Le dossier semble administratif mais la famille de recours reste trop "
            "incertaine pour orienter vers un specialiste plus fin sans revue humaine."
        ),
        "examples": (
            "Fais relire le dossier pour arbitrer entre annulation d'une decision, "
            "contentieux indemnitaire, fiscal ou recours proceduraux techniques."
        ),
    }


def _get_similarity_vectorizer(model):
    pipeline_steps = getattr(model, "named_steps", {})
    direct_vectorizer = pipeline_steps.get("tfidf")
    if direct_vectorizer is not None and hasattr(direct_vectorizer, "transform"):
        return direct_vectorizer

    feature_union = pipeline_steps.get("features")
    transformer_list = getattr(feature_union, "transformer_list", [])
    for name, transformer in transformer_list:
        if name == "word_tfidf" and hasattr(transformer, "transform"):
            return transformer

    return None


def _base_probability_df(model, facts_summary: str) -> Optional[pd.DataFrame]:
    labels = getattr(model, "classes_", None)
    if labels is None:
        return None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([facts_summary])[0]
    elif hasattr(model, "decision_function"):
        decision_scores = model.decision_function([facts_summary])
        decision_scores = np.asarray(decision_scores)
        if decision_scores.ndim == 1:
            decision_scores = decision_scores.reshape(1, -1)
        stabilized = decision_scores[0] - np.max(decision_scores[0])
        exp_scores = np.exp(stabilized)
        probabilities = exp_scores / exp_scores.sum()
    else:
        return None

    return pd.DataFrame(
        {"case_category": labels, "probability": probabilities}
    ).sort_values("probability", ascending=False)


def _admin_rule_scores(facts_summary: str) -> tuple[dict[str, float], dict[str, list[str]]]:
    normalized = _normalize_text(facts_summary)
    scores: dict[str, float] = {}
    matches: dict[str, list[str]] = {}

    for category, groups in ADMIN_CATEGORY_RULES.items():
        group_scores = []
        category_matches: list[str] = []
        for keywords in groups.values():
            group_matches = [keyword for keyword in keywords if keyword in normalized]
            group_scores.append(len(group_matches) / max(len(keywords), 1))
            category_matches.extend(group_matches)

        max_group_score = max(group_scores) if group_scores else 0.0
        avg_group_score = sum(group_scores) / len(group_scores) if group_scores else 0.0
        scores[category] = round((0.65 * max_group_score) + (0.35 * avg_group_score), 4)
        matches[category] = list(dict.fromkeys(category_matches))

    return scores, matches


def _triage_case(model, facts_summary: str) -> dict[str, object]:
    model_prediction = model.predict([facts_summary])[0]
    probability_df = _base_probability_df(model, facts_summary)

    base_scores = {
        category: 0.0 for category in CATEGORY_LABELS
    }
    if probability_df is not None:
        for row in probability_df.itertuples(index=False):
            base_scores[str(row.case_category)] = float(row.probability)
    else:
        base_scores[model_prediction] = 1.0

    rule_scores, rule_matches = _admin_rule_scores(facts_summary)
    combined_scores = {}
    for category in CATEGORY_LABELS:
        combined_scores[category] = (0.55 * base_scores.get(category, 0.0)) + (
            0.45 * rule_scores.get(category, 0.0)
        )

    total_score = sum(combined_scores.values())
    if total_score > 0:
        normalized_scores = {
            category: score / total_score for category, score in combined_scores.items()
        }
    else:
        normalized_scores = combined_scores

    blended_df = (
        pd.DataFrame(
            {
                "case_category": list(normalized_scores.keys()),
                "probability": list(normalized_scores.values()),
            }
        )
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )

    final_category = str(blended_df.iloc[0]["case_category"])
    final_confidence = float(blended_df.iloc[0]["probability"])
    second_confidence = (
        float(blended_df.iloc[1]["probability"]) if len(blended_df) > 1 else 0.0
    )

    return {
        "model_prediction": model_prediction,
        "final_category": final_category,
        "probability_df": blended_df,
        "base_probability_df": probability_df,
        "rule_scores": rule_scores,
        "rule_matches": rule_matches,
        "final_confidence": final_confidence,
        "final_margin": final_confidence - second_confidence,
    }


def _load_dataset() -> Optional[pd.DataFrame]:
    if not PROCESSED_DATA_FILE.exists():
        return None
    return pd.read_csv(PROCESSED_DATA_FILE)


def _load_metrics() -> Optional[pd.DataFrame]:
    if not MODEL_METRICS_FILE.exists():
        return None
    return pd.read_csv(MODEL_METRICS_FILE)


def _load_robustness_metrics() -> Optional[pd.DataFrame]:
    if not ROBUSTNESS_METRICS_FILE.exists():
        return None
    return pd.read_csv(ROBUSTNESS_METRICS_FILE)


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
    metrics_df = _load_metrics()
    if metrics_df is not None and not metrics_df.empty and "f1_macro" in metrics_df.columns:
        ranked = metrics_df.sort_values("f1_macro", ascending=False)
        for _, row in ranked.iterrows():
            model_key = row.get("model_key")
            model_config = MODELS.get(model_key)
            if model_config and model_config["path"].exists():
                return {**model_config, "key": model_key}, load_model(model_config["path"])

    for model_key in (
        "hybrid_log_reg_legal",
        "char_svm_legal",
        "lsa_log_reg_legal",
        "linear_svm_legal",
        "log_reg_legal",
    ):
        model_config = MODELS.get(model_key)
        if model_config and model_config["path"].exists():
            return {**model_config, "key": model_key}, load_model(model_config["path"])

    return None, None


def _available_model_options() -> list[dict[str, object]]:
    options = []
    metrics_df = _load_metrics()
    robustness_df = _load_robustness_metrics()
    metric_lookup = {}
    robustness_lookup = {}
    if metrics_df is not None and not metrics_df.empty:
        metric_lookup = metrics_df.set_index("model_key").to_dict(orient="index")
    if robustness_df is not None and not robustness_df.empty:
        robustness_lookup = robustness_df.set_index("model_key").to_dict(orient="index")

    for model_key, model_config in MODELS.items():
        if not model_config["path"].exists():
            continue
        option = {
            "model_key": model_key,
            "label": model_config["name"],
            "config": model_config,
            "f1_macro": metric_lookup.get(model_key, {}).get("f1_macro"),
            "robustness": robustness_lookup.get(model_key, {}).get("accuracy_expected_category"),
        }
        options.append(option)

    options.sort(
        key=lambda item: item["f1_macro"] if item["f1_macro"] is not None else -1,
        reverse=True,
    )
    return options


def _best_model_summary(metrics_df: Optional[pd.DataFrame]) -> tuple[str, str]:
    if metrics_df is None or metrics_df.empty:
        return "Aucun modele", "0%"

    ranked = metrics_df.sort_values("f1_macro", ascending=False).iloc[0]
    return str(ranked["model_name"]), f"{ranked['f1_macro'] * 100:.1f}%"


def _best_robust_model_summary(
    robustness_df: Optional[pd.DataFrame],
) -> tuple[str, str]:
    if robustness_df is None or robustness_df.empty:
        return "Aucun modele", "0%"

    ranked = robustness_df.sort_values(
        "accuracy_expected_category", ascending=False
    ).iloc[0]
    return (
        str(ranked["model_name"]),
        f"{ranked['accuracy_expected_category'] * 100:.1f}%",
    )


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


def _build_metrics_chart(metrics_df: Optional[pd.DataFrame]) -> go.Figure:
    figure = go.Figure()

    if metrics_df is None or metrics_df.empty:
        figure.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return figure

    chart_df = metrics_df.copy()
    chart_df["f1_macro"] = chart_df["f1_macro"] * 100
    chart_df = chart_df.sort_values("f1_macro", ascending=True)

    figure.add_trace(
        go.Bar(
            x=chart_df["f1_macro"],
            y=chart_df["model_name"],
            orientation="h",
            marker=dict(
                color=["rgba(15, 23, 42, 0.82)", "rgba(107, 114, 128, 0.58)"][: len(chart_df)],
                line=dict(color="rgba(255,255,255,0.85)", width=1.2),
            ),
            hovertemplate="%{y}<br>F1 macro: %{x:.1f}%<extra></extra>",
        )
    )

    figure.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.38,
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="rgba(15, 23, 42, 0.08)",
            zeroline=False,
            ticksuffix="%",
            color="#667085",
        ),
        yaxis=dict(title="", showgrid=False, color="#0f172a"),
        showlegend=False,
        font=dict(
            family="SF Pro Display, SF Pro Text, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
            color="#0f172a",
        ),
    )
    return figure


def _build_category_chart(dataset_df: Optional[pd.DataFrame]) -> go.Figure:
    figure = go.Figure()

    if dataset_df is None or dataset_df.empty:
        figure.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return figure

    chart_df = (
        dataset_df[TARGET_COLUMN]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )
    chart_df["label"] = chart_df["category"].map(_humanize_category)
    chart_df = chart_df.sort_values("count", ascending=True)

    colors = ["rgba(15, 23, 42, 0.90)", "rgba(99, 102, 241, 0.48)", "rgba(203, 213, 225, 0.95)"]

    figure.add_trace(
        go.Bar(
            x=chart_df["count"],
            y=chart_df["label"],
            orientation="h",
            marker=dict(
                color=colors[: len(chart_df)],
                line=dict(color="rgba(255,255,255,0.85)", width=1.2),
            ),
            hovertemplate="%{y}<br>%{x} decisions<extra></extra>",
        )
    )

    figure.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.34,
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="rgba(15, 23, 42, 0.08)",
            zeroline=False,
            color="#667085",
        ),
        yaxis=dict(title="", showgrid=False, color="#0f172a"),
        showlegend=False,
        font=dict(
            family="SF Pro Display, SF Pro Text, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
            color="#0f172a",
        ),
    )
    return figure


def _predict_case(model, facts_summary: str) -> tuple[str, Optional[pd.DataFrame]]:
    triage = _triage_case(model, facts_summary)
    return str(triage["final_category"]), triage["probability_df"]


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
    display_df["user_meaning"] = display_df[OUTCOME_COLUMN].map(OUTCOME_USER_MEANINGS)
    return display_df.rename(
        columns={
            "outcome_label": "Lecture metier",
            "winner_label": "Qui est le plus souvent avantage",
            "user_meaning": "Si vous etes le requerant, cela signifie",
            "share": "Part observee (%)",
        }
    )[
        [
            "Lecture metier",
            "Qui est le plus souvent avantage",
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

    vectorizer = _get_similarity_vectorizer(model)
    if vectorizer is None or not hasattr(vectorizer, "transform"):
        return None

    dataset_vectors = vectorizer.transform(dataset_df[TEXT_COLUMN].astype(str))
    query_vector = vectorizer.transform([facts_summary])
    similarities = cosine_similarity(query_vector, dataset_vectors).ravel()

    similar_df = dataset_df.copy()
    similar_df["similarity"] = similarities
    similar_df = similar_df.sort_values("similarity", ascending=False).head(top_n)

    if reference_df is not None and "source_file" in similar_df.columns:
        similar_df = similar_df.merge(
            reference_df,
            on="source_file",
            how="left",
            suffixes=("", "_ref"),
        )

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

    for column in ("numero_ecli", "numero_dossier", "nom_juridiction"):
        if column not in similar_df.columns:
            similar_df[column] = ""

    return similar_df.rename(
        columns={
            "numero_ecli": "ECLI",
            "numero_dossier": "Numero dossier",
            "date_lecture": "Date",
            "nom_juridiction": "Juridiction",
            "categorie_lisible": "Categorie",
            "solution": "Issue observee",
            "similarity": "Proximite textuelle (%)",
            "resume_court": "Extrait de faits proches",
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
    feature_extractor = pipeline_steps.get("tfidf") or pipeline_steps.get("features")
    classifier = pipeline_steps.get("classifier")
    latent_step = pipeline_steps.get("lsa")

    if feature_extractor is None or classifier is None:
        return None

    if not hasattr(feature_extractor, "transform") or not hasattr(
        feature_extractor, "get_feature_names_out"
    ):
        return None

    if not hasattr(classifier, "coef_") or not hasattr(classifier, "classes_"):
        return None

    # For latent semantic pipelines, classifier coefficients live in the
    # reduced latent space rather than directly in the TF-IDF feature space.
    # A token-level contribution table would be misleading here, so we skip it.
    if latent_step is not None:
        return None

    class_labels = list(classifier.classes_)
    if predicted_category not in class_labels:
        return None

    row = feature_extractor.transform([facts_summary])
    class_index = class_labels.index(predicted_category)
    contributions = row.multiply(classifier.coef_[class_index]).tocsr()
    feature_names = feature_extractor.get_feature_names_out()

    evidence = []
    for feature_index, contribution in zip(contributions.indices, contributions.data):
        contribution_value = float(contribution)
        if contribution_value <= 0:
            continue
        feature_name = str(feature_names[feature_index])
        feature_name = feature_name.replace("word_tfidf__", "")
        feature_name = feature_name.replace("signals__", "")
        feature_name = feature_name.replace("signal_", "Signal metier : ")
        evidence.append(
            {
                "Terme repere dans le texte": feature_name,
                "Contribution": round(contribution_value, 4),
            }
        )

    if not evidence:
        return None

    evidence_df = pd.DataFrame(evidence).sort_values("Contribution", ascending=False)
    return evidence_df.head(top_n)


def _model_reading_copy(model_key: str) -> tuple[str, str]:
    if model_key == "hybrid_log_reg_legal":
        return (
            "Probabilites, termes du texte et signaux metier administratifs qui ont influence la sortie actuelle.",
            "Le classifieur combine toujours un signal lexical avec quelques indices metier explicites "
            "comme annulation, fiscalite, indemnisation ou urgence. Il reste sensible a la formulation "
            "et ne remplace pas une qualification juridique humaine.",
        )

    return (
        "Probabilites et indices lexicaux qui ont influence la sortie actuelle.",
        "Le classifieur actuel reste lexical. Il repere des mots et groupes de mots, "
        "pas une comprehension semantique profonde. Une reformulation peut donc faire varier la sortie.",
    )


def _render_sidebar(
    dataset_df: Optional[pd.DataFrame], metrics_df: Optional[pd.DataFrame]
) -> tuple[str, Optional[str]]:
    with st.sidebar:
        robustness_df = _load_robustness_metrics()
        st.markdown(
            """
            <div style="margin-bottom: 1.4rem;">
                <div class="eyebrow">Legal ML</div>
                <h3 style="margin: 0.85rem 0 0.35rem 0; font-size: 1.4rem; letter-spacing: -0.03em;">
                    Control Center
                </h3>
                <p style="margin: 0; color: #667085; line-height: 1.55;">
                    MVP premium pour l'orientation de dossiers administratifs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section = st.radio(
            "Navigation",
            ["Vue d'ensemble", "Case Studio", "Corpus & Jurisprudence"],
            label_visibility="collapsed",
        )

        model_options = _available_model_options()
        selected_model_key = None
        if model_options:
            option_labels = []
            for option in model_options:
                f1_score = option["f1_macro"]
                robustness_score = option["robustness"]
                if f1_score is None and robustness_score is None:
                    option_labels.append(option["label"])
                else:
                    parts = [option["label"]]
                    if f1_score is not None:
                        parts.append(f"F1 {f1_score * 100:.1f}%")
                    if robustness_score is not None:
                        parts.append(f"Robustesse {robustness_score * 100:.1f}%")
                    option_labels.append("  •  ".join(parts))

            default_index = 0
            robust_candidates = [
                idx for idx, option in enumerate(model_options) if option["robustness"] is not None
            ]
            if robust_candidates:
                default_index = max(
                    robust_candidates,
                    key=lambda idx: model_options[idx]["robustness"],
                )

            selected_label = st.selectbox(
                "Modele actif",
                option_labels,
                index=default_index,
            )
            selected_model_key = model_options[option_labels.index(selected_label)]["model_key"]

        st.markdown(
            """
            <div class="callout callout-strong" style="margin-top: 1rem;">
                <p class="callout-title">Perimetre</p>
                <p class="callout-copy">
                    Cette interface est reservee au droit administratif. Les cas
                    de travail, penal, famille ou contrats peuvent etre mal interpretes.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if dataset_df is not None:
            st.markdown(
                f"""
                <div style="margin-top: 1rem;">
                    <span class="micro-chip">{len(dataset_df)} decisions chargees</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if metrics_df is not None and not metrics_df.empty:
            model_name, best_score = _best_model_summary(metrics_df)
            robust_model_name, robust_score = _best_robust_model_summary(robustness_df)
            st.markdown(
                f"""
                <div style="margin-top: 0.85rem;">
                    <span class="micro-chip">Best F1: {best_score}</span>
                </div>
                <div style="margin-top: 0.55rem;">
                    <span class="micro-chip">Best robustness: {robust_score}</span>
                </div>
                <p style="margin-top: 0.65rem; color: #475467; font-size: 0.9rem; line-height: 1.55;">
                    Leader benchmark : {model_name}<br/>
                    Leader reformulations : {robust_model_name}
                </p>
            """,
            unsafe_allow_html=True,
        )

    return section, selected_model_key


def _render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="eyebrow">Administrative Law Triage</div>
            <h1 class="hero-title">{PROJECT_TITLE}</h1>
            <p class="hero-subtitle">
                {PROJECT_SUBTITLE}. Une interface premium pour lire le perimetre
                du dossier, estimer sa famille de contentieux et rapprocher des
                jurisprudences administratives voisines.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_kpis(
    dataset_df: Optional[pd.DataFrame],
    metrics_df: Optional[pd.DataFrame],
    robustness_df: Optional[pd.DataFrame],
    model_config,
) -> None:
    model_name, best_score = _best_model_summary(metrics_df)
    robust_name, robust_score = _best_robust_model_summary(robustness_df)
    categories_count = dataset_df[TARGET_COLUMN].nunique() if dataset_df is not None else 0
    cases_count = len(dataset_df) if dataset_df is not None else 0
    active_model = model_config["name"] if model_config is not None else "Aucun"

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    with col1:
        st.markdown(
            _metric_card(
                "Corpus actif",
                f"{cases_count}",
                "Decisions administratives nettoyees et exploitables.",
            ),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            _metric_card(
                "Familles de recours",
                f"{categories_count}",
                "Le MVP reste volontairement compact pour garder un signal stable.",
            ),
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            _metric_card(
                "Meilleur F1 macro",
                best_score,
                f"Modele leader actuel : {model_name}.",
            ),
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            _metric_card(
                "Robustesse paraphrases",
                robust_score,
                f"Leader reformulations : {robust_name}. Modele actif : {active_model}.",
            ),
            unsafe_allow_html=True,
        )


def _render_overview(
    dataset_df: Optional[pd.DataFrame], metrics_df: Optional[pd.DataFrame]
) -> None:
    left, right = st.columns((1.15, 1), gap="large")

    with left:
        _glass_open()
        _section_header(
            "Distribution du corpus",
            "Les trois familles de recours du MVP servent de colonne vertebrale a l'orientation.",
        )
        st.plotly_chart(
            _build_category_chart(dataset_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        _glass_close()

    with right:
        _glass_open()
        _section_header(
            "Performance modele",
            "Lecture rapide du niveau de fiabilite actuel pour comparer les baselines.",
        )
        st.plotly_chart(
            _build_metrics_chart(metrics_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        _glass_close()

    st.markdown("<div style='height: 1.15rem;'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns((1, 1), gap="large")
    with col1:
        _glass_open()
        _section_header(
            "Ce que fait vraiment le produit",
            "Positionnement volontairement simple, assume et defendable en soutenance.",
        )
        st.markdown(
            """
            <div class="footer-note">
                <p>
                    L'app ne remplace pas l'analyse d'un avocat. Elle sert a pre-qualifier
                    un dossier administratif, a verifier s'il reste dans le bon perimetre
                    juridique et a recuperer des references de jurisprudence proches du corpus.
                </p>
                <p>
                    L'approche actuelle repose sur un modele lexical. Elle est utile pour un
                    MVP de triage, mais pas pour trancher seule un dossier complexe.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _glass_close()

    with col2:
        _glass_open()
        _section_header(
            "Source officielle",
            "Le corpus provient de l'open data de la justice administrative.",
        )
        st.markdown(
            """
            <div class="footer-note">
                <p><strong>Source principale :</strong> opendata.justice-administrative.fr</p>
                <p>
                    Le MVP tourne aujourd'hui sur un lot Conseil d'Etat, mais la
                    structure retenue est compatible avec des extensions futures vers
                    les CAA et les tribunaux administratifs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _glass_close()


def _render_case_studio(
    dataset_df: Optional[pd.DataFrame],
    reference_df: Optional[pd.DataFrame],
    model_config,
    model,
) -> None:
    _glass_open()
    _section_header(
        "Case Studio",
        "Analyse un dossier, teste le perimetre juridique et rapproche des jurisprudences proches.",
    )
    st.markdown(
        """
        <div class="callout callout-danger" style="margin-bottom: 1rem;">
            <p class="callout-title">Droit administratif uniquement</p>
            <p class="callout-copy">
                Saisis de preference un cas de titre de sejour, decision de prefecture,
                permis, refus d'autorisation, contentieux fiscal, marche public ou acte
                administratif contestable.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("case_studio_form", clear_on_submit=False):
        facts_summary = st.text_area(
            "Resume des faits",
            placeholder=(
                "Exemple : Une personne conteste un refus de titre de sejour "
                "oppose par la prefecture apres plusieurs demandes."
            ),
            height=170,
        )
        submitted = st.form_submit_button("Analyser le dossier")

    if not submitted:
        st.markdown(
            """
            <div class="footer-note" style="margin-top: 0.4rem;">
                Lance une analyse pour obtenir un cadrage de domaine, une orientation
                vers le bon specialiste et un rapprochement avec des decisions du corpus.
            </div>
            """,
            unsafe_allow_html=True,
        )
        _glass_close()
        return

    if not facts_summary.strip():
        st.warning("Saisis d'abord un resume de faits.")
        _glass_close()
        return

    if dataset_df is None or model is None or model_config is None:
        st.error(
            "Le dataset ou le modele manque. Recharge le projet puis relance les modeles."
        )
        _glass_close()
        return

    triage_result = _triage_case(model, facts_summary)
    predicted_category = str(triage_result["final_category"])
    probability_df = triage_result["probability_df"]
    domain_signal = _detect_legal_domain(facts_summary)
    specialist_guidance = _specialist_guidance(predicted_category)
    top_domain = str(domain_signal["top_domain"])
    is_confident = bool(domain_signal["is_confident"])
    matched_terms = ", ".join(domain_signal["matched_terms"][top_domain][:6]) or "aucun terme fort"
    rule_matches = triage_result["rule_matches"].get(predicted_category, [])

    if probability_df is not None:
        top_probability = float(triage_result["final_confidence"])
        confidence_display = f"{top_probability * 100:.1f}%"
    else:
        top_probability = 0.0
        confidence_display = "n/a"
    final_margin = float(triage_result["final_margin"])
    rule_display = ", ".join(rule_matches[:5]) if rule_matches else "pas de signal metier fort"

    if is_confident and top_domain != "administratif":
        specialist_title = DOMAIN_SPECIALISTS[top_domain]
        specialist_copy = (
            "Le texte ressemble davantage a un dossier hors perimetre administratif. "
            "Cette orientation doit primer sur la prediction du classifieur administratif."
        )
    elif top_probability < 0.5 or final_margin < 0.1:
        generic_guidance = _generic_admin_specialist()
        specialist_title = generic_guidance["specialist"]
        specialist_copy = generic_guidance["orientation"]
    else:
        specialist_title = specialist_guidance["specialist"]
        specialist_copy = specialist_guidance["orientation"]

    scope_value = DOMAIN_LABELS.get(top_domain, top_domain.title())
    scope_copy = (
        f"Indices reperes : {matched_terms}."
        if matched_terms
        else "Peu d'indices de domaine clairement distinctifs."
    )
    reading_subtitle, transparency_copy = _model_reading_copy(
        model_config.get("key", "")
    )

    kpi_cols = st.columns(4, gap="medium")
    with kpi_cols[0]:
        st.markdown(
            _signal_card(
                "Categorie de triage",
                _humanize_category(predicted_category),
                f"Code interne : {predicted_category} | Regles reperees : {rule_display}",
            ),
            unsafe_allow_html=True,
        )
    with kpi_cols[1]:
        st.markdown(
            _signal_card(
                "Perimetre detecte",
                scope_value,
                scope_copy,
            ),
            unsafe_allow_html=True,
        )
    with kpi_cols[2]:
        st.markdown(
            _signal_card(
                "Confiance",
                confidence_display,
                "Score combine modele + regles metier, puis normalise pour le triage.",
            ),
            unsafe_allow_html=True,
        )
    with kpi_cols[3]:
        st.markdown(
            _signal_card(
                "Specialiste conseille",
                specialist_title,
                specialist_copy,
            ),
            unsafe_allow_html=True,
        )

    if is_confident and top_domain != "administratif":
        st.markdown(
            """
            <div class="callout callout-danger" style="margin-top: 1rem;">
                <p class="callout-title">Hors perimetre probable</p>
                <p class="callout-copy">
                    Le texte semble davantage relever d'une autre branche du droit que
                    du contentieux administratif. Le modele actuel ne doit pas etre lu
                    comme une orientation finale fiable dans ce cas.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif top_probability < 0.5 or final_margin < 0.1:
        st.markdown(
            """
            <div class="callout callout-strong" style="margin-top: 1rem;">
                <p class="callout-title">Confiance moderee</p>
                <p class="callout-copy">
                    Le triage hesite encore. Utilise la sortie comme un signal de triage,
                    pas comme une conclusion juridique. L'orientation specialistique reste
                    volontairement prudente dans cette zone.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="callout callout-ok" style="margin-top: 1rem;">
                <p class="callout-title">Signal compatible avec le corpus</p>
                <p class="callout-copy">
                    Les indices textuels semblent rester dans le perimetre du MVP.
                    L'analyse ci-dessous peut donc etre lue comme une aide de triage plus cohérente.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    analysis_left, analysis_right = st.columns((1, 1), gap="large")

    with analysis_left:
        _glass_open()
        _section_header(
            "Lecture du modele",
            reading_subtitle,
        )
        if probability_df is not None:
            st.dataframe(
                _format_probability_df(probability_df.head(3)),
                width="stretch",
                hide_index=True,
            )

        st.markdown(
            f"""
            <div class="callout callout-strong" style="margin-top: 1rem; margin-bottom: 1rem;">
                <p class="callout-title">Transparence sur la limite du modele</p>
                <p class="callout-copy">
                    {transparency_copy}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        evidence_df = _top_linear_evidence(model, facts_summary, predicted_category)
        if evidence_df is not None:
            st.dataframe(evidence_df, width="stretch", hide_index=True)
        else:
            st.info("Aucune evidence lexicale exploitable n'a pu etre extraite.")
        _glass_close()

    with analysis_right:
        _glass_open()
        _section_header(
            "Lecture contentieuse",
            "Qui semble avantagé dans les cas proches et comment lire l'issue.",
        )
        outcome_df = _empirical_outcomes(dataset_df, predicted_category)
        if outcome_df is not None:
            st.markdown(
                """
                <div class="footer-note" style="margin-bottom: 0.9rem;">
                    Dans cette lecture, le requerant est la personne ou l'entite qui
                    saisit le juge administratif. Une issue defavorable au requerant
                    signifie donc que le requerant perd et que la partie defenderesse
                    est avantagée.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                _format_outcome_df(outcome_df),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Pas d'estimation empirique disponible pour cette categorie.")
        _glass_close()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    _glass_open()
    _section_header(
        "Jurisprudences proches",
        "Decisions du corpus dont le resume de faits est le plus proche textuellement de votre saisie.",
    )
    if is_confident and top_domain != "administratif":
        st.warning(
            "Le texte semble hors perimetre administratif. Je n'utilise donc pas la "
            "jurisprudence administrative du corpus comme reference principale."
        )
    else:
        similar_cases_df = _find_similar_jurisprudence(
            model,
            dataset_df,
            reference_df,
            facts_summary,
        )
        if similar_cases_df is not None:
            st.dataframe(similar_cases_df, width="stretch", hide_index=True)
        else:
            st.info("Impossible de calculer un rapprochement fiable avec le corpus.")
    _glass_close()
    _glass_close()


def _render_corpus(
    dataset_df: Optional[pd.DataFrame],
    metrics_df: Optional[pd.DataFrame],
    robustness_df: Optional[pd.DataFrame],
) -> None:
    left, right = st.columns((1.1, 0.9), gap="large")

    with left:
        _glass_open()
        _section_header(
            "Apercu du dataset",
            "Vue rapide sur les donnees qui alimentent le moteur de triage.",
        )
        if dataset_df is not None and not dataset_df.empty:
            preview_df = dataset_df.head(8).copy()
            preview_df[TARGET_COLUMN] = preview_df[TARGET_COLUMN].map(_humanize_category)
            st.dataframe(preview_df, width="stretch", hide_index=True)
        else:
            st.info("Dataset indisponible.")
        _glass_close()

    with right:
        _glass_open()
        _section_header(
            "Metriques",
            "Reference compacte pour relire le niveau du modele pendant la demo.",
        )
        if metrics_df is not None and not metrics_df.empty:
            st.dataframe(_format_metrics(metrics_df), width="stretch", hide_index=True)
        else:
            st.info("Aucune metrique disponible.")
        if robustness_df is not None and not robustness_df.empty:
            robustness_display = robustness_df.copy()
            robustness_display["accuracy_expected_category"] = (
                robustness_display["accuracy_expected_category"] * 100
            ).round(1)
            robustness_display["paraphrase_consistency"] = (
                robustness_display["paraphrase_consistency"] * 100
            ).round(1)
            if "mean_top_probability" in robustness_display.columns:
                robustness_display["mean_top_probability"] = (
                    pd.to_numeric(
                        robustness_display["mean_top_probability"], errors="coerce"
                    )
                    * 100
                ).round(1)
            st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="footer-note" style="margin-bottom: 0.65rem;">
                    Lecture supplementaire : tenue du modele sur un mini-benchmark de reformulations.
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                robustness_display.rename(
                    columns={
                        "model_name": "Modele",
                        "accuracy_expected_category": "Exactitude benchmark (%)",
                        "paraphrase_consistency": "Stabilite reformulations (%)",
                        "mean_top_probability": "Confiance moyenne (%)",
                    }
                )[
                    [
                        "Modele",
                        "Exactitude benchmark (%)",
                        "Stabilite reformulations (%)",
                        "Confiance moyenne (%)",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        _glass_close()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    _glass_open()
    _section_header(
        "Pourquoi seulement 3 categories ?",
        "Choix methodologique volontaire pour un MVP stable, lisible et defendable.",
    )
    st.markdown(
        """
        <div class="footer-note">
            <p>
                Le corpus actuel vient surtout d'un lot Conseil d'Etat en droit administratif.
                Les labels du MVP derivent donc de trois grandes familles procedurales :
                recours pour exces de pouvoir, plein contentieux et autres recours administratifs.
            </p>
            <p>
                Ce n'est pas une cartographie complete de tous les specialistes du droit francais.
                C'est une premiere couche de triage administatif, volontairement restreinte pour
                conserver un produit plus honnete, plus lisible et plus fiable.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _glass_close()


def build_app() -> None:
    st.set_page_config(
        page_title=PROJECT_TITLE,
        page_icon=".",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    _inject_css()

    dataset_df = _load_dataset()
    reference_df = _load_reference_metadata()
    metrics_df = _load_metrics()
    robustness_df = _load_robustness_metrics()
    model_config, model = _load_demo_model()

    section, selected_model_key = _render_sidebar(dataset_df, metrics_df)
    if selected_model_key is not None:
        selected_config = MODELS.get(selected_model_key)
        if selected_config and selected_config["path"].exists():
            model_config = {**selected_config, "key": selected_model_key}
            model = load_model(selected_config["path"])

    _render_hero()
    _render_kpis(dataset_df, metrics_df, robustness_df, model_config)
    st.markdown("<div style='height: 1.1rem;'></div>", unsafe_allow_html=True)

    if section == "Vue d'ensemble":
        _render_overview(dataset_df, metrics_df)
    elif section == "Case Studio":
        _render_case_studio(dataset_df, reference_df, model_config, model)
    else:
        _render_corpus(dataset_df, metrics_df, robustness_df)


if __name__ == "__main__":
    build_app()
