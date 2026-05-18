"""Premium Streamlit dashboard for the legal case triage demo."""

from __future__ import annotations

import html
import math
import re
import sys
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

try:
    from wordcloud import STOPWORDS as WORDCLOUD_STOPWORDS
    from wordcloud import WordCloud
except ImportError:  # pragma: no cover - graceful fallback for local envs
    WORDCLOUD_STOPWORDS = set()
    WordCloud = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import (
    CLASS_METRICS_FILE,
    CONFUSION_MATRIX_FILE,
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

ANNOTATION_COLORS = {
    "exces_de_pouvoir": "rgba(244, 114, 182, 0.22)",
    "plein_contentieux": "rgba(59, 130, 246, 0.20)",
    "autres_recours": "rgba(148, 163, 184, 0.26)",
    "travail": "rgba(249, 115, 22, 0.22)",
    "penal": "rgba(239, 68, 68, 0.20)",
    "famille": "rgba(16, 185, 129, 0.20)",
    "administratif": "rgba(99, 102, 241, 0.18)",
}

ADMIN_GROUP_EXPLANATIONS = {
    ("exces_de_pouvoir", "decision"): (
        "Signal d'acte administratif contesté",
        "Ce terme renforce l'hypothèse d'un recours pour exces de pouvoir contre une decision administrative.",
    ),
    ("exces_de_pouvoir", "autority"): (
        "Signal d'autorité publique",
        "La presence d'une autorite administrative pousse le dossier vers le contentieux administratif de legalite.",
    ),
    ("exces_de_pouvoir", "immigration"): (
        "Signal d'immigration",
        "Le vocabulaire du sejour, du visa ou de l'asile renforce souvent les recours contre des refus ou decisions prefectorales.",
    ),
    ("plein_contentieux", "money"): (
        "Signal financier / indemnitaire",
        "Ce terme suggere une demande de somme, de condamnation ou de remboursement, typique du plein contentieux.",
    ),
    ("plein_contentieux", "harm"): (
        "Signal de responsabilite",
        "Le vocabulaire du prejudice ou du dommage oriente vers une logique indemnitaire de plein contentieux.",
    ),
    ("plein_contentieux", "fiscal_social"): (
        "Signal fiscal ou social",
        "Ce terme renforce les litiges ou l'on discute une charge, une cotisation, une decharge ou une prestation.",
    ),
    ("autres_recours", "procedure"): (
        "Signal procedural",
        "Ce terme correspond davantage a un recours technique ou procedural qu'a une contestation classique de legalite ou d'indemnisation.",
    ),
}

DOMAIN_EXPLANATIONS = {
    "travail": "Signal hors perimetre administratif : vocabulaire typique d'un litige de travail.",
    "penal": "Signal hors perimetre administratif : vocabulaire associe a une procedure ou qualification penale.",
    "famille": "Signal hors perimetre administratif : vocabulaire lie au droit de la famille.",
    "administratif": "Signal de contexte administratif general.",
}

FRENCH_WORDCLOUD_STOPWORDS = {
    "alors",
    "apres",
    "ainsi",
    "au",
    "aux",
    "avec",
    "avoir",
    "ce",
    "ces",
    "cet",
    "cette",
    "comme",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "elles",
    "en",
    "entre",
    "est",
    "et",
    "etre",
    "fait",
    "faits",
    "il",
    "ils",
    "la",
    "le",
    "les",
    "leur",
    "mais",
    "meme",
    "ne",
    "ni",
    "nous",
    "ou",
    "par",
    "pas",
    "plus",
    "pour",
    "que",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sur",
    "une",
    "un",
    "vos",
    "votre",
    "requete",
    "requérant",
    "requerant",
    "demande",
    "demander",
    "decision",
    "administrative",
}

WORDCLOUD_VISUAL_BLACKLIST = {
    "janvier",
    "fevrier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "aout",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "decembre",
    "décembre",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
    "2025",
    "2026",
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

            .annotated-sentence {
                margin-top: 1rem;
                padding: 1rem 1.05rem;
                border-radius: 18px;
                border: 1px solid rgba(255, 255, 255, 0.58);
                background: rgba(255, 255, 255, 0.55);
                color: var(--text);
                line-height: 1.9;
                font-size: 1rem;
            }

            .annotated-token {
                position: relative;
                display: inline;
                padding: 0.08rem 0.28rem;
                border-radius: 0.55rem;
                border: 1px solid rgba(15, 23, 42, 0.08);
                cursor: help;
                box-decoration-break: clone;
                -webkit-box-decoration-break: clone;
                transition: filter 140ms ease, transform 140ms ease;
            }

            .annotated-token:hover {
                filter: brightness(0.98);
            }

            .annotated-token .token-tooltip {
                visibility: hidden;
                opacity: 0;
                position: absolute;
                left: 50%;
                bottom: calc(100% + 10px);
                transform: translateX(-50%);
                min-width: 240px;
                max-width: 320px;
                padding: 0.72rem 0.82rem;
                border-radius: 14px;
                background: rgba(15, 23, 42, 0.96);
                color: #f8fafc;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.22);
                font-size: 0.83rem;
                line-height: 1.45;
                z-index: 30;
                pointer-events: none;
                transition: opacity 160ms ease, visibility 160ms ease;
            }

            .annotated-token:hover .token-tooltip {
                visibility: visible;
                opacity: 1;
            }

            .annotation-exces_de_pouvoir {
                background: rgba(244, 114, 182, 0.22);
            }

            .annotation-plein_contentieux {
                background: rgba(59, 130, 246, 0.20);
            }

            .annotation-autres_recours {
                background: rgba(148, 163, 184, 0.26);
            }

            .annotation-travail {
                background: rgba(249, 115, 22, 0.22);
            }

            .annotation-penal {
                background: rgba(239, 68, 68, 0.20);
            }

            .annotation-famille {
                background: rgba(16, 185, 129, 0.20);
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


def _load_class_metrics() -> Optional[pd.DataFrame]:
    if not CLASS_METRICS_FILE.exists():
        return None
    return pd.read_csv(CLASS_METRICS_FILE)


def _load_confusion_matrix() -> Optional[pd.DataFrame]:
    if not CONFUSION_MATRIX_FILE.exists():
        return None
    return pd.read_csv(CONFUSION_MATRIX_FILE)


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


def _format_class_metrics(
    class_metrics_df: pd.DataFrame, selected_model_key: Optional[str] = None
) -> pd.DataFrame:
    display_df = class_metrics_df.copy()
    if selected_model_key:
        display_df = display_df[display_df["model_key"] == selected_model_key]

    if display_df.empty:
        return display_df

    display_df["case_category"] = display_df["case_category"].map(_humanize_category)
    for column in ("precision", "recall", "f1_score"):
        display_df[column] = (display_df[column] * 100).round(1)

    return display_df.rename(
        columns={
            "case_category": "Classe",
            "precision": "Precision (%)",
            "recall": "Recall (%)",
            "f1_score": "F1 (%)",
            "support": "Support",
        }
    )[
        ["Classe", "Precision (%)", "Recall (%)", "F1 (%)", "Support"]
    ]


def _build_confusion_chart(
    confusion_df: Optional[pd.DataFrame], selected_model_key: Optional[str]
) -> go.Figure:
    figure = go.Figure()
    if confusion_df is None or confusion_df.empty or selected_model_key is None:
        figure.update_layout(
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return figure

    model_df = confusion_df[confusion_df["model_key"] == selected_model_key].copy()
    if model_df.empty:
        figure.update_layout(
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return figure

    category_columns = [
        column
        for column in model_df.columns
        if column not in {"actual_category", "model_key", "model_name"}
    ]
    heatmap_df = model_df.set_index("actual_category")[category_columns]
    heatmap_df.index = [_humanize_category(value) for value in heatmap_df.index]
    heatmap_df.columns = [_humanize_category(value) for value in heatmap_df.columns]

    figure = go.Figure(
        data=[
            go.Heatmap(
                z=heatmap_df.values,
                x=heatmap_df.columns.tolist(),
                y=heatmap_df.index.tolist(),
                colorscale=[
                    [0.0, "rgba(226,232,240,0.35)"],
                    [0.5, "rgba(148,163,184,0.62)"],
                    [1.0, "rgba(15,23,42,0.92)"],
                ],
                text=heatmap_df.values,
                texttemplate="%{text}",
                hovertemplate="Reel: %{y}<br>Predit: %{x}<br>Count: %{z}<extra></extra>",
            )
        ]
    )
    figure.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Prediction", color="#667085"),
        yaxis=dict(title="Reel", color="#667085"),
        font=dict(
            family="SF Pro Display, SF Pro Text, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
            color="#0f172a",
        ),
    )
    return figure


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


def _build_text_length_chart(dataset_df: Optional[pd.DataFrame]) -> go.Figure:
    figure = go.Figure()

    if dataset_df is None or dataset_df.empty or "text_length" not in dataset_df.columns:
        figure.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return figure

    chart_df = dataset_df.copy()
    chart_df["Categorie"] = chart_df[TARGET_COLUMN].map(_humanize_category)

    ordered_labels = [
        _humanize_category(category)
        for category in dataset_df[TARGET_COLUMN].value_counts().index.tolist()
    ]
    colors = {
        "Recours pour exces de pouvoir": "rgba(15, 23, 42, 0.82)",
        "Plein contentieux": "rgba(99, 102, 241, 0.52)",
        "Autres recours administratifs": "rgba(148, 163, 184, 0.88)",
    }

    for label in ordered_labels:
        label_df = chart_df[chart_df["Categorie"] == label]
        figure.add_trace(
            go.Box(
                x=label_df["text_length"],
                name=label,
                marker_color=colors.get(label, "rgba(15, 23, 42, 0.72)"),
                line=dict(width=1.3),
                boxmean=True,
                hovertemplate="%{x} caracteres<extra>" + label + "</extra>",
            )
        )

    figure.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title="Longueur du resume (caracteres)",
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


def _build_outcome_chart(dataset_df: Optional[pd.DataFrame]) -> go.Figure:
    figure = go.Figure()

    if dataset_df is None or dataset_df.empty or OUTCOME_COLUMN not in dataset_df.columns:
        figure.update_layout(
            height=340,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return figure

    chart_df = (
        dataset_df[OUTCOME_COLUMN]
        .value_counts()
        .rename_axis("outcome")
        .reset_index(name="count")
        .head(6)
    )
    chart_df["label"] = chart_df["outcome"].map(_humanize_outcome)
    chart_df = chart_df.sort_values("count", ascending=True)

    figure.add_trace(
        go.Bar(
            x=chart_df["count"],
            y=chart_df["label"],
            orientation="h",
            marker=dict(
                color="rgba(15, 23, 42, 0.86)",
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
            title="Nombre de decisions",
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


def _dataset_profile_table(dataset_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if dataset_df is None or dataset_df.empty:
        return pd.DataFrame()

    date_series = dataset_df["date_lecture"].dropna().astype(str)
    if date_series.empty:
        period_label = "non renseignee"
    else:
        period_label = f"{date_series.min()} -> {date_series.max()}"

    median_length = (
        int(dataset_df["text_length"].median())
        if "text_length" in dataset_df.columns
        else 0
    )
    mean_length = (
        int(dataset_df["text_length"].mean())
        if "text_length" in dataset_df.columns
        else 0
    )

    profile_rows = [
        {
            "Indicateur": "Nombre de decisions",
            "Valeur": f"{len(dataset_df)}",
        },
        {
            "Indicateur": "Longueur mediane du resume",
            "Valeur": f"{median_length:,} caracteres".replace(",", " "),
        },
        {
            "Indicateur": "Longueur moyenne du resume",
            "Valeur": f"{mean_length:,} caracteres".replace(",", " "),
        },
        {
            "Indicateur": "Periode couverte",
            "Valeur": period_label,
        },
        {
            "Indicateur": "Issue la plus frequente",
            "Valeur": _humanize_outcome(
                str(dataset_df[OUTCOME_COLUMN].value_counts().index[0])
            ),
        },
        {
            "Indicateur": "Categorie la plus frequente",
            "Valeur": _humanize_category(
                str(dataset_df[TARGET_COLUMN].value_counts().index[0])
            ),
        },
    ]
    return pd.DataFrame(profile_rows)


def _typical_examples_df(dataset_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if dataset_df is None or dataset_df.empty:
        return pd.DataFrame()

    examples = []
    for category in dataset_df[TARGET_COLUMN].value_counts().index.tolist():
        category_df = dataset_df[dataset_df[TARGET_COLUMN] == category].copy()
        if category_df.empty:
            continue
        category_df["rule_strength"] = category_df[TEXT_COLUMN].astype(str).apply(
            lambda text: _admin_rule_scores(text)[0].get(str(category), 0.0)
        )
        category_df["distance_to_median"] = (
            category_df["text_length"] - category_df["text_length"].median()
        ).abs()
        example_row = category_df.sort_values(
            ["rule_strength", "distance_to_median"],
            ascending=[False, True],
        ).iloc[0]
        examples.append(
            {
                "Categorie": _humanize_category(str(category)),
                "Issue observee": _humanize_outcome(str(example_row[OUTCOME_COLUMN])),
                "Exemple de faits": str(example_row[TEXT_COLUMN])[:260].strip() + "...",
            }
        )

    return pd.DataFrame(examples)


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


def _similar_jurisprudence_matches(
    model,
    dataset_df: Optional[pd.DataFrame],
    reference_df: Optional[pd.DataFrame],
    facts_summary: str,
    top_n: int = 8,
) -> Optional[pd.DataFrame]:
    if dataset_df is None or dataset_df.empty:
        return None

    vectorizer = _get_similarity_vectorizer(model)
    if vectorizer is None or not hasattr(vectorizer, "transform"):
        return None

    dataset_vectors = vectorizer.transform(dataset_df[TEXT_COLUMN].astype(str))
    query_vector = vectorizer.transform([facts_summary])
    similarities = cosine_similarity(query_vector, dataset_vectors).ravel()

    similar_df = dataset_df.reset_index(drop=False).rename(columns={"index": "row_id"}).copy()
    similar_df["similarity_raw"] = similarities
    similar_df = similar_df.sort_values("similarity_raw", ascending=False).head(top_n)

    if reference_df is not None and "source_file" in similar_df.columns:
        similar_df = similar_df.merge(
            reference_df,
            on="source_file",
            how="left",
            suffixes=("", "_ref"),
        )

    similar_df["categorie_lisible"] = similar_df[TARGET_COLUMN].map(_humanize_category)
    similar_df["similarity"] = (similar_df["similarity_raw"] * 100).round(1)
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

    return similar_df


def _format_similar_jurisprudence(similar_df: pd.DataFrame) -> pd.DataFrame:
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


def _build_jurisprudence_network(
    facts_summary: str,
    similar_df: Optional[pd.DataFrame],
) -> go.Figure:
    figure = go.Figure()
    if similar_df is None or similar_df.empty:
        figure.update_layout(
            height=430,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return figure

    chart_df = similar_df.copy().reset_index(drop=True)
    if "similarity_raw" not in chart_df.columns:
        chart_df["similarity_raw"] = chart_df["similarity"] / 100.0

    base_angles = np.linspace(0, 2 * np.pi, len(chart_df), endpoint=False)
    radii = 1.6 - (chart_df["similarity_raw"].clip(0, 1) * 0.8)
    chart_df["x"] = np.cos(base_angles) * radii
    chart_df["y"] = np.sin(base_angles) * radii

    edge_x: list[float] = []
    edge_y: list[float] = []
    for _, row in chart_df.iterrows():
        edge_x.extend([0.0, float(row["x"]), None])
        edge_y.extend([0.0, float(row["y"]), None])

    figure.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="rgba(148,163,184,0.55)", width=1.8),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    if len(chart_df) > 1:
        inter_x: list[float] = []
        inter_y: list[float] = []
        for left_index in range(len(chart_df)):
            for right_index in range(left_index + 1, len(chart_df)):
                left_row = chart_df.iloc[left_index]
                right_row = chart_df.iloc[right_index]
                pair_similarity = 1 - min(
                    abs(float(left_row["similarity_raw"]) - float(right_row["similarity_raw"])),
                    1.0,
                )
                if pair_similarity >= 0.82:
                    inter_x.extend([float(left_row["x"]), float(right_row["x"]), None])
                    inter_y.extend([float(left_row["y"]), float(right_row["y"]), None])

        if inter_x:
            figure.add_trace(
                go.Scatter(
                    x=inter_x,
                    y=inter_y,
                    mode="lines",
                    line=dict(color="rgba(99,102,241,0.18)", width=1.1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    node_colors = [
        ANNOTATION_COLORS.get(str(category), "rgba(15,23,42,0.22)")
        for category in chart_df[TARGET_COLUMN].tolist()
    ]
    node_sizes = (chart_df["similarity_raw"] * 28).clip(lower=16).tolist()
    node_text = []
    for row in chart_df.itertuples(index=False):
        node_text.append(
            "<br>".join(
                [
                    f"<b>{_humanize_category(str(getattr(row, TARGET_COLUMN)))}</b>",
                    f"Proximite textuelle : {float(row.similarity):.1f}%",
                    f"Juridiction : {getattr(row, 'nom_juridiction', '') or 'n/r'}",
                    f"ECLI : {getattr(row, 'numero_ecli', '') or 'n/r'}",
                    f"Extrait : {html.escape(str(row.resume_court))}",
                ]
            )
        )

    figure.add_trace(
        go.Scatter(
            x=chart_df["x"],
            y=chart_df["y"],
            mode="markers+text",
            text=[f"J{i + 1}" for i in range(len(chart_df))],
            textposition="middle center",
            textfont=dict(color="#0f172a", size=11),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(color="rgba(15,23,42,0.18)", width=1.6),
            ),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=node_text,
            name="Jurisprudences",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[0.0],
            y=[0.0],
            mode="markers+text",
            text=["Cas"],
            textposition="middle center",
            textfont=dict(color="#ffffff", size=12),
            marker=dict(
                size=34,
                color="rgba(15,23,42,0.94)",
                line=dict(color="rgba(255,255,255,0.92)", width=2),
            ),
            hovertemplate=(
                "<b>Votre cas</b><br>"
                + html.escape(facts_summary[:220].strip())
                + "<extra></extra>"
            ),
            name="Cas utilisateur",
        )
    )

    figure.update_layout(
        height=430,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        font=dict(
            family="SF Pro Display, SF Pro Text, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
            color="#0f172a",
        ),
    )
    return figure


def _linear_evidence_items(
    model, facts_summary: str, predicted_category: str, top_n: int = 6
) -> list[dict[str, object]]:
    pipeline_steps = getattr(model, "named_steps", {})
    feature_extractor = pipeline_steps.get("tfidf") or pipeline_steps.get("features")
    classifier = pipeline_steps.get("classifier")
    latent_step = pipeline_steps.get("lsa")

    if feature_extractor is None or classifier is None:
        return []

    if not hasattr(feature_extractor, "transform") or not hasattr(
        feature_extractor, "get_feature_names_out"
    ):
        return []

    if not hasattr(classifier, "coef_") or not hasattr(classifier, "classes_"):
        return []

    # For latent semantic pipelines, classifier coefficients live in the
    # reduced latent space rather than directly in the TF-IDF feature space.
    # A token-level contribution table would be misleading here, so we skip it.
    if latent_step is not None:
        return []

    class_labels = list(classifier.classes_)
    if predicted_category not in class_labels:
        return []

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
        display_name = feature_name.replace("word_tfidf__", "")
        display_name = display_name.replace("signals__", "")
        display_name = display_name.replace("signal_", "Signal metier : ")
        evidence.append(
            {
                "feature_name": feature_name,
                "display_name": display_name,
                "Contribution": round(contribution_value, 4),
            }
        )

    return sorted(evidence, key=lambda item: item["Contribution"], reverse=True)[:top_n]


def _top_linear_evidence(
    model, facts_summary: str, predicted_category: str, top_n: int = 6
) -> Optional[pd.DataFrame]:
    evidence = _linear_evidence_items(model, facts_summary, predicted_category, top_n=top_n)
    if not evidence:
        return None

    evidence_df = pd.DataFrame(
        [
            {
                "Terme repere dans le texte": item["display_name"],
                "Contribution": item["Contribution"],
            }
            for item in evidence
        ]
    )
    return evidence_df


def _normalized_text_with_map(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for index, character in enumerate(text):
        normalized_character = _normalize_text(character)
        for normalized_unit in normalized_character:
            normalized_chars.append(normalized_unit)
            index_map.append(index)
    return "".join(normalized_chars), index_map


def _find_keyword_spans(text: str, keyword: str) -> list[tuple[int, int]]:
    normalized_text, index_map = _normalized_text_with_map(text)
    normalized_keyword = _normalize_text(keyword).strip()
    if not normalized_keyword:
        return []

    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        position = normalized_text.find(normalized_keyword, start)
        if position == -1:
            break
        end_position = position + len(normalized_keyword)
        spans.append((index_map[position], index_map[end_position - 1] + 1))
        start = end_position
    return spans


def _is_lexical_term_useful(term: str) -> bool:
    cleaned = term.strip().lower()
    if len(cleaned) <= 2:
        return False
    if cleaned in FRENCH_WORDCLOUD_STOPWORDS:
        return False
    if cleaned.startswith("signal metier"):
        return False
    if re.fullmatch(r"[0-9]+", cleaned):
        return False
    return True


def _build_inline_annotations(
    facts_summary: str,
    predicted_category: str,
    top_domain: str,
    domain_signal: dict[str, object],
    evidence_items: list[dict[str, object]],
) -> list[dict[str, object]]:
    annotation_candidates: list[dict[str, object]] = []

    for group_name, keywords in ADMIN_CATEGORY_RULES.get(predicted_category, {}).items():
        title, explanation = ADMIN_GROUP_EXPLANATIONS.get(
            (predicted_category, group_name),
            (
                "Signal juridique",
                f"Ce terme pousse le dossier vers {_humanize_category(predicted_category).lower()}.",
            ),
        )
        for keyword in keywords:
            for start, end in _find_keyword_spans(facts_summary, keyword):
                annotation_candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "tone": predicted_category,
                        "weight": 5.0 + (end - start),
                        "tooltip": (
                            f"<strong>{html.escape(keyword)}</strong><br>"
                            f"{html.escape(title)}<br>"
                            f"{html.escape(explanation)}"
                        ),
                    }
                )

    if top_domain != "administratif":
        for keyword in domain_signal.get("matched_terms", {}).get(top_domain, []):
            for start, end in _find_keyword_spans(facts_summary, keyword):
                annotation_candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "tone": top_domain,
                        "weight": 4.0 + (end - start),
                        "tooltip": (
                            f"<strong>{html.escape(keyword)}</strong><br>"
                            f"Signal de domaine : {html.escape(DOMAIN_LABELS.get(top_domain, top_domain.title()))}<br>"
                            f"{html.escape(DOMAIN_EXPLANATIONS.get(top_domain, 'Indice de domaine.'))}"
                        ),
                    }
                )

    for item in evidence_items:
        feature_name = str(item.get("feature_name", ""))
        display_name = str(item.get("display_name", ""))
        contribution = float(item.get("Contribution", 0.0))
        if feature_name.startswith("word_tfidf__") and _is_lexical_term_useful(display_name):
            for start, end in _find_keyword_spans(facts_summary, display_name):
                annotation_candidates.append(
                    {
                        "start": start,
                        "end": end,
                        "tone": predicted_category,
                        "weight": 2.0 + contribution,
                        "tooltip": (
                            f"<strong>{html.escape(display_name)}</strong><br>"
                            f"Indice lexical repere par le modele<br>"
                            f"Contribution positive vers {_humanize_category(predicted_category).lower()} "
                            f"({contribution:.3f})."
                        ),
                    }
                )

    selected_annotations: list[dict[str, object]] = []
    occupied_positions: set[int] = set()
    for candidate in sorted(
        annotation_candidates,
        key=lambda item: (-float(item["weight"]), -(int(item["end"]) - int(item["start"]))),
    ):
        span_positions = set(range(int(candidate["start"]), int(candidate["end"])))
        if occupied_positions.intersection(span_positions):
            continue
        selected_annotations.append(candidate)
        occupied_positions.update(span_positions)

    return sorted(selected_annotations, key=lambda item: int(item["start"]))


def _render_annotated_sentence(
    facts_summary: str,
    annotations: list[dict[str, object]],
) -> str:
    if not annotations:
        return (
            '<div class="annotated-sentence">'
            + html.escape(facts_summary)
            + "</div>"
        )

    chunks: list[str] = []
    cursor = 0
    for annotation in annotations:
        start = int(annotation["start"])
        end = int(annotation["end"])
        tone = str(annotation["tone"])
        tooltip = str(annotation["tooltip"])

        chunks.append(html.escape(facts_summary[cursor:start]))
        surface = html.escape(facts_summary[start:end])
        chunks.append(
            f'<span class="annotated-token annotation-{tone}" title="{html.escape(re.sub("<[^>]+>", " ", tooltip))}">'
            f"{surface}<span class=\"token-tooltip\">{tooltip}</span></span>"
        )
        cursor = end

    chunks.append(html.escape(facts_summary[cursor:]))
    return '<div class="annotated-sentence">' + "".join(chunks) + "</div>"


def _build_category_wordcloud(
    dataset_df: Optional[pd.DataFrame], category: str
) -> Optional[plt.Figure]:
    if dataset_df is None or dataset_df.empty or WordCloud is None:
        return None

    category_df = dataset_df[dataset_df[TARGET_COLUMN] == category]
    if category_df.empty:
        return None

    texts = category_df[TEXT_COLUMN].astype(str).tolist()
    vectorizer = None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            min_df=2,
            max_df=0.9,
            stop_words=sorted(FRENCH_WORDCLOUD_STOPWORDS.union(WORDCLOUD_STOPWORDS)),
        )
        matrix = vectorizer.fit_transform(texts)
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        terms = vectorizer.get_feature_names_out()
        frequencies = {
            term: float(score)
            for term, score in zip(terms, scores)
            if float(score) > 0
        }
    except ValueError:
        frequencies = {}

    def _is_visual_wordcloud_term_allowed(term: str) -> bool:
        cleaned = _normalize_text(str(term)).strip()
        if not cleaned:
            return False
        if cleaned in WORDCLOUD_VISUAL_BLACKLIST:
            return False
        if re.fullmatch(r"(19|20)\d{2}", cleaned):
            return False
        if re.fullmatch(r"\d+", cleaned):
            return False
        if re.fullmatch(r"\d+(er|e|eme)", cleaned):
            return False
        if re.fullmatch(r"\d+[a-z]{0,2}", cleaned):
            return False
        return True

    frequencies = {
        term: weight
        for term, weight in frequencies.items()
        if _is_visual_wordcloud_term_allowed(term)
    }

    if not frequencies:
        return None

    cloud = WordCloud(
        width=900,
        height=480,
        background_color="white",
        max_words=70,
        prefer_horizontal=0.9,
        stopwords=FRENCH_WORDCLOUD_STOPWORDS.union(WORDCLOUD_STOPWORDS),
        colormap={
            "exces_de_pouvoir": "RdPu",
            "plein_contentieux": "Blues",
            "autres_recours": "Greys",
        }.get(category, "viridis"),
    ).generate_from_frequencies(frequencies)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.imshow(cloud, interpolation="bilinear")
    axis.axis("off")
    figure.tight_layout(pad=0)
    return figure


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

    evidence_items = _linear_evidence_items(
        model,
        facts_summary,
        predicted_category,
        top_n=8,
    )
    inline_annotations = _build_inline_annotations(
        facts_summary,
        predicted_category,
        top_domain,
        domain_signal,
        evidence_items,
    )

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    _glass_open()
    _section_header(
        "Phrase analysee",
        "Les mots ou expressions les plus influents sont surlignes directement dans la phrase. Survole-les pour comprendre pourquoi ils comptent.",
    )
    st.markdown(
        _render_annotated_sentence(facts_summary, inline_annotations),
        unsafe_allow_html=True,
    )
    _glass_close()

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
        similar_cases_raw = _similar_jurisprudence_matches(
            model,
            dataset_df,
            reference_df,
            facts_summary,
            top_n=8,
        )
        if similar_cases_raw is not None:
            st.plotly_chart(
                _build_jurisprudence_network(facts_summary, similar_cases_raw),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.dataframe(
                _format_similar_jurisprudence(similar_cases_raw),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Impossible de calculer un rapprochement fiable avec le corpus.")
    _glass_close()
    _glass_close()


def _render_corpus(
    dataset_df: Optional[pd.DataFrame],
    metrics_df: Optional[pd.DataFrame],
    robustness_df: Optional[pd.DataFrame],
    class_metrics_df: Optional[pd.DataFrame],
    confusion_df: Optional[pd.DataFrame],
    selected_model_key: Optional[str],
) -> None:
    profile_df = _dataset_profile_table(dataset_df)
    examples_df = _typical_examples_df(dataset_df)
    left, right = st.columns((1.1, 0.9), gap="large")

    with left:
        _glass_open()
        _section_header(
            "Profil du corpus",
            "Lecture rapide de la matiere premiere qui alimente le moteur de triage.",
        )
        if not profile_df.empty:
            st.dataframe(profile_df, width="stretch", hide_index=True)
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
    upper_left, upper_right = st.columns((1, 1), gap="large")

    with upper_left:
        _glass_open()
        _section_header(
            "Distribution des categories",
            "Equilibre du corpus entre les trois familles de recours administatifs.",
        )
        st.plotly_chart(
            _build_category_chart(dataset_df),
            width="stretch",
            config={"displayModeBar": False},
        )
        _glass_close()

    with upper_right:
        _glass_open()
        _section_header(
            "Longueur des resumes",
            "Dispersion de la longueur des decisions resumees selon la categorie.",
        )
        st.plotly_chart(
            _build_text_length_chart(dataset_df),
            width="stretch",
            config={"displayModeBar": False},
        )
        _glass_close()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    outcome_left, outcome_right = st.columns((1, 1), gap="large")

    with outcome_left:
        _glass_open()
        _section_header(
            "Issues observees",
            "Vue rapide des solutions les plus frequentes dans les decisions du corpus.",
        )
        st.plotly_chart(
            _build_outcome_chart(dataset_df),
            width="stretch",
            config={"displayModeBar": False},
        )
        _glass_close()

    with outcome_right:
        _glass_open()
        _section_header(
            "Exemples typiques",
            "Un exemple representatif par famille de recours pour lire concretement le corpus.",
        )
        if not examples_df.empty:
            st.dataframe(examples_df, width="stretch", hide_index=True)
        else:
            st.info("Exemples indisponibles.")
        _glass_close()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    _glass_open()
    _section_header(
        "Nuages de mots par categorie",
        "Lecture visuelle du vocabulaire qui ressort le plus dans chaque famille de recours du corpus.",
    )
    if WordCloud is None:
        st.info(
            "Le package `wordcloud` n'est pas encore disponible dans l'environnement. "
            "Installe les dependances du projet pour activer cette visualisation."
        )
    else:
        cloud_columns = st.columns(3, gap="medium")
        for column, category in zip(cloud_columns, CATEGORY_LABELS.keys()):
            with column:
                st.markdown(
                    f"""
                    <div class="micro-chip" style="margin-bottom: 0.7rem;">
                        {_humanize_category(str(category))}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                figure = _build_category_wordcloud(dataset_df, str(category))
                if figure is not None:
                    st.pyplot(figure, clear_figure=True, use_container_width=True)
                else:
                    st.info("Nuage indisponible pour cette categorie.")
    _glass_close()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    lower_left, lower_right = st.columns((0.95, 1.05), gap="large")

    with lower_left:
        _glass_open()
        _section_header(
            "Metriques par classe",
            "Ou le modele est fort ou fragile selon chaque famille de recours.",
        )
        if class_metrics_df is not None and not class_metrics_df.empty:
            formatted_class_metrics = _format_class_metrics(
                class_metrics_df,
                selected_model_key,
            )
            if not formatted_class_metrics.empty:
                st.dataframe(
                    formatted_class_metrics,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("Aucune metrique par classe disponible pour ce modele.")
        else:
            st.info("Metriques par classe indisponibles.")
        _glass_close()

    with lower_right:
        _glass_open()
        _section_header(
            "Matrice de confusion",
            "Lecture rapide des confusions entre familles de recours sur le jeu de test.",
        )
        confusion_chart = _build_confusion_chart(confusion_df, selected_model_key)
        st.plotly_chart(confusion_chart, width="stretch", config={"displayModeBar": False})
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
    class_metrics_df = _load_class_metrics()
    confusion_df = _load_confusion_matrix()
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
        active_model_key = selected_model_key or (
            model_config.get("key") if model_config is not None else None
        )
        _render_corpus(
            dataset_df,
            metrics_df,
            robustness_df,
            class_metrics_df,
            confusion_df,
            active_model_key,
        )


if __name__ == "__main__":
    build_app()
