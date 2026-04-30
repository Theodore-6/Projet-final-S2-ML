from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


class LegalSignalTransformer(BaseEstimator, TransformerMixin):
    """Add compact domain signals on top of raw administrative-law text.

    The goal is not to replace TF-IDF, but to give the classifier a few
    explicit legal cues that remain stable across paraphrases.
    """

    SIGNAL_PATTERNS = {
        "annulation": (
            "annuler",
            "annulation",
            "abroger",
            "abrogation",
            "retirer",
            "retrait",
            "injonction",
        ),
        "decision_administrative": (
            "decision",
            "arrete",
            "decret",
            "circulaire",
            "deliberation",
            "refus",
            "autorisation",
            "agrement",
            "permis",
        ),
        "immigration": (
            "titre de sejour",
            "sejour",
            "asile",
            "visa",
            "oqtf",
            "etranger",
            "nationalite",
        ),
        "autorite_publique": (
            "prefet",
            "prefecture",
            "ministre",
            "maire",
            "commune",
            "recteur",
            "ofpra",
            "administration",
        ),
        "financier": (
            "euro",
            "euros",
            "somme",
            "verser",
            "condamner",
            "interets",
            "capitalises",
        ),
        "fiscal": (
            "impot",
            "taxe",
            "tva",
            "cotisation",
            "decharge",
            "imposition",
            "fiscal",
        ),
        "indemnitaire": (
            "responsabilite",
            "indemnisation",
            "indemnitaire",
            "reparation",
            "prejudice",
            "dommage",
            "faute",
        ),
        "social": (
            "allocation",
            "pension",
            "invalidite",
            "emploi",
            "chomage",
            "rsa",
            "aide au retour a l'emploi",
        ),
        "contrat_public": (
            "marche public",
            "concession",
            "delegation de service public",
            "contrat public",
            "travaux publics",
        ),
        "urgence": (
            "refere",
            "suspension",
            "liberte",
            "urgence",
            "mesures utiles",
        ),
        "renvoi_procedural": (
            "renvoi",
            "sursis a statuer",
            "question prejudicielle",
            "avis contentieux",
            "transmis",
            "saisi le tribunal",
        ),
        "juridiction_externe": (
            "cour d'appel",
            "prud'hommes",
            "tribunal judiciaire",
            "tribunal des pensions",
            "cour de cassation",
        ),
    }

    FEATURE_NAMES = [
        "signal_log_tokens",
        "signal_log_chars",
        "signal_ratio_digits",
        "signal_annulation",
        "signal_decision_administrative",
        "signal_immigration",
        "signal_autorite_publique",
        "signal_financier",
        "signal_fiscal",
        "signal_indemnitaire",
        "signal_social",
        "signal_contrat_public",
        "signal_urgence",
        "signal_renvoi_procedural",
        "signal_juridiction_externe",
        "signal_score_exces_pouvoir",
        "signal_score_plein_contentieux",
        "signal_score_autres_recours",
    ]

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> csr_matrix:
        rows = [self._vectorize_text(text) for text in X]
        return csr_matrix(np.asarray(rows, dtype=float))

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(self.FEATURE_NAMES, dtype=object)

    def _vectorize_text(self, text: object) -> list[float]:
        raw_text = "" if text is None else str(text)
        normalized = self._normalize_text(raw_text)
        token_count = len(re.findall(r"\b\w+\b", normalized))
        char_count = len(normalized)
        digit_count = sum(character.isdigit() for character in raw_text)

        scores = {
            name: self._pattern_score(normalized, patterns)
            for name, patterns in self.SIGNAL_PATTERNS.items()
        }

        exces_score = (
            scores["annulation"]
            + scores["decision_administrative"]
            + scores["immigration"]
            + scores["autorite_publique"]
        ) / 4.0
        plein_score = (
            scores["financier"]
            + scores["fiscal"]
            + scores["indemnitaire"]
            + scores["social"]
            + scores["contrat_public"]
        ) / 5.0
        autres_score = (
            scores["urgence"]
            + scores["renvoi_procedural"]
            + scores["juridiction_externe"]
        ) / 3.0

        return [
            math.log1p(token_count),
            math.log1p(char_count),
            digit_count / max(len(raw_text), 1),
            scores["annulation"],
            scores["decision_administrative"],
            scores["immigration"],
            scores["autorite_publique"],
            scores["financier"],
            scores["fiscal"],
            scores["indemnitaire"],
            scores["social"],
            scores["contrat_public"],
            scores["urgence"],
            scores["renvoi_procedural"],
            scores["juridiction_externe"],
            exces_score,
            plein_score,
            autres_score,
        ]

    @staticmethod
    def _pattern_score(text: str, patterns: tuple[str, ...]) -> float:
        matches = sum(1 for pattern in patterns if pattern in text)
        return matches / max(len(patterns), 1)

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(character for character in text if not unicodedata.combining(character))
        return text.lower()
