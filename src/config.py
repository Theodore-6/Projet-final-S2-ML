from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PLOTS_DIR = PROJECT_ROOT / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"

for dir in [
    DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    NOTEBOOKS_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    SCRIPTS_DIR,
    TESTS_DIR,
]:
    dir.mkdir(exist_ok=True)

ENV_FILE = PROJECT_ROOT / ".env"
APP_ENTRYPOINT = PROJECT_ROOT / "src" / "app.py"
MODEL_METRICS_FILE = RESULTS_DIR / "model_metrics.csv"
ROBUSTNESS_METRICS_FILE = RESULTS_DIR / "robustness_metrics.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_legal_cases_admin.csv"

STREAMLIT_HOST = "localhost"
STREAMLIT_PORT = 8501
RANDOM_STATE = 42
TEST_SIZE = 0.25

PROJECT_TITLE = "Assistant d'aide a l'analyse juridique"
PROJECT_SUBTITLE = (
    "Triage de dossiers en droit administratif a partir d'un resume de faits"
)

TEXT_COLUMN = "facts_summary"
TARGET_COLUMN = "case_category"
OUTCOME_COLUMN = "likely_outcome"

MODELS = {
    "log_reg_legal": {
        "name": "TF-IDF + Logistic Regression",
        "description": "Baseline text classifier for legal case triage.",
        "path": MODELS_DIR / "log_reg_legal.joblib",
    },
    "linear_svm_legal": {
        "name": "TF-IDF + Linear SVM",
        "description": "Margin-based baseline for short legal fact summaries.",
        "path": MODELS_DIR / "linear_svm_legal.joblib",
    },
    "char_svm_legal": {
        "name": "Character TF-IDF + Linear SVM",
        "description": "More formulation-robust linear model based on character n-grams.",
        "path": MODELS_DIR / "char_svm_legal.joblib",
    },
    "lsa_log_reg_legal": {
        "name": "Latent Semantic TF-IDF + Logistic Regression",
        "description": "Dense semantic projection over TF-IDF features for improved robustness.",
        "path": MODELS_DIR / "lsa_log_reg_legal.joblib",
    },
    "hybrid_log_reg_legal": {
        "name": "Hybrid TF-IDF + Legal Signals",
        "description": "Word-level text features reinforced with handcrafted administrative-law signals.",
        "path": MODELS_DIR / "hybrid_log_reg_legal.joblib",
    },
}
