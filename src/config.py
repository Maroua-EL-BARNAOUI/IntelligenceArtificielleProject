"""
Configuration globale du projet.
Centralise tous les hyperparamètres, chemins et constantes.
"""
from pathlib import Path

# ── Chemins ────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
FIGS_DIR   = ROOT / "figures"
MODELS_DIR = ROOT / "models"

DATA_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────
DATA_FILE   = DATA_DIR / "creditcard.csv"
N_SAMPLES   = 284_807
FRAUD_RATIO = 0.00173   # ~0.17 % (Credit Card Fraud Kaggle)
RANDOM_SEED = 42
TEST_SIZE   = 0.20

# ── Features ───────────────────────────────────────────────────────────────
PCA_FEATURES     = [f"V{i}" for i in range(1, 29)]
NUMERIC_FEATURES = PCA_FEATURES + ["Amount"]
TARGET           = "Class"

# ── Palette visuelle ───────────────────────────────────────────────────────
PALETTE = {
    "bg":      "#0D1117",
    "surface": "#161B22",
    "border":  "#30363D",
    "accent1": "#58A6FF",
    "accent2": "#F78166",
    "accent3": "#3FB950",
    "accent4": "#D2A8FF",
    "text":    "#E6EDF3",
    "subtext": "#8B949E",
}

# ── Modèle 1 – Logistic Regression (Elastic Net) ───────────────────────────
LR_PARAMS = {
    "penalty":      "elasticnet",
    "solver":       "saga",
    "l1_ratio":     0.5,
    "C":            0.1,
    "max_iter":     1000,
    "random_state": RANDOM_SEED,
    "n_jobs":       -1,
}

# ── Modèle 2 – Random Forest ───────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators":     100,
    "max_depth":        10,
    "min_samples_leaf": 5,
    "max_features":     "sqrt",
    "class_weight":     "balanced",
    "oob_score":        True,
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
}
RF_PROXIMITY_SAMPLE = 200  # nb observations pour la matrice de proximité

# ── Modèle 3 – XGBoost ────────────────────────────────────────────────────
XGB_BASE_PARAMS = {
    "n_estimators":     150,
    "max_depth":        5,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "eval_metric":      "aucpr",
    "random_state":     RANDOM_SEED,
    "n_jobs":           -1,
    "verbosity":        0,
}

# Espace de recherche Optuna
OPTUNA_SEARCH_SPACE = {
    "n_estimators":     (50,   300),
    "max_depth":        (3,    8),
    "learning_rate":    (0.01, 0.3),   # échelle log
    "subsample":        (0.6,  1.0),
    "colsample_bytree": (0.6,  1.0),
    "reg_alpha":        (1e-3, 5.0),   # échelle log
    "reg_lambda":       (1e-3, 5.0),   # échelle log
}
OPTUNA_N_TRIALS = 30
OPTUNA_CV_FOLDS = 3

# Focal Loss
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.75   # poids pour la classe positive (fraude)

# ── Calibration ────────────────────────────────────────────────────────────
CALIBRATION_METHOD = "sigmoid"  # Platt Scaling
CALIBRATION_N_BINS = 8

# ── SHAP ────────────────────────────────────────────────────────────────────
SHAP_SAMPLE_SIZE = 300
