"""
models.py — Définition et entraînement des trois modèles.

Classes
-------
LogisticModel    : Régression Logistique Elastic Net
RandomForestModel: Random Forest + matrice de proximité
XGBoostModel     : XGBoost avec scale_pos_weight ou Focal Loss

Chaque classe expose :
    .fit(X_train, y_train)
    .predict_proba(X)  → np.ndarray shape (n, 2)
    .feature_importances_ (si disponible)
"""

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import (
    LR_PARAMS, RF_PARAMS, RF_PROXIMITY_SAMPLE,
    XGB_BASE_PARAMS, OPTUNA_SEARCH_SPACE,
    OPTUNA_N_TRIALS, OPTUNA_CV_FOLDS,
    FOCAL_GAMMA, FOCAL_ALPHA, RANDOM_SEED,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLE 1 – Régression Logistique (Elastic Net)
# ─────────────────────────────────────────────────────────────────────────────

class LogisticModel:
    """
    Régression Logistique avec pénalité Elastic Net.

    Justification Elastic Net
    -------------------------
    - L1 (Lasso) : sélectionne les variables (coefficients → 0)
    - L2 (Ridge) : stabilise les coefficients corrélés
    - l1_ratio=0.5 : compromis sélection / stabilité
    - C=0.1 : forte régularisation (C = 1/λ)

    Paramètres
    ----------
    class_weight : "balanced" | None
        Si "balanced", pondère les classes par fréquence inverse.
    """

    def __init__(self, class_weight: str | None = "balanced"):
        params = {**LR_PARAMS, "class_weight": class_weight}
        self.model = LogisticRegression(**params)
        self.class_weight = class_weight

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def coef_(self):
        return self.model.coef_

    def __repr__(self):
        return f"LogisticModel(class_weight={self.class_weight!r})"


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLE 2 – Random Forest + Matrice de Proximité
# ─────────────────────────────────────────────────────────────────────────────

class RandomForestModel:
    """
    Random Forest avec calcul de la matrice de proximité.

    Matrice de Proximité
    --------------------
    Deux observations i et j sont "proches" si elles finissent
    dans la même feuille terminale. La proximité est la fraction
    d'arbres où cela se produit :

        prox(i, j) = (1/T) Σ_t 1[feuille_t(i) == feuille_t(j)]

    Score d'outlier : obs_score(i) = 1 / Σ_j prox(i,j)²
    Une valeur élevée indique une observation difficile à classifier.
    """

    def __init__(self):
        self.model = RandomForestClassifier(**RF_PARAMS)
        self.proximity_matrix_: np.ndarray | None = None
        self.outlier_scores_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self.model.fit(X, y)
        print(f"[RF] OOB Score : {self.model.oob_score_:.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        return self.model.feature_importances_

    def compute_proximity(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_size: int = RF_PROXIMITY_SAMPLE,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calcule la matrice de proximité sur un sous-échantillon équilibré.

        Retourne
        --------
        proximity_matrix : np.ndarray shape (n, n)
        outlier_scores   : np.ndarray shape (n,) normalisés dans [0, 1]
        """
        # Sous-échantillon : moitié fraudes, moitié légitimes
        idx_fraud = np.where(y == 1)[0][: sample_size // 5]
        idx_legit = np.where(y == 0)[0][: sample_size - len(idx_fraud)]
        idx = np.concatenate([idx_fraud, idx_legit])
        rng = np.random.default_rng(RANDOM_SEED)
        rng.shuffle(idx)

        X_sub, y_sub = X[idx], y[idx]
        n = len(X_sub)

        # Feuilles terminales : shape (n_samples, n_trees)
        leaves = self.model.apply(X_sub)

        # Matrice de co-occurrence vectorisée
        proximity = np.zeros((n, n), dtype=np.float32)
        for tree_leaves in leaves.T:
            same = (tree_leaves[:, None] == tree_leaves[None, :]).astype(np.float32)
            proximity += same
        proximity /= self.model.n_estimators
        np.fill_diagonal(proximity, 1.0)

        # Scores d'outliers normalisés
        raw_scores = 1.0 / ((proximity ** 2).sum(axis=1) + 1e-9)
        mn, mx = float(raw_scores.min()), float(raw_scores.max())
        outlier_scores = (raw_scores - mn) / (mx - mn + 1e-9)

        self.proximity_matrix_  = proximity
        self.outlier_scores_    = outlier_scores
        self._proximity_y       = y_sub

        n_out = int((outlier_scores > outlier_scores.mean() + 2 * outlier_scores.std()).sum())
        print(f"[RF] Outliers détectés (μ+2σ) : {n_out} / {n}")
        return proximity, outlier_scores

    def __repr__(self):
        return f"RandomForestModel(n_estimators={self.model.n_estimators})"


# ─────────────────────────────────────────────────────────────────────────────
# MODÈLE 3 – XGBoost + Focal Loss + Optuna
# ─────────────────────────────────────────────────────────────────────────────

def _focal_loss(
    y_pred: np.ndarray,
    dtrain: xgb.DMatrix,
    gamma: float = FOCAL_GAMMA,
    alpha: float = FOCAL_ALPHA,
):
    """
    Focal Loss pour l'apprentissage sensible au coût.

    Formule
    -------
    FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

    Le gradient et le hessien sont dérivés analytiquement :
        grad = α_t · (1 − p_t)^γ · (p − y)
        hess = α_t · (1 − p_t)^γ · p · (1 − p)

    Paramètres
    ----------
    gamma : paramètre de focalisation (2.0 = standard)
        - γ=0 → Cross-Entropy classique
        - γ>0 → réduit la contribution des exemples faciles
    alpha : poids pour la classe positive (fraude)
        - 0.75 → 3× plus d'importance aux fraudes
    """
    y_true = dtrain.get_label()
    p      = 1.0 / (1.0 + np.exp(-y_pred))
    p_t    = np.where(y_true == 1, p, 1.0 - p)
    alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
    w      = alpha_t * (1.0 - p_t) ** gamma
    grad   = w * (p - y_true)
    hess   = w * p * (1.0 - p)
    return grad, hess


class XGBoostModel:
    """
    XGBoost avec deux stratégies de gestion du déséquilibre :

    1. scale_pos_weight
       - Multiplie le gradient de la classe positive par le ratio
         n_négatifs / n_positifs (~461 pour ce dataset)
       - Simple et efficace, mais uniforme sur tous les exemples

    2. Focal Loss (personnalisée)
       - Réduit le gradient des exemples bien classés ("faciles")
       - Concentre l'apprentissage sur les cas difficiles/ambigus
       - Préférable quand le déséquilibre + la difficulté se cumulent

    Optimisation
    ------------
    Recherche bayésienne via Optuna (TPE Sampler) plutôt que GridSearch.
    Le TPE modélise P(hyperparamètres | bonne valeur) pour explorer
    efficacement l'espace de recherche.
    """

    def __init__(
        self,
        mode: str = "focal",           # "spw" | "focal"
        scale_pos_weight: float | None = None,
        params: dict | None = None,
    ):
        if mode not in ("spw", "focal"):
            raise ValueError("mode doit être 'spw' ou 'focal'")
        self.mode = mode
        self.scale_pos_weight = scale_pos_weight
        self.params = params or {**XGB_BASE_PARAMS}
        self._booster  = None
        self._sklearn  = None
        self.study_    = None

    # ── Optimisation Optuna ───────────────────────────────────────────────

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_trials: int = OPTUNA_N_TRIALS,
    ) -> dict:
        """
        Recherche Bayésienne des hyperparamètres via Optuna (TPE).

        L'espace de recherche est défini dans config.OPTUNA_SEARCH_SPACE.
        La métrique optimisée est l'AUPRC (average_precision) en CV 3-fold.

        Retourne
        --------
        dict des meilleurs hyperparamètres
        """
        spw = self.scale_pos_weight

        def objective(trial: optuna.Trial) -> float:
            sp = OPTUNA_SEARCH_SPACE
            p = {
                "n_estimators":     trial.suggest_int(   "n_estimators",     *sp["n_estimators"]),
                "max_depth":        trial.suggest_int(   "max_depth",         *sp["max_depth"]),
                "learning_rate":    trial.suggest_float( "learning_rate",     *sp["learning_rate"],    log=True),
                "subsample":        trial.suggest_float( "subsample",         *sp["subsample"]),
                "colsample_bytree": trial.suggest_float( "colsample_bytree",  *sp["colsample_bytree"]),
                "reg_alpha":        trial.suggest_float( "reg_alpha",         *sp["reg_alpha"],        log=True),
                "reg_lambda":       trial.suggest_float( "reg_lambda",        *sp["reg_lambda"],       log=True),
                "scale_pos_weight": spw,
                "eval_metric":      "aucpr",
                "random_state":     RANDOM_SEED,
                "n_jobs":           -1,
                "verbosity":        0,
            }
            clf = xgb.XGBClassifier(**p)
            cv  = StratifiedKFold(
                n_splits=OPTUNA_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED
            )
            scores = cross_validate(
                clf, X_train, y_train,
                cv=cv, scoring="average_precision", n_jobs=-1,
            )
            return float(scores["test_score"].mean())

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        self.study_  = study
        self.params  = {k: v for k, v in study.best_params.items()}
        print(f"[XGB] Optuna best AUPRC (CV) : {study.best_value:.4f}")
        print(f"[XGB] Meilleurs params : {self.params}")
        return self.params

    # ── Entraînement ──────────────────────────────────────────────────────

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostModel":
        if self.mode == "spw":
            self._fit_spw(X_train, y_train)
        else:
            self._fit_focal(X_train, y_train)
        return self

    def _fit_spw(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entraînement avec scale_pos_weight."""
        if self.scale_pos_weight is None:
            self.scale_pos_weight = float((y == 0).sum()) / float((y == 1).sum())
            print(f"[XGB] scale_pos_weight auto = {self.scale_pos_weight:.1f}")

        clf = xgb.XGBClassifier(
            **self.params,
            scale_pos_weight=self.scale_pos_weight,
        )
        clf.fit(X, y)
        self._sklearn = clf
        self.feature_importances_ = clf.feature_importances_

    def _fit_focal(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entraînement avec la Focal Loss personnalisée."""
        dtrain = xgb.DMatrix(X, label=y)
        params = {k: v for k, v in self.params.items() if k != "n_estimators"}
        params.setdefault("seed", RANDOM_SEED)
        params.setdefault("verbosity", 0)

        n_rounds = self.params.get("n_estimators", XGB_BASE_PARAMS["n_estimators"])
        self._booster = xgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            obj=_focal_loss,
            verbose_eval=False,
        )
        # Importances depuis le booster
        scores = self._booster.get_score(importance_type="gain")
        self.feature_importances_ = scores

    # ── Prédiction ────────────────────────────────────────────────────────

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "spw":
            return self._sklearn.predict_proba(X)
        else:
            dmat = xgb.DMatrix(X)
            raw  = self._booster.predict(dmat)
            proba = 1.0 / (1.0 + np.exp(-raw))   # sigmoid
            return np.column_stack([1 - proba, proba])

    def __repr__(self):
        return f"XGBoostModel(mode={self.mode!r})"
