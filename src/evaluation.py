"""
evaluation.py — Métriques, calibration et courbes d'évaluation.

Fonctions principales
---------------------
compute_metrics(y_true, proba)   → dict avec F1-Macro, MCC, AUPRC, AUC, Brier
calibrate(model, X_cal, y_cal)   → modèle calibré (Platt Scaling)
plot_precision_recall(results)   → courbes PR pour tous les modèles
plot_confusion_matrices(results) → matrices de confusion
plot_reliability_diagrams(...)   → diagrammes de fiabilité avant/après calibration
metrics_table(results)           → pd.DataFrame récapitulatif
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve,
)
from sklearn.linear_model import LogisticRegression as _CalLR

from src.config import PALETTE, CALIBRATION_METHOD, CALIBRATION_N_BINS, FIGS_DIR

warnings.filterwarnings("ignore")

# ─── Style matplotlib ────────────────────────────────────────────────────────
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["surface"],
    "axes.edgecolor":   PALETTE["border"],
    "axes.labelcolor":  PALETTE["text"],
    "xtick.color":      PALETTE["subtext"],
    "ytick.color":      PALETTE["subtext"],
    "text.color":       PALETTE["text"],
    "grid.color":       PALETTE["border"],
    "grid.alpha":       0.5,
    "font.family":      "monospace",
})


# ─────────────────────────────────────────────────────────────────────────────
# Métriques
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    proba:  np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Calcule les métriques d'évaluation pour données déséquilibrées.

    Métriques retournées (l'Accuracy est volontairement exclue)
    -----------------------------------------------------------
    F1-Macro : moyenne des F1 par classe — équilibre précision/rappel
    MCC      : Matthews Correlation Coefficient — non biaisé par déséquilibre
    AUPRC    : Aire sous la courbe Précision-Rappel — focus classe positive
    AUC-ROC  : Area Under the ROC Curve
    Brier    : MSE des probabilités — mesure la calibration

    Paramètres
    ----------
    threshold : seuil de classification (défaut 0.5)
    """
    pred = (proba >= threshold).astype(int)
    return {
        "F1-Macro": round(f1_score(y_true, pred, average="macro"),   4),
        "MCC":      round(matthews_corrcoef(y_true, pred),            4),
        "AUPRC":    round(average_precision_score(y_true, proba),     4),
        "AUC-ROC":  round(roc_auc_score(y_true, proba),              4),
        "Brier":    round(brier_score_loss(y_true, proba),            4),
    }


def metrics_table(results: dict) -> pd.DataFrame:
    """
    Construit un DataFrame récapitulatif depuis un dictionnaire
    {nom_modèle: {"proba": ..., "y_true": ...}}.
    """
    rows = {}
    for name, res in results.items():
        rows[name] = compute_metrics(res["y_true"], res["proba"])
    df = pd.DataFrame(rows).T
    df.index.name = "Modèle"
    return df.sort_values("AUPRC", ascending=False)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def platt_scale(
    proba_raw: np.ndarray,
    y_true:    np.ndarray,
) -> np.ndarray:
    """
    Platt Scaling : ajuste une sigmoïde σ(az + b) sur les probabilités brutes.

    Plus simple qu'Isotonic Regression, robuste sur petits datasets de calibration.
    Suppose que la transformation calibrante est sigmoïdale.

    Retourne
    --------
    proba calibrées (np.ndarray de même forme)
    """
    cal = _CalLR()
    cal.fit(proba_raw.reshape(-1, 1), y_true)
    return cal.predict_proba(proba_raw.reshape(-1, 1))[:, 1]


def calibrate_sklearn(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = CALIBRATION_METHOD,
):
    """
    Calibre un modèle scikit-learn via CalibratedClassifierCV.

    Paramètres
    ----------
    method : "sigmoid" (Platt) | "isotonic" (Isotonic Regression)
    """
    cal_model = CalibratedClassifierCV(model, method=method, cv="prefit")
    cal_model.fit(X_cal, y_cal)
    return cal_model


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_STYLES = {
    "LR_classweight": (PALETTE["accent4"], "-",  "LR (class_weight)"),
    "LR_smote":       (PALETTE["subtext"], "--", "LR + SMOTE"),
    "RF":             (PALETTE["accent3"], "-",  "Random Forest"),
    "XGB_spw":        (PALETTE["accent1"], "-",  "XGBoost (scale_pos_weight)"),
    "XGB_focal":      (PALETTE["accent2"], "-",  "XGBoost (Focal Loss)"),
}


def _save(name: str) -> None:
    plt.savefig(
        FIGS_DIR / name, dpi=150, bbox_inches="tight",
        facecolor=PALETTE["bg"],
    )
    plt.close()


def plot_precision_recall(results: dict, save: bool = True) -> None:
    """
    Trace les courbes Precision-Recall pour tous les modèles.
    results = {nom: {"proba": ..., "y_true": ...}}
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, res in results.items():
        col, ls, lbl = _MODEL_STYLES.get(name, (PALETTE["accent1"], "-", name))
        p, r, _ = precision_recall_curve(res["y_true"], res["proba"])
        ap = average_precision_score(res["y_true"], res["proba"])
        ax.plot(r, p, color=col, linestyle=ls, lw=2,
                label=f"{lbl}  (AUPRC={ap:.4f})")

    baseline = list(results.values())[0]["y_true"].mean()
    ax.axhline(baseline, color=PALETTE["subtext"], linestyle=":",
               label=f"Baseline aléatoire ({baseline:.4f})")

    ax.set_title("Courbes Precision-Recall", fontsize=14)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        _save("09_precision_recall_curves.png")
    return fig


def plot_confusion_matrices(results: dict, save: bool = True) -> None:
    """Matrices de confusion côte-à-côte pour tous les modèles."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm  = confusion_matrix(res["y_true"], (res["proba"] >= 0.5).astype(int))
        lbl = _MODEL_STYLES.get(name, (None, None, name))[2]
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                    cbar=False, annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(lbl, fontsize=10)
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")

    fig.suptitle("Matrices de Confusion", fontsize=14, y=1.02)
    plt.tight_layout()
    if save:
        _save("10_confusion_matrices.png")
    return fig


def plot_reliability_diagrams(
    results: dict,
    n_bins:  int  = CALIBRATION_N_BINS,
    save:    bool = True,
) -> None:
    """
    Reliability Diagrams (diagrammes de fiabilité) avant et après Platt Scaling.

    Un modèle parfaitement calibré suit la diagonale y=x.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (do_cal, title) in zip(
        axes,
        [(False, "Avant Calibration"),
         (True,  "Après Calibration (Platt Scaling)")],
    ):
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Calibration parfaite")

        for name, res in results.items():
            if name == "XGB_focal":
                continue  # booster XGB custom → calibration séparée
            col, ls, lbl = _MODEL_STYLES.get(name, (PALETTE["accent1"], "-", name))
            proba = res["proba"].copy()

            if do_cal:
                proba = platt_scale(proba, res["y_true"])

            frac_pos, mean_pred = calibration_curve(
                res["y_true"], proba, n_bins=n_bins
            )
            ax.plot(mean_pred, frac_pos, "s-", color=col,
                    lw=1.8, markersize=5, label=lbl)

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Probabilité prédite moyenne")
        ax.set_ylabel("Fraction de positifs réels")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        _save("11_reliability_diagram.png")
    return fig
