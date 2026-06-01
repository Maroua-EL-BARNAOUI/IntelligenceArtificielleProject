"""
visualization.py — Toutes les visualisations du projet.

Fonctions
---------
plot_class_distribution(df)
plot_amount_analysis(df)
plot_correlation_matrix(df)
plot_vif(vif_df)
plot_resampling_comparison(counts_dict)
plot_rf_importances(importances, feature_cols)
plot_proximity_matrix(proximity, outlier_scores, y)
plot_optuna_convergence(study)
plot_shap_bar(shap_values, feature_cols)
plot_shap_beeswarm(shap_values, X, feature_cols)
plot_shap_waterfall(shap_values_sample, feature_cols, base_val)
plot_dashboard(results, vif_df, shap_mean, study_trials)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve

from src.config import PALETTE, FIGS_DIR

# ─── Style ───────────────────────────────────────────────────────────────────
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

_MODEL_STYLES = {
    "LR_classweight": (PALETTE["accent4"], "-",  "LR (class_weight)"),
    "LR_smote":       (PALETTE["subtext"], "--", "LR + SMOTE"),
    "RF":             (PALETTE["accent3"], "-",  "Random Forest"),
    "XGB_spw":        (PALETTE["accent1"], "-",  "XGBoost (scale_pos_weight)"),
    "XGB_focal":      (PALETTE["accent2"], "-",  "XGBoost (Focal Loss)"),
}


def _save(name: str) -> None:
    plt.savefig(FIGS_DIR / name, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# EDA
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, save: bool = True) -> None:
    counts = df["Class"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        ["Légitimes\n(Classe 0)", "Fraudes\n(Classe 1)"],
        counts.values,
        color=[PALETTE["accent1"], PALETTE["accent2"]],
        width=0.5, edgecolor=PALETTE["border"], linewidth=1.5,
    )
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                f"{b.get_height():,}", ha="center", va="bottom",
                color=PALETTE["text"], fontsize=12, fontweight="bold")
    ax.set_title("Distribution des Classes (Déséquilibre Extrême)",
                 fontsize=14, pad=15)
    ax.set_ylabel("Nombre de transactions")
    ax.set_yscale("log")
    ax.yaxis.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        _save("01_class_distribution.png")


def plot_amount_analysis(df: pd.DataFrame, save: bool = True) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for cls, col, lbl in [
        (0, PALETTE["accent1"], "Légitimes"),
        (1, PALETTE["accent2"], "Fraudes"),
    ]:
        data = df[df.Class == cls]["Amount"]
        axes[0].hist(data.clip(upper=500), bins=60,
                     alpha=0.7, color=col, label=lbl, density=True)
        axes[1].boxplot(
            [data], positions=[cls], patch_artist=True,
            boxprops=dict(facecolor=col, alpha=0.6),
            medianprops=dict(color=PALETTE["text"], linewidth=2),
            whiskerprops=dict(color=PALETTE["subtext"]),
            capprops=dict(color=PALETTE["subtext"]),
            flierprops=dict(marker=".", markersize=2,
                            markerfacecolor=PALETTE["subtext"], alpha=0.3),
        )
    axes[0].set_title("Distribution du Montant", fontsize=12)
    axes[0].set_xlabel("Montant (€)")
    axes[0].legend()
    axes[1].set_title("Boxplot Montant par Classe", fontsize=12)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Légitimes", "Fraudes"])
    fig.suptitle("Analyse du Montant des Transactions", fontsize=14, y=1.02)
    plt.tight_layout()
    if save:
        _save("02_amount_analysis.png")


def plot_correlation_matrix(df: pd.DataFrame, save: bool = True) -> None:
    from src.config import PCA_FEATURES
    feat_cols = PCA_FEATURES + ["Amount"]
    corr  = df[feat_cols + ["Class"]].corr()
    cmap  = LinearSegmentedColormap.from_list(
        "custom", [PALETTE["accent2"], PALETTE["bg"], PALETTE["accent1"]]
    )
    mask  = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                vmin=-1, vmax=1, ax=ax, linewidths=0.3,
                linecolor=PALETTE["border"],
                cbar_kws={"shrink": 0.8})
    ax.set_title("Matrice de Corrélation (Triangle Inférieur)",
                 fontsize=14, pad=15)
    plt.tight_layout()
    if save:
        _save("03_correlation_matrix.png")


def plot_vif(vif_df: pd.DataFrame, save: bool = True) -> None:
    colors = [
        PALETTE["accent2"] if (np.isinf(v) or v > 10) else
        PALETTE["accent3"] if v > 5 else
        PALETTE["accent1"]
        for v in vif_df["VIF"]
    ]
    vif_display = vif_df["VIF"].clip(upper=50)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(vif_df["Feature"][::-1], vif_display[::-1], color=colors[::-1])
    ax.axvline(5,  color=PALETTE["accent3"], linestyle="--",
               lw=1.5, label="Seuil modéré (5)")
    ax.axvline(10, color=PALETTE["accent2"], linestyle="--",
               lw=1.5, label="Seuil élevé (10)")
    ax.set_xlabel("VIF")
    ax.set_title("Variance Inflation Factor par Feature", fontsize=13)
    ax.legend()
    plt.tight_layout()
    if save:
        _save("04_vif.png")


def plot_resampling_comparison(counts_dict: dict, save: bool = True) -> None:
    """
    counts_dict = {"Original": y_original, "SMOTE": y_smote, ...}
    """
    n = len(counts_dict)
    colors_map = [PALETTE["accent1"], PALETTE["accent3"],
                  PALETTE["accent4"], PALETTE["accent2"]]
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

    for ax, ((title, y_), col) in zip(axes, zip(counts_dict.items(), colors_map)):
        vals = pd.Series(y_).value_counts()
        ax.bar(["Légitimes", "Fraudes"],
               [vals.get(0, 0), vals.get(1, 0)],
               color=[PALETTE["accent1"], col],
               edgecolor=PALETTE["border"])
        ax.set_title(title, fontsize=12)
        ax.set_yscale("log")
        ax.yaxis.grid(True, alpha=0.3)

    fig.suptitle("Comparaison des Techniques de Rééchantillonnage",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    if save:
        _save("05_resampling_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def plot_rf_importances(
    importances: np.ndarray,
    feature_cols: list[str],
    top_n: int = 20,
    save: bool = True,
) -> None:
    series = pd.Series(importances, index=feature_cols).nlargest(top_n)
    colors = [
        PALETTE["accent2"] if i < 5 else
        PALETTE["accent1"] if i < 10 else PALETTE["subtext"]
        for i in range(top_n)
    ]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(series.index[::-1], series.values[::-1], color=colors[::-1])
    ax.set_title(f"Top {top_n} Feature Importances – Random Forest",
                 fontsize=13)
    ax.set_xlabel("Importance (Gini)")
    plt.tight_layout()
    if save:
        _save("06_rf_importances.png")


def plot_proximity_matrix(
    proximity: np.ndarray,
    outlier_scores: np.ndarray,
    y: np.ndarray,
    save: bool = True,
) -> None:
    cmap_prox = LinearSegmentedColormap.from_list(
        "prox", [PALETTE["bg"], PALETTE["accent1"]]
    )
    n = len(outlier_scores)
    threshold = outlier_scores.mean() + 2 * outlier_scores.std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap de proximité
    im = axes[0].imshow(proximity, cmap=cmap_prox, aspect="auto",
                        vmin=0, vmax=1)
    axes[0].set_title("Matrice de Proximité", fontsize=12)
    axes[0].set_xlabel("Observation")
    axes[0].set_ylabel("Observation")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    # Scatter des scores d'outliers
    colors_out = [PALETTE["accent2"] if c == 1 else PALETTE["accent1"]
                  for c in y]
    axes[1].scatter(range(n), outlier_scores, c=colors_out, s=35, alpha=0.8)
    axes[1].axhline(threshold, color=PALETTE["accent3"],
                    linestyle="--", lw=1.5, label="Seuil μ+2σ")
    axes[1].set_title("Scores d'Outliers de Prédiction (RF Proximity)",
                      fontsize=12)
    axes[1].set_xlabel("Index observation")
    axes[1].set_ylabel("Outlier Score (normalisé)")
    legend_elements = [
        Patch(facecolor=PALETTE["accent2"], label="Fraude"),
        Patch(facecolor=PALETTE["accent1"], label="Légitime"),
        plt.Line2D([0], [0], color=PALETTE["accent3"],
                   linestyle="--", label="Seuil μ+2σ"),
    ]
    axes[1].legend(handles=legend_elements)

    plt.tight_layout()
    if save:
        _save("07_proximity_outliers.png")


# ─────────────────────────────────────────────────────────────────────────────
# Optuna
# ─────────────────────────────────────────────────────────────────────────────

def plot_optuna_convergence(
    trials_df: pd.DataFrame,
    best_value: float,
    save: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(trials_df.index, trials_df["value"],
                    c=PALETTE["accent1"], s=30, alpha=0.6, label="Essai")
    axes[0].plot(trials_df.index, trials_df["value"].cummax(),
                 color=PALETTE["accent2"], lw=2, label="Meilleur cumulé")
    axes[0].set_title("Optuna – Optimization History (TPE Sampler)",
                      fontsize=12)
    axes[0].set_xlabel("Numéro d'essai")
    axes[0].set_ylabel("AUPRC (CV 3-fold)")
    axes[0].legend()

    axes[1].hist(trials_df["value"].dropna(), bins=15,
                 color=PALETTE["accent4"], edgecolor=PALETTE["border"])
    axes[1].axvline(best_value, color=PALETTE["accent2"],
                    linestyle="--", lw=2, label=f"Best = {best_value:.4f}")
    axes[1].set_title("Optuna – Distribution des Valeurs Objectif",
                      fontsize=12)
    axes[1].set_xlabel("AUPRC")
    axes[1].set_ylabel("Fréquence")
    axes[1].legend()

    plt.tight_layout()
    if save:
        _save("08_optuna_convergence.png")


# ─────────────────────────────────────────────────────────────────────────────
# SHAP
# ─────────────────────────────────────────────────────────────────────────────

def plot_shap_bar(
    mean_abs_shap: np.ndarray,
    feature_cols: list[str],
    top_n: int = 20,
    save: bool = True,
) -> None:
    series = pd.Series(mean_abs_shap, index=feature_cols).nlargest(top_n)
    colors = [
        PALETTE["accent2"] if i < 5 else
        PALETTE["accent1"] if i < 10 else PALETTE["subtext"]
        for i in range(top_n)
    ]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(series.index[::-1], series.values[::-1], color=colors[::-1])
    ax.set_title(f"SHAP – Importance Globale (Top {top_n} features, XGBoost)",
                 fontsize=13)
    ax.set_xlabel("|SHAP value| moyen")
    plt.tight_layout()
    if save:
        _save("12_shap_bar.png")


def plot_shap_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_cols: list[str],
    top_n: int = 15,
    save: bool = True,
) -> None:
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[-top_n:][::-1]
    shap_top = shap_values[:, top_idx]
    feat_top = [feature_cols[i] for i in top_idx]
    vals_top = X[:, top_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    rng = np.random.default_rng(42)
    for fi in range(top_n):
        sv  = shap_top[:, -(fi + 1)]
        fv  = vals_top[:, -(fi + 1)]
        mn, mx = fv.min(), fv.max()
        fv_norm = (fv - mn) / (mx - mn + 1e-9)
        colors  = plt.cm.RdYlBu_r(fv_norm)
        jitter  = rng.uniform(-0.3, 0.3, len(sv))
        ax.scatter(sv, fi + jitter, c=colors, s=8, alpha=0.6)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feat_top[::-1], fontsize=9)
    ax.axvline(0, color=PALETTE["subtext"], lw=1)
    ax.set_title(f"SHAP Beeswarm – Top {top_n} Features", fontsize=13)
    ax.set_xlabel("Valeur SHAP (impact sur la prédiction)")

    sm = plt.cm.ScalarMappable(cmap="RdYlBu_r")
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label("Valeur de la feature\n(faible → élevé)", fontsize=9)
    plt.tight_layout()
    if save:
        _save("13_shap_beeswarm.png")


def plot_shap_waterfall(
    shap_sample: np.ndarray,
    feature_cols: list[str],
    base_val: float,
    top_n: int = 10,
    save: bool = True,
    label: str = "fraude",
) -> None:
    top_idx = np.argsort(np.abs(shap_sample))[-top_n:][::-1]
    sv_top  = shap_sample[top_idx]
    fn_top  = [feature_cols[i] for i in top_idx]
    colors  = [PALETTE["accent2"] if s > 0 else PALETTE["accent1"]
               for s in sv_top]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(fn_top[::-1], sv_top[::-1], color=colors[::-1])
    ax.axvline(0, color=PALETTE["subtext"], lw=1)
    ax.set_title(
        f"SHAP Waterfall – Exemple {label} détecté\n(valeur de base = {base_val:.3f})",
        fontsize=12,
    )
    ax.set_xlabel("Valeur SHAP")
    plt.tight_layout()
    if save:
        _save(f"14_shap_waterfall_{label}.png")
