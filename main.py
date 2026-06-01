"""
main.py — Pipeline principal du projet.

Usage
-----
    python main.py                    # exécution complète
    python main.py --skip-optuna      # saute la recherche bayésienne
    python main.py --quick            # dataset réduit, moins d'essais Optuna
    python main.py --data path/to/creditcard.csv

Le script exécute les 4 étapes dans l'ordre :
    1. EDA & Préparation
    2. Développement des modèles
    3. Évaluation & Calibration
    4. Interprétabilité (SHAP)

Résultats
---------
    figures/   : 14 visualisations PNG
    models/    : modèles sérialisés (.pkl)
    figures/metrics_table.csv
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.config import (
    DATA_FILE, FIGS_DIR, MODELS_DIR, RANDOM_SEED, SHAP_SAMPLE_SIZE,
)
from src.data import (
    load_dataset, feature_engineering,
    split_and_scale, resample,
)
from src.models import LogisticModel, RandomForestModel, XGBoostModel
from src.evaluation import (
    compute_metrics, metrics_table,
    plot_precision_recall, plot_confusion_matrices,
    plot_reliability_diagrams,
)
from src.visualization import (
    plot_class_distribution, plot_amount_analysis,
    plot_correlation_matrix, plot_vif,
    plot_resampling_comparison, plot_rf_importances,
    plot_proximity_matrix, plot_optuna_convergence,
    plot_shap_bar, plot_shap_beeswarm, plot_shap_waterfall,
)

# ─── VIF helper ─────────────────────────────────────────────────────────────
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    sample = df[feature_cols].sample(3000, random_state=RANDOM_SEED)
    vif = pd.DataFrame({
        "Feature": feature_cols,
        "VIF": [variance_inflation_factor(sample.values, i)
                for i in range(sample.shape[1])],
    }).sort_values("VIF", ascending=False)
    return vif


# ─────────────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:

    print("\n" + "=" * 65)
    print("  PROJET IA – Classification Robuste (Détection de Fraude)")
    print("=" * 65)

    # ── ÉTAPE 1 : EDA & Préparation ───────────────────────────────────────
    print("\n[1/4] EDA & Préparation")

    data_path = Path(args.data) if args.data else DATA_FILE
    df = load_dataset(data_path)
    print(f"  → {df.shape[0]:,} lignes  |  Fraudes : {df['Class'].sum()}"
          f" ({df['Class'].mean()*100:.3f} %)")

    plot_class_distribution(df)
    plot_amount_analysis(df)
    plot_correlation_matrix(df)

    # VIF sur les features PCA + Amount
    from src.config import PCA_FEATURES
    vif_df = compute_vif(df, PCA_FEATURES + ["Amount"])
    vif_df.to_csv(FIGS_DIR / "vif_table.csv", index=False)
    plot_vif(vif_df)
    print(f"  → VIF calculé sur {len(PCA_FEATURES)+1} features")

    # Feature Engineering
    df, feature_cols = feature_engineering(df)

    # Split + scaling
    X_train, X_test, y_train, y_test, scaler = split_and_scale(df, feature_cols)

    # Rééchantillonnage
    X_sm, y_sm = resample(X_train, y_train, "smote")
    X_ad, y_ad = resample(X_train, y_train, "adasyn")
    X_nm, y_nm = resample(X_train, y_train, "nearmiss")

    plot_resampling_comparison({
        "Original": y_train,
        "SMOTE":    y_sm,
        "ADASYN":   y_ad,
        "NearMiss": y_nm,
    })

    results = {}

    # ── ÉTAPE 2 : Modèles ─────────────────────────────────────────────────
    print("\n[2/4] Développement des Modèles")

    # — Modèle 1 : LR class_weight ──────────────────────────────────────
    print("\n  ► LR Elastic Net (class_weight='balanced')")
    lr_cw = LogisticModel(class_weight="balanced")
    lr_cw.fit(X_train, y_train)
    proba_lrcw = lr_cw.predict_proba(X_test)[:, 1]
    results["LR_classweight"] = {"proba": proba_lrcw, "y_true": y_test}
    m = compute_metrics(y_test, proba_lrcw)
    print(f"     {m}")

    # — Modèle 1 bis : LR + SMOTE ────────────────────────────────────────
    print("\n  ► LR Elastic Net + SMOTE")
    lr_sm = LogisticModel(class_weight=None)
    lr_sm.fit(X_sm, y_sm)
    proba_lrsm = lr_sm.predict_proba(X_test)[:, 1]
    results["LR_smote"] = {"proba": proba_lrsm, "y_true": y_test}
    print(f"     {compute_metrics(y_test, proba_lrsm)}")

    # — Modèle 2 : Random Forest ─────────────────────────────────────────
    print("\n  ► Random Forest + Matrice de Proximité")
    rf = RandomForestModel()
    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    results["RF"] = {"proba": proba_rf, "y_true": y_test}
    print(f"     {compute_metrics(y_test, proba_rf)}")

    plot_rf_importances(rf.feature_importances_, feature_cols)
    proximity, outlier_scores = rf.compute_proximity(X_test, y_test)
    plot_proximity_matrix(proximity, outlier_scores, rf._proximity_y)

    # — Modèle 3 : XGBoost scale_pos_weight ──────────────────────────────
    print("\n  ► XGBoost (scale_pos_weight)")
    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    xgb_spw = XGBoostModel(mode="spw", scale_pos_weight=spw)

    if not args.skip_optuna:
        print("     Optimisation Optuna (TPE)…")
        n_trials = 10 if args.quick else 30
        xgb_spw.optimize(X_train, y_train, n_trials=n_trials)
        trials_df = xgb_spw.study_.trials_dataframe()
        plot_optuna_convergence(trials_df, xgb_spw.study_.best_value)
    else:
        print("     (Optuna ignoré — utilisation des hyperparamètres par défaut)")

    xgb_spw.fit(X_train, y_train)
    proba_spw = xgb_spw.predict_proba(X_test)[:, 1]
    results["XGB_spw"] = {"proba": proba_spw, "y_true": y_test}
    print(f"     {compute_metrics(y_test, proba_spw)}")

    # — Modèle 3 bis : XGBoost Focal Loss ────────────────────────────────
    print("\n  ► XGBoost (Focal Loss, γ=2, α=0.75)")
    xgb_focal = XGBoostModel(mode="focal", params=dict(xgb_spw.params))
    xgb_focal.fit(X_train, y_train)
    proba_focal = xgb_focal.predict_proba(X_test)[:, 1]
    results["XGB_focal"] = {"proba": proba_focal, "y_true": y_test}
    print(f"     {compute_metrics(y_test, proba_focal)}")

    # ── ÉTAPE 3 : Évaluation & Calibration ────────────────────────────────
    print("\n[3/4] Évaluation & Calibration")

    df_metrics = metrics_table(results)
    df_metrics.to_csv(FIGS_DIR / "metrics_table.csv")
    print("\n" + df_metrics.to_string())

    best = df_metrics.index[0]
    print(f"\n  ✔ Meilleur modèle (AUPRC) : {best}")

    plot_precision_recall(results)
    plot_confusion_matrices(results)
    plot_reliability_diagrams(results)

    # ── ÉTAPE 4 : SHAP ────────────────────────────────────────────────────
    print("\n[4/4] Interprétabilité (SHAP)")
    import shap as shap_lib

    explainer = shap_lib.TreeExplainer(xgb_spw._sklearn)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X_test), min(SHAP_SAMPLE_SIZE, len(X_test)),
                     replace=False)
    shap_values = explainer.shap_values(X_test[idx])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    plot_shap_bar(mean_abs, feature_cols)
    plot_shap_beeswarm(shap_values, X_test[idx], feature_cols)

    fraud_idx = np.where(y_test[idx] == 1)[0]
    if len(fraud_idx) > 0:
        base_val = float(explainer.expected_value) if not isinstance(
            explainer.expected_value, np.ndarray
        ) else float(explainer.expected_value[0])
        plot_shap_waterfall(shap_values[fraud_idx[0]], feature_cols, base_val)

    # ── Sauvegarde des modèles ────────────────────────────────────────────
    print("\n  Sauvegarde des modèles…")
    with open(MODELS_DIR / "lr_classweight.pkl", "wb") as f:
        pickle.dump(lr_cw.model, f)
    with open(MODELS_DIR / "random_forest.pkl", "wb") as f:
        pickle.dump(rf.model, f)
    with open(MODELS_DIR / "xgb_spw.pkl", "wb") as f:
        pickle.dump(xgb_spw._sklearn, f)
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "feature_cols.pkl", "wb") as f:
        pickle.dump(feature_cols, f)

    print("\n" + "=" * 65)
    print("  ✅  Pipeline terminé.")
    print(f"  📊  Figures    → {FIGS_DIR}/")
    print(f"  🤖  Modèles    → {MODELS_DIR}/")
    print(f"  📋  Métriques  → {FIGS_DIR}/metrics_table.csv")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Projet IA – Détection de Fraude Bancaire"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Chemin vers creditcard.csv (défaut : data/creditcard.csv)"
    )
    parser.add_argument(
        "--skip-optuna", action="store_true",
        help="Ignore l'optimisation Optuna (hyperparamètres par défaut)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Mode rapide : 10 essais Optuna au lieu de 30"
    )
    args = parser.parse_args()
    main(args)
