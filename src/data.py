"""
data.py — Chargement, génération et feature engineering.

Fonctions principales
---------------------
generate_dataset()      : génère un dataset synthétique (si pas de données réelles)
load_dataset()          : charge le CSV et valide le schéma
feature_engineering()   : crée les features dérivées et les interactions
split_and_scale()       : train/test split stratifié + RobustScaler
resample(strategy)      : SMOTE | ADASYN | NearMiss | class_weight
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import NearMiss

from src.config import (
    DATA_FILE, N_SAMPLES, FRAUD_RATIO, RANDOM_SEED,
    TEST_SIZE, PCA_FEATURES, TARGET,
)


# ─────────────────────────────────────────────────────────────────────────────
# Génération du dataset synthétique
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(path: Path = DATA_FILE, force: bool = False) -> pd.DataFrame:
    """
    Génère un dataset synthétique imitant le Credit Card Fraud Detection (Kaggle).

    Si le fichier existe déjà et que force=False, il est chargé directement.

    Paramètres
    ----------
    path  : chemin de sauvegarde du CSV
    force : régénérer même si le fichier existe

    Retourne
    --------
    pd.DataFrame avec colonnes V1–V28, Amount, Time, Class
    """
    if path.exists() and not force:
        print(f"[data] Dataset existant chargé : {path}")
        return pd.read_csv(path)

    print(f"[data] Génération d'un dataset synthétique ({N_SAMPLES:,} lignes)…")
    np.random.seed(RANDOM_SEED)

    X, y = make_classification(
        n_samples=N_SAMPLES,
        n_features=28,
        n_informative=15,
        n_redundant=5,
        n_clusters_per_class=3,
        weights=[1 - FRAUD_RATIO, FRAUD_RATIO],
        flip_y=0.001,
        class_sep=0.8,
        random_state=RANDOM_SEED,
    )

    df = pd.DataFrame(X, columns=PCA_FEATURES)
    df["Amount"] = np.abs(np.random.exponential(scale=88, size=N_SAMPLES))
    df["Amount"] = np.where(y == 1, df["Amount"] * 2.5, df["Amount"])
    df["Time"]   = np.sort(np.random.uniform(0, 172_792, N_SAMPLES))
    df[TARGET]   = y

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[data] Sauvegardé : {path}  |  Fraudes : {y.sum()} ({y.mean()*100:.3f} %)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(path: Path = DATA_FILE) -> pd.DataFrame:
    """
    Charge le dataset depuis le CSV.
    Si le fichier n'existe pas, génère un dataset synthétique.
    """
    if not path.exists():
        print(f"[data] {path} introuvable — génération automatique.")
        return generate_dataset(path)
    df = pd.read_csv(path)
    _validate(df)
    return df


def _validate(df: pd.DataFrame) -> None:
    """Vérifie que les colonnes attendues sont présentes."""
    missing = set(PCA_FEATURES + ["Amount", TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"[data] Colonnes manquantes : {missing}")


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Crée les features dérivées et les interactions entre les top features.

    Features créées
    ---------------
    Amount_log   : log(1 + Amount)   — normalise la distribution log-normale
    Amount_sqrt  : √Amount           — compresse les outliers
    Hour         : heure de la journée (0–23) depuis Time
    IsNight      : 1 si 22h ≤ heure ≤ 6h, sinon 0
    inter_Vi_Vj  : produit croisé entre le top-5 features (corrélation avec Class)

    Retourne
    --------
    (df enrichi, liste des noms de features)
    """
    df = df.copy()

    # Features temporelles et de montant
    df["Amount_log"]  = np.log1p(df["Amount"])
    df["Amount_sqrt"] = np.sqrt(df["Amount"])
    df["Hour"]        = (df["Time"] // 3600) % 24
    df["IsNight"]     = ((df["Hour"] >= 22) | (df["Hour"] <= 6)).astype(int)

    # Top features corrélées avec la cible → interactions
    corr_with_target = df[PCA_FEATURES + [TARGET]].corr()[TARGET].abs()
    top5 = corr_with_target.drop(TARGET).nlargest(5).index.tolist()

    interaction_cols = []
    for i in range(len(top5)):
        for j in range(i + 1, len(top5)):
            col = f"inter_{top5[i]}_{top5[j]}"
            df[col] = df[top5[i]] * df[top5[j]]
            interaction_cols.append(col)

    feature_cols = (
        PCA_FEATURES
        + ["Amount_log", "Amount_sqrt", "Hour", "IsNight"]
        + interaction_cols
    )

    print(f"[data] Features : {len(feature_cols)} "
          f"({len(interaction_cols)} interactions)")
    return df, feature_cols


# ─────────────────────────────────────────────────────────────────────────────
# Split & Scaling
# ─────────────────────────────────────────────────────────────────────────────

def split_and_scale(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple:
    """
    Split stratifié train/test + normalisation RobustScaler.

    RobustScaler est préféré à StandardScaler car il utilise la médiane et
    l'IQR, ce qui le rend résistant aux outliers présents dans Amount.

    Retourne
    --------
    X_train_sc, X_test_sc, y_train, y_test, scaler, feature_cols
    """
    X = df[feature_cols].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    scaler = RobustScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[data] Train : {X_train_sc.shape}  |  Test : {X_test_sc.shape}")
    print(f"[data] Fraudes — train : {y_train.sum()}  |  test : {y_test.sum()}")

    return X_train_sc, X_test_sc, y_train, y_test, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Rééchantillonnage
# ─────────────────────────────────────────────────────────────────────────────

RESAMPLERS = {
    "smote":    lambda: SMOTE(random_state=RANDOM_SEED, k_neighbors=5),
    "adasyn":   lambda: ADASYN(random_state=RANDOM_SEED),
    "nearmiss": lambda: NearMiss(version=1),
}


def resample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str = "smote",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Rééchantillonne les données d'entraînement.

    Paramètres
    ----------
    strategy : "smote" | "adasyn" | "nearmiss"

    Retourne
    --------
    (X_resampled, y_resampled)
    """
    if strategy not in RESAMPLERS:
        raise ValueError(f"Stratégie inconnue : {strategy}. "
                         f"Choisir parmi {list(RESAMPLERS)}")

    resampler = RESAMPLERS[strategy]()
    X_res, y_res = resampler.fit_resample(X_train, y_train)
    print(f"[data] {strategy.upper()} : {X_res.shape}  "
          f"|  ratio fraudes = {y_res.mean():.3f}")
    return X_res, y_res
