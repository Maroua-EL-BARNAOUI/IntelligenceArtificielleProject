# 🔐 Projet IA — Classification Robuste en Environnement Critique

> Détection de Fraude Bancaire · Credit Card Fraud Detection  
> Déséquilibre extrême (0.22%) · XGBoost Focal Loss · Optuna TPE · SHAP

---

## 📋 Description

Ce projet implémente un système de classification robuste pour détecter les fraudes bancaires dans un contexte de **déséquilibre extrême des classes** (ratio 461:1). Il couvre l'intégralité du pipeline ML :

| Étape | Contenu |
|-------|---------|
| **EDA** | Matrice de corrélation, VIF, analyse du montant |
| **Feature Engineering** | Variables temporelles, log/sqrt, interactions croisées |
| **Modèles** | LR Elastic Net, Random Forest + Proximité, XGBoost Focal Loss |
| **Évaluation** | F1-Macro, MCC, AUPRC, Brier Score (pas d'Accuracy) |
| **Calibration** | Platt Scaling, Reliability Diagrams |
| **Interprétabilité** | SHAP TreeExplainer (Bar, Beeswarm, Waterfall) |

### Résultats clés

| Modèle | AUPRC | F1-Macro | MCC | Brier |
|--------|-------|----------|-----|-------|
| LR (class_weight) | 0.0734 | 0.4478 | 0.0451 | 0.1552 |
| LR + SMOTE | 0.0790 | 0.4481 | 0.0000 | 0.1501 |
| Random Forest | 0.0266 | 0.5214 | 0.0750 | 0.0316 |
| XGBoost scale_pos_w. | 0.1492 | 0.5463 | 0.1552 | 0.0180 |
| **🏆 XGBoost Focal Loss** | **0.2559** | **0.5804** | **0.2747** | **0.0058** |

---

## 🗂️ Structure du Projet

```
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparamètres, chemins, constantes
│   ├── data.py            # Chargement, feature engineering, rééchantillonnage
│   ├── models.py          # LR, RF, XGBoost (Focal Loss + Optuna)
│   ├── evaluation.py      # Métriques, calibration, courbes PR
│   └── visualization.py   # Toutes les visualisations matplotlib
├── notebooks/
│   └── Projet_IA_Fraude_Bancaire.ipynb
├── figures/               # Visualisations générées (PNG)
├── data/                  # Dataset CSV (voir "Données" ci-dessous)
├── models/                # Modèles sérialisés (.pkl) — gitignored
├── main.py                # Pipeline complet
├── requirements.txt
└── .gitignore
```

---

## 🚀 Installation & Utilisation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<ton-username>/projet-ia-fraude.git
cd projet-ia-fraude
```

### 2. Créer l'environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

### 3. Données

**Option A — Dataset réel (recommandé)**  
Télécharger depuis Kaggle :  
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
Placer le fichier dans `data/creditcard.csv`.

**Option B — Dataset synthétique (automatique)**  
Si `data/creditcard.csv` est absent, un dataset synthétique (~284 000 lignes, 0.22% de fraudes) est généré automatiquement :

```bash
python -c "from src.data import generate_dataset; generate_dataset()"
```

### 4. Exécuter le pipeline

```bash
# Pipeline complet (EDA → Modèles → Évaluation → SHAP)
python main.py

# Sans optimisation Optuna (plus rapide)
python main.py --skip-optuna

# Mode rapide (10 essais Optuna au lieu de 30)
python main.py --quick

# Avec un chemin de données personnalisé
python main.py --data /path/to/creditcard.csv
```

### 5. Notebook Jupyter

```bash
jupyter notebook notebooks/Projet_IA_Fraude_Bancaire.ipynb
```

---

## 🏗️ Architecture des Modules

### `src/config.py`
Centralise **tous** les hyperparamètres et constantes. Modifier ici pour changer le comportement global.

```python
XGB_BASE_PARAMS = {
    "n_estimators": 150,
    "max_depth": 5,
    "learning_rate": 0.1,
    ...
}
FOCAL_GAMMA = 2.0   # paramètre de focalisation
FOCAL_ALPHA = 0.75  # poids pour la classe fraude
```

### `src/models.py`

Trois classes principales :

```python
# Régression Logistique Elastic Net
lr = LogisticModel(class_weight="balanced")
lr.fit(X_train, y_train)

# Random Forest + matrice de proximité
rf = RandomForestModel()
rf.fit(X_train, y_train)
proximity, outlier_scores = rf.compute_proximity(X_test, y_test)

# XGBoost avec Focal Loss + Optuna
xgb = XGBoostModel(mode="focal")
xgb.optimize(X_train, y_train, n_trials=30)  # Recherche Bayésienne
xgb.fit(X_train, y_train)
```

### Focal Loss (Apprentissage Sensible au Coût)

```
FL(p_t) = -α_t · (1 − p_t)^γ · log(p_t)

γ = 0 → Cross-Entropy classique
γ > 0 → réduit la contribution des exemples faciles
α = 0.75 → 3× plus d'importance aux fraudes
```

### Recherche Bayésienne (Optuna TPE)

L'optimiseur TPE (Tree-structured Parzen Estimator) modélise `P(hyperparamètres | bonne valeur)` pour explorer intelligemment l'espace de recherche, bien plus efficacement qu'un GridSearch exhaustif.

| Hyperparamètre | Plage | Justification |
|----------------|-------|---------------|
| `max_depth` | [3, 8] | Profondeurs > 8 sur-ajustent sur classes déséquilibrées |
| `learning_rate` | [0.01, 0.3] log | Compromis vitesse/précision |
| `subsample` | [0.6, 1.0] | Stochastisation réduit la variance |
| `reg_alpha/lambda` | [1e-3, 5] log | Régularisation L1/L2 |

---

## 📊 Métriques Utilisées

L'Accuracy est **volontairement exclue** (un modèle naïf "tout légitime" atteint 99.78%).

| Métrique | Raison |
|----------|--------|
| **F1-Macro** | Équilibre précision/rappel, traite chaque classe également |
| **MCC** | Non biaisé par le déséquilibre, utilise les 4 éléments de la matrice |
| **AUPRC** | Robuste au déséquilibre, focus sur la classe positive |
| **Brier Score** | Mesure la qualité de calibration des probabilités |

---

## 🔬 Techniques de Gestion du Déséquilibre

| Niveau | Technique | Implémentation |
|--------|-----------|----------------|
| **Algorithmique** | `class_weight="balanced"` | LR, RF |
| **Algorithmique** | `scale_pos_weight` | XGBoost |
| **Algorithmique** | Focal Loss | XGBoost custom |
| **Données** | SMOTE | `imblearn` |
| **Données** | ADASYN | `imblearn` |
| **Données** | NearMiss | `imblearn` |

---

## 📦 Dépendances

```
scikit-learn >= 1.3
xgboost >= 2.0
imbalanced-learn >= 0.12
optuna >= 3.5
shap >= 0.45
statsmodels >= 0.14
numpy, pandas, matplotlib, seaborn
```

---

## 📄 Licence

MIT — libre d'utilisation et de modification.

---

*Projet réalisé dans le cadre du cours Intelligence Artificielle.*
