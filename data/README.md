# Données

## Dataset réel (recommandé)

Télécharger depuis Kaggle :
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Placer le fichier `creditcard.csv` dans ce dossier.

## Dataset synthétique (automatique)

Si `creditcard.csv` est absent, le script `main.py` génère automatiquement
un dataset synthétique de 284 807 lignes avec 0.22% de fraudes.

```bash
python -c "from src.data import generate_dataset; generate_dataset()"
```
