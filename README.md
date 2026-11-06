# Titanic_Project

Projet de groupe ECES : prédire la survie des passagers du Titanic (classification binaire, variable cible `Survived`).
Il comprend un modèle de classification (scikit-learn), une API FastAPI et une application Streamlit.

## Structure du dépôt

| Dossier | Rôle |
|---|---|
| `data/` | jeu de données `titanic.csv` (891 passagers) |
| `src/` | code partagé : config, données, entraînement, prédiction, évaluation |
| `models/` | modèle sauvegardé (`titanic_model.joblib`) |
| `reports/` | métriques, matrice de confusion, courbe ROC, interprétation |
| `tests/` | tests pytest |
| `notebooks/` | exploration des données (`eda.ipynb`) |
| `streamlit_app/` | application Streamlit |
| `api/` | API FastAPI |
| `scripts/` | scripts utilitaires |
| `docs/` | feuille de route du projet |

## Installation

```bash
git clone <url-du-depot>
cd Titanic_Project
uv sync
```

Python 3.12 et [uv](https://docs.astral.sh/uv/) sont nécessaires. Aucune autre installation n'est requise.

## Commandes

```bash
uv run python -m src.train                 # entraîner les modèles et sauvegarder le meilleur
uv run python -m src.evaluate              # évaluer le modèle sur le jeu de test
uv run uvicorn api.main:app --reload       # lancer l'API (http://localhost:8000/docs)
uv run streamlit run streamlit_app/app.py  # lancer l'application (http://localhost:8501)
uv run pytest                              # lancer les tests
```

## Démarche

**Préparation des données.** Elle est intégrée dans le pipeline scikit-learn de `src/train.py` : âge manquant remplacé par la médiane, port manquant par la valeur la plus fréquente, sexe et port encodés en 0/1, variables numériques mises à l'échelle. Les statistiques sont calculées sur le seul jeu d'entraînement, pour éviter toute fuite d'information vers le test. Le notebook `notebooks/eda.ipynb` explore les données et justifie ces choix.

**Modèle.** Quatre modèles sont comparés par validation croisée stratifiée à 5 plis sur le jeu d'entraînement : régression logistique, arbre de décision, forêt aléatoire et KNN. Le meilleur, un arbre de décision de profondeur 5, est sauvegardé avec Joblib. Le pipeline complet (préparation et modèle) est sauvegardé ensemble : l'API et l'application acceptent donc des valeurs brutes.

**Découpage.** 80 % entraînement, 20 % test, stratifié, `random_state = 42`. Le jeu de test n'est utilisé qu'à l'évaluation.

## Résultats

Validation croisée (entraînement) : exactitude d'environ **80,3 %**. Évaluation sur le jeu de test (179 passagers) :

| Exactitude | Précision | Rappel | F1 | ROC AUC |
|---|---|---|---|---|
| 76,5 % | 75,5 % | 58,0 % | 65,6 % | 79,7 % |

Matrice de confusion : 97 décédés bien classés, 13 faux survivants, 29 survivants manqués, 40 survivants bien classés. Le point faible du modèle est le rappel : il détecte environ 58 % des survivants. L'écart entre validation croisée et test (environ 3,8 points) reste modéré. Détails dans `reports/interpretation.md`.

## API

- `GET /health` : vérifie que le modèle est chargé.
- `POST /predict` : prédit la survie d'un passager.

Exemple de requête :

```json
{"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0, "fare": 100, "embarked": "C"}
```

Réponse : `{"survived": true, "probability": 1.0}`. L'âge est facultatif (`null` accepté). Les valeurs invalides renvoient une erreur 422.

## Qui fait quoi

| Membre | Rôle | Fichiers / livrables |
|---|---|---|
| Membre 1 | Données | `notebooks/eda.ipynb` (exploration et choix de préparation) |
| Membre 2 | Modèle | `src/train.py`, 4 modèles comparés en validation croisée |
| Membre 3 | Évaluation | `src/evaluate.py`, métriques, matrice de confusion, courbe ROC, `reports/interpretation.md` |
| Membre 4 | Joblib, tests | test de sauvegarde et rechargement du modèle (`tests/test_base.py`) |
| Membre 5 | Application Streamlit | `streamlit_app/app.py` |
| Membre 6 | API FastAPI | `api/main.py`, `tests/test_api.py` |

Détail de ce que chaque membre reçoit et livre : `docs/FEUILLE_DE_ROUTE.md`